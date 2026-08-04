//! `GraphSession` drives steps.
//!
//! The session is the primary orchestrator, and until it could carry an
//! [`EffectDriver`] a pure-Rust caller could not run a graph that mixed
//! filters and steps — only the Python bindings could, because only they
//! built the executor contexts by hand. These tests pin the seam at the
//! session level: `with_driver` once, and `run`, `fit` and `forward` all
//! reach the steps.

use somatize_compiler::CompileMode;
use somatize_core::cache::CacheKey;
use somatize_core::effect::{Effect, EffectResult, LlmRequest, LlmResponse, StopReason};
use somatize_core::error::Result;
use somatize_core::filter::{Distribution, Filter, FilterKind, FilterMeta, StreamMode};
use somatize_core::graph::{Edge, Graph, Node};
use somatize_core::message::Message;
use somatize_core::step::{Step, StepCtx, StepMeta, Transition};
use somatize_core::value::Value;
use somatize_runtime::GraphSession;
use somatize_runtime::cache::fs_store::FsActionStore;
use somatize_runtime::effects::{EffectDriver, EffectHandler, EffectJournal};
use somatize_runtime::node_catalog::NodeCatalog;
use std::sync::Arc;

// ── Test doubles ──

/// Uppercases text, so a filter's effect on the value is visible.
struct Shout;

impl Filter for Shout {
    fn config_hash(&self) -> CacheKey {
        CacheKey::from_parts(&[b"Shout"])
    }
    fn fit(&self, _x: &Value, _y: Option<&Value>) -> Result<Value> {
        Ok(Value::Empty)
    }
    fn forward(&self, x: &Value, _state: &Value) -> Result<Value> {
        Ok(Value::text(x.as_text().unwrap_or_default().to_uppercase()))
    }
    fn meta(&self) -> FilterMeta {
        FilterMeta {
            name: "Shout".into(),
            kind: FilterKind::Stateless,
            cacheable: false,
            differentiable: false,
            deterministic: true,
            stream_mode: StreamMode::FixedState,
            distribution: Distribution::Local,
            input_schema: None,
            output_schema: None,
        }
    }
}

/// Answers with a reply derived from the question.
struct FakeLlm;

impl EffectHandler for FakeLlm {
    fn handles(&self, effect: &Effect) -> bool {
        matches!(effect, Effect::Llm(_))
    }
    fn perform(&self, effect: &Effect) -> Result<EffectResult> {
        let Effect::Llm(req) = effect else {
            unreachable!()
        };
        let asked = req.messages.last().map(|m| m.text()).unwrap_or_default();
        Ok(EffectResult::Llm(LlmResponse {
            message: Message::assistant(format!("answer to: {asked}")),
            stop_reason: StopReason::EndTurn,
            usage: Default::default(),
            model: None,
        }))
    }
}

/// Asks the model once with whatever came in, returns the reply as text.
struct AskOnce;

impl Step for AskOnce {
    fn config_hash(&self) -> CacheKey {
        CacheKey::from_parts(&[b"AskOnce"])
    }
    fn meta(&self) -> StepMeta {
        StepMeta::new("AskOnce")
    }
    fn poll(&self, ctx: &StepCtx<'_>) -> Result<Transition> {
        match ctx.result() {
            None => Ok(Transition::Await(vec![Effect::Llm(LlmRequest::new(
                "claude-opus-5",
                vec![Message::user(ctx.input.as_text().unwrap_or_default())].into(),
            ))])),
            Some(EffectResult::Llm(r)) => Ok(Transition::Done(Value::text(r.message.text()))),
            Some(other) => Err(somatize_core::error::SomaError::Execution {
                node_id: ctx.node_id.to_string(),
                message: format!("unexpected effect result: {other:?}"),
            }),
        }
    }
}

// ── Harness ──

fn driver(dir: &tempfile::TempDir) -> EffectDriver {
    let store = Arc::new(FsActionStore::new(dir.path()).unwrap());
    let journal = EffectJournal::new(store.clone(), store);
    EffectDriver::new(journal).with_handler(Arc::new(FakeLlm))
}

fn catalog() -> NodeCatalog {
    let mut catalog = NodeCatalog::new();
    catalog.register("prep", Box::new(Shout));
    catalog.register("shout", Box::new(Shout));
    catalog.register_step("ask", Box::new(AskOnce));
    catalog
}

/// `prep(filter) -> ask(step) -> shout(filter)`
fn mixed_graph() -> Graph {
    let mut g = Graph::new();
    g.add_node(Node::filter_with_id("prep", "prep"));
    g.add_node(Node::step("ask", "AskOnce"));
    g.add_node(Node::filter_with_id("shout", "shout"));
    g.add_edge(Edge::data("e1", "prep", "ask"));
    g.add_edge(Edge::data("e2", "ask", "shout"));
    g
}

// ── Tests ──

/// `forward` on a mixed graph: the filter's output reaches the step, the
/// step's answer reaches the next filter. One `with_driver`, nothing else.
#[test]
fn graph_session_runs_a_mixed_graph() {
    let dir = tempfile::tempdir().unwrap();
    let session = GraphSession::new(mixed_graph(), catalog()).with_driver(driver(&dir));

    let out = session.forward(&Value::text("hi")).unwrap();
    assert_eq!(out.as_text(), Some("ANSWER TO: HI"));
}

/// `run` builds its context on a different path than `forward`; both must
/// carry the driver.
#[test]
fn graph_session_run_drives_steps() {
    let dir = tempfile::tempdir().unwrap();
    let mut g = Graph::new();
    g.add_node(Node::step("ask", "AskOnce"));

    let mut session = GraphSession::new(g, catalog()).with_driver(driver(&dir));
    let outputs = session.run(CompileMode::NoCache).unwrap();
    assert!(
        outputs.get("ask").and_then(Value::as_text).is_some(),
        "the step should have produced an answer, got {outputs:?}"
    );
}

/// `fit` on a mixed graph reaches the step too — the fit path used to be
/// the one that stopped at the first step for want of a driver.
#[test]
fn graph_session_fit_drives_steps() {
    let dir = tempfile::tempdir().unwrap();
    let mut session = GraphSession::new(mixed_graph(), catalog()).with_driver(driver(&dir));

    let outputs = session.fit(&Value::text("hi"), None).unwrap();
    assert_eq!(
        outputs.get("shout").and_then(Value::as_text),
        Some("ANSWER TO: HI"),
        "the whole chain should have run, got {outputs:?}"
    );
}

/// Without a driver the failure must keep naming the fix — that error is
/// load-bearing, and attaching a driver by default would hide the choice.
#[test]
fn graph_session_without_a_driver_names_the_fix() {
    let session = GraphSession::new(mixed_graph(), catalog());

    let err = session.forward(&Value::text("hi")).unwrap_err();
    assert!(
        err.to_string().contains("effect driver"),
        "the error should say what is missing: {err}"
    );
}
