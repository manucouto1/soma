//! `fit` and `forward` are one walk.
//!
//! Fitting used to have an execution loop of its own: it flattened the
//! plan into a list of node ids and ran them in order. That loop was
//! filter-only and topology-blind, so four things were true at once —
//! a graph containing a step could not be fitted at all, a fan-out was
//! fitted as a chain, a branch fitted both arms, and no cache event was
//! ever emitted while fitting. These check that none of them is true now.

use somatize_compiler::{CompileMode, SimpleNodeRegistry, compile};
use somatize_core::agentic::effect::{
    Effect, EffectResult, LlmRequest, LlmResponse, StopReason, Usage,
};
use somatize_core::agentic::message::Message;
use somatize_core::cache::CacheKey;
use somatize_core::data::keys::node_of_state_key;
use somatize_core::data::value::Value;
use somatize_core::error::Result;
use somatize_core::graph::filter::{Distribution, Filter, FilterKind, FilterMeta, StreamMode};
use somatize_core::graph::step::{Step, StepCtx, StepMeta, Transition};
use somatize_core::graph::{Edge, Graph, Node};
use somatize_core::tracking::EventSink;
use somatize_core::tracking::event::Event;
use somatize_runtime::agentic::{EffectDriver, EffectHandler, EffectJournal};
use somatize_runtime::cache::MemoryCache;
use somatize_runtime::cache::fs_store::FsActionStore;
use somatize_runtime::execution::executor::GraphInfo;
use somatize_runtime::execution::node_catalog::NodeCatalog;
use somatize_runtime::execution::runner::{LocalRunner, RunContext, Runner};
use somatize_runtime::tracking::event_bus::EventBus;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// ── Test doubles ──

/// Learns the mean of its input, then subtracts it.
struct Centre;

impl Filter for Centre {
    fn config_hash(&self) -> CacheKey {
        CacheKey::from_parts(&[b"Centre"])
    }
    fn fit(&self, x: &Value, _y: Option<&Value>) -> Result<Value> {
        let (data, _) = x.as_tensor().unwrap_or((&[], &[]));
        let mean = if data.is_empty() {
            0.0
        } else {
            data.iter().sum::<f64>() / data.len() as f64
        };
        Ok(Value::json(serde_json::json!({ "mean": mean })))
    }
    fn forward(&self, x: &Value, state: &Value) -> Result<Value> {
        let (data, shape) = x.as_tensor().unwrap_or((&[], &[]));
        let mean = state
            .as_json()
            .and_then(|j| j["mean"].as_f64())
            .unwrap_or(0.0);
        Ok(Value::tensor(
            data.iter().map(|v| v - mean).collect(),
            shape.to_vec(),
        ))
    }
    fn meta(&self) -> FilterMeta {
        meta("Centre", FilterKind::Trainable)
    }
}

/// Stateless doubler, so a node's effect on the value is visible.
struct Double;

impl Filter for Double {
    fn config_hash(&self) -> CacheKey {
        CacheKey::from_parts(&[b"Double"])
    }
    fn fit(&self, _x: &Value, _y: Option<&Value>) -> Result<Value> {
        Ok(Value::Empty)
    }
    fn forward(&self, x: &Value, _state: &Value) -> Result<Value> {
        let (data, shape) = x.as_tensor().unwrap_or((&[], &[]));
        Ok(Value::tensor(
            data.iter().map(|v| v * 2.0).collect(),
            shape.to_vec(),
        ))
    }
    fn meta(&self) -> FilterMeta {
        meta("Double", FilterKind::Stateless)
    }
}

fn meta(name: &str, kind: FilterKind) -> FilterMeta {
    FilterMeta {
        name: name.into(),
        kind,
        cacheable: true,
        differentiable: false,
        deterministic: true,
        stream_mode: StreamMode::FixedState,
        distribution: Distribution::Local,
        input_schema: None,
        output_schema: None,
    }
}

/// Passes its input through, so a selector can name an arm.
struct Echo;

impl Filter for Echo {
    fn config_hash(&self) -> CacheKey {
        CacheKey::from_parts(&[b"Echo"])
    }
    fn fit(&self, _x: &Value, _y: Option<&Value>) -> Result<Value> {
        Ok(Value::Empty)
    }
    fn forward(&self, x: &Value, _state: &Value) -> Result<Value> {
        Ok(x.clone())
    }
    fn meta(&self) -> FilterMeta {
        meta("Echo", FilterKind::Stateless)
    }
}

/// Answers with a fixed reply.
struct FakeLlm;

impl EffectHandler for FakeLlm {
    fn handles(&self, effect: &Effect) -> bool {
        matches!(effect, Effect::Llm(_))
    }
    fn perform(&self, _effect: &Effect) -> Result<EffectResult> {
        Ok(EffectResult::Llm(LlmResponse {
            message: Message::assistant("ok"),
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
            model: None,
        }))
    }
}

/// Asks once, answers with the reply.
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
                "m",
                vec![Message::user("hi")].into(),
            ))])),
            Some(EffectResult::Llm(r)) => Ok(Transition::Done(Value::text(r.message.text()))),
            Some(other) => Err(somatize_core::error::SomaError::Execution {
                node_id: ctx.node_id.to_string(),
                message: format!("unexpected: {other:?}"),
            }),
        }
    }
}

/// Records every event, so a test can ask what a run emitted.
#[derive(Default)]
struct Recorder {
    events: Mutex<Vec<Event>>,
}

impl EventSink for Recorder {
    fn record(&self, event: &Event) {
        self.events.lock().unwrap().push(event.clone());
    }
    fn flush(&self) {}
}

impl Recorder {
    fn names(&self) -> Vec<String> {
        self.events
            .lock()
            .unwrap()
            .iter()
            .map(|e| {
                format!("{e:?}")
                    .split_whitespace()
                    .next()
                    .unwrap()
                    .to_string()
            })
            .collect()
    }
}

// ── Harness ──

fn registry(nodes: &[(&str, FilterMeta)]) -> SimpleNodeRegistry {
    let mut reg = SimpleNodeRegistry::new();
    for (id, m) in nodes {
        reg.register_meta(*id, m.clone(), CacheKey::from_parts(&[id.as_bytes()]));
    }
    reg
}

fn states(produced: &HashMap<String, Value>) -> Vec<String> {
    let mut ids: Vec<String> = produced
        .keys()
        .filter_map(|k| node_of_state_key(k))
        .map(str::to_string)
        .collect();
    ids.sort();
    ids
}

// ── Tests ──

/// A graph that contains a step can be fitted.
///
/// The old fit walk resolved every node id through the filter half of the
/// catalog, so a step id came back as `NodeNotFound` and the whole fit
/// failed — a graph you could `run` was a graph you could not `fit`.
#[test]
fn a_graph_containing_a_step_can_be_fitted() {
    let mut g = Graph::new();
    g.add_node(Node::filter_with_id("centre", "centre"));
    g.add_node(Node::step("ask", "AskOnce"));
    g.add_edge(Edge::data("e1", "centre", "ask"));

    let reg = registry(&[("centre", Centre.meta())]);
    let plan = compile(&g, &reg, CompileMode::NoCache, None)
        .expect("compiles")
        .plan;

    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(FsActionStore::new(dir.path()).unwrap());
    let driver =
        EffectDriver::new(EffectJournal::new(store.clone(), store)).with_handler(Arc::new(FakeLlm));

    let mut catalog = NodeCatalog::new();
    catalog.register("centre", Box::new(Centre));
    catalog.register_step("ask", Box::new(AskOnce));

    let bus = Arc::new(EventBus::new(256));
    let cache = MemoryCache::default();
    let ctx = RunContext::new(&catalog, &cache, &bus, "fit_run", GraphInfo::from_graph(&g))
        .with_driver(driver);

    let (_out, produced) = LocalRunner
        .fit(&plan, &ctx, &Value::tensor(vec![1.0, 3.0], vec![2]), None)
        .expect("fitting a graph with a step must work");

    assert_eq!(states(&produced), vec!["centre"], "the filter was fitted");
    assert!(produced.contains_key("ask"), "the step produced an output");
}

/// A fan-out is fitted as a fan-out.
///
/// Both branches read the same source, so both must be fitted against
/// *its* output. Flattened into a chain, the second branch was fitted on
/// the first branch's output instead.
#[test]
fn both_branches_of_a_fan_out_are_fitted_from_their_own_predecessor() {
    let mut g = Graph::new();
    g.add_node(Node::filter_with_id("src", "src"));
    g.add_node(Node::filter_with_id("left", "left"));
    g.add_node(Node::filter_with_id("right", "right"));
    g.add_edge(Edge::data("e1", "src", "left"));
    g.add_edge(Edge::data("e2", "src", "right"));

    let reg = registry(&[
        ("src", Double.meta()),
        ("left", Centre.meta()),
        ("right", Centre.meta()),
    ]);
    let plan = compile(&g, &reg, CompileMode::NoCache, None)
        .expect("compiles")
        .plan;

    let mut catalog = NodeCatalog::new();
    catalog.register("src", Box::new(Double));
    catalog.register("left", Box::new(Centre));
    catalog.register("right", Box::new(Centre));

    let bus = Arc::new(EventBus::new(256));
    let cache = MemoryCache::default();
    let ctx = RunContext::new(&catalog, &cache, &bus, "fan_out", GraphInfo::from_graph(&g));

    let (_out, produced) = LocalRunner
        .fit(&plan, &ctx, &Value::tensor(vec![1.0, 2.0], vec![2]), None)
        .expect("fit");

    assert_eq!(states(&produced), vec!["left", "right"]);

    // src doubles [1,2] → [2,4]; both branches centre THAT, so both learn
    // mean 3. A chain would have fitted `right` on `left`'s output (mean 0).
    for id in ["left", "right"] {
        let state = produced
            .get(&somatize_core::data::keys::state_key(id))
            .expect("state");
        let mean = state.as_json().and_then(|j| j["mean"].as_f64()).unwrap();
        assert_eq!(mean, 3.0, "`{id}` was fitted on the wrong predecessor");
    }
}

/// Fitting emits the same cache events a run does.
///
/// The old walk emitted `NodeStarted`/`NodeCompleted` by hand and no cache
/// event at all, so a tracked fit reported every node as a miss-free
/// mystery: the report's cache panel was empty for training runs.
#[test]
fn a_fit_reports_cache_activity_like_a_run() {
    let mut g = Graph::new();
    g.add_node(Node::filter_with_id("centre", "centre"));
    g.add_node(Node::filter_with_id("double", "double"));
    g.add_edge(Edge::data("e1", "centre", "double"));

    let reg = registry(&[("centre", Centre.meta()), ("double", Double.meta())]);
    let plan = compile(&g, &reg, CompileMode::NoCache, None)
        .expect("compiles")
        .plan;

    let mut catalog = NodeCatalog::new();
    catalog.register("centre", Box::new(Centre));
    catalog.register("double", Box::new(Double));

    let cache = MemoryCache::default();
    let x = Value::tensor(vec![1.0, 3.0], vec![2]);

    let first = Arc::new(Recorder::default());
    let bus = Arc::new(EventBus::new(256));
    bus.add_sink(first.clone());
    let ctx = RunContext::new(&catalog, &cache, &bus, "fit1", GraphInfo::from_graph(&g));
    LocalRunner.fit(&plan, &ctx, &x, None).expect("fit");

    let names = first.names();
    assert!(
        names.iter().any(|n| n.contains("NodeCacheMiss")),
        "a cold fit must report misses, got {names:?}"
    );

    // Same everything: the second fit should be served from cache.
    let second = Arc::new(Recorder::default());
    let bus2 = Arc::new(EventBus::new(256));
    bus2.add_sink(second.clone());
    let ctx2 = RunContext::new(&catalog, &cache, &bus2, "fit2", GraphInfo::from_graph(&g));
    LocalRunner.fit(&plan, &ctx2, &x, None).expect("fit again");

    let names = second.names();
    assert!(
        names.iter().any(|n| n.contains("NodeCacheHit")),
        "a warm fit must report hits, got {names:?}"
    );
}

/// A fit and a run of the same graph agree on what each node produced.
///
/// Two walks meant two answers were possible; one walk means this is
/// true by construction, and the test is what keeps it that way.
#[test]
fn fit_and_forward_agree_on_every_node_output() {
    let mut g = Graph::new();
    g.add_node(Node::filter_with_id("centre", "centre"));
    g.add_node(Node::filter_with_id("double", "double"));
    g.add_edge(Edge::data("e1", "centre", "double"));

    let reg = registry(&[("centre", Centre.meta()), ("double", Double.meta())]);
    let plan = compile(&g, &reg, CompileMode::NoCache, None)
        .expect("compiles")
        .plan;

    let mut catalog = NodeCatalog::new();
    catalog.register("centre", Box::new(Centre));
    catalog.register("double", Box::new(Double));

    let bus = Arc::new(EventBus::new(256));
    let cache = MemoryCache::default();
    let x = Value::tensor(vec![1.0, 3.0], vec![2]);

    let ctx = RunContext::new(&catalog, &cache, &bus, "fit", GraphInfo::from_graph(&g));
    let (fit_out, produced) = LocalRunner.fit(&plan, &ctx, &x, None).expect("fit");

    // Carry the learned states over, exactly as a session does.
    for (key, state) in &produced {
        if let Some(node) = node_of_state_key(key) {
            catalog.try_set_state(node.to_string(), state.clone()).ok();
        }
    }

    let ctx2 = RunContext::new(&catalog, &cache, &bus, "fwd", GraphInfo::from_graph(&g));
    let fwd_out = LocalRunner.forward(&plan, &ctx2, &x).expect("forward");

    assert_eq!(fit_out, fwd_out);
}

/// A fit trains the arm that ran, and only that one.
///
/// The old walk took `plan.node_ids()` — which lists *every* arm — and ran
/// them in sequence, so fitting a branch trained the path not taken as
/// well, on whatever value happened to be lying around.
#[test]
fn a_branch_fits_only_the_arm_that_runs() {
    let mut g = Graph::new();
    g.add_node(Node::branch("pick"));
    g.add_node(Node::filter_with_id("taken", "taken"));
    g.add_node(Node::filter_with_id("skipped", "skipped"));
    g.add_edge(Edge::control("c1", "pick", "taken").with_label("taken"));
    g.add_edge(Edge::control("c2", "pick", "skipped").with_label("skipped"));

    let plan = somatize_compiler::ExecutionPlan::Branch {
        node_id: "pick".into(),
        arms: vec![
            (
                "taken".into(),
                somatize_compiler::ExecutionPlan::Execute {
                    node_id: "taken".into(),
                },
            ),
            (
                "skipped".into(),
                somatize_compiler::ExecutionPlan::Execute {
                    node_id: "skipped".into(),
                },
            ),
        ],
    };

    // The selector is a node like any other: it computes the label. Here
    // it simply echoes the input, so the input names the arm.
    let mut catalog = NodeCatalog::new();
    catalog.register("pick", Box::new(Echo));
    catalog.register("taken", Box::new(Centre));
    catalog.register("skipped", Box::new(Centre));

    let bus = Arc::new(EventBus::new(256));
    let cache = MemoryCache::default();
    let ctx = RunContext::new(
        &catalog,
        &cache,
        &bus,
        "branch_fit",
        GraphInfo::from_graph(&g),
    );

    let (_out, produced) = LocalRunner
        .fit(&plan, &ctx, &Value::text("taken"), None)
        .expect("fit");

    assert_eq!(
        states(&produced),
        vec!["taken"],
        "the arm that did not run must not be fitted"
    );
}
