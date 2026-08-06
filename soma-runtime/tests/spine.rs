//! Soma in one file, executable.
//!
//! The Internals section describes this system across eleven pages and some
//! seven hundred `file:line` anchors. Prose that precise goes stale silently:
//! a rename does not make a sentence wrong, it makes it *almost* right, and
//! nothing fails. This file makes the central claims fail out loud instead.
//!
//! It is deliberately not a second implementation. There is exactly one
//! execution site, and a "minimal Soma" written beside it would be a second
//! thing to keep true — which is precisely how the stream path drifted away
//! from `run_node` while claiming to share it. This file *asserts* the one
//! implementation.
//!
//! # The spine
//!
//! ```text
//!   Graph              nodes and edges, no behaviour
//!     │
//!   compile()          → ExecutionPlan  (+ schema and topology diagnostics)
//!     │
//!   LocalRunner        → Context, built from the plan AND the topology
//!     │
//!   execute()          recurses over the plan; every leaf reaches ↓
//!     │
//!   run_node()         the one execution site, for filters and steps alike
//!     │
//!     ├─ output_key    hash(config ‖ state ‖ input) — the memoization guard
//!     ├─ compute_node  catch_unwind(forward | driver.run)
//!     └─ store_output  the value, with provenance
//! ```
//!
//! # The one distinction to internalize
//!
//! **A filter memoizes by content. A step journals by site.**
//!
//! A filter's output is a function of its config, its state and its input, so
//! an identical call *anywhere* may reuse the result — `output_key` does not
//! contain the node id, and [`a_filter_memoizes_by_content_not_by_where_it_sits`]
//! is what says so. A step's effects are not: asking a model the same question
//! twice can give two answers, so an impure effect is keyed by *where and when*
//! — `(run, node, turn, index)` — recorded once and replayed on resume, never
//! re-run. [`a_step_journals_by_site_and_replays_rather_than_re_running`] is
//! what says that.
//!
//! Everything else about caching, resumption and reproducibility follows from
//! those two sentences.

use somatize_compiler::{CompileMode, SimpleNodeRegistry, compile};
use somatize_core::cache::CacheKey;
use somatize_core::effect::{Effect, EffectResult, LlmRequest, LlmResponse, StopReason, Usage};
use somatize_core::error::{Result, SomaError};
use somatize_core::event::Event;
use somatize_core::filter::{Distribution, Filter, FilterKind, FilterMeta, StreamMode};
use somatize_core::graph::{Edge, Graph, Node};
use somatize_core::message::Message;
use somatize_core::step::{Step, StepCtx, StepMeta, Transition};
use somatize_core::tracking::EventSink;
use somatize_core::value::Value;
use somatize_runtime::cache::MemoryCache;
use somatize_runtime::cache::fs_store::FsActionStore;
use somatize_runtime::effects::{EffectDriver, EffectHandler, EffectJournal};
use somatize_runtime::event_bus::EventBus;
use somatize_runtime::executor::GraphInfo;
use somatize_runtime::node_catalog::NodeCatalog;
use somatize_runtime::runner::{LocalRunner, RunContext, Runner};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

// ── Test doubles ─────────────────────────────────────────────────────────────

/// Passes its input through unchanged, but carries a config that changes its
/// identity. That gap — same output, different key — is what makes the early
/// cutoff observable.
struct Tag(&'static str);

impl Filter for Tag {
    fn config_hash(&self) -> CacheKey {
        CacheKey::from_parts(&[b"Tag", self.0.as_bytes()])
    }
    fn fit(&self, _x: &Value, _y: Option<&Value>) -> Result<Value> {
        Ok(Value::Empty)
    }
    fn forward(&self, x: &Value, _state: &Value) -> Result<Value> {
        Ok(x.clone())
    }
    fn meta(&self) -> FilterMeta {
        meta("Tag")
    }
}

/// Doubles every element, so a node's effect on the value is visible.
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
        meta("Double")
    }
}

fn meta(name: &str) -> FilterMeta {
    FilterMeta {
        name: name.into(),
        kind: FilterKind::Stateless,
        cacheable: true,
        differentiable: false,
        deterministic: true,
        stream_mode: StreamMode::FixedState,
        distribution: Distribution::Local,
        input_schema: None,
        output_schema: None,
    }
}

/// Answers differently every time, and counts. A model that returned the same
/// thing twice could not tell replay from re-execution.
#[derive(Default)]
struct CountingLlm {
    calls: AtomicUsize,
}

impl EffectHandler for CountingLlm {
    fn handles(&self, effect: &Effect) -> bool {
        matches!(effect, Effect::Llm(_))
    }
    fn perform(&self, _effect: &Effect) -> Result<EffectResult> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        Ok(EffectResult::Llm(LlmResponse {
            message: Message::assistant(format!("answer-{n}")),
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
            Some(other) => Err(SomaError::Execution {
                node_id: ctx.node_id.to_string(),
                message: format!("unexpected: {other:?}"),
            }),
        }
    }
}

/// Records every event, so a test can ask what a run emitted, in order.
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
    /// The node-level trace: what happened, to which node, in what order.
    fn trace(&self) -> Vec<String> {
        self.events
            .lock()
            .unwrap()
            .iter()
            .filter_map(|e| match e {
                Event::NodeCacheHit { node_id, .. } => Some(format!("{node_id}: hit")),
                Event::NodeCacheMiss { node_id, .. } => Some(format!("{node_id}: miss")),
                Event::NodeStarted { node_id, .. } => Some(format!("{node_id}: started")),
                Event::NodeCompleted { node_id, .. } => Some(format!("{node_id}: completed")),
                _ => None,
            })
            .collect()
    }
}

// ── Harness ──────────────────────────────────────────────────────────────────

/// `tag → mid → out`, the smallest graph with an interior node.
fn chain() -> Graph {
    let mut g = Graph::new();
    g.add_node(Node::filter_with_id("tag", "tag"));
    g.add_node(Node::filter_with_id("mid", "mid"));
    g.add_node(Node::filter_with_id("out", "out"));
    g.add_edge(Edge::data("e1", "tag", "mid"));
    g.add_edge(Edge::data("e2", "mid", "out"));
    g
}

fn plan_for(g: &Graph, nodes: &[&str]) -> somatize_compiler::ExecutionPlan {
    let mut reg = SimpleNodeRegistry::new();
    for id in nodes {
        reg.register_meta(*id, meta(id), CacheKey::from_parts(&[id.as_bytes()]));
    }
    compile(g, &reg, CompileMode::NoCache, None)
        .expect("the chain compiles")
        .plan
}

fn catalog_with(tag: &'static str) -> NodeCatalog {
    let mut catalog = NodeCatalog::new();
    catalog.register("tag", Box::new(Tag(tag)));
    catalog.register("mid", Box::new(Double));
    catalog.register("out", Box::new(Double));
    catalog
}

/// One `forward`, returning what it produced and the trace it emitted.
fn run(
    plan: &somatize_compiler::ExecutionPlan,
    g: &Graph,
    catalog: &NodeCatalog,
    cache: &MemoryCache,
    run_id: &str,
    x: &Value,
) -> (Value, Vec<String>) {
    let recorder = Arc::new(Recorder::default());
    let bus = Arc::new(EventBus::new(256));
    bus.add_sink(recorder.clone());
    let ctx = RunContext::new(catalog, cache, &bus, run_id, GraphInfo::from_graph(g));
    let out = LocalRunner.forward(plan, &ctx, x).expect("forward");
    (out, recorder.trace())
}

// ── The trace ────────────────────────────────────────────────────────────────

/// A cold run is miss → started → completed, once per node, in topological
/// order.
///
/// This is the whole of trace `b`: `execute` recurses to a leaf, `run_node`
/// derives the key with `output_key`, misses, calls `compute_node`, and hands
/// the result to `store_output`. The events are the observable shadow of those
/// three primitives, and their order is the claim.
#[test]
fn a_cold_run_is_miss_started_completed_for_every_node() {
    let g = chain();
    let plan = plan_for(&g, &["tag", "mid", "out"]);
    let catalog = catalog_with("a");
    let cache = MemoryCache::default();

    let (out, trace) = run(
        &plan,
        &g,
        &catalog,
        &cache,
        "cold",
        &Value::tensor(vec![1.0, 2.0], vec![2]),
    );

    assert_eq!(
        trace,
        vec![
            "tag: miss",
            "tag: started",
            "tag: completed",
            "mid: miss",
            "mid: started",
            "mid: completed",
            "out: miss",
            "out: started",
            "out: completed",
        ],
        "the walk is one bracket per node, in topological order"
    );

    // tag passes [1,2] through; mid doubles to [2,4]; out doubles to [4,8].
    let (data, _) = out.as_tensor().expect("a tensor");
    assert_eq!(data, &[4.0, 8.0], "the chain computed what the edges say");
}

/// A warm run is the same walk, served from the cache.
///
/// Nothing about the topology changes: `run_node` still visits all three. What
/// changes is that `output_key` finds what `store_output` wrote, so
/// `compute_node` is never reached — which is why there is no `started`.
#[test]
fn a_warm_run_is_the_same_walk_served_from_the_cache() {
    let g = chain();
    let plan = plan_for(&g, &["tag", "mid", "out"]);
    let catalog = catalog_with("a");
    let cache = MemoryCache::default();
    let x = Value::tensor(vec![1.0, 2.0], vec![2]);

    let (first, _) = run(&plan, &g, &catalog, &cache, "cold", &x);
    let (second, trace) = run(&plan, &g, &catalog, &cache, "warm", &x);

    assert_eq!(
        trace,
        vec!["tag: hit", "mid: hit", "out: hit"],
        "every node was served from cache, and none started"
    );
    assert_eq!(first, second, "the cache returned what was computed");
}

/// An unchanged intermediate cuts off everything downstream.
///
/// The surprising property, and the reason cache keys are resolved at runtime
/// rather than at compile time. `tag`'s config changed, so *its* key changed
/// and it recomputes. But a downstream key is `hash(config ‖ state ‖ input)`
/// where the input term is the **content** of what arrived — and `tag` produced
/// exactly what it produced before. So `mid` and `out` hit.
///
/// Compile-time keys cannot express this: at compile time the only honest thing
/// to say is "an ancestor changed, so everything below is stale".
#[test]
fn an_unchanged_intermediate_cuts_off_everything_downstream() {
    let g = chain();
    let plan = plan_for(&g, &["tag", "mid", "out"]);
    let cache = MemoryCache::default();
    let x = Value::tensor(vec![1.0, 2.0], vec![2]);

    run(&plan, &g, &catalog_with("a"), &cache, "first", &x);

    // Same graph, same input, different `tag` config — and Tag is the identity,
    // so the value crossing the first edge is byte-identical.
    let (_, trace) = run(&plan, &g, &catalog_with("b"), &cache, "second", &x);

    assert_eq!(
        trace,
        vec![
            "tag: miss",
            "tag: started",
            "tag: completed",
            "mid: hit",
            "out: hit",
        ],
        "a changed node recomputes; an unchanged output stops the invalidation there"
    );
}

// ── The one distinction ──────────────────────────────────────────────────────

/// A filter memoizes by content, not by where it sits.
///
/// `output_key` is `hash(config ‖ state ‖ input)`. There is no node id in it,
/// on purpose: two nodes running the same filter over the same value are the
/// same computation, and computing it twice would be waste the system can see.
///
/// The test builds a *different graph* — one node, differently named — over the
/// same cache, and the work already done is reused.
#[test]
fn a_filter_memoizes_by_content_not_by_where_it_sits() {
    let cache = MemoryCache::default();
    let x = Value::tensor(vec![3.0], vec![1]);

    let mut here = Graph::new();
    here.add_node(Node::filter_with_id("here", "here"));
    let mut catalog = NodeCatalog::new();
    catalog.register("here", Box::new(Double));
    let plan = plan_for(&here, &["here"]);
    run(&plan, &here, &catalog, &cache, "r1", &x);

    // Another graph, another node id, same filter and same input.
    let mut elsewhere = Graph::new();
    elsewhere.add_node(Node::filter_with_id("elsewhere", "elsewhere"));
    let mut catalog2 = NodeCatalog::new();
    catalog2.register("elsewhere", Box::new(Double));
    let plan2 = plan_for(&elsewhere, &["elsewhere"]);
    let (out, trace) = run(&plan2, &elsewhere, &catalog2, &cache, "r2", &x);

    assert_eq!(
        trace,
        vec!["elsewhere: hit"],
        "the same computation under another name is still the same computation"
    );
    assert_eq!(out.as_tensor().unwrap().0, &[6.0]);
}

/// A step journals by site, and replays rather than re-running.
///
/// The mirror image. A model asked the same question twice may answer
/// differently, so an impure effect cannot be keyed by content — it is keyed by
/// `(run, node, turn, index)`, recorded once, and replayed for *that* run.
///
/// The handler counts and answers differently every call, because a fixed reply
/// could not distinguish replay from re-execution: both would look identical.
#[test]
fn a_step_journals_by_site_and_replays_rather_than_re_running() {
    let mut g = Graph::new();
    g.add_node(Node::step("ask", "AskOnce"));
    let plan = plan_for(&g, &[]);

    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(FsActionStore::new(dir.path()).unwrap());
    let llm = Arc::new(CountingLlm::default());

    let mut catalog = NodeCatalog::new();
    catalog.register_step("ask", Box::new(AskOnce));
    let cache = MemoryCache::default();
    let x = Value::text("go");

    let once = |run_id: &str| {
        let driver = EffectDriver::new(EffectJournal::new(store.clone(), store.clone()))
            .with_handler(llm.clone());
        let bus = Arc::new(EventBus::new(256));
        let ctx = RunContext::new(&catalog, &cache, &bus, run_id, GraphInfo::from_graph(&g))
            .with_driver(driver);
        LocalRunner.forward(&plan, &ctx, &x).expect("forward")
    };

    let first = once("run-1");
    assert_eq!(first, Value::text("answer-1"));
    assert_eq!(llm.calls.load(Ordering::SeqCst), 1, "the model was asked");

    // The same site, in the same run: a resume. A fresh driver, so nothing but
    // the journal on disk can carry the answer across.
    let replayed = once("run-1");
    assert_eq!(replayed, first, "replay returns what was recorded");
    assert_eq!(
        llm.calls.load(Ordering::SeqCst),
        1,
        "an impure effect is replayed, never performed twice for the same site"
    );

    // A different run is a different site, so it is genuinely asked again —
    // and, this being a model, gets a different answer.
    let elsewhere = once("run-2");
    assert_eq!(
        llm.calls.load(Ordering::SeqCst),
        2,
        "another run, another ask"
    );
    assert_eq!(
        elsewhere,
        Value::text("answer-2"),
        "a step's answer does not travel between runs the way a filter's does"
    );
}
