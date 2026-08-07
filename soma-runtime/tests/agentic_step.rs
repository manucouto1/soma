//! An effectful step as a node in an ordinary graph.
//!
//! What these check is the seam: a `Step` compiles like any other node, runs
//! through the same executor, reads its input from its predecessors, and
//! hands its output to its successors — so a graph can mix computation and
//! effects without either half knowing about the other.

use somatize_compiler::{CompileMode, SimpleNodeRegistry, compile};
use somatize_core::agentic::effect::{
    Effect, EffectResult, LlmRequest, LlmResponse, StopReason, Usage,
};
use somatize_core::agentic::message::Message;
use somatize_core::cache::CacheKey;
use somatize_core::data::value::Value;
use somatize_core::error::Result;
use somatize_core::graph::filter::{Distribution, Filter, FilterKind, FilterMeta, StreamMode};
use somatize_core::graph::step::{Step, StepCtx, StepMeta, Transition};
use somatize_core::graph::{Edge, Graph, Node};
use somatize_runtime::agentic::{EffectDriver, EffectHandler, EffectJournal};
use somatize_runtime::cache::MemoryCache;
use somatize_runtime::cache::fs_store::FsActionStore;
use somatize_runtime::execution::executor::{Context, GraphInfo, execute};
use somatize_runtime::execution::node_catalog::NodeCatalog;
use somatize_runtime::tracking::event_bus::EventBus;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

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

/// Uppercases like `Shout`, but cacheable — the double the output-cache
/// tests need, since `Shout` opts out of caching.
struct Memo;

impl Filter for Memo {
    fn config_hash(&self) -> CacheKey {
        CacheKey::from_parts(&[b"Memo"])
    }
    fn fit(&self, _x: &Value, _y: Option<&Value>) -> Result<Value> {
        Ok(Value::Empty)
    }
    fn forward(&self, x: &Value, _state: &Value) -> Result<Value> {
        Ok(Value::text(x.as_text().unwrap_or_default().to_uppercase()))
    }
    fn meta(&self) -> FilterMeta {
        FilterMeta {
            name: "Memo".into(),
            cacheable: true,
            ..Shout.meta()
        }
    }
}

/// Answers with a fixed reply, counting how often it is actually called.
struct FakeLlm {
    calls: AtomicUsize,
}

impl FakeLlm {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            calls: AtomicUsize::new(0),
        })
    }
    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl EffectHandler for FakeLlm {
    fn handles(&self, effect: &Effect) -> bool {
        matches!(effect, Effect::Llm(_))
    }
    fn perform(&self, effect: &Effect) -> Result<EffectResult> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let Effect::Llm(req) = effect else {
            unreachable!()
        };
        let asked = req.messages.last().map(|m| m.text()).unwrap_or_default();
        Ok(EffectResult::Llm(LlmResponse {
            message: Message::assistant(format!("answer to: {asked}")),
            stop_reason: StopReason::EndTurn,
            usage: Usage {
                input_tokens: 7,
                output_tokens: 11,
                ..Default::default()
            },
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

struct Harness {
    _dir: tempfile::TempDir,
    /// Filters and steps together — the catalog is one registry now, and
    /// the executor gets the same value the compiler read.
    catalog: NodeCatalog,
    driver: EffectDriver,
    llm: Arc<FakeLlm>,
}

impl Harness {
    /// The harness catalog plus whatever filters this test needs.
    fn with_filters(&self, ids: &[&str]) -> NodeCatalog {
        let mut catalog = self.catalog.clone();
        for id in ids {
            catalog.register(*id, Box::new(Shout));
        }
        catalog
    }
}

fn harness() -> Harness {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(FsActionStore::new(dir.path()).unwrap());
    let journal = EffectJournal::new(store.clone(), store);
    let llm = FakeLlm::new();

    let mut catalog = NodeCatalog::new();
    catalog.register_step("ask", Box::new(AskOnce));

    Harness {
        _dir: dir,
        catalog,
        driver: EffectDriver::new(journal).with_handler(llm.clone()),
        llm,
    }
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

/// A `Step` node compiles to `ExecutionPlan::Step`, not `Execute`. The
/// distinction matters: the runtime drives a turn loop for one and calls a
/// function once for the other.
#[test]
fn a_step_node_compiles_to_a_step_plan() {
    let g = mixed_graph();
    let mut reg = SimpleNodeRegistry::new();
    for id in ["prep", "shout"] {
        reg.register_meta(id, Shout.meta(), CacheKey::from_parts(&[id.as_bytes()]));
    }

    let plan = compile(&g, &reg, CompileMode::Inference, None)
        .expect("compiles")
        .plan;

    let rendered = plan.to_string();
    assert!(rendered.contains("Step(ask)"), "{rendered}");
    assert!(rendered.contains("Execute(prep)"), "{rendered}");
    assert!(rendered.contains("Execute(shout)"), "{rendered}");
}

/// The value flows filter → step → filter, unchanged in kind.
#[test]
fn a_step_reads_from_and_writes_to_its_neighbours() {
    let h = harness();
    let bus = Arc::new(EventBus::new(256));
    let cache = MemoryCache::default();

    let filters = h.with_filters(&["prep", "shout"]);

    let g = mixed_graph();
    let mut ctx = Context::new(bus, "run-1")
        .with_graph_info(GraphInfo::from_graph(&g))
        .with_driver(h.driver.clone().with_catalog(Arc::new(filters.clone())));
    ctx.set("prep", Value::text("what is soma?"));

    let mut reg = SimpleNodeRegistry::new();
    for id in ["prep", "shout"] {
        reg.register_meta(id, Shout.meta(), CacheKey::from_parts(&[id.as_bytes()]));
    }
    let plan = compile(&g, &reg, CompileMode::Inference, None)
        .unwrap()
        .plan;

    execute(&plan, &mut ctx, &filters, &cache).unwrap();

    // `prep` uppercases, the step answers it, `shout` uppercases the answer.
    assert_eq!(
        ctx.get("shout").and_then(|v| v.as_text()),
        Some("ANSWER TO: WHAT IS SOMA?")
    );
    assert_eq!(h.llm.calls(), 1);
}

/// Re-running the same run id replays from the journal: the model is not
/// called again, and the answer is identical. This is what makes an
/// agentic run resumable after a crash.
#[test]
fn re_running_the_same_run_replays_instead_of_calling() {
    let h = harness();
    let cache = MemoryCache::default();

    let filters = h.with_filters(&["prep", "shout"]);

    let g = mixed_graph();
    let mut reg = SimpleNodeRegistry::new();
    for id in ["prep", "shout"] {
        reg.register_meta(id, Shout.meta(), CacheKey::from_parts(&[id.as_bytes()]));
    }
    let plan = compile(&g, &reg, CompileMode::Inference, None)
        .unwrap()
        .plan;

    let mut outputs = Vec::new();
    for _ in 0..2 {
        let bus = Arc::new(EventBus::new(256));
        let mut ctx = Context::new(bus, "run-same")
            .with_graph_info(GraphInfo::from_graph(&g))
            .with_driver(h.driver.clone().with_catalog(Arc::new(filters.clone())));
        ctx.set("prep", Value::text("hello"));
        execute(&plan, &mut ctx, &filters, &cache).unwrap();
        outputs.push(ctx.get("shout").and_then(|v| v.as_text()).map(String::from));
    }

    assert_eq!(outputs[0], outputs[1], "replay produced a different answer");
    assert_eq!(h.llm.calls(), 1, "the replay called the model again");
}

/// A plan containing a step, run without a step library, must say exactly
/// what is missing rather than fail as a missing node.
#[test]
fn a_step_without_a_library_explains_itself() {
    let bus = Arc::new(EventBus::new(64));
    let cache = MemoryCache::default();
    let filters = NodeCatalog::new();

    let mut ctx = Context::new(bus, "run-x");
    ctx.set("ask", Value::text("hi"));

    let plan = somatize_compiler::ExecutionPlan::Step {
        node_id: "ask".into(),
        handoffs: vec![],
    };
    let err = execute(&plan, &mut ctx, &filters, &cache).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("ask"), "should name the node: {msg}");
}

// ── Handoffs ──

/// Hands control to whichever target its input names.
struct Router;

impl Step for Router {
    fn config_hash(&self) -> CacheKey {
        CacheKey::from_parts(&[b"Router"])
    }
    fn meta(&self) -> StepMeta {
        StepMeta::new("Router")
    }
    fn poll(&self, ctx: &StepCtx<'_>) -> Result<Transition> {
        Ok(Transition::Goto {
            target: ctx.input.as_text().unwrap_or_default().to_string(),
            carry: Value::text("routed payload"),
        })
    }
}

/// `router` may hand off to `billing` or `tech`; only the chosen one runs,
/// and it reads the carried value as an ordinary predecessor output.
fn routed_graph() -> Graph {
    let mut g = Graph::new();
    g.add_node(Node::step("router", "Router"));
    g.add_node(Node::filter_with_id("billing", "billing"));
    g.add_node(Node::filter_with_id("tech", "tech"));
    g.add_edge(Edge::control("e1", "router", "billing"));
    g.add_edge(Edge::control("e2", "router", "tech"));
    g
}

fn routed_plan() -> somatize_compiler::ExecutionPlan {
    let g = routed_graph();
    let mut reg = SimpleNodeRegistry::new();
    for id in ["billing", "tech"] {
        reg.register_meta(id, Shout.meta(), CacheKey::from_parts(&[id.as_bytes()]));
    }
    compile(&g, &reg, CompileMode::Inference, None)
        .expect("compiles")
        .plan
}

#[test]
fn a_handoff_runs_only_the_chosen_target() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(FsActionStore::new(dir.path()).unwrap());
    let journal = EffectJournal::new(store.clone(), store);
    let driver = EffectDriver::new(journal);

    let mut filters = NodeCatalog::new();
    filters.register_step("router", Box::new(Router));
    filters.register("billing", Box::new(Shout));
    filters.register("tech", Box::new(Shout));

    let g = routed_graph();
    let plan = routed_plan();
    let cache = MemoryCache::default();

    let mut ctx = Context::new(Arc::new(EventBus::new(64)), "run-handoff")
        .with_graph_info(GraphInfo::from_graph(&g))
        .with_driver(driver.with_catalog(Arc::new(filters.clone())));
    ctx.set("router", Value::text("tech"));

    execute(&plan, &mut ctx, &filters, &cache).unwrap();

    assert_eq!(
        ctx.get("tech").and_then(|v| v.as_text()),
        Some("ROUTED PAYLOAD"),
        "the chosen target should have run on the carried value"
    );
    assert!(
        ctx.get("billing").is_none(),
        "the target that was not chosen must not run"
    );
}

/// Each handoff target is compiled once, inside the step — not again as a
/// top-level node, which would run both specialists regardless.
#[test]
fn handoff_targets_are_compiled_exactly_once() {
    let rendered = routed_plan().to_string();
    assert_eq!(
        rendered.matches("Execute(billing)").count(),
        1,
        "{rendered}"
    );
    assert_eq!(rendered.matches("Execute(tech)").count(), 1, "{rendered}");
    assert!(rendered.contains("Step(router)"), "{rendered}");
}

/// Handing off somewhere undeclared names the declared targets rather than
/// failing obscurely.
#[test]
fn an_undeclared_handoff_target_is_reported() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(FsActionStore::new(dir.path()).unwrap());
    let driver = EffectDriver::new(EffectJournal::new(store.clone(), store));

    let mut filters = NodeCatalog::new();
    filters.register_step("router", Box::new(Router));

    let g = routed_graph();
    let plan = routed_plan();
    let cache = MemoryCache::default();

    let mut ctx = Context::new(Arc::new(EventBus::new(64)), "run-bad-handoff")
        .with_graph_info(GraphInfo::from_graph(&g))
        .with_driver(driver.with_catalog(Arc::new(filters.clone())));
    ctx.set("router", Value::text("legal"));

    let err = execute(&plan, &mut ctx, &filters, &cache).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("legal"), "should name the target: {msg}");
    assert!(
        msg.contains("billing"),
        "should list what is declared: {msg}"
    );
}

// ── Suspension through the executor ──

/// Pauses for a person, then reports their decision.
struct NeedsApproval;

impl Step for NeedsApproval {
    fn config_hash(&self) -> CacheKey {
        CacheKey::from_parts(&[b"NeedsApproval"])
    }
    fn meta(&self) -> StepMeta {
        StepMeta::new("NeedsApproval")
    }
    fn poll(&self, ctx: &StepCtx<'_>) -> Result<Transition> {
        match ctx.result() {
            None => Ok(Transition::Suspend {
                reason: somatize_core::agentic::effect::SuspendReason::Human {
                    prompt: "approve?".into(),
                    schema: None,
                },
            }),
            Some(EffectResult::Node(a)) => Ok(Transition::Done(Value::text(
                a.as_text().unwrap_or_default(),
            ))),
            Some(other) => Ok(Transition::Done(Value::text(format!("{other:?}")))),
        }
    }
}

/// A suspended run stops the *whole* plan — nodes downstream of the pause
/// must not run — and resumes to completion once answered.
#[test]
fn a_suspended_run_halts_the_plan_and_then_resumes() {
    use somatize_core::error::SomaError;

    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(FsActionStore::new(dir.path()).unwrap());
    let driver = EffectDriver::new(EffectJournal::new(store.clone(), store));

    let mut filters = NodeCatalog::new();
    filters.register_step("approve", Box::new(NeedsApproval));
    filters.register("after", Box::new(Shout));

    let mut g = Graph::new();
    g.add_node(Node::step("approve", "NeedsApproval"));
    g.add_node(Node::filter_with_id("after", "after"));
    g.add_edge(Edge::data("e", "approve", "after"));

    let mut reg = SimpleNodeRegistry::new();
    reg.register_meta("after", Shout.meta(), CacheKey::from_parts(&[b"after"]));
    let plan = compile(&g, &reg, CompileMode::Inference, None)
        .unwrap()
        .plan;
    let cache = MemoryCache::default();

    let run = |driver: EffectDriver| {
        let mut ctx = Context::new(Arc::new(EventBus::new(64)), "run-approve")
            .with_graph_info(GraphInfo::from_graph(&g))
            .with_driver(driver.with_catalog(Arc::new(filters.clone())));
        let outcome = execute(&plan, &mut ctx, &filters, &cache);
        (outcome, ctx)
    };

    let (outcome, ctx) = run(driver.clone());
    let err = outcome.expect_err("the run should have paused");
    let SomaError::Suspended { node_id, turn, .. } = &err else {
        panic!("expected Suspended, got {err}");
    };
    assert_eq!(node_id, "approve");
    assert!(
        ctx.get("after").is_none(),
        "a node downstream of the pause ran anyway"
    );

    driver
        .resume_with(
            "run-approve",
            "approve",
            *turn,
            &somatize_core::agentic::effect::SuspendReason::Human {
                prompt: "approve?".into(),
                schema: None,
            },
            Value::text("granted"),
        )
        .unwrap();

    let (outcome, ctx) = run(driver);
    outcome.expect("the resumed run should finish");
    assert_eq!(
        ctx.get("after").and_then(|v| v.as_text()),
        Some("GRANTED"),
        "the downstream node should have seen the human's answer"
    );
}

/// Steps emit the agent-level events, on the same bus as everything else.
#[test]
fn a_step_emits_agent_events() {
    use somatize_core::tracking::event::Event;

    let h = harness();
    let bus = Arc::new(EventBus::new(256));
    let mut rx = bus.subscribe();
    let cache = MemoryCache::default();
    let filters = h.catalog.clone();

    let mut ctx = Context::new(bus.clone(), "run-events").with_driver(
        h.driver
            .clone()
            .with_event_bus(bus.clone())
            .with_catalog(Arc::new(filters.clone())),
    );
    ctx.set("ask", Value::text("hi"));

    let plan = somatize_compiler::ExecutionPlan::Step {
        node_id: "ask".into(),
        handoffs: vec![],
    };
    execute(&plan, &mut ctx, &filters, &cache).unwrap();
    drop(ctx);

    let mut turns = 0;
    let mut requested = 0;
    let mut completed = 0;
    let mut finished = 0;
    while let Ok(event) = rx.try_recv() {
        match event {
            Event::AgentTurnStarted { .. } => turns += 1,
            Event::EffectRequested { effect, .. } => {
                assert!(effect.starts_with("llm:"), "{effect}");
                requested += 1;
            }
            Event::EffectCompleted { replayed, .. } => {
                assert!(!replayed, "first run should not be a replay");
                completed += 1;
            }
            Event::AgentStepCompleted {
                turns: n,
                output_tokens,
                ..
            } => {
                assert_eq!(n, 2, "one turn to ask, one to finish");
                assert_eq!(output_tokens, 11, "usage was not accumulated");
                finished += 1;
            }
            _ => {}
        }
    }

    assert_eq!(turns, 2);
    assert_eq!(requested, 1);
    assert_eq!(completed, 1);
    assert_eq!(finished, 1);
}

// ── One registry ──

/// Produces a tensor and says so, which no agent can read.
struct Numeric;

impl Filter for Numeric {
    fn config_hash(&self) -> CacheKey {
        CacheKey::from_parts(&[b"Numeric"])
    }
    fn fit(&self, _x: &Value, _y: Option<&Value>) -> Result<Value> {
        Ok(Value::Empty)
    }
    fn forward(&self, _x: &Value, _state: &Value) -> Result<Value> {
        Ok(Value::tensor(vec![1.0], vec![1]))
    }
    fn meta(&self) -> FilterMeta {
        FilterMeta {
            output_schema: Some(somatize_core::data::schema::Schema::scalar(
                somatize_core::data::schema::DataType::Float64,
            )),
            ..Shout.meta()
        }
    }
}

/// Insists on a conversation.
struct WantsMessages;

impl Step for WantsMessages {
    fn config_hash(&self) -> CacheKey {
        CacheKey::from_parts(&[b"WantsMessages"])
    }
    fn meta(&self) -> StepMeta {
        StepMeta::new("WantsMessages")
            .with_input_schema(somatize_core::data::schema::Schema::messages())
    }
    fn poll(&self, _ctx: &StepCtx<'_>) -> Result<Transition> {
        Ok(Transition::Done(Value::Empty))
    }
}

/// Filters and steps share one registry, so a step's declared input is
/// checked against its predecessor's output like any other edge.
///
/// It used to depend on which registry the caller happened to pass: the
/// filter library alone answered `None` for every step, so every edge into
/// an agent went unchecked — and `Graph.compile()` in Python passed exactly
/// that, while `run()` and `forward()` passed a registry spanning both.
#[test]
fn a_step_edge_is_schema_checked_like_any_other() {
    let mut catalog = NodeCatalog::new();
    catalog.register("numeric", Box::new(Numeric));
    catalog.register_step("agent", Box::new(WantsMessages));

    let mut g = Graph::new();
    g.add_node(Node::filter_with_id("numeric", "numeric"));
    g.add_node(Node::step("agent", "WantsMessages"));
    g.add_edge(Edge::data("e", "numeric", "agent"));

    let err = compile(&g, &catalog, CompileMode::Inference, None)
        .expect_err("a float cannot become a conversation");
    let msg = err.to_string();
    assert!(msg.contains("numeric") && msg.contains("agent"), "{msg}");
}

/// A step's `distribution` was a field nothing ever read: the compiler
/// asked the filter registry, which knew nothing about steps. Now that
/// there is one registry, a step marked remote is wrapped like a filter.
#[test]
fn a_remote_step_is_wrapped_for_dispatch() {
    struct RemoteStep;

    impl Step for RemoteStep {
        fn config_hash(&self) -> CacheKey {
            CacheKey::from_parts(&[b"RemoteStep"])
        }
        fn meta(&self) -> StepMeta {
            let mut m = StepMeta::new("RemoteStep");
            m.distribution = Distribution::Remote(somatize_core::graph::filter::RemoteTarget::Tag(
                "gpu".into(),
            ));
            m
        }
        fn poll(&self, _ctx: &StepCtx<'_>) -> Result<Transition> {
            Ok(Transition::Done(Value::Empty))
        }
    }

    let mut catalog = NodeCatalog::new();
    catalog.register_step("far", Box::new(RemoteStep));

    let mut g = Graph::new();
    g.add_node(Node::step("far", "RemoteStep"));

    let plan = compile(&g, &catalog, CompileMode::Inference, None)
        .expect("compiles")
        .plan;

    assert!(
        matches!(plan, somatize_compiler::ExecutionPlan::Remote { .. }),
        "expected the step to be wrapped for dispatch, got {plan}"
    );
}

// ── One execution site ──

/// A step deciding which branch arm runs.
struct RoutingStep;

impl Step for RoutingStep {
    fn config_hash(&self) -> CacheKey {
        CacheKey::from_parts(&[b"RoutingStep"])
    }
    fn meta(&self) -> StepMeta {
        StepMeta::new("RoutingStep")
    }
    fn poll(&self, ctx: &StepCtx<'_>) -> Result<Transition> {
        Ok(Transition::Goto {
            target: ctx.input.as_text().unwrap_or_default().to_string(),
            carry: Value::text("routed"),
        })
    }
}

/// A step can be a branch's condition and name the arm with `Goto`.
///
/// This could not work before. The branch called the step with an empty
/// handoff list — its control edges having been consumed as the branch's
/// arms — so `Goto` always hit "it declares no handoffs". The branch reads
/// the outcome now, and a handoff names an arm as directly as a returned
/// label does.
#[test]
fn a_step_can_decide_a_branch_by_handing_off() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(FsActionStore::new(dir.path()).unwrap());
    let driver = EffectDriver::new(EffectJournal::new(store.clone(), store));

    let mut catalog = NodeCatalog::new();
    catalog.register_step("router", Box::new(RoutingStep));
    catalog.register("billing", Box::new(Shout));
    catalog.register("tech", Box::new(Shout));

    let mut g = Graph::new();
    g.add_node(Node::branch("router"));
    g.add_node(Node::filter_with_id("billing", "billing"));
    g.add_node(Node::filter_with_id("tech", "tech"));
    g.add_edge(Edge::control("c1", "router", "billing").with_label("billing"));
    g.add_edge(Edge::control("c2", "router", "tech").with_label("tech"));

    let plan = somatize_compiler::ExecutionPlan::Branch {
        node_id: "router".into(),
        arms: vec![
            (
                "billing".into(),
                somatize_compiler::ExecutionPlan::Execute {
                    node_id: "billing".into(),
                },
            ),
            (
                "tech".into(),
                somatize_compiler::ExecutionPlan::Execute {
                    node_id: "tech".into(),
                },
            ),
        ],
    };

    let cache = MemoryCache::default();
    let mut ctx = Context::new(Arc::new(EventBus::new(64)), "run-branch-step")
        .with_graph_info(GraphInfo::from_graph(&g))
        .with_driver(driver.with_catalog(Arc::new(catalog.clone())));
    ctx.set("__input__", Value::text("tech"));
    ctx.set("router", Value::text("tech"));

    execute(&plan, &mut ctx, &catalog, &cache).expect("the step should pick an arm");

    assert!(ctx.get("tech").is_some(), "the named arm should have run");
    assert!(
        ctx.get("billing").is_none(),
        "the arm that was not named must not run"
    );
}

/// Panics in `poll`, the way a Python step with a bug does.
struct PanickingStep;

impl Step for PanickingStep {
    fn config_hash(&self) -> CacheKey {
        CacheKey::from_parts(&[b"PanickingStep"])
    }
    fn meta(&self) -> StepMeta {
        StepMeta::new("PanickingStep")
    }
    fn poll(&self, _ctx: &StepCtx<'_>) -> Result<Transition> {
        panic!("the step fell over");
    }
}

/// A panicking step is an error, not a dead process.
///
/// `catch_unwind` used to wrap only the filter path, on the argument that
/// a user's filter must not take the runtime down. A step written in
/// Python is user code by exactly the same argument, and it had no such
/// protection — until both went through one execution site.
#[test]
fn a_panicking_step_is_contained_like_a_panicking_filter() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(FsActionStore::new(dir.path()).unwrap());
    let driver = EffectDriver::new(EffectJournal::new(store.clone(), store));

    let mut catalog = NodeCatalog::new();
    catalog.register_step("boom", Box::new(PanickingStep));

    let cache = MemoryCache::default();
    let mut ctx = Context::new(Arc::new(EventBus::new(64)), "run-panic-step")
        .with_driver(driver.with_catalog(Arc::new(catalog.clone())));
    ctx.set("boom", Value::text("go"));

    let plan = somatize_compiler::ExecutionPlan::Step {
        node_id: "boom".into(),
        handoffs: vec![],
    };

    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = execute(&plan, &mut ctx, &catalog, &cache);
    std::panic::set_hook(previous);

    let err = result.expect_err("a panicking step must not be a success");
    assert!(err.to_string().contains("the step fell over"), "{err}");
}

// ── The output cache and a step ──

/// `memo(filter) -> ask(step)`, compiled with the catalog itself.
fn memoized_graph() -> Graph {
    let mut g = Graph::new();
    g.add_node(Node::filter_with_id("memo", "memo"));
    g.add_node(Node::step("ask", "AskOnce"));
    g.add_edge(Edge::data("e", "memo", "ask"));
    g
}

/// The guard at the single execution site is `cacheable && deterministic`,
/// and a step reaches it declaring `false/false` via `From<StepMeta>`. This
/// pins that as *observable* behavior: the cacheable filter misses then
/// hits, while the step's id appears in no cache event, ever. Flipping
/// `StepMeta -> NodeMeta` to `cacheable: true` would memoize a model call
/// by content — the frozen-answer foot-gun — and fail here first.
#[test]
fn the_output_cache_never_touches_a_step() {
    use somatize_core::tracking::event::Event;

    let h = harness();
    let mut catalog = h.catalog.clone();
    catalog.register("memo", Box::new(Memo));

    let g = memoized_graph();
    let plan = compile(&g, &catalog, CompileMode::Inference, None)
        .unwrap()
        .plan;

    let cache = MemoryCache::default();
    let bus = Arc::new(EventBus::new(256));
    let mut rx = bus.subscribe();

    // Two runs, one cache: the second run's filter output is already there.
    for run_id in ["run-cache-1", "run-cache-2"] {
        let mut ctx = Context::new(bus.clone(), run_id)
            .with_graph_info(GraphInfo::from_graph(&g))
            .with_driver(h.driver.clone().with_catalog(Arc::new(catalog.clone())));
        ctx.set("memo", Value::text("the same question"));
        execute(&plan, &mut ctx, &catalog, &cache).unwrap();
    }

    let mut misses = 0;
    let mut hits = 0;
    while let Ok(event) = rx.try_recv() {
        match event {
            Event::NodeCacheMiss { node_id, .. } => {
                assert_eq!(node_id, "memo", "a step's node id reached the output cache");
                misses += 1;
            }
            Event::NodeCacheHit { node_id, .. } => {
                assert_eq!(node_id, "memo", "a step's node id reached the output cache");
                hits += 1;
            }
            _ => {}
        }
    }
    assert_eq!(misses, 1, "the cacheable filter should miss exactly once");
    assert_eq!(hits, 1, "the second run should serve the filter from cache");
}

/// A filter's output cache key is `hash(config + state + input content)` —
/// nothing about the graph around it. Wiring an agent downstream must not
/// invalidate what a plain pipeline already computed, or adding a step to
/// an existing experiment would silently recompute every filter.
#[test]
fn a_filter_cache_key_ignores_its_agentic_neighbours() {
    use somatize_core::tracking::event::Event;

    let h = harness();
    let mut catalog = h.catalog.clone();
    catalog.register("memo", Box::new(Memo));

    // Graph A: the filter alone. Graph B: the same filter, a step downstream.
    let mut graph_a = Graph::new();
    graph_a.add_node(Node::filter_with_id("memo", "memo"));
    let graph_b = memoized_graph();

    let cache = MemoryCache::default();
    let bus = Arc::new(EventBus::new(256));

    let plan_a = compile(&graph_a, &catalog, CompileMode::Inference, None)
        .unwrap()
        .plan;
    let mut ctx =
        Context::new(bus.clone(), "run-plain").with_graph_info(GraphInfo::from_graph(&graph_a));
    ctx.set("memo", Value::text("stable input"));
    execute(&plan_a, &mut ctx, &catalog, &cache).unwrap();

    // Same filter, same input, same cache — now with an agent attached.
    let mut rx = bus.subscribe();
    let plan_b = compile(&graph_b, &catalog, CompileMode::Inference, None)
        .unwrap()
        .plan;
    let mut ctx = Context::new(bus.clone(), "run-agentic")
        .with_graph_info(GraphInfo::from_graph(&graph_b))
        .with_driver(h.driver.clone().with_catalog(Arc::new(catalog.clone())));
    ctx.set("memo", Value::text("stable input"));
    execute(&plan_b, &mut ctx, &catalog, &cache).unwrap();

    let mut hit = false;
    while let Ok(event) = rx.try_recv() {
        if let Event::NodeCacheHit { node_id, .. } = event {
            assert_eq!(node_id, "memo");
            hit = true;
        }
    }
    assert!(
        hit,
        "adding a step downstream moved the filter's cache key — the \
         plain run's entry was not reused"
    );
}

// ── Node events ──

/// `NodeStarted` reports each kind as itself. `effectful` exists because an
/// agent reported as an `Opaque` filter was indistinguishable from one —
/// every event consumer saw a filter it could not look inside.
#[test]
fn a_step_and_a_filter_start_with_their_own_kind() {
    use somatize_core::tracking::event::Event;

    let h = harness();
    let bus = Arc::new(EventBus::new(256));
    let mut rx = bus.subscribe();
    let cache = MemoryCache::default();
    let filters = h.with_filters(&["prep", "shout"]);

    let g = mixed_graph();
    let mut reg = SimpleNodeRegistry::new();
    for id in ["prep", "shout"] {
        reg.register_meta(id, Shout.meta(), CacheKey::from_parts(&[id.as_bytes()]));
    }
    let plan = compile(&g, &reg, CompileMode::Inference, None)
        .unwrap()
        .plan;

    let mut ctx = Context::new(bus.clone(), "run-kinds")
        .with_graph_info(GraphInfo::from_graph(&g))
        .with_driver(h.driver.clone().with_catalog(Arc::new(filters.clone())));
    ctx.set("prep", Value::text("hi"));
    execute(&plan, &mut ctx, &filters, &cache).unwrap();

    let mut saw_step = false;
    let mut saw_filter = false;
    while let Ok(event) = rx.try_recv() {
        if let Event::NodeStarted {
            node_id, effectful, ..
        } = event
        {
            match node_id.as_str() {
                "ask" => {
                    assert!(effectful, "the step must start as effectful");
                    saw_step = true;
                }
                "prep" => {
                    assert!(!effectful, "the filter must not start as effectful");
                    saw_filter = true;
                }
                _ => {}
            }
        }
    }
    assert!(saw_step && saw_filter, "both kinds should have started");
}

/// Fails in `poll`, the way a step with a broken tool configuration does.
struct FailingStep;

impl Step for FailingStep {
    fn config_hash(&self) -> CacheKey {
        CacheKey::from_parts(&[b"FailingStep"])
    }
    fn meta(&self) -> StepMeta {
        StepMeta::new("FailingStep")
    }
    fn poll(&self, ctx: &StepCtx<'_>) -> Result<Transition> {
        Err(somatize_core::error::SomaError::Execution {
            node_id: ctx.node_id.to_string(),
            message: "the model is unreachable".into(),
        })
    }
}

/// A step failing must be visible on the bus, not only in the returned
/// error: run trackers and health flags read `NodeFailed`, and a step that
/// died silently would show as a run that never finished.
#[test]
fn a_failing_step_emits_node_failed() {
    use somatize_core::tracking::event::Event;

    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(FsActionStore::new(dir.path()).unwrap());
    let driver = EffectDriver::new(EffectJournal::new(store.clone(), store));

    let mut catalog = NodeCatalog::new();
    catalog.register_step("fails", Box::new(FailingStep));

    let bus = Arc::new(EventBus::new(64));
    let mut rx = bus.subscribe();
    let cache = MemoryCache::default();
    let mut ctx = Context::new(bus.clone(), "run-fail-step")
        .with_driver(driver.with_catalog(Arc::new(catalog.clone())));
    ctx.set("fails", Value::text("go"));

    let plan = somatize_compiler::ExecutionPlan::Step {
        node_id: "fails".into(),
        handoffs: vec![],
    };
    let err = execute(&plan, &mut ctx, &catalog, &cache).expect_err("the step fails");
    assert!(err.to_string().contains("unreachable"), "{err}");

    let mut failed = false;
    while let Ok(event) = rx.try_recv() {
        if let Event::NodeFailed { node_id, error, .. } = event {
            assert_eq!(node_id, "fails");
            assert!(error.contains("unreachable"), "{error}");
            failed = true;
        }
    }
    assert!(
        failed,
        "no NodeFailed event was emitted for the failing step"
    );
}

/// The existing panic test checks the returned error; this one pins the
/// event. A contained panic that emitted nothing would leave the run's
/// record claiming the node is still running.
#[test]
fn a_panicking_step_emits_node_failed() {
    use somatize_core::tracking::event::Event;

    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(FsActionStore::new(dir.path()).unwrap());
    let driver = EffectDriver::new(EffectJournal::new(store.clone(), store));

    let mut catalog = NodeCatalog::new();
    catalog.register_step("boom", Box::new(PanickingStep));

    let bus = Arc::new(EventBus::new(64));
    let mut rx = bus.subscribe();
    let cache = MemoryCache::default();
    let mut ctx = Context::new(bus.clone(), "run-panic-event")
        .with_driver(driver.with_catalog(Arc::new(catalog.clone())));
    ctx.set("boom", Value::text("go"));

    let plan = somatize_compiler::ExecutionPlan::Step {
        node_id: "boom".into(),
        handoffs: vec![],
    };

    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = execute(&plan, &mut ctx, &catalog, &cache);
    std::panic::set_hook(previous);
    result.expect_err("a panicking step must not be a success");

    let mut failed = false;
    while let Ok(event) = rx.try_recv() {
        if let Event::NodeFailed { node_id, error, .. } = event {
            assert_eq!(node_id, "boom");
            assert!(error.contains("the step fell over"), "{error}");
            failed = true;
        }
    }
    assert!(
        failed,
        "no NodeFailed event was emitted for the panicking step"
    );
}

/// A handoff is a control-flow fact a run viewer cannot reconstruct from
/// node events alone — `Event::Handoff` is the record of who passed the
/// conversation to whom.
#[test]
fn a_handoff_emits_its_event() {
    use somatize_core::tracking::event::Event;

    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(FsActionStore::new(dir.path()).unwrap());
    let bus = Arc::new(EventBus::new(64));
    let mut rx = bus.subscribe();
    let driver =
        EffectDriver::new(EffectJournal::new(store.clone(), store)).with_event_bus(bus.clone());

    let mut filters = NodeCatalog::new();
    filters.register_step("router", Box::new(Router));
    filters.register("billing", Box::new(Shout));
    filters.register("tech", Box::new(Shout));

    let g = routed_graph();
    let plan = routed_plan();
    let cache = MemoryCache::default();

    let mut ctx = Context::new(bus.clone(), "run-handoff-event")
        .with_graph_info(GraphInfo::from_graph(&g))
        .with_driver(driver.with_catalog(Arc::new(filters.clone())));
    ctx.set("router", Value::text("tech"));
    execute(&plan, &mut ctx, &filters, &cache).unwrap();

    let mut seen = false;
    while let Ok(event) = rx.try_recv() {
        if let Event::Handoff { from, to, .. } = event {
            assert_eq!(from, "router");
            assert_eq!(to, "tech");
            seen = true;
        }
    }
    assert!(seen, "no Handoff event was emitted");
}

/// A pause and its pickup are the events a human-in-the-loop UI subscribes
/// to; without them the only sign of a suspension is an error type.
#[test]
fn a_suspension_emits_suspended_and_resuming_emits_resumed() {
    use somatize_core::tracking::event::Event;

    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(FsActionStore::new(dir.path()).unwrap());
    let bus = Arc::new(EventBus::new(64));
    let mut rx = bus.subscribe();
    let driver =
        EffectDriver::new(EffectJournal::new(store.clone(), store)).with_event_bus(bus.clone());

    let mut filters = NodeCatalog::new();
    filters.register_step("approve", Box::new(NeedsApproval));

    let cache = MemoryCache::default();
    let plan = somatize_compiler::ExecutionPlan::Step {
        node_id: "approve".into(),
        handoffs: vec![],
    };
    let run = |driver: EffectDriver| {
        let mut ctx = Context::new(bus.clone(), "run-suspend-events")
            .with_driver(driver.with_catalog(Arc::new(filters.clone())));
        ctx.set("approve", Value::text("go"));
        execute(&plan, &mut ctx, &filters, &cache)
    };

    run(driver.clone()).expect_err("the run should pause");
    let mut suspended = false;
    let mut completed_during_pause = false;
    while let Ok(event) = rx.try_recv() {
        match event {
            Event::Suspended {
                node_id,
                reason,
                turns,
                ..
            } => {
                assert_eq!(node_id, "approve");
                assert_eq!(reason, "human");
                assert_eq!(turns, 1, "the suspension should carry the cost so far");
                suspended = true;
            }
            // A suspended step has not finished; its final completion event
            // carries cumulative totals, so an extra one here would make a
            // reader that sums per node double-count.
            Event::AgentStepCompleted { .. } => completed_during_pause = true,
            _ => {}
        }
    }
    assert!(suspended, "no Suspended event was emitted");
    assert!(
        !completed_during_pause,
        "a suspension must not emit AgentStepCompleted"
    );

    driver
        .resume_with(
            "run-suspend-events",
            "approve",
            0,
            &somatize_core::agentic::effect::SuspendReason::Human {
                prompt: "approve?".into(),
                schema: None,
            },
            Value::text("granted"),
        )
        .unwrap();
    run(driver).expect("the resumed run should finish");

    let mut resumed = false;
    while let Ok(event) = rx.try_recv() {
        if let Event::Resumed { node_id, turn, .. } = event {
            assert_eq!(node_id, "approve");
            assert_eq!(turn, 0);
            resumed = true;
        }
    }
    assert!(resumed, "no Resumed event was emitted");
}

// ── A step inside a loop ──

/// Finishes with a "done" verdict once it has been polled `stop_at` times,
/// counting so the test can prove the loop actually iterated.
struct CountingVerdict {
    calls: Arc<AtomicUsize>,
    stop_at: usize,
}

impl Step for CountingVerdict {
    fn config_hash(&self) -> CacheKey {
        CacheKey::from_parts(&[b"CountingVerdict"])
    }
    fn meta(&self) -> StepMeta {
        StepMeta::new("CountingVerdict")
    }
    fn poll(&self, _ctx: &StepCtx<'_>) -> Result<Transition> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        Ok(Transition::Done(Value::json(
            serde_json::json!({ "done": n >= self.stop_at }),
        )))
    }
}

/// A step as a loop body, end to end: the compiler claims it by dominance,
/// resolves `BodyTerminal` to it, and the executor drives it once per
/// iteration until its verdict says stop. Each pass is a fresh driver run —
/// a step being uncacheable is what makes iteration N+1 actually run
/// rather than replay iteration N.
#[test]
fn a_step_runs_inside_a_loop() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut catalog = NodeCatalog::new();
    catalog.register_step(
        "verdict",
        Box::new(CountingVerdict {
            calls: calls.clone(),
            stop_at: 2,
        }),
    );

    let mut g = Graph::new();
    g.add_node(Node::loop_node("refine", Some(5)));
    g.add_node(Node::step("verdict", "CountingVerdict"));
    g.add_edge(Edge::control("e", "refine", "verdict"));

    let plan = compile(&g, &catalog, CompileMode::Inference, None)
        .unwrap()
        .plan;

    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(FsActionStore::new(dir.path()).unwrap());
    let driver = EffectDriver::new(EffectJournal::new(store.clone(), store));

    let cache = MemoryCache::default();
    let mut ctx = Context::new(Arc::new(EventBus::new(64)), "run-loop-step")
        .with_graph_info(GraphInfo::from_graph(&g))
        .with_driver(driver.with_catalog(Arc::new(catalog.clone())));
    ctx.set("refine", Value::text("go"));

    execute(&plan, &mut ctx, &catalog, &cache).unwrap();

    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "the step should run once per iteration and stop on its own verdict"
    );
}
