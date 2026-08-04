//! Running a Soma graph as an effect.
//!
//! This is what makes a computational pipeline a first-class thing an agent
//! can reach for. No `publish` mechanism, no tool wrapper, no bridge: the
//! agent emits [`Effect::Graph`] and gets the output back, with the graph's
//! own cache, schema checks and events all applying as usual.
//!
//! It also means an agentic run is journaled at the pipeline boundary. A
//! research loop that crashes after its fourth experiment replays the first
//! three from the journal instead of paying for them again.

use crate::cache::MemoryCache;
use crate::effects::{EffectDriver, EffectHandler, EffectJournal};
use crate::event_bus::EventBus;
use crate::graph_session::GraphSession;
use crate::node_catalog::NodeCatalog;
use somatize_core::cache::CacheStore;
use somatize_core::effect::{Effect, EffectResult, GraphEffectMode};
use somatize_core::error::{Result, SomaError};
use std::sync::Arc;

/// How deep agent → pipeline → agent nesting may go.
///
/// Each level is a step whose sub-graph contains another step. Real flows
/// are one or two levels; a graph that reaches eight is almost certainly
/// recursing on itself, and stopping with a readable failure beats a stack
/// that grows until the OS ends the process.
pub const MAX_GRAPH_DEPTH: usize = 8;

/// Everything needed to drive steps inside a sub-graph.
///
/// `handlers` are the *sibling* handlers (llm, tools, sleep, custom) — never
/// a `GraphHandler`. The recursion is rebuilt one level deeper instead, so
/// each level carries its own depth and the nesting can be capped.
#[derive(Clone)]
struct StepRuntime {
    handlers: Vec<Arc<dyn EffectHandler>>,
    journal: EffectJournal,
    event_bus: Option<Arc<EventBus>>,
    depth: usize,
}

/// Runs graphs on behalf of a step.
///
/// Holds the filters those graphs are built from — a graph names its nodes,
/// it does not carry their implementations — plus the cache they share.
pub struct GraphHandler {
    library: NodeCatalog,
    cache: Arc<dyn CacheStore>,
    step_runtime: Option<StepRuntime>,
}

impl GraphHandler {
    /// A handler over `library`, with an in-memory cache.
    ///
    /// Without [`Self::with_step_runtime`] the sub-graphs it runs must be
    /// purely computational; one that contains a step fails as an
    /// [`EffectResult::Failed`] naming the missing runtime.
    pub fn new(library: NodeCatalog) -> Self {
        Self {
            library,
            cache: Arc::new(MemoryCache::new(64 * 1024 * 1024)),
            step_runtime: None,
        }
    }

    /// Share the caller's cache, so a pipeline the agent runs hits the same
    /// entries the user's own runs wrote.
    pub fn with_cache(mut self, cache: Arc<dyn CacheStore>) -> Self {
        self.cache = cache;
        self
    }

    /// Let sub-graphs contain steps of their own: agent → pipeline → agent.
    ///
    /// `handlers` are the sibling handlers the parent driver carries (llm,
    /// tools, sleep — everything except graph handlers), and `journal` is
    /// the parent's journal, so an inner model call is journaled in the same
    /// store as an outer one. Nesting is capped at [`MAX_GRAPH_DEPTH`].
    pub fn with_step_runtime(
        mut self,
        handlers: Vec<Arc<dyn EffectHandler>>,
        journal: EffectJournal,
    ) -> Self {
        self.step_runtime = Some(StepRuntime {
            handlers,
            journal,
            event_bus: None,
            depth: 0,
        });
        self
    }

    /// Forward the sub-graphs' step events to this bus.
    ///
    /// Only meaningful together with [`Self::with_step_runtime`]; without
    /// one there is no inner driver to emit anything.
    pub fn with_event_bus(mut self, bus: Arc<EventBus>) -> Self {
        if let Some(rt) = &mut self.step_runtime {
            rt.event_bus = Some(bus);
        }
        self
    }

    /// The filters available to graphs run through this handler.
    pub fn library(&self) -> &NodeCatalog {
        &self.library
    }

    /// The driver a step-containing sub-graph gets: the sibling handlers,
    /// plus a `GraphHandler` one level deeper. `None` past the depth cap.
    fn child_driver(&self) -> Option<EffectDriver> {
        let rt = self.step_runtime.as_ref()?;
        if rt.depth + 1 >= MAX_GRAPH_DEPTH {
            return None;
        }
        let mut child_rt = rt.clone();
        child_rt.depth += 1;
        let mut driver = EffectDriver::new(rt.journal.clone())
            .with_catalog(Arc::new(self.library.clone()))
            .with_handler(Arc::new(GraphHandler {
                library: self.library.clone(),
                cache: self.cache.clone(),
                step_runtime: Some(child_rt),
            }));
        for handler in &rt.handlers {
            driver = driver.with_handler(handler.clone());
        }
        if let Some(bus) = &rt.event_bus {
            driver = driver.with_event_bus(bus.clone());
        }
        Some(driver)
    }
}

impl EffectHandler for GraphHandler {
    fn handles(&self, effect: &Effect) -> bool {
        matches!(effect, Effect::Graph { .. })
    }

    fn perform(&self, effect: &Effect) -> Result<EffectResult> {
        let Effect::Graph { graph, input, mode } = effect else {
            return Err(SomaError::Other("not a graph effect".into()));
        };

        // The clone shares the state store, so a graph fitted by one effect
        // is fitted for the next one.
        let mut session = GraphSession::new((**graph).clone(), self.library.clone())
            .with_cache(self.cache.clone());

        if graph.contains_steps() {
            match self.child_driver() {
                Some(driver) => session = session.with_driver(driver),
                // The executor's own "no effect driver" error is accurate
                // but does not say *why* there is none here; the reason —
                // no runtime, or too deep — is information the agent needs.
                None => {
                    return Ok(EffectResult::Failed {
                        message: match &self.step_runtime {
                            None => "the sub-graph contains a step, but this graph handler \
                                     was built without a step runtime; build it with \
                                     `GraphHandler::with_step_runtime(...)`"
                                .into(),
                            Some(rt) => format!(
                                "the sub-graph contains a step, but nesting agents inside \
                                 pipelines inside agents stops at depth {MAX_GRAPH_DEPTH} \
                                 (this call is at depth {})",
                                rt.depth + 1
                            ),
                        },
                    });
                }
            }
        }

        let outcome = match mode {
            GraphEffectMode::Fit => session
                .fit(input, None)
                .map(|outputs| somatize_core::value::Value::json(outputs_summary(&outputs))),
            // `GraphEffectMode` is `#[non_exhaustive]`; anything added later
            // is a mode this build does not know how to run, and guessing
            // `forward` would silently skip a fit.
            GraphEffectMode::Forward => session.forward(input),
            other => Err(SomaError::Other(format!(
                "unsupported graph effect mode: {other:?}"
            ))),
        };

        match outcome {
            Ok(value) => Ok(EffectResult::Graph(value)),
            // A pipeline that fails is a result the agent has to read and
            // act on — an unfittable configuration is information, and one
            // of the more valuable kinds. Ending the run instead would
            // throw away everything learned up to that point.
            Err(e) => Ok(EffectResult::Failed {
                message: e.to_string(),
            }),
        }
    }
}

/// What a fit pass produced, as something a model can read.
///
/// One entry per node, minus the bulk: a node that produced a score or a
/// threshold has said something the caller needs, and one that produced a
/// 40-million-element tensor has not — and JSON-encoding that into an effect
/// result would put it in the journal forever.
///
/// The runtime's own bookkeeping keys (`__input_*`, `__state_*`) are not
/// results and do not belong in front of a model.
fn outputs_summary(
    outputs: &std::collections::HashMap<String, somatize_core::value::Value>,
) -> serde_json::Value {
    let mut summary = serde_json::Map::new();
    for (node_id, value) in outputs {
        if node_id.starts_with("__") {
            continue;
        }
        summary.insert(node_id.clone(), summarize_state(value));
    }
    serde_json::Value::Object(summary)
}

/// How many elements a learned array can have before it counts as weights.
const WEIGHTS_THRESHOLD: usize = 32;

fn summarize_state(state: &somatize_core::value::Value) -> serde_json::Value {
    let json = state.to_plain_json();
    if is_bulk(&json) {
        return serde_json::json!({ "fitted": true });
    }
    json
}

fn is_bulk(json: &serde_json::Value) -> bool {
    match json {
        serde_json::Value::Array(items) => {
            items.len() > WEIGHTS_THRESHOLD || items.iter().any(is_bulk)
        }
        serde_json::Value::Object(map) => map.values().any(is_bulk),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use somatize_core::filter::{Filter, FilterKind, FilterMeta, StreamMode};
    use somatize_core::graph::{Graph, Node};
    use somatize_core::value::Value;

    struct Doubler;

    impl Filter for Doubler {
        fn meta(&self) -> FilterMeta {
            FilterMeta {
                name: "doubler".into(),
                kind: FilterKind::Stateless,
                cacheable: true,
                differentiable: false,
                deterministic: true,
                stream_mode: StreamMode::FixedState,
                distribution: somatize_core::filter::Distribution::Local,
                input_schema: None,
                output_schema: None,
            }
        }
        fn fit(&self, _x: &Value, _y: Option<&Value>) -> Result<Value> {
            Ok(Value::Empty)
        }
        fn forward(&self, x: &Value, _state: &Value) -> Result<Value> {
            let (data, shape) = x
                .as_tensor()
                .ok_or(SomaError::Other("not a tensor".into()))?;
            Ok(Value::tensor(
                data.iter().map(|v| v * 2.0).collect(),
                shape.to_vec(),
            ))
        }
        fn config_hash(&self) -> somatize_core::cache::CacheKey {
            somatize_core::cache::CacheKey::from_parts(&[b"doubler"])
        }
    }

    fn handler() -> GraphHandler {
        let mut library = NodeCatalog::new();
        library.register("double", Box::new(Doubler));
        GraphHandler::new(library)
    }

    fn one_node_graph() -> Graph {
        let mut graph = Graph::new();
        graph.add_node(Node::filter_with_id("double", "doubler"));
        graph
    }

    #[test]
    fn a_graph_effect_runs_the_graph() {
        let result = handler()
            .perform(&Effect::Graph {
                graph: Box::new(one_node_graph()),
                input: Value::tensor(vec![1.0, 2.0], vec![2]),
                mode: GraphEffectMode::Forward,
            })
            .unwrap();

        match result {
            EffectResult::Graph(value) => {
                let (data, _) = value.as_tensor().unwrap();
                assert_eq!(data, &[2.0, 4.0]);
            }
            other => panic!("{other:?}"),
        }
    }

    /// A pipeline that will not run is a finding, not a crash: the agent
    /// reads it and tries something else.
    #[test]
    fn a_failing_graph_comes_back_as_a_result() {
        let mut graph = Graph::new();
        graph.add_node(Node::filter_with_id("missing", "nowhere"));

        let result = handler()
            .perform(&Effect::Graph {
                graph: Box::new(graph),
                input: Value::tensor(vec![1.0], vec![1]),
                mode: GraphEffectMode::Forward,
            })
            .unwrap();

        assert!(matches!(result, EffectResult::Failed { .. }));
    }

    use somatize_core::effect::{LlmRequest, LlmResponse, StopReason};
    use somatize_core::message::Message;
    use somatize_core::step::{StepCtx, StepMeta, Transition};

    /// Answers every model call with a fixed string.
    struct CannedLlm(&'static str);

    impl EffectHandler for CannedLlm {
        fn handles(&self, effect: &Effect) -> bool {
            matches!(effect, Effect::Llm(_))
        }
        fn perform(&self, _effect: &Effect) -> Result<EffectResult> {
            Ok(EffectResult::Llm(LlmResponse {
                message: Message::assistant(self.0),
                stop_reason: StopReason::EndTurn,
                usage: Default::default(),
                model: None,
            }))
        }
    }

    /// Asks the model once, then hands back what it said.
    struct AskOnce;

    impl somatize_core::step::Step for AskOnce {
        fn config_hash(&self) -> somatize_core::cache::CacheKey {
            somatize_core::cache::CacheKey::from_parts(&[b"AskOnce"])
        }
        fn meta(&self) -> StepMeta {
            StepMeta::new("AskOnce")
        }
        fn poll(&self, ctx: &StepCtx<'_>) -> Result<Transition> {
            if ctx.turn == 0 {
                return Ok(Transition::Await(vec![Effect::Llm(LlmRequest::new(
                    "claude-opus-5",
                    vec![Message::user("inner question")].into(),
                ))]));
            }
            let text = match ctx.result() {
                Some(EffectResult::Llm(r)) => r.message.text(),
                other => format!("unexpected result: {other:?}"),
            };
            Ok(Transition::Done(Value::text(text)))
        }
    }

    fn journal() -> EffectJournal {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(crate::cache::FsActionStore::new(dir.keep()).unwrap());
        EffectJournal::new(store.clone(), store)
    }

    /// Agent → pipeline → agent: a graph effect whose sub-graph itself
    /// contains a step. The handler builds the inner step a driver of its
    /// own, sharing the sibling handlers and the journal.
    #[test]
    fn a_sub_graph_containing_a_step_runs() {
        let mut library = NodeCatalog::new();
        library.register_step("ask", Box::new(AskOnce));

        let llm: Arc<dyn EffectHandler> = Arc::new(CannedLlm("the inner answer"));
        let handler = GraphHandler::new(library).with_step_runtime(vec![llm], journal());

        let mut graph = Graph::new();
        graph.add_node(Node::step("ask", "AskOnce"));

        let result = handler
            .perform(&Effect::Graph {
                graph: Box::new(graph),
                input: Value::text("outer input"),
                mode: GraphEffectMode::Forward,
            })
            .unwrap();

        match result {
            EffectResult::Graph(value) => {
                assert_eq!(value.as_text(), Some("the inner answer"));
            }
            other => panic!("expected the inner step's output, got {other:?}"),
        }
    }

    /// Without a step runtime, a step-containing sub-graph is a readable
    /// failure that names the fix, not the executor's generic error.
    #[test]
    fn a_step_sub_graph_without_a_runtime_names_the_fix() {
        let mut library = NodeCatalog::new();
        library.register_step("ask", Box::new(AskOnce));

        let mut graph = Graph::new();
        graph.add_node(Node::step("ask", "AskOnce"));

        let result = GraphHandler::new(library)
            .perform(&Effect::Graph {
                graph: Box::new(graph),
                input: Value::Empty,
                mode: GraphEffectMode::Forward,
            })
            .unwrap();

        match result {
            EffectResult::Failed { message } => {
                assert!(message.contains("with_step_runtime"), "{message}");
            }
            other => panic!("expected a failure, got {other:?}"),
        }
    }

    /// A step that keeps running its own graph again. The recursion must
    /// end at the depth cap with a failure the agent can read — not a stack
    /// overflow.
    struct Recurse;

    impl somatize_core::step::Step for Recurse {
        fn config_hash(&self) -> somatize_core::cache::CacheKey {
            somatize_core::cache::CacheKey::from_parts(&[b"Recurse"])
        }
        fn meta(&self) -> StepMeta {
            StepMeta::new("Recurse")
        }
        fn poll(&self, ctx: &StepCtx<'_>) -> Result<Transition> {
            if ctx.turn == 0 {
                let mut graph = Graph::new();
                graph.add_node(Node::step("recurse", "Recurse"));
                return Ok(Transition::Await(vec![Effect::Graph {
                    graph: Box::new(graph),
                    input: Value::text("again"),
                    mode: GraphEffectMode::Forward,
                }]));
            }
            let text = match ctx.result() {
                Some(EffectResult::Graph(v)) => v.as_text().unwrap_or_default().to_string(),
                Some(EffectResult::Failed { message }) => message.clone(),
                other => format!("unexpected: {other:?}"),
            };
            Ok(Transition::Done(Value::text(text)))
        }
    }

    #[test]
    fn nesting_stops_at_the_depth_cap() {
        let mut library = NodeCatalog::new();
        library.register_step("recurse", Box::new(Recurse));

        let handler = GraphHandler::new(library).with_step_runtime(Vec::new(), journal());

        let mut graph = Graph::new();
        graph.add_node(Node::step("recurse", "Recurse"));

        let result = handler
            .perform(&Effect::Graph {
                graph: Box::new(graph),
                input: Value::text("go"),
                mode: GraphEffectMode::Forward,
            })
            .unwrap();

        // The innermost level fails at the cap; every level above hands the
        // message outward as its own output.
        match result {
            EffectResult::Graph(value) => {
                let text = value.as_text().unwrap_or_default();
                assert!(
                    text.contains(&format!("depth {MAX_GRAPH_DEPTH}")),
                    "the failure should name the cap, got: {text}"
                );
            }
            other => panic!("expected the propagated cap message, got {other:?}"),
        }
    }

    use crate::effects::NodeOutcome;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Doubles, counting how often the graph actually runs it.
    ///
    /// `cacheable: false`, deliberately: the sub-graph's own output cache
    /// must not be able to spare the second run, so the only thing that
    /// can is the journal — which is what the test is about.
    struct CountingDoubler {
        calls: Arc<AtomicUsize>,
    }

    impl Filter for CountingDoubler {
        fn config_hash(&self) -> somatize_core::cache::CacheKey {
            somatize_core::cache::CacheKey::from_parts(&[b"CountingDoubler"])
        }
        fn fit(&self, _x: &Value, _y: Option<&Value>) -> Result<Value> {
            Ok(Value::Empty)
        }
        fn forward(&self, x: &Value, _state: &Value) -> Result<Value> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let (data, shape) = x
                .as_tensor()
                .ok_or(SomaError::Other("not a tensor".into()))?;
            Ok(Value::tensor(
                data.iter().map(|v| v * 2.0).collect(),
                shape.to_vec(),
            ))
        }
        fn meta(&self) -> FilterMeta {
            FilterMeta {
                name: "counting".into(),
                kind: FilterKind::Stateless,
                cacheable: false,
                differentiable: false,
                deterministic: true,
                stream_mode: StreamMode::FixedState,
                distribution: somatize_core::filter::Distribution::Local,
                input_schema: None,
                output_schema: None,
            }
        }
    }

    /// Awaits one filter-only Forward graph effect, then reports its output.
    struct RunsPipeline;

    impl somatize_core::step::Step for RunsPipeline {
        fn config_hash(&self) -> somatize_core::cache::CacheKey {
            somatize_core::cache::CacheKey::from_parts(&[b"RunsPipeline"])
        }
        fn meta(&self) -> StepMeta {
            StepMeta::new("RunsPipeline")
        }
        fn poll(&self, ctx: &StepCtx<'_>) -> Result<Transition> {
            if ctx.turn == 0 {
                let mut graph = Graph::new();
                graph.add_node(Node::filter_with_id("count", "counting"));
                return Ok(Transition::Await(vec![Effect::Graph {
                    graph: Box::new(graph),
                    input: Value::tensor(vec![3.0], vec![1]),
                    mode: GraphEffectMode::Forward,
                }]));
            }
            match ctx.result() {
                Some(EffectResult::Graph(v)) => Ok(Transition::Done(v.clone())),
                other => Ok(Transition::Done(Value::text(format!(
                    "unexpected: {other:?}"
                )))),
            }
        }
    }

    /// A filter-only Forward graph effect is *pure*: it keys on content, so
    /// the journal serves it to any run, like the filter cache it rides on.
    /// Two different runs asking for the identical pipeline must cost one
    /// execution — the second is a journal hit, not a re-run. This is the
    /// property that makes a crashed research loop replay its finished
    /// experiments instead of paying for them again.
    #[test]
    fn an_identical_pure_graph_effect_is_served_from_the_journal() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut library = NodeCatalog::new();
        library.register(
            "count",
            Box::new(CountingDoubler {
                calls: calls.clone(),
            }),
        );

        let d = EffectDriver::new(journal()).with_handler(Arc::new(GraphHandler::new(library)));

        // Different run ids on purpose: an *impure* effect would re-perform
        // for run-B, a pure one must not.
        let first = d
            .run(&RunsPipeline, "run-A", "agent", &Value::Empty)
            .unwrap();
        let second = d
            .run(&RunsPipeline, "run-B", "agent", &Value::Empty)
            .unwrap();

        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "an identical pure graph effect re-ran the pipeline"
        );
        match (first, second) {
            (NodeOutcome::Produced(a), NodeOutcome::Produced(b)) => {
                assert_eq!(a.as_tensor().map(|(d, _)| d.to_vec()), Some(vec![6.0]));
                assert_eq!(a, b, "the journal served a different answer");
            }
            other => panic!("expected two Done outcomes, got {other:?}"),
        }
    }

    /// Learns the mean, then subtracts it — a filter whose fitted state has
    /// a visible effect on a later forward.
    struct MeanFilter;
    impl Filter for MeanFilter {
        fn config_hash(&self) -> somatize_core::cache::CacheKey {
            somatize_core::cache::CacheKey::from_parts(&[b"Mean"])
        }
        fn fit(&self, x: &Value, _y: Option<&Value>) -> Result<Value> {
            let (data, _) = x
                .as_tensor()
                .ok_or(SomaError::Other("need tensor".into()))?;
            let mean = data.iter().sum::<f64>() / data.len() as f64;
            Ok(Value::json(serde_json::json!({ "mean": mean })))
        }
        fn forward(&self, x: &Value, state: &Value) -> Result<Value> {
            let (data, shape) = x
                .as_tensor()
                .ok_or(SomaError::Other("need tensor".into()))?;
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
            FilterMeta {
                name: "mean".into(),
                kind: FilterKind::Trainable,
                cacheable: true,
                differentiable: false,
                deterministic: true,
                stream_mode: StreamMode::FixedState,
                distribution: somatize_core::filter::Distribution::Local,
                input_schema: None,
                output_schema: None,
            }
        }
    }

    /// `Fit` mode answers with a JSON summary an agent can read — never the
    /// bulk outputs, which would sit in the journal forever — and the fitted
    /// states land in the handler's shared state store, so the *next*
    /// Forward through the same handler runs fitted. That second half is the
    /// claim the handler's session-clone comment makes; this is the test
    /// that would catch a session that stopped sharing states.
    #[test]
    fn fit_mode_fits_and_summarizes() {
        let mut library = NodeCatalog::new();
        library.register("mean", Box::new(MeanFilter));
        let handler = GraphHandler::new(library);

        let mut graph = Graph::new();
        graph.add_node(Node::filter_with_id("mean", "mean"));
        let input = Value::tensor(vec![10.0, 20.0, 30.0], vec![3]);

        let fitted = handler
            .perform(&Effect::Graph {
                graph: Box::new(graph.clone()),
                input: input.clone(),
                mode: GraphEffectMode::Fit,
            })
            .unwrap();
        let EffectResult::Graph(summary) = fitted else {
            panic!("expected a graph result, got {fitted:?}");
        };
        let json = summary
            .as_json()
            .expect("a fit answers with a JSON summary, not bulk output");
        assert!(json.get("mean").is_some(), "no entry for the node: {json}");

        // mean = 20 was learned above: forward must subtract it. An
        // unfitted graph would fall back to 0 and echo the input.
        let forwarded = handler
            .perform(&Effect::Graph {
                graph: Box::new(graph),
                input,
                mode: GraphEffectMode::Forward,
            })
            .unwrap();
        let EffectResult::Graph(out) = forwarded else {
            panic!("expected a graph result, got {forwarded:?}");
        };
        let (data, _) = out.as_tensor().expect("a tensor");
        assert_eq!(
            data,
            &[-10.0, 0.0, 10.0],
            "the forward did not see the state the fit just learned"
        );
    }

    #[test]
    fn the_handler_claims_only_graph_effects() {
        let h = handler();
        assert!(h.handles(&Effect::Graph {
            graph: Box::new(Graph::new()),
            input: Value::Empty,
            mode: GraphEffectMode::Forward,
        }));
        assert!(!h.handles(&Effect::Sleep(std::time::Duration::from_secs(1))));
    }
}
