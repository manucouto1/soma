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
use crate::effects::EffectHandler;
use crate::graph_session::GraphSession;
use crate::node_catalog::NodeCatalog;
use somatize_core::cache::CacheStore;
use somatize_core::effect::{Effect, EffectResult, GraphEffectMode};
use somatize_core::error::{Result, SomaError};
use std::sync::Arc;

/// Runs graphs on behalf of a step.
///
/// Holds the filters those graphs are built from — a graph names its nodes,
/// it does not carry their implementations — plus the cache they share.
pub struct GraphHandler {
    library: NodeCatalog,
    cache: Arc<dyn CacheStore>,
}

impl GraphHandler {
    /// A handler over `library`, with an in-memory cache.
    pub fn new(library: NodeCatalog) -> Self {
        Self {
            library,
            cache: Arc::new(MemoryCache::new(64 * 1024 * 1024)),
        }
    }

    /// Share the caller's cache, so a pipeline the agent runs hits the same
    /// entries the user's own runs wrote.
    pub fn with_cache(mut self, cache: Arc<dyn CacheStore>) -> Self {
        self.cache = cache;
        self
    }

    /// The filters available to graphs run through this handler.
    pub fn library(&self) -> &NodeCatalog {
        &self.library
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
