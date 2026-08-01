//! A forward pass must follow the graph, not the plan's node order.

use somatize_core::cache::CacheKey;
use somatize_core::error::Result;
use somatize_core::filter::{Distribution, Filter, FilterKind, FilterMeta, StreamMode};
use somatize_core::graph::{Edge, Graph, Node};
use somatize_core::value::Value;
use somatize_runtime::{GraphSession, NodeCatalog};

/// Reports what it received, so the shape of the answer shows the wiring.
struct Echo(String);

impl Filter for Echo {
    fn config_hash(&self) -> CacheKey {
        CacheKey::from_parts(&[self.0.as_bytes()])
    }
    fn fit(&self, _x: &Value, _y: Option<&Value>) -> Result<Value> {
        Ok(Value::Empty)
    }
    fn forward(&self, x: &Value, _s: &Value) -> Result<Value> {
        let seen = match x {
            Value::Text(t) => t.to_string(),
            Value::Json(j) => j.to_string(),
            other => format!("{other:?}"),
        };
        Ok(Value::text(format!("{}[{seen}]", self.0)))
    }
    fn meta(&self) -> FilterMeta {
        FilterMeta {
            name: self.0.clone(),
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

/// `a → {b, c} → d`.
///
/// The runner derived its own topology from the plan's flattened node
/// order, chaining `a → b → c → d`. So `d` saw only `c`, `b`'s output went
/// nowhere, and `a` — no longer believed to be a root — never received the
/// input at all. The answer was `d[c[…]]`.
#[test]
fn a_diamond_reaches_both_branches() {
    let mut g = Graph::new();
    for id in ["a", "b", "c", "d"] {
        g.add_node(Node::filter_with_id(id, id));
    }
    g.add_edge(Edge::data("e1", "a", "b"));
    g.add_edge(Edge::data("e2", "a", "c"));
    g.add_edge(Edge::data("e3", "b", "d"));
    g.add_edge(Edge::data("e4", "c", "d"));

    let mut lib = NodeCatalog::new();
    for id in ["a", "b", "c", "d"] {
        lib.register(id, Box::new(Echo(id.to_string())));
    }

    let session = GraphSession::new(g, lib);
    let out = session.forward(&Value::text("X")).unwrap();
    let text = out.as_text().unwrap_or_default().to_string();

    assert!(
        text.starts_with("d["),
        "the leaf should be the answer: {text}"
    );
    assert!(
        text.contains("b[a[X]]"),
        "`d` must see the branch through `b`: {text}"
    );
    assert!(
        text.contains("c[a[X]]"),
        "`d` must see the branch through `c`: {text}"
    );
}

/// A trainable node behind a fan-out is fitted on what its predecessors
/// produced.
///
/// `forward` on this shape has a test; `fit` did not, and it was broken —
/// twice over, in opposite directions. `LocalRunner::fit` builds an
/// output store, and `fit_sequence` used to hand each step of a sequence
/// a *fresh* one. So `sink`, running in a later step than the parallel
/// pair that feeds it, looked its predecessors up in an empty map and was
/// fitted on `{}`.
///
/// Before the runner was given the real topology it was wrong the other
/// way: `GraphInfo::for_linear` said `sink` had one predecessor, and the
/// one-predecessor arm falls back to the threaded input, so the fit saw
/// whichever branch happened to be last in plan order.
#[test]
fn a_fan_in_is_fitted_on_both_branches() {
    use somatize_core::error::Result as SomaResult;
    use std::sync::{Arc, Mutex};

    /// Records what it was asked to fit on.
    struct Recorder(Arc<Mutex<Option<Value>>>);

    impl Filter for Recorder {
        fn config_hash(&self) -> CacheKey {
            CacheKey::from_parts(&[b"Recorder"])
        }
        fn fit(&self, x: &Value, _y: Option<&Value>) -> SomaResult<Value> {
            *self.0.lock().unwrap() = Some(x.clone());
            Ok(Value::Empty)
        }
        fn forward(&self, x: &Value, _s: &Value) -> SomaResult<Value> {
            Ok(x.clone())
        }
        fn meta(&self) -> FilterMeta {
            FilterMeta {
                name: "Recorder".into(),
                kind: FilterKind::Trainable,
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

    let seen = Arc::new(Mutex::new(None));

    let mut g = Graph::new();
    for id in ["src", "left", "right"] {
        g.add_node(Node::filter_with_id(id, id));
    }
    g.add_node(Node::filter_with_id("sink", "sink"));
    g.add_edge(Edge::data("e1", "src", "left"));
    g.add_edge(Edge::data("e2", "src", "right"));
    g.add_edge(Edge::data("e3", "left", "sink"));
    g.add_edge(Edge::data("e4", "right", "sink"));

    let mut lib = NodeCatalog::new();
    for id in ["src", "left", "right"] {
        lib.register(id, Box::new(Echo(id.to_string())));
    }
    lib.register("sink", Box::new(Recorder(seen.clone())));

    let mut session = GraphSession::new(g, lib);
    session.fit(&Value::text("X"), None).unwrap();

    let fitted = seen
        .lock()
        .unwrap()
        .clone()
        .expect("sink should have been fitted");
    let json = fitted
        .as_json()
        .unwrap_or_else(|| panic!("a fan-in should be fitted on a map, got {fitted:?}"));
    let obj = json
        .as_object()
        .expect("a JSON object keyed by predecessor");

    assert!(
        obj.contains_key("left") && obj.contains_key("right"),
        "the fit must see both branches, got {obj:?}"
    );
}
