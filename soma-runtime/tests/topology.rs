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
