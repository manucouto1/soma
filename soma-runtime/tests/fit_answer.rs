//! What a fit answers with, and what it says on the bus while doing it.
//!
//! Two things used to depend on which branch ran rather than on the type:
//! whether `fit`'s map held node outputs or trained states, and whether
//! the run bracket got closed at all.

use somatize_core::cache::CacheKey;
use somatize_core::data::value::Value;
use somatize_core::error::SomaError;
use somatize_core::graph::filter::{Distribution, Filter, FilterKind, FilterMeta, StreamMode};
use somatize_core::graph::{Edge, Graph, Node};
use somatize_core::tracking::EventSink;
use somatize_core::tracking::event::Event;
use somatize_runtime::EventBus;
use somatize_runtime::execution::graph_session::GraphSession;
use somatize_runtime::execution::node_catalog::NodeCatalog;
use std::sync::{Arc, Mutex};

/// Learns the mean, subtracts it. State and output are different values,
/// which is the whole point of asking which is which.
struct Centre;

impl Filter for Centre {
    fn config_hash(&self) -> CacheKey {
        CacheKey::from_parts(&[b"Centre"])
    }
    fn fit(&self, x: &Value, _y: Option<&Value>) -> somatize_core::Result<Value> {
        let (data, _) = x.as_tensor().unwrap();
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        Ok(Value::json(serde_json::json!({ "mean": mean })))
    }
    fn forward(&self, x: &Value, state: &Value) -> somatize_core::Result<Value> {
        let (data, shape) = x.as_tensor().unwrap();
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

/// Refuses to fit. The failing arm of the bracket was the one nobody
/// wrote consistently.
struct Refuses;

impl Filter for Refuses {
    fn config_hash(&self) -> CacheKey {
        CacheKey::from_parts(&[b"Refuses"])
    }
    fn fit(&self, _x: &Value, _y: Option<&Value>) -> somatize_core::Result<Value> {
        Err(SomaError::Other("this filter does not fit".into()))
    }
    fn forward(&self, _x: &Value, _state: &Value) -> somatize_core::Result<Value> {
        Err(SomaError::Other("this filter does not run either".into()))
    }
    fn meta(&self) -> FilterMeta {
        meta("Refuses", FilterKind::Trainable)
    }
}

fn meta(name: &str, kind: FilterKind) -> FilterMeta {
    FilterMeta {
        name: name.into(),
        kind,
        cacheable: false,
        differentiable: false,
        deterministic: true,
        stream_mode: StreamMode::FixedState,
        distribution: Distribution::Local,
        input_schema: None,
        output_schema: None,
    }
}

/// Collects everything the bus emits.
#[derive(Default)]
struct Recorder(Mutex<Vec<Event>>);

impl EventSink for Recorder {
    fn record(&self, event: &Event) {
        self.0.lock().unwrap().push(event.clone());
    }
    fn flush(&self) {}
}

impl Recorder {
    fn kinds(&self) -> Vec<String> {
        self.0
            .lock()
            .unwrap()
            .iter()
            .filter_map(|e| match e {
                Event::RunStarted { .. } => Some("RunStarted".to_string()),
                Event::RunCompleted { .. } => Some("RunCompleted".to_string()),
                Event::RunFailed { .. } => Some("RunFailed".to_string()),
                _ => None,
            })
            .collect()
    }
}

fn mean_of(state: &Value) -> f64 {
    state
        .as_json()
        .and_then(|j| j["mean"].as_f64())
        .expect("the state Centre learned")
}

fn one_node(id: &str, filter: Box<dyn Filter>) -> (Graph, NodeCatalog) {
    let mut graph = Graph::new();
    graph.add_node(Node::filter_with_id(id, id));
    let mut catalog = NodeCatalog::new();
    catalog.register(id.to_string(), filter);
    (graph, catalog)
}

#[test]
fn a_fit_says_which_half_is_which() {
    let mut graph = Graph::new();
    graph.add_node(Node::filter_with_id("centre", "centre"));
    graph.add_node(Node::filter_with_id("twice", "twice"));
    graph.add_edge(Edge::data("e0", "centre", "twice"));
    let mut catalog = NodeCatalog::new();
    catalog.register("centre".to_string(), Box::new(Centre));
    catalog.register("twice".to_string(), Box::new(Centre));

    let mut session = GraphSession::new(graph, catalog);
    let fitted = session
        .fit(&Value::tensor(vec![10.0, 20.0, 30.0], vec![3]), None)
        .expect("fit");

    // Outputs are what the nodes computed…
    let (data, _) = fitted.outputs["centre"].as_tensor().unwrap();
    assert_eq!(data, &[-10.0, 0.0, 10.0]);

    // …and states are what they learned. Different values, same node id.
    assert_eq!(mean_of(&fitted.states["centre"]), 20.0);
    assert_eq!(mean_of(&fitted.states["twice"]), 0.0);

    // The `__state_` prefix is a key inside the runner's value store. It
    // used to leak into every caller, which is how the Python
    // differentiable path came to file an output as a state.
    for key in fitted.outputs.keys().chain(fitted.states.keys()) {
        assert!(
            !key.starts_with("__"),
            "`{key}` is a store key, not an answer"
        );
    }

    // The last node that ran, for a caller that wants one value.
    let (last, _) = fitted.last.as_tensor().unwrap();
    assert_eq!(last, &[-10.0, 0.0, 10.0]);
}

#[test]
fn a_failed_fit_closes_its_bracket() {
    let recorder = Arc::new(Recorder::default());
    let bus = Arc::new(EventBus::new(64));
    bus.add_sink(recorder.clone());

    let (graph, catalog) = one_node("nope", Box::new(Refuses));
    let mut session = GraphSession::new(graph, catalog).with_event_bus(bus);
    let err = session.fit(&Value::tensor(vec![1.0], vec![1]), None);

    assert!(err.is_err(), "the filter refuses to fit");
    assert_eq!(
        recorder.kinds(),
        vec!["RunStarted", "RunFailed"],
        "a run that fails must say so, not leave the bracket open"
    );
}

#[test]
fn a_failed_run_closes_its_bracket() {
    let recorder = Arc::new(Recorder::default());
    let bus = Arc::new(EventBus::new(64));
    bus.add_sink(recorder.clone());

    let (graph, catalog) = one_node("nope", Box::new(Refuses));
    let mut session = GraphSession::new(graph, catalog).with_event_bus(bus);
    let err = session.run(somatize_compiler::CompileMode::NoCache);

    assert!(err.is_err());
    assert_eq!(recorder.kinds(), vec!["RunStarted", "RunFailed"]);
}
