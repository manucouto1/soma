//! Integration tests: PBT with real trainable filters and GraphSession.

use soma_core::cache::CacheKey;
use soma_core::error::Result;
use soma_core::event::Event;
use soma_core::filter::{Filter, FilterKind, FilterMeta, StreamMode};
use soma_core::graph::{Edge, Graph, Node};
use soma_core::search::{Scale, SearchDimension, SearchSpace};
use soma_core::strategy::{ExploitStrategy, ExploreStrategy};
use soma_core::value::Value;
use soma_runtime::EventBus;
use soma_runtime::cache::MemoryCache;
use soma_runtime::filter_library::FilterLibrary;
use soma_runtime::graph_session::GraphSession;
use soma_runtime::pbt_runner::{FnPbtExecutor, PbtConfig, PbtRunner, PopulationMember};
use std::sync::Arc;

/// Trainable scaler that learns mean from data.
struct TrainableScaler {
    scale: f64,
}

impl Filter for TrainableScaler {
    fn config_hash(&self) -> CacheKey {
        CacheKey::from_parts(&[b"TrainableScaler", &self.scale.to_le_bytes()])
    }
    fn fit(&self, x: &Value, _y: Option<&Value>) -> Result<Value> {
        let (data, _) = x
            .as_tensor()
            .ok_or(soma_core::error::SomaError::Other("need tensor".into()))?;
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        Ok(Value::json(serde_json::json!({"mean": mean})))
    }
    fn forward(&self, x: &Value, state: &Value) -> Result<Value> {
        let (data, shape) = x
            .as_tensor()
            .ok_or(soma_core::error::SomaError::Other("need tensor".into()))?;
        let mean = state
            .as_json()
            .and_then(|j| j["mean"].as_f64())
            .unwrap_or(0.0);
        let result: Vec<f64> = data.iter().map(|v| (v - mean) * self.scale).collect();
        Ok(Value::tensor(result, shape.to_vec()))
    }
    fn meta(&self) -> FilterMeta {
        FilterMeta {
            name: "TrainableScaler".into(),
            kind: FilterKind::Trainable,
            cacheable: true,
            differentiable: true,
            stream_mode: StreamMode::FixedState,
            distribution: soma_core::filter::Distribution::Local,
            input_schema: None,
            output_schema: None,
        }
    }
}

fn make_graph_and_library(scale: f64) -> (Graph, FilterLibrary) {
    let mut g = Graph::new();
    g.nodes
        .push(Node::new("scaler", "Scaler", "TrainableScaler"));
    let mut lib = FilterLibrary::new();
    lib.register("scaler", Box::new(TrainableScaler { scale }));
    (g, lib)
}

#[test]
fn pbt_with_graph_session_converges() {
    let bus = Arc::new(EventBus::new(512));
    let runner = PbtRunner::new(bus.clone());

    let mut space = SearchSpace::new();
    space.add(SearchDimension::Float {
        name: "scale".into(),
        low: 0.01,
        high: 10.0,
        scale: Scale::Log,
        default: None,
    });

    let config = PbtConfig {
        population_size: 8,
        generations: 5,
        exploit: ExploitStrategy::Truncation { fraction: 0.25 },
        explore: ExploreStrategy::Perturbation { factor: 0.3 },
        search_space: space,
        train_steps_per_generation: 1,
    };

    let train_data = Value::tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]);

    let executor = FnPbtExecutor {
        train_fn: move |member: &PopulationMember| {
            let scale = member
                .params
                .get("scale")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0);

            let (graph, lib) = make_graph_and_library(scale);
            let mut session = GraphSession::new(graph, lib);
            let outputs = session.fit(&train_data, None)?;

            // Return the scaler output as state
            Ok(outputs.get("scaler").cloned().unwrap_or(Value::Empty))
        },
        eval_fn: |member: &PopulationMember| {
            // Fitness: scale closest to 2.0 is best
            let scale = member
                .params
                .get("scale")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0);
            Ok(-(scale - 2.0).abs())
        },
    };

    let result = runner.run(&config, &executor).unwrap();

    assert_eq!(result.len(), 8);
    assert!(result.iter().all(|m| m.fitness.is_some()));

    // Best should have better fitness than worst
    let best_fitness = result[0].fitness.unwrap();
    let worst_fitness = result.last().unwrap().fitness.unwrap();
    assert!(best_fitness >= worst_fitness);
}

#[test]
fn pbt_emits_generation_events() {
    let bus = Arc::new(EventBus::new(512));
    let mut rx = bus.subscribe();
    let runner = PbtRunner::new(bus);

    let mut space = SearchSpace::new();
    space.add(SearchDimension::Float {
        name: "x".into(),
        low: 0.0,
        high: 1.0,
        scale: Scale::Linear,
        default: None,
    });

    let config = PbtConfig {
        population_size: 4,
        generations: 3,
        exploit: ExploitStrategy::Truncation { fraction: 0.5 },
        explore: ExploreStrategy::Perturbation { factor: 0.1 },
        search_space: space,
        train_steps_per_generation: 1,
    };

    let executor = FnPbtExecutor {
        train_fn: |_: &PopulationMember| Ok(Value::Empty),
        eval_fn: |member: &PopulationMember| {
            Ok(member
                .params
                .get("x")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0))
        },
    };

    runner.run(&config, &executor).unwrap();

    let mut events = Vec::new();
    while let Ok(e) = rx.try_recv() {
        events.push(e);
    }

    let gen_started = events
        .iter()
        .filter(|e| matches!(e, Event::GenerationStarted { .. }))
        .count();
    let gen_completed = events
        .iter()
        .filter(|e| matches!(e, Event::GenerationCompleted { .. }))
        .count();
    let exploits = events
        .iter()
        .filter(|e| matches!(e, Event::MemberExploited { .. }))
        .count();

    assert_eq!(gen_started, 3);
    assert_eq!(gen_completed, 3);
    assert!(exploits > 0, "should have exploit events with truncation");
}

#[test]
fn graph_session_fit_forward_roundtrip() {
    let mut graph = Graph::new();
    graph
        .nodes
        .push(Node::new("scaler", "Scaler", "TrainableScaler"));
    graph
        .nodes
        .push(Node::new("scaler2", "Scaler2", "TrainableScaler"));
    graph.edges.push(Edge::data("e1", "scaler", "scaler2"));

    let mut lib = FilterLibrary::new();
    lib.register("scaler", Box::new(TrainableScaler { scale: 2.0 }));
    lib.register("scaler2", Box::new(TrainableScaler { scale: 1.0 }));

    let mut session = GraphSession::new(graph, lib).with_cache(Arc::new(MemoryCache::default()));

    // Fit
    let train = Value::tensor(vec![10.0, 20.0, 30.0], vec![3]);
    let outputs = session.fit(&train, None).unwrap();

    // scaler: mean=20, forward: [(10-20)*2, (20-20)*2, (30-20)*2] = [-20, 0, 20]
    let scaler_out = outputs.get("scaler").unwrap();
    let (data, _) = scaler_out.as_tensor().unwrap();
    assert_eq!(data, &[-20.0, 0.0, 20.0]);

    // scaler2: mean=0, forward: [(-20-0)*1, (0-0)*1, (20-0)*1] = [-20, 0, 20]
    let scaler2_out = outputs.get("scaler2").unwrap();
    let (data2, _) = scaler2_out.as_tensor().unwrap();
    assert_eq!(data2, &[-20.0, 0.0, 20.0]);

    assert!(session.is_fitted());
}

#[test]
fn graph_session_subscribe_events() {
    let (graph, lib) = make_graph_and_library(1.0);
    let bus = Arc::new(EventBus::new(128));
    let mut rx = bus.subscribe();

    let mut session = GraphSession::new(graph, lib).with_event_bus(bus);

    let train = Value::tensor(vec![1.0, 2.0, 3.0], vec![3]);
    session.fit(&train, None).unwrap();

    let mut events = Vec::new();
    while let Ok(e) = rx.try_recv() {
        events.push(e);
    }

    assert!(
        events
            .iter()
            .any(|e| matches!(e, Event::NodeStarted { .. }))
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Event::NodeCompleted { .. }))
    );
}
