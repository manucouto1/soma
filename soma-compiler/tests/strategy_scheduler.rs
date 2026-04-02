//! Integration tests: TrainingStrategy affects Scheduler output.

use somatize_compiler::scheduler::{WorkerInfo, schedule};
use somatize_compiler::{CompileMode, compile};
use somatize_core::cache::CacheKey;
use somatize_core::error::Result;
use somatize_core::filter::{Filter, FilterKind, FilterMeta, StreamMode};
use somatize_core::graph::{Edge, Graph, Node};
use somatize_core::strategy::{
    CommunicationProtocol, GradientAggregation, Partition, TrainingStrategy,
};
use somatize_core::value::Value;

struct DummyFilter(String);

impl Filter for DummyFilter {
    fn config_hash(&self) -> CacheKey {
        CacheKey::from_parts(&[self.0.as_bytes()])
    }
    fn fit(&self, _x: &Value, _y: Option<&Value>) -> Result<Value> {
        Ok(Value::Empty)
    }
    fn forward(&self, x: &Value, _state: &Value) -> Result<Value> {
        Ok(x.clone())
    }
    fn meta(&self) -> FilterMeta {
        FilterMeta {
            name: self.0.clone(),
            kind: FilterKind::Trainable,
            cacheable: true,
            differentiable: true,
            stream_mode: StreamMode::FixedState,
            distribution: somatize_core::filter::Distribution::Local,
            input_schema: None,
            output_schema: None,
        }
    }
}

fn linear_graph(ids: &[&str]) -> Graph {
    let mut g = Graph::new();
    for &id in ids {
        g.nodes.push(Node::new(id, id, id));
    }
    for (i, pair) in ids.windows(2).enumerate() {
        g.edges.push(Edge::data(format!("e{i}"), pair[0], pair[1]));
    }
    g
}

fn test_workers() -> Vec<WorkerInfo> {
    vec![
        WorkerInfo {
            id: "w1".into(),
            name: "GPU Worker 1".into(),
            tags: vec!["gpu".into(), "training".into()],
            gpu: true,
            cpu_cores: 8,
            active_jobs: 0,
            max_concurrent: 4,
        },
        WorkerInfo {
            id: "w2".into(),
            name: "GPU Worker 2".into(),
            tags: vec!["gpu".into(), "training".into()],
            gpu: true,
            cpu_cores: 8,
            active_jobs: 0,
            max_concurrent: 4,
        },
        WorkerInfo {
            id: "w3".into(),
            name: "CPU Worker".into(),
            tags: vec!["cpu".into(), "inference".into()],
            gpu: false,
            cpu_cores: 16,
            active_jobs: 0,
            max_concurrent: 8,
        },
    ]
}

#[test]
fn local_strategy_compiles_normally() {
    let graph = linear_graph(&["scaler", "model"]);

    let mut registry = somatize_compiler::SimpleFilterRegistry::new();
    registry.register("scaler", &DummyFilter("Scaler".into()));
    registry.register("model", &DummyFilter("Model".into()));

    let result = compile(&graph, &registry, CompileMode::NoCache, None).unwrap();
    assert_eq!(result.plan.node_count(), 2);
}

#[test]
fn scheduler_distributes_parallel_branches() {
    let mut graph = Graph::new();
    graph.nodes.push(Node::new("input", "Input", "Input"));
    graph.nodes.push(Node::new("branch_a", "A", "A"));
    graph.nodes.push(Node::new("branch_b", "B", "B"));
    graph.nodes.push(Node::new("merge", "Merge", "Merge"));
    graph.edges.push(Edge::data("e1", "input", "branch_a"));
    graph.edges.push(Edge::data("e2", "input", "branch_b"));
    graph.edges.push(Edge::data("e3", "branch_a", "merge"));
    graph.edges.push(Edge::data("e4", "branch_b", "merge"));

    let mut registry = somatize_compiler::SimpleFilterRegistry::new();
    for id in ["input", "branch_a", "branch_b", "merge"] {
        registry.register(id, &DummyFilter(id.into()));
    }

    let result = compile(&graph, &registry, CompileMode::NoCache, None).unwrap();

    let workers = test_workers();
    let dist_plan = schedule(&result.plan, &workers, &[]);

    // Should have assignments
    assert!(!dist_plan.assignments.is_empty());
    // At least 2 nodes should be assigned
    assert!(dist_plan.assignments.len() >= 2);
}

#[test]
fn scheduler_groups_differentiable_nodes() {
    let graph = linear_graph(&["encoder", "decoder"]);

    let mut registry = somatize_compiler::SimpleFilterRegistry::new();
    registry.register("encoder", &DummyFilter("Encoder".into()));
    registry.register("decoder", &DummyFilter("Decoder".into()));

    let result = compile(&graph, &registry, CompileMode::NoCache, None).unwrap();

    let workers = test_workers();
    let diff_nodes = vec!["encoder".to_string(), "decoder".to_string()];
    let dist_plan = schedule(&result.plan, &workers, &diff_nodes);

    // Differentiable nodes should be on the same worker
    let encoder_worker = dist_plan
        .assignments
        .iter()
        .find(|a| a.node_id == "encoder")
        .map(|a| &a.worker_id);
    let decoder_worker = dist_plan
        .assignments
        .iter()
        .find(|a| a.node_id == "decoder")
        .map(|a| &a.worker_id);

    if let (Some(ew), Some(dw)) = (encoder_worker, decoder_worker) {
        assert_eq!(ew, dw, "differentiable nodes should be on the same worker");
    }
}

#[test]
fn training_strategy_serializes_on_graph() {
    let mut graph = linear_graph(&["a", "b"]);
    graph.set_strategy(TrainingStrategy::DataParallel {
        num_replicas: 4,
        aggregation: GradientAggregation::AllReduce,
    });

    assert!(graph.training_strategy.is_some());

    // Serialize and deserialize
    let json = serde_json::to_string(&graph).unwrap();
    let parsed: Graph = serde_json::from_str(&json).unwrap();
    assert!(parsed.training_strategy.is_some());
    assert!(matches!(
        parsed.training_strategy.unwrap(),
        TrainingStrategy::DataParallel {
            num_replicas: 4,
            ..
        }
    ));
}

#[test]
fn model_parallel_partition_validates() {
    let mut graph = linear_graph(&["embed", "backbone", "head"]);
    graph.set_strategy(TrainingStrategy::ModelParallel {
        partitions: vec![
            Partition {
                node_ids: vec!["embed".into(), "backbone".into()],
                target: somatize_core::filter::RemoteTarget::Tag("gpu-0".into()),
            },
            Partition {
                node_ids: vec!["head".into()],
                target: somatize_core::filter::RemoteTarget::Tag("gpu-1".into()),
            },
        ],
        communication: CommunicationProtocol::Pipeline {
            micro_batch_size: 4,
        },
    });

    // Graph should still validate (strategy doesn't affect topology)
    assert!(graph.validate().is_ok());

    // All nodes accounted for in partitions
    if let Some(TrainingStrategy::ModelParallel { partitions, .. }) = &graph.training_strategy {
        let all_nodes: Vec<&str> = partitions
            .iter()
            .flat_map(|p| p.node_ids.iter().map(|s| s.as_str()))
            .collect();
        assert!(all_nodes.contains(&"embed"));
        assert!(all_nodes.contains(&"backbone"));
        assert!(all_nodes.contains(&"head"));
    }
}

#[test]
fn effective_strategy_defaults_to_local() {
    let graph = Graph::new();
    assert!(matches!(
        graph.effective_strategy(),
        TrainingStrategy::Local
    ));
}

#[test]
fn with_strategy_builder() {
    let graph = Graph::new().with_strategy(TrainingStrategy::DataParallel {
        num_replicas: 2,
        aggregation: GradientAggregation::ParameterServer,
    });
    assert!(matches!(
        graph.effective_strategy(),
        TrainingStrategy::DataParallel {
            num_replicas: 2,
            ..
        }
    ));
}
