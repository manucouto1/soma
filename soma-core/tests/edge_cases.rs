//! Core-type edge cases: odd values, degenerate graphs, boundary shapes.

use somatize_core::cache::CacheKey;
use somatize_core::data::value::Value;
use somatize_core::error::SomaError;
use somatize_core::graph::{Edge, Graph, Node, linear_pipeline};
use somatize_core::optimizer::search::{Scale, SearchDimension, SearchSpace};
use somatize_core::optimizer::study::{
    Direction, Objective, SearchStrategy, Study, Trial, TrialState,
};
use somatize_core::tracking::event::MetricRecord;

// ── Value edge cases ──

#[test]
fn value_empty_tensor() {
    let v = Value::tensor(vec![], vec![0]);
    assert_eq!(v.size(), 0);
    let (data, shape) = v.as_tensor().unwrap();
    assert!(data.is_empty());
    assert_eq!(shape, &[0]);
}

#[test]
fn value_single_element_tensor() {
    let v = Value::tensor(vec![42.0], vec![1]);
    assert_eq!(v.size(), 1);
}

#[test]
fn value_empty_json() {
    let v = Value::json(serde_json::json!({}));
    assert!(v.size() > 0); // "{}" has length 2
}

#[test]
fn value_nested_json() {
    let v = Value::json(serde_json::json!({"a": {"b": {"c": 1}}}));
    assert!(v.as_json().unwrap()["a"]["b"]["c"] == 1);
}

// ── CacheKey edge cases ──

#[test]
fn cache_key_empty_data() {
    let k = CacheKey::hash_data(b"");
    let k2 = CacheKey::hash_data(b"");
    assert_eq!(k, k2); // deterministic
}

#[test]
fn cache_key_empty_parts() {
    let k = CacheKey::from_parts(&[]);
    let k2 = CacheKey::from_parts(&[]);
    assert_eq!(k, k2);
}

#[test]
fn cache_key_single_byte_differs() {
    let k1 = CacheKey::hash_data(b"\x00");
    let k2 = CacheKey::hash_data(b"\x01");
    assert_ne!(k1, k2);
}

// ── Graph edge cases ──

#[test]
fn graph_self_loop_detected_as_cycle() {
    let mut g = Graph::new();
    g.add_node(Node::new("a", "A", "F"));
    g.add_edge(Edge::data("e1", "a", "a")); // self-loop
    assert!(matches!(g.validate(), Err(SomaError::CycleDetected)));
}

#[test]
fn graph_empty_linear_pipeline() {
    let g = linear_pipeline(vec![]);
    assert!(g.nodes.is_empty());
    assert!(g.edges.is_empty());
    assert!(g.validate().is_ok());
}

#[test]
fn graph_disconnected_components() {
    let mut g = Graph::new();
    g.add_node(Node::new("a", "A", "F"));
    g.add_node(Node::new("b", "B", "F"));
    g.add_node(Node::new("c", "C", "F"));
    g.add_edge(Edge::data("e1", "a", "b"));
    // c is disconnected from a-b
    assert!(g.validate().is_ok());
    let sorted = g.topological_sort().unwrap();
    assert_eq!(sorted.len(), 3);
    // c must appear somewhere (level 0 alongside a)
    assert!(sorted.contains(&"c"));
}

#[test]
fn graph_diamond_roots_and_leaves() {
    let mut g = Graph::new();
    g.add_node(Node::new("r", "Root", "F"));
    g.add_node(Node::new("l", "Left", "F"));
    g.add_node(Node::new("ri", "Right", "F"));
    g.add_node(Node::new("m", "Merge", "F"));
    g.add_edge(Edge::data("e1", "r", "l"));
    g.add_edge(Edge::data("e2", "r", "ri"));
    g.add_edge(Edge::data("e3", "l", "m"));
    g.add_edge(Edge::data("e4", "ri", "m"));

    assert_eq!(g.roots(), vec!["r"]);
    assert_eq!(g.leaves(), vec!["m"]);
}

// ── Search edge cases ──

#[test]
fn search_space_merge_empty() {
    let mut space = SearchSpace::new();
    space.merge_with_prefix("Empty", SearchSpace::new());
    assert!(space.is_empty());
}

#[test]
fn search_space_freeze_nonexistent() {
    let mut space = SearchSpace::new();
    space.add(SearchDimension::Float {
        name: "lr".into(),
        low: 0.001,
        high: 0.1,
        scale: Scale::Log,
        default: None,
    });
    space.freeze("nonexistent_param", serde_json::json!(42));
    assert_eq!(space.len(), 1); // lr still there
}

#[test]
fn search_dimension_int_single_value_range() {
    let dim = SearchDimension::Int {
        name: "n".into(),
        low: 5,
        high: 5, // low == high
        scale: Scale::Linear,
    };
    // This should fail validation (low must be < high)
    assert!(dim.validate().is_err());
}

// ── Study edge cases ──

#[test]
fn study_best_trial_single_trial() {
    let mut study = Study::new(
        "test",
        SearchSpace::new(),
        SearchStrategy::Random {
            n_trials: 1,
            seed: None,
        },
        vec![Objective {
            metric: "f1".into(),
            direction: Direction::Maximize,
        }],
    );
    let mut t = Trial::new("t1", std::collections::HashMap::new());
    t.state = TrialState::Completed;
    t.metrics.push(MetricRecord {
        name: "f1".into(),
        value: 0.5,
        step: 0,
        timestamp: chrono::Utc::now(),
    });
    study.trials.push(t);

    assert_eq!(study.best_trial().unwrap().id, "t1");
}

#[test]
fn study_best_trial_all_same_value() {
    let mut study = Study::new(
        "test",
        SearchSpace::new(),
        SearchStrategy::Random {
            n_trials: 3,
            seed: None,
        },
        vec![Objective {
            metric: "f1".into(),
            direction: Direction::Maximize,
        }],
    );
    for i in 0..3 {
        let mut t = Trial::new(format!("t{i}"), std::collections::HashMap::new());
        t.state = TrialState::Completed;
        t.metrics.push(MetricRecord {
            name: "f1".into(),
            value: 0.8, // all same
            step: 0,
            timestamp: chrono::Utc::now(),
        });
        study.trials.push(t);
    }
    // Should return some trial (deterministic)
    assert!(study.best_trial().is_some());
}

#[test]
fn study_progress_no_total() {
    let study = Study::new(
        "test",
        SearchSpace::new(),
        SearchStrategy::Grid { points_per_dim: 5 }, // n_trials unknown
        vec![],
    );
    assert_eq!(study.progress(), 0.0);
}

#[test]
fn trial_best_metric_empty_metrics() {
    let t = Trial::new("t1", std::collections::HashMap::new());
    assert!(t.best_metric("f1", Direction::Maximize).is_none());
}

#[test]
fn trial_best_metric_multiple_steps() {
    let mut t = Trial::new("t1", std::collections::HashMap::new());
    t.state = TrialState::Completed;
    for (step, val) in [(0, 0.5), (1, 0.7), (2, 0.6), (3, 0.9)] {
        t.metrics.push(MetricRecord {
            name: "f1".into(),
            value: val,
            step,
            timestamp: chrono::Utc::now(),
        });
    }
    // Maximize: should find 0.9
    assert_eq!(t.best_metric("f1", Direction::Maximize), Some(0.9));
    // Minimize: should find 0.5
    assert_eq!(t.best_metric("f1", Direction::Minimize), Some(0.5));
}
