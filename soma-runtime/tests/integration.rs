//! End-to-end integration tests for full Soma workflows.
//!
//! These tests verify realistic researcher use cases:
//! define filters → build graph → fit → forward → cache hit → invalidation.

use somatize_compiler::{CompileMode, SimpleFilterRegistry, compile};
use somatize_core::cache::CacheKey;
use somatize_core::error::{Result, SomaError};
use somatize_core::event::MetricRecord;
use somatize_core::filter::{Filter, FilterKind, FilterMeta, StreamMode};
use somatize_core::graph::{Edge, Graph, Node, linear_pipeline};
use somatize_core::search::{Scale, SearchDimension, SearchSpace};
use somatize_core::study::{Direction, Objective, SearchStrategy, Study};
use somatize_core::value::Value;
use somatize_runtime::*;
use std::sync::Arc;

// ═══════════════════════════════════════════════
// Test Filters
// ═══════════════════════════════════════════════

/// Computes mean and std, normalizes: (x - mean) / std
struct Normalizer;

impl Filter for Normalizer {
    fn config_hash(&self) -> CacheKey {
        CacheKey::from_parts(&[b"Normalizer"])
    }
    fn fit(&self, x: &Value, _y: Option<&Value>) -> Result<Value> {
        let (data, _) = x
            .as_tensor()
            .ok_or(SomaError::Other("need tensor".into()))?;
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let std = (data.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
        let std = if std == 0.0 { 1.0 } else { std };
        Ok(Value::json(serde_json::json!({"mean": mean, "std": std})))
    }
    fn forward(&self, x: &Value, state: &Value) -> Result<Value> {
        let (data, shape) = x
            .as_tensor()
            .ok_or(SomaError::Other("need tensor".into()))?;
        let j = state
            .as_json()
            .ok_or(SomaError::Other("need json state".into()))?;
        let mean = j["mean"].as_f64().unwrap_or(0.0);
        let std = j["std"].as_f64().unwrap_or(1.0);
        let result: Vec<f64> = data.iter().map(|v| (v - mean) / std).collect();
        Ok(Value::tensor(result, shape.to_vec()))
    }
    fn meta(&self) -> FilterMeta {
        FilterMeta {
            name: "Normalizer".into(),
            kind: FilterKind::Trainable,
            cacheable: true,
            differentiable: true,
            stream_mode: StreamMode::FixedState,
            distribution: somatize_core::filter::Distribution::Local,
            input_schema: None,
            output_schema: None,
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Linear transformation: y = x * weight + bias
struct LinearModel {
    learning_rate: f64,
}

impl Filter for LinearModel {
    fn config_hash(&self) -> CacheKey {
        CacheKey::from_parts(&[b"LinearModel", &self.learning_rate.to_le_bytes()])
    }
    fn fit(&self, x: &Value, y: Option<&Value>) -> Result<Value> {
        let (x_data, _) = x
            .as_tensor()
            .ok_or(SomaError::Other("need tensor".into()))?;
        let y_data = y.and_then(|v| v.as_tensor().map(|(d, _)| d));
        let n = x_data.len() as f64;
        let x_mean = x_data.iter().sum::<f64>() / n;
        match y_data {
            Some(yd) => {
                let y_mean = yd.iter().sum::<f64>() / n;
                let cov: f64 = x_data
                    .iter()
                    .zip(yd)
                    .map(|(xi, yi)| (xi - x_mean) * (yi - y_mean))
                    .sum::<f64>()
                    / n;
                let var: f64 = x_data.iter().map(|xi| (xi - x_mean).powi(2)).sum::<f64>() / n;
                let weight = if var == 0.0 { 0.0 } else { cov / var };
                let bias = y_mean - weight * x_mean;
                Ok(Value::json(
                    serde_json::json!({"weight": weight, "bias": bias}),
                ))
            }
            None => Ok(Value::json(serde_json::json!({"weight": 1.0, "bias": 0.0}))),
        }
    }
    fn forward(&self, x: &Value, state: &Value) -> Result<Value> {
        let (data, shape) = x
            .as_tensor()
            .ok_or(SomaError::Other("need tensor".into()))?;
        let j = state
            .as_json()
            .ok_or(SomaError::Other("need json state".into()))?;
        let w = j["weight"].as_f64().unwrap_or(1.0);
        let b = j["bias"].as_f64().unwrap_or(0.0);
        let result: Vec<f64> = data.iter().map(|v| v * w + b).collect();
        Ok(Value::tensor(result, shape.to_vec()))
    }
    fn meta(&self) -> FilterMeta {
        FilterMeta {
            name: "LinearModel".into(),
            kind: FilterKind::Trainable,
            cacheable: true,
            differentiable: true,
            stream_mode: StreamMode::FixedState,
            distribution: somatize_core::filter::Distribution::Local,
            input_schema: None,
            output_schema: None,
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Always fails
struct FailingFilter;

impl Filter for FailingFilter {
    fn config_hash(&self) -> CacheKey {
        CacheKey::from_parts(&[b"Fail"])
    }
    fn fit(&self, _x: &Value, _y: Option<&Value>) -> Result<Value> {
        Err(SomaError::Other("intentional failure".into()))
    }
    fn forward(&self, _x: &Value, _s: &Value) -> Result<Value> {
        Err(SomaError::Other("intentional failure".into()))
    }
    fn meta(&self) -> FilterMeta {
        FilterMeta {
            name: "FailingFilter".into(),
            kind: FilterKind::Opaque,
            cacheable: false,
            differentiable: false,
            stream_mode: StreamMode::FixedState,
            distribution: somatize_core::filter::Distribution::Local,
            input_schema: None,
            output_schema: None,
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn make_linear_graph(ids: &[&str]) -> Graph {
    let mut g = Graph::new();
    for &id in ids {
        g.nodes.push(Node::new(id, id, id));
    }
    for (i, pair) in ids.windows(2).enumerate() {
        g.edges.push(Edge::data(format!("e{i}"), pair[0], pair[1]));
    }
    g
}

// ═══════════════════════════════════════════════
// Full Workflow Tests
// ═══════════════════════════════════════════════

#[test]
fn full_workflow_fit_forward_cache_rerun() {
    let cache = Arc::new(MemoryCache::default());

    // Run 1: fit + forward
    let graph = make_linear_graph(&["normalizer", "model"]);
    let mut lib = FilterLibrary::new();
    lib.register("normalizer", Box::new(Normalizer));
    lib.register(
        "model",
        Box::new(LinearModel {
            learning_rate: 0.01,
        }),
    );

    let mut session = GraphSession::new(graph, lib).with_cache(cache.clone());

    let train_x = Value::tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]);
    let train_y = Value::tensor(vec![2.0, 4.0, 6.0, 8.0, 10.0], vec![5]);

    session.fit(&train_x, Some(&train_y)).unwrap();

    // Cache should have entries
    assert!(!cache.is_empty());

    // Run 2: same config + same data → states should be cached
    let graph2 = make_linear_graph(&["normalizer", "model"]);
    let mut lib2 = FilterLibrary::new();
    lib2.register("normalizer", Box::new(Normalizer));
    lib2.register(
        "model",
        Box::new(LinearModel {
            learning_rate: 0.01,
        }),
    );

    let mut session2 = GraphSession::new(graph2, lib2).with_cache(cache.clone());
    session2.fit(&train_x, Some(&train_y)).unwrap();

    // Both sessions should have fitted states
    assert!(session.is_fitted());
    assert!(session2.is_fitted());
}

#[test]
fn cache_invalidation_on_config_change() {
    let cache = Arc::new(MemoryCache::default());

    // Run 1: lr = 0.01
    let graph = make_linear_graph(&["model"]);
    let mut lib = FilterLibrary::new();
    lib.register(
        "model",
        Box::new(LinearModel {
            learning_rate: 0.01,
        }),
    );
    let mut s1 = GraphSession::new(graph, lib).with_cache(cache.clone());

    let data = Value::tensor(vec![1.0, 2.0, 3.0], vec![3]);
    let r1 = s1.fit(&data, None).unwrap();

    // Run 2: lr = 0.1 (different config)
    let graph2 = make_linear_graph(&["model"]);
    let mut lib2 = FilterLibrary::new();
    lib2.register("model", Box::new(LinearModel { learning_rate: 0.1 }));
    let mut s2 = GraphSession::new(graph2, lib2).with_cache(cache.clone());

    let r2 = s2.fit(&data, None).unwrap();

    // Both succeed independently
    assert!(r1.get("model").unwrap().as_tensor().is_some());
    assert!(r2.get("model").unwrap().as_tensor().is_some());
}

#[test]
fn cache_invalidation_on_data_change() {
    let cache = Arc::new(MemoryCache::default());

    let graph = make_linear_graph(&["normalizer"]);
    let mut lib = FilterLibrary::new();
    lib.register("normalizer", Box::new(Normalizer));
    let mut s = GraphSession::new(graph, lib).with_cache(cache.clone());

    // Fit with data A
    let data_a = Value::tensor(vec![10.0, 20.0, 30.0], vec![3]);
    let r_a = s.fit(&data_a, None).unwrap();
    let (a_vals, _) = r_a.get("normalizer").unwrap().as_tensor().unwrap();

    // Fit with data B
    let graph2 = make_linear_graph(&["normalizer"]);
    let mut lib2 = FilterLibrary::new();
    lib2.register("normalizer", Box::new(Normalizer));
    let mut s2 = GraphSession::new(graph2, lib2).with_cache(cache.clone());

    let data_b = Value::tensor(vec![100.0, 200.0, 300.0], vec![3]);
    let r_b = s2.fit(&data_b, None).unwrap();
    let (b_vals, _) = r_b.get("normalizer").unwrap().as_tensor().unwrap();

    // Both normalize their respective means to ~0 (middle element)
    assert!(
        (a_vals[1] - 0.0).abs() < 0.01,
        "a middle should be ~0, got {}",
        a_vals[1]
    );
    assert!(
        (b_vals[1] - 0.0).abs() < 0.01,
        "b middle should be ~0, got {}",
        b_vals[1]
    );
}

// ═══════════════════════════════════════════════
// Study + Graph Integration
// ═══════════════════════════════════════════════

#[test]
fn study_with_graph_integration() {
    let bus = Arc::new(EventBus::new(512));
    let runner = StudyRunner::new(bus);

    let mut space = SearchSpace::new();
    space.add(SearchDimension::Float {
        name: "lr".into(),
        low: 0.001,
        high: 1.0,
        scale: Scale::Log,
        default: None,
    });

    let mut study = Study::new(
        "graph_study",
        space,
        SearchStrategy::Random {
            n_trials: 10,
            seed: Some(42),
        },
        vec![Objective {
            metric: "mse".into(),
            direction: Direction::Minimize,
        }],
    );

    let executor = FnTrialExecutor(
        |params: &std::collections::HashMap<String, serde_json::Value>| {
            let lr = params["lr"].as_f64().unwrap();

            let graph = make_linear_graph(&["normalizer", "model"]);
            let mut lib = FilterLibrary::new();
            lib.register("normalizer", Box::new(Normalizer));
            lib.register("model", Box::new(LinearModel { learning_rate: lr }));

            let mut session = GraphSession::new(graph, lib);

            let train_x = Value::tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
            let train_y = Value::tensor(vec![2.0, 4.0, 6.0, 8.0], vec![4]);
            let outputs = session.fit(&train_x, Some(&train_y)).unwrap();

            let pred = outputs.get("model").unwrap();
            let (pred_data, _) = pred.as_tensor().unwrap();
            let (y_data, _) = train_y.as_tensor().unwrap();

            let mse: f64 = pred_data
                .iter()
                .zip(y_data)
                .map(|(p, y)| (p - y).powi(2))
                .sum::<f64>()
                / pred_data.len() as f64;

            Ok(TrialOutcome::Completed(vec![MetricRecord {
                name: "mse".into(),
                value: mse,
                step: 0,
                timestamp: chrono::Utc::now(),
            }]))
        },
    );

    let mut sampler = RandomSampler::new(10, Some(42));
    runner.run(&mut study, &mut sampler, &executor).unwrap();

    assert_eq!(study.trials.len(), 10);
    let best = study.best_trial().unwrap();
    assert!(best.best_metric("mse", Direction::Minimize).is_some());
}

// ═══════════════════════════════════════════════
// Error Resilience
// ═══════════════════════════════════════════════

#[test]
fn graph_fit_error_propagates() {
    let graph = make_linear_graph(&["normalizer", "fail"]);
    let mut lib = FilterLibrary::new();
    lib.register("normalizer", Box::new(Normalizer));
    lib.register("fail", Box::new(FailingFilter));

    let mut session = GraphSession::new(graph, lib);

    let data = Value::tensor(vec![1.0, 2.0], vec![2]);
    let result = session.fit(&data, None);
    assert!(result.is_err());
    assert!(!session.is_fitted());
}

#[test]
fn study_continues_after_failed_trials() {
    let bus = Arc::new(EventBus::new(256));
    let runner = StudyRunner::new(bus);

    let mut space = SearchSpace::new();
    space.add(SearchDimension::Float {
        name: "x".into(),
        low: -1.0,
        high: 1.0,
        scale: Scale::Linear,
        default: None,
    });

    let mut study = Study::new(
        "resilience_test",
        space,
        SearchStrategy::Random {
            n_trials: 10,
            seed: Some(42),
        },
        vec![Objective {
            metric: "score".into(),
            direction: Direction::Maximize,
        }],
    );

    let executor = FnTrialExecutor(
        |params: &std::collections::HashMap<String, serde_json::Value>| {
            let x = params["x"].as_f64().unwrap();
            if x < 0.0 {
                return Err(SomaError::Other("negative x".into()));
            }
            Ok(TrialOutcome::Completed(vec![MetricRecord {
                name: "score".into(),
                value: x,
                step: 0,
                timestamp: chrono::Utc::now(),
            }]))
        },
    );

    let mut sampler = RandomSampler::new(10, Some(42));
    runner.run(&mut study, &mut sampler, &executor).unwrap();

    assert_eq!(study.trials.len(), 10);
    let completed = study.completed_trials().len();
    let failed = study
        .trials
        .iter()
        .filter(|t| matches!(t.state, somatize_core::study::TrialState::Failed { .. }))
        .count();
    assert!(completed > 0, "some trials should succeed");
    assert!(failed > 0, "some trials should fail");
    assert_eq!(completed + failed, 10);
}

// ═══════════════════════════════════════════════
// Edge Cases
// ═══════════════════════════════════════════════

#[test]
fn graph_single_filter() {
    let graph = make_linear_graph(&["normalizer"]);
    let mut lib = FilterLibrary::new();
    lib.register("normalizer", Box::new(Normalizer));

    let mut session = GraphSession::new(graph, lib);

    let data = Value::tensor(vec![10.0, 20.0, 30.0], vec![3]);
    let outputs = session.fit(&data, None).unwrap();
    let result = outputs.get("normalizer").unwrap();

    let (vals, _) = result.as_tensor().unwrap();
    assert!((vals[1] - 0.0).abs() < 0.01, "middle value should be ~0");
    assert!(vals[0] < 0.0, "below mean should be negative");
    assert!(vals[2] > 0.0, "above mean should be positive");
}

#[test]
fn graph_single_sample() {
    let graph = make_linear_graph(&["normalizer"]);
    let mut lib = FilterLibrary::new();
    lib.register("normalizer", Box::new(Normalizer));

    let mut session = GraphSession::new(graph, lib);

    let data = Value::tensor(vec![42.0], vec![1]);
    let outputs = session.fit(&data, None).unwrap();
    let result = outputs.get("normalizer").unwrap();

    let (vals, _) = result.as_tensor().unwrap();
    assert!((vals[0] - 0.0).abs() < 0.01);
}

// ═══════════════════════════════════════════════
// Compiler + Runtime Integration
// ═══════════════════════════════════════════════

#[test]
fn compile_then_execute_with_cache() {
    let graph = linear_pipeline(vec![
        Node::new("normalizer", "Normalizer", "Normalizer"),
        Node::new("model", "Model", "LinearModel"),
    ]);

    let mut registry = SimpleFilterRegistry::new();
    registry.register("normalizer", &Normalizer as &dyn Filter);
    registry.register(
        "model",
        &LinearModel {
            learning_rate: 0.01,
        } as &dyn Filter,
    );

    let cache = MemoryCache::default();

    // Compile
    let result = compile(&graph, &registry, CompileMode::Inference, Some(&cache)).unwrap();
    assert_eq!(result.plan.node_count(), 2);
    assert_eq!(result.plan.cached_count(), 0);

    // Execute
    let bus = Arc::new(EventBus::new(64));
    let graph_info = somatize_runtime::executor::GraphInfo::from_graph(&graph);
    let mut ctx = Context::new(bus, "run_1").with_graph_info(graph_info);
    ctx.set("input", Value::tensor(vec![1.0, 2.0, 3.0], vec![3]));

    let normalizer = Normalizer;
    let model = LinearModel {
        learning_rate: 0.01,
    };

    let input = Value::tensor(vec![1.0, 2.0, 3.0], vec![3]);
    let norm_state = normalizer.fit(&input, None).unwrap();
    let norm_output = normalizer.forward(&input, &norm_state).unwrap();
    let model_state = model.fit(&norm_output, None).unwrap();

    let mut filters = FilterLibrary::new();
    filters.register("normalizer", Box::new(Normalizer));
    filters.try_set_state("normalizer", norm_state).unwrap();
    filters.register(
        "model",
        Box::new(LinearModel {
            learning_rate: 0.01,
        }),
    );
    filters.try_set_state("model", model_state).unwrap();

    execute(&result.plan, &mut ctx, &filters, &cache).unwrap();

    assert!(ctx.get("model").is_some());
}
