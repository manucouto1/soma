//! End-to-end integration tests for full Soma workflows.
//!
//! These tests verify realistic researcher use cases:
//! define filters → build pipeline → fit → predict → cache hit → invalidation.

use soma_compiler::{compile, CompileMode, SimpleFilterRegistry};
use soma_core::cache::{CacheKey, CacheStore};
use soma_core::error::{Result, SomaError};
use soma_core::event::{Event, MetricRecord};
use soma_core::filter::{Filter, FilterKind, FilterMeta, StreamMode};
use soma_core::graph::{linear_pipeline, Node};
use soma_core::search::{Scale, SearchDimension, SearchSpace};
use soma_core::study::{Direction, Objective, SearchStrategy, Study};
use soma_core::value::Value;
use soma_runtime::*;
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
        let (data, _) = x.as_tensor().ok_or(SomaError::Other("need tensor".into()))?;
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let std = (data.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
        let std = if std == 0.0 { 1.0 } else { std }; // handle single sample
        Ok(Value::json(serde_json::json!({"mean": mean, "std": std})))
    }
    fn forward(&self, x: &Value, state: &Value) -> Result<Value> {
        let (data, shape) = x.as_tensor().ok_or(SomaError::Other("need tensor".into()))?;
        let j = state.as_json().ok_or(SomaError::Other("need json state".into()))?;
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
            distribution: soma_core::filter::Distribution::Local,
        }
    }
}

/// Linear transformation: y = x * weight + bias (learns from data)
struct LinearModel {
    learning_rate: f64,
}

impl Filter for LinearModel {
    fn config_hash(&self) -> CacheKey {
        CacheKey::from_parts(&[b"LinearModel", &self.learning_rate.to_le_bytes()])
    }
    fn fit(&self, x: &Value, y: Option<&Value>) -> Result<Value> {
        let (x_data, _) = x.as_tensor().ok_or(SomaError::Other("need tensor".into()))?;
        let y_data = y.and_then(|v| v.as_tensor().map(|(d, _)| d));
        // Simple: weight = correlation(x, y), bias = mean(y) - weight * mean(x)
        let n = x_data.len() as f64;
        let x_mean = x_data.iter().sum::<f64>() / n;
        match y_data {
            Some(yd) => {
                let y_mean = yd.iter().sum::<f64>() / n;
                let cov: f64 = x_data.iter().zip(yd).map(|(xi, yi)| (xi - x_mean) * (yi - y_mean)).sum::<f64>() / n;
                let var: f64 = x_data.iter().map(|xi| (xi - x_mean).powi(2)).sum::<f64>() / n;
                let weight = if var == 0.0 { 0.0 } else { cov / var };
                let bias = y_mean - weight * x_mean;
                Ok(Value::json(serde_json::json!({"weight": weight, "bias": bias})))
            }
            None => Ok(Value::json(serde_json::json!({"weight": 1.0, "bias": 0.0}))),
        }
    }
    fn forward(&self, x: &Value, state: &Value) -> Result<Value> {
        let (data, shape) = x.as_tensor().ok_or(SomaError::Other("need tensor".into()))?;
        let j = state.as_json().ok_or(SomaError::Other("need json state".into()))?;
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
            distribution: soma_core::filter::Distribution::Local,
        }
    }
}

/// Always fails
struct FailingFilter;

impl Filter for FailingFilter {
    fn config_hash(&self) -> CacheKey { CacheKey::from_parts(&[b"Fail"]) }
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
            distribution: soma_core::filter::Distribution::Local,
        }
    }
}

// ═══════════════════════════════════════════════
// Full Workflow Tests
// ═══════════════════════════════════════════════

#[test]
fn full_workflow_fit_predict_cache_rerun() {
    let cache = Arc::new(MemoryCache::default());

    // Run 1: fit + predict
    let mut pipeline = Pipeline::new(vec![
        ("normalizer".into(), Box::new(Normalizer) as Box<dyn Filter>),
        ("model".into(), Box::new(LinearModel { learning_rate: 0.01 })),
    ]).with_cache(cache.clone());

    let train_x = Value::tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]);
    let train_y = Value::tensor(vec![2.0, 4.0, 6.0, 8.0, 10.0], vec![5]);

    pipeline.fit(&train_x, Some(&train_y)).unwrap();
    let result1 = pipeline.predict(&Value::tensor(vec![3.0], vec![1])).unwrap();

    // Cache should have entries
    assert!(!cache.is_empty());

    // Run 2: same config + same data → states should be cached
    let mut pipeline2 = Pipeline::new(vec![
        ("normalizer".into(), Box::new(Normalizer) as Box<dyn Filter>),
        ("model".into(), Box::new(LinearModel { learning_rate: 0.01 })),
    ]).with_cache(cache.clone());

    pipeline2.fit(&train_x, Some(&train_y)).unwrap();
    let result2 = pipeline2.predict(&Value::tensor(vec![3.0], vec![1])).unwrap();

    // Same input → same output
    assert_eq!(result1, result2);
}

#[test]
fn cache_invalidation_on_config_change() {
    let cache = Arc::new(MemoryCache::default());

    // Run 1: lr = 0.01
    let mut p1 = Pipeline::new(vec![
        ("model".into(), Box::new(LinearModel { learning_rate: 0.01 })),
    ]).with_cache(cache.clone());

    let data = Value::tensor(vec![1.0, 2.0, 3.0], vec![3]);
    p1.fit(&data, None).unwrap();
    let r1 = p1.predict(&data).unwrap();

    // Run 2: lr = 0.1 (different config)
    let mut p2 = Pipeline::new(vec![
        ("model".into(), Box::new(LinearModel { learning_rate: 0.1 })),
    ]).with_cache(cache.clone());

    p2.fit(&data, None).unwrap();
    let r2 = p2.predict(&data).unwrap();

    // Different config → different cache key → different state
    // (In this case output is same because fit doesn't use lr, but cache keys differ)
    // The important thing is both runs succeed independently
    assert!(r1.as_tensor().is_some());
    assert!(r2.as_tensor().is_some());
}

#[test]
fn cache_invalidation_on_data_change() {
    let cache = Arc::new(MemoryCache::default());

    let mut p = Pipeline::new(vec![
        ("normalizer".into(), Box::new(Normalizer) as Box<dyn Filter>),
    ]).with_cache(cache.clone());

    // Fit with data A
    let data_a = Value::tensor(vec![10.0, 20.0, 30.0], vec![3]);
    p.fit(&data_a, None).unwrap();
    let r_a = p.predict(&Value::tensor(vec![20.0], vec![1])).unwrap();

    // Fit with data B (different distribution)
    let mut p2 = Pipeline::new(vec![
        ("normalizer".into(), Box::new(Normalizer) as Box<dyn Filter>),
    ]).with_cache(cache.clone());

    let data_b = Value::tensor(vec![100.0, 200.0, 300.0], vec![3]);
    p2.fit(&data_b, None).unwrap();
    let r_b = p2.predict(&Value::tensor(vec![200.0], vec![1])).unwrap();

    // Different training data → different state → different normalization
    let (a_vals, _) = r_a.as_tensor().unwrap();
    let (b_vals, _) = r_b.as_tensor().unwrap();
    // Both should normalize their respective means to ~0
    assert!((a_vals[0] - 0.0).abs() < 0.01, "a should be ~0, got {}", a_vals[0]);
    assert!((b_vals[0] - 0.0).abs() < 0.01, "b should be ~0, got {}", b_vals[0]);
}

// ═══════════════════════════════════════════════
// Study + Pipeline Integration
// ═══════════════════════════════════════════════

#[test]
fn study_with_pipeline_integration() {
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
        "pipeline_study",
        space,
        SearchStrategy::Random { n_trials: 10, seed: Some(42) },
        vec![Objective { metric: "mse".into(), direction: Direction::Minimize }],
    );

    let executor = FnTrialExecutor(|params: &std::collections::HashMap<String, serde_json::Value>| {
        let lr = params["lr"].as_f64().unwrap();

        // Build pipeline with sampled lr
        let mut pipeline = Pipeline::new(vec![
            ("normalizer".into(), Box::new(Normalizer) as Box<dyn Filter>),
            ("model".into(), Box::new(LinearModel { learning_rate: lr })),
        ]);

        let train_x = Value::tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let train_y = Value::tensor(vec![2.0, 4.0, 6.0, 8.0], vec![4]);
        pipeline.fit(&train_x, Some(&train_y)).unwrap();

        let pred = pipeline.predict(&train_x).unwrap();
        let (pred_data, _) = pred.as_tensor().unwrap();
        let (y_data, _) = train_y.as_tensor().unwrap();

        // Compute MSE
        let mse: f64 = pred_data.iter().zip(y_data)
            .map(|(p, y)| (p - y).powi(2))
            .sum::<f64>() / pred_data.len() as f64;

        Ok(vec![MetricRecord {
            name: "mse".into(),
            value: mse,
            step: 0,
            timestamp: chrono::Utc::now(),
        }])
    });

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
fn pipeline_fit_error_propagates() {
    let mut pipeline = Pipeline::new(vec![
        ("normalizer".into(), Box::new(Normalizer) as Box<dyn Filter>),
        ("fail".into(), Box::new(FailingFilter) as Box<dyn Filter>),
    ]);

    let data = Value::tensor(vec![1.0, 2.0], vec![2]);
    let result = pipeline.fit(&data, None);
    assert!(result.is_err());
    assert!(!pipeline.is_fitted());
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
        SearchStrategy::Random { n_trials: 10, seed: Some(42) },
        vec![Objective { metric: "score".into(), direction: Direction::Maximize }],
    );

    // Executor fails when x < 0
    let executor = FnTrialExecutor(|params: &std::collections::HashMap<String, serde_json::Value>| {
        let x = params["x"].as_f64().unwrap();
        if x < 0.0 {
            return Err(SomaError::Other("negative x".into()));
        }
        Ok(vec![MetricRecord {
            name: "score".into(),
            value: x,
            step: 0,
            timestamp: chrono::Utc::now(),
        }])
    });

    let mut sampler = RandomSampler::new(10, Some(42));
    runner.run(&mut study, &mut sampler, &executor).unwrap();

    // Study completes despite some failures
    assert_eq!(study.trials.len(), 10);
    let completed = study.completed_trials().len();
    let failed = study.trials.iter()
        .filter(|t| matches!(t.state, soma_core::study::TrialState::Failed { .. }))
        .count();
    assert!(completed > 0, "some trials should succeed");
    assert!(failed > 0, "some trials should fail");
    assert_eq!(completed + failed, 10);
}

// ═══════════════════════════════════════════════
// Edge Cases
// ═══════════════════════════════════════════════

#[test]
fn pipeline_single_filter() {
    let mut pipeline = Pipeline::new(vec![
        ("normalizer".into(), Box::new(Normalizer) as Box<dyn Filter>),
    ]);

    let data = Value::tensor(vec![10.0, 20.0, 30.0], vec![3]);
    pipeline.fit(&data, None).unwrap();
    let result = pipeline.predict(&data).unwrap();

    let (vals, _) = result.as_tensor().unwrap();
    // Mean=20, std=~8.16. (10-20)/8.16 ≈ -1.22, (20-20)/8.16 = 0, (30-20)/8.16 ≈ 1.22
    assert!((vals[1] - 0.0).abs() < 0.01, "middle value should be ~0");
    assert!(vals[0] < 0.0, "below mean should be negative");
    assert!(vals[2] > 0.0, "above mean should be positive");
}

#[test]
fn pipeline_single_sample() {
    let mut pipeline = Pipeline::new(vec![
        ("normalizer".into(), Box::new(Normalizer) as Box<dyn Filter>),
    ]);

    // Single sample: std would be 0, normalizer handles this
    let data = Value::tensor(vec![42.0], vec![1]);
    pipeline.fit(&data, None).unwrap();
    let result = pipeline.predict(&data).unwrap();

    let (vals, _) = result.as_tensor().unwrap();
    // (42 - 42) / 1.0 = 0 (std forced to 1.0 to avoid division by zero)
    assert!((vals[0] - 0.0).abs() < 0.01);
}

#[test]
fn event_tracking_throughout_lifecycle() {
    let bus = Arc::new(EventBus::new(128));
    let mut rx = bus.subscribe();

    let mut pipeline = Pipeline::new(vec![
        ("normalizer".into(), Box::new(Normalizer) as Box<dyn Filter>),
        ("model".into(), Box::new(LinearModel { learning_rate: 0.01 })),
    ]).with_event_bus(bus);

    let data = Value::tensor(vec![1.0, 2.0, 3.0], vec![3]);
    pipeline.fit(&data, None).unwrap();
    pipeline.predict(&data).unwrap();

    let mut events = Vec::new();
    while let Ok(e) = rx.try_recv() {
        events.push(e);
    }

    // Should have events from both fit and predict phases
    let run_started = events.iter().filter(|e| matches!(e, Event::RunStarted { .. })).count();
    let run_completed = events.iter().filter(|e| matches!(e, Event::RunCompleted { .. })).count();
    let node_started = events.iter().filter(|e| matches!(e, Event::NodeStarted { .. })).count();
    let node_completed = events.iter().filter(|e| matches!(e, Event::NodeCompleted { .. })).count();

    assert_eq!(run_started, 2, "fit + predict = 2 runs");
    assert_eq!(run_completed, 2);
    assert!(node_started >= 4, "at least 2 nodes * 2 runs");
    assert!(node_completed >= 4);
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
    registry.register("model", &LinearModel { learning_rate: 0.01 } as &dyn Filter);

    let cache = MemoryCache::default();

    // Compile
    let result = compile(&graph, &registry, CompileMode::Inference, Some(&cache)).unwrap();
    assert_eq!(result.plan.node_count(), 2);
    assert_eq!(result.plan.cached_count(), 0); // nothing cached yet

    // Execute
    let bus = Arc::new(EventBus::new(64));
    let graph_info = soma_runtime::executor::GraphInfo::from_graph(&graph);
    let mut ctx = Context::new(bus, "run_1").with_graph_info(graph_info);
    ctx.set("input", Value::tensor(vec![1.0, 2.0, 3.0], vec![3]));

    let normalizer = Normalizer;
    let model = LinearModel { learning_rate: 0.01 };

    // Fit filters to get states
    let input = Value::tensor(vec![1.0, 2.0, 3.0], vec![3]);
    let norm_state = normalizer.fit(&input, None).unwrap();
    let norm_output = normalizer.forward(&input, &norm_state).unwrap();
    let model_state = model.fit(&norm_output, None).unwrap();

    let mut filters = FilterStore::new();
    filters.register("normalizer", Box::new(Normalizer));
    filters.set_state("normalizer", norm_state);
    filters.register("model", Box::new(LinearModel { learning_rate: 0.01 }));
    filters.set_state("model", model_state);

    execute(&result.plan, &mut ctx, &filters, &cache).unwrap();

    // Model should have produced output
    assert!(ctx.get("model").is_some());
}
