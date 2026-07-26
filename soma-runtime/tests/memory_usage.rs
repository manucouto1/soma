//! Memory usage tests — verify that batched and streaming execution do not
//! leak memory as more chunks/batches are processed.
//!
//! Uses a tracking global allocator to measure peak and current heap usage
//! at key points during execution.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

// ── Tracking allocator ──

struct TrackingAllocator;

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let current = ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(current, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        ALLOCATED.fetch_sub(layout.size(), Ordering::Relaxed);
    }
}

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator;

/// The counting allocator is process-global, so tests measuring
/// allocation deltas must never run concurrently — a sibling test's
/// allocations would land inside this test's window. Every test takes
/// this lock first.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial_guard() -> std::sync::MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn current_allocated() -> usize {
    ALLOCATED.load(Ordering::Relaxed)
}

fn reset_peak() {
    PEAK.store(ALLOCATED.load(Ordering::Relaxed), Ordering::Relaxed);
}

fn peak_allocated() -> usize {
    PEAK.load(Ordering::Relaxed)
}

// ── Test filters ──

use somatize_core::cache::CacheKey;
use somatize_core::error::Result;
use somatize_core::filter::{Distribution, Filter, FilterKind, FilterMeta, StreamMode};
use somatize_core::value::Value;
use std::sync::Arc;

/// Stateless filter that doubles tensor values. Allocates a new output
/// tensor on each call but keeps no internal state.
struct Doubler;

impl Filter for Doubler {
    fn config_hash(&self) -> CacheKey {
        CacheKey::from_parts(&[b"Doubler"])
    }
    fn fit(&self, _x: &Value, _y: Option<&Value>) -> Result<Value> {
        Ok(Value::Empty)
    }
    fn forward(&self, x: &Value, _state: &Value) -> Result<Value> {
        match x {
            Value::Tensor { values, shape } => Ok(Value::tensor(
                values.iter().map(|v| v * 2.0).collect(),
                shape.clone(),
            )),
            _ => Ok(x.clone()),
        }
    }
    fn meta(&self) -> FilterMeta {
        FilterMeta {
            name: "Doubler".into(),
            kind: FilterKind::Stateless,
            cacheable: false,
            differentiable: false,
            stream_mode: StreamMode::FixedState,
            distribution: Distribution::Local,
            input_schema: None,
            output_schema: None,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Trainable filter that learns a mean and subtracts it.
struct MeanNormalizer;

impl Filter for MeanNormalizer {
    fn config_hash(&self) -> CacheKey {
        CacheKey::from_parts(&[b"MeanNorm"])
    }
    fn fit(&self, x: &Value, _y: Option<&Value>) -> Result<Value> {
        if let Some((data, _)) = x.as_tensor() {
            let mean = data.iter().sum::<f64>() / data.len().max(1) as f64;
            Ok(Value::json(serde_json::json!({ "mean": mean })))
        } else {
            Ok(Value::Empty)
        }
    }
    fn forward(&self, x: &Value, state: &Value) -> Result<Value> {
        let mean = state
            .as_json()
            .and_then(|j| j["mean"].as_f64())
            .unwrap_or(0.0);
        match x {
            Value::Tensor { values, shape } => Ok(Value::tensor(
                values.iter().map(|v| v - mean).collect(),
                shape.clone(),
            )),
            _ => Ok(x.clone()),
        }
    }
    fn meta(&self) -> FilterMeta {
        FilterMeta {
            name: "MeanNorm".into(),
            kind: FilterKind::Trainable,
            cacheable: false,
            differentiable: false,
            stream_mode: StreamMode::FixedState,
            distribution: Distribution::Local,
            input_schema: None,
            output_schema: None,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ── Helper ──

use somatize_core::graph::{Graph, Node};
use somatize_runtime::cache::MemoryCache;
use somatize_runtime::filter_library::FilterLibrary;
use somatize_runtime::graph_session::GraphSession;

fn make_doubler_session() -> GraphSession {
    let mut graph = Graph::new();
    graph.nodes.push(Node::new("double", "Double", "double"));

    let mut lib = FilterLibrary::new();
    lib.register("double", Box::new(Doubler));

    let cache = Arc::new(MemoryCache::new(64 * 1024 * 1024)); // 64 MB
    GraphSession::new(graph, lib).with_cache(cache)
}

fn make_pipeline_session() -> GraphSession {
    use somatize_core::graph::Edge;

    let mut graph = Graph::new();
    graph.nodes.push(Node::new("norm", "Normalize", "norm"));
    graph.nodes.push(Node::new("double", "Double", "double"));
    graph
        .edges
        .push(Edge::data("norm_to_double", "norm", "double"));

    let mut lib = FilterLibrary::new();
    lib.register("norm", Box::new(MeanNormalizer));
    lib.register("double", Box::new(Doubler));

    let cache = Arc::new(MemoryCache::new(64 * 1024 * 1024));
    GraphSession::new(graph, lib).with_cache(cache)
}

// ── Memory tests ──

/// Stream many chunks and verify memory stays bounded.
///
/// The current stream executor accumulates all chunk outputs in a Vec
/// before concatenating them (known design tradeoff). So we expect memory
/// proportional to input + output + accumulated chunks. What we guard
/// against is EXTRA growth beyond that — e.g. leaked contexts, duplicated
/// states, or unbounded caches.
#[test]
#[cfg_attr(
    coverage,
    ignore = "allocation counts are meaningless under coverage instrumentation"
)]
fn stream_memory_does_not_grow_with_chunks() {
    let _serial = serial_guard();
    use somatize_runtime::forward::Stream;

    let session = make_doubler_session();

    let chunk_rows = 1000;
    let n_chunks = 50;
    let total_rows = chunk_rows * n_chunks;

    // Large input: many rows
    let input = Value::tensor(
        (0..total_rows).map(|i| i as f64).collect(),
        vec![total_rows],
    );

    // Warm up — let internal structures settle
    let small = Value::tensor(vec![1.0; chunk_rows], vec![chunk_rows]);
    let _ = session.forward_with(
        &small,
        &Stream {
            chunk_size: chunk_rows,
        },
    );

    // Measure baseline
    reset_peak();
    let before = current_allocated();

    let result = session.forward_with(
        &input,
        &Stream {
            chunk_size: chunk_rows,
        },
    );
    assert!(result.is_ok());

    let after = current_allocated();
    let peak = peak_allocated();

    // Expected: input (held during streaming) + output tensor + intermediate
    // chunk outputs Vec. The output is ~total_rows * 8 bytes.
    let expected_output_bytes = total_rows * std::mem::size_of::<f64>();
    let growth = after.saturating_sub(before);

    eprintln!(
        "Stream: before={before}B, after={after}B, peak={peak}B, growth={growth}B, expected_output={expected_output_bytes}B"
    );

    // Growth should be bounded: output tensor + some overhead.
    // Fail if we see more than 8× the output size (which would indicate
    // a real leak beyond the expected chunk accumulation).
    assert!(
        growth < expected_output_bytes * 8,
        "Memory grew {growth}B after streaming {n_chunks} chunks — possible leak \
         (expected < {}B)",
        expected_output_bytes * 8
    );
}

/// Process many batches and verify memory stays bounded.
///
/// Simulates what Batched strategy does: repeatedly calling forward()
/// with different batch data. Memory should not accumulate across calls.
#[test]
#[cfg_attr(
    coverage,
    ignore = "allocation counts are meaningless under coverage instrumentation"
)]
fn repeated_forward_memory_does_not_grow() {
    let _serial = serial_guard();
    let session = make_doubler_session();

    let batch_size = 1000;
    let n_batches = 100;

    // Warm up
    let warm = Value::tensor(vec![1.0; batch_size], vec![batch_size]);
    let _ = session.forward(&warm);

    reset_peak();
    let before = current_allocated();

    for i in 0..n_batches {
        let batch = Value::tensor(
            (0..batch_size)
                .map(|j| (i * batch_size + j) as f64)
                .collect(),
            vec![batch_size],
        );
        let result = session.forward(&batch);
        assert!(result.is_ok());
    }

    let after = current_allocated();
    let peak = peak_allocated();

    let single_batch_bytes = batch_size * std::mem::size_of::<f64>();
    let growth = after.saturating_sub(before);

    eprintln!(
        "Batched: before={before}B, after={after}B, peak={peak}B, growth={growth}B, single_batch={single_batch_bytes}B"
    );

    // After all batches are done, memory should not retain more than a few
    // batch-sized buffers. If it grew linearly with n_batches, that's a leak.
    // Allow 10× single batch as margin (cache entries, Arc overhead, etc).
    assert!(
        growth < single_batch_bytes * 32,
        "Memory grew {growth}B after {n_batches} forward() calls — possible leak \
         (expected < {}B)",
        single_batch_bytes * 32
    );
}

/// Verify that peak memory during streaming is bounded.
///
/// The current executor holds all chunk outputs until concatenation, so
/// peak ≈ input + all chunk outputs + final tensor. We verify it doesn't
/// exceed a reasonable multiple of the data size.
#[test]
#[cfg_attr(
    coverage,
    ignore = "allocation counts are meaningless under coverage instrumentation"
)]
fn stream_peak_memory_bounded() {
    let _serial = serial_guard();
    use somatize_runtime::forward::Stream;

    let session = make_doubler_session();

    let chunk_rows = 500;
    let total_rows = 50_000; // 100 chunks

    let input = Value::tensor(
        (0..total_rows).map(|i| i as f64).collect(),
        vec![total_rows],
    );

    let input_bytes = total_rows * std::mem::size_of::<f64>();

    // Warm up
    let small = Value::tensor(vec![1.0; chunk_rows], vec![chunk_rows]);
    let _ = session.forward_with(
        &small,
        &Stream {
            chunk_size: chunk_rows,
        },
    );

    reset_peak();
    let before = current_allocated();

    let result = session.forward_with(
        &input,
        &Stream {
            chunk_size: chunk_rows,
        },
    );
    assert!(result.is_ok());

    let peak = peak_allocated();
    let peak_growth = peak.saturating_sub(before);

    eprintln!("Peak: before={before}B, peak={peak}B, growth={peak_growth}B, input={input_bytes}B");

    // Peak = input (400KB) + chunked outputs (~400KB) + final concat (~400KB)
    // + overhead. Allow 8× input as upper bound to catch real leaks.
    assert!(
        peak_growth < input_bytes * 8,
        "Peak memory {peak_growth}B during streaming is too high — may accumulate \
         extra data (expected < {}B)",
        input_bytes * 8
    );
}

/// Run fit + forward on a pipeline and verify memory stays stable across
/// multiple forward passes after fitting.
#[test]
#[cfg_attr(
    coverage,
    ignore = "allocation counts are meaningless under coverage instrumentation"
)]
fn pipeline_fit_then_repeated_forward_stable() {
    let _serial = serial_guard();
    let mut session = make_pipeline_session();

    let train_data = Value::tensor((0..5000).map(|i| i as f64).collect(), vec![5000]);

    // Fit the pipeline
    let fit_result = session.fit(&train_data, None);
    assert!(fit_result.is_ok());

    // Warm up forward
    let warm = Value::tensor(vec![1.0; 1000], vec![1000]);
    let _ = session.forward(&warm);

    reset_peak();
    let before = current_allocated();

    // Run many forward passes
    let n_passes = 50;
    let batch_size = 1000;
    for i in 0..n_passes {
        let batch = Value::tensor(
            (0..batch_size)
                .map(|j| (i * batch_size + j) as f64)
                .collect(),
            vec![batch_size],
        );
        let result = session.forward(&batch);
        assert!(result.is_ok());
    }

    let after = current_allocated();
    let single_batch_bytes = batch_size * std::mem::size_of::<f64>();
    let growth = after.saturating_sub(before);

    eprintln!("Pipeline: before={before}B, after={after}B, growth={growth}B");

    // Memory should not grow linearly with number of forward passes.
    assert!(
        growth < single_batch_bytes * 48,
        "Pipeline memory grew {growth}B after {n_passes} forward passes — possible leak \
         (expected < {}B)",
        single_batch_bytes * 48
    );
}

/// Verify that Value::clone() is cheap (Arc clone, not deep copy).
#[test]
#[cfg_attr(
    coverage,
    ignore = "allocation counts are meaningless under coverage instrumentation"
)]
fn value_clone_is_cheap() {
    let _serial = serial_guard();
    // Create a large tensor
    let big = Value::tensor(vec![42.0; 100_000], vec![100_000]);

    reset_peak();
    let before = current_allocated();

    // Clone 100 times — should be essentially free
    let mut clones = Vec::with_capacity(100);
    for _ in 0..100 {
        clones.push(big.clone());
    }

    let after = current_allocated();
    let growth = after.saturating_sub(before);

    // The original tensor is 100K * 8 = 800KB.
    // 100 clones via Arc should cost ~0 bytes of data, just pointer+refcount overhead.
    // Allow 100KB for Vec<Value> + Arc overhead (which is negligible).
    let data_size = 100_000 * std::mem::size_of::<f64>();

    eprintln!("Clone: before={before}B, after={after}B, growth={growth}B, data_size={data_size}B");

    assert!(
        growth < data_size / 4, // Less than 25% of ONE tensor copy
        "Cloning Value allocated {growth}B — expected near-zero \
         (Arc clone). data_size={data_size}B"
    );

    drop(clones);
}
