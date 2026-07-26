//! Integration tests for the local tracking backend: JSONL sink,
//! LocalTracker run directories, and EventBus sink wiring.

use somatize_core::event::{Event, MetricRecord};
use somatize_core::study::{Direction, Objective, SearchStrategy, Study};
use somatize_core::tracking::{EventSink, RunKind, RunState, Tracker};
use somatize_runtime::EventBus;
use somatize_runtime::tracking::{JsonlEventSink, LocalTracker, load_manifest, load_status};
use std::sync::Arc;

fn metric(name: &str, value: f64, step: usize) -> MetricRecord {
    MetricRecord {
        name: name.into(),
        value,
        step,
        timestamp: chrono::Utc::now(),
    }
}

fn step_event(run_id: &str, step: usize) -> Event {
    Event::StepCompleted {
        run_id: run_id.into(),
        step,
        epoch: None,
    }
}

fn read_lines(path: &std::path::Path) -> Vec<serde_json::Value> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .map(|l| serde_json::from_str(l).expect("valid JSON line"))
        .collect()
}

#[test]
fn sink_records_all_events_in_envelope() {
    let dir = tempfile::tempdir().unwrap();
    let events_path = dir.path().join("events.jsonl");
    let sink = JsonlEventSink::create(&events_path, None, 5).unwrap();

    for i in 0..17 {
        sink.record(&step_event("r1", i));
    }
    sink.flush();

    let lines = read_lines(&events_path);
    assert_eq!(lines.len(), 17);
    for (i, line) in lines.iter().enumerate() {
        assert_eq!(line["seq"], i as u64);
        assert_eq!(line["event_type"], "StepCompleted");
        assert_eq!(line["step"], i as u64);
        assert!(line["ts"].is_string());
    }
}

#[test]
fn sink_is_lossless_under_concurrent_emit() {
    let dir = tempfile::tempdir().unwrap();
    let events_path = dir.path().join("events.jsonl");
    let bus = Arc::new(EventBus::new(4));
    let sink = Arc::new(JsonlEventSink::create(&events_path, None, 7).unwrap());
    bus.add_sink(sink.clone());

    let mut handles = Vec::new();
    for t in 0..8 {
        let bus = bus.clone();
        handles.push(std::thread::spawn(move || {
            for i in 0..50 {
                bus.emit(step_event(&format!("thread{t}"), i));
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    bus.flush_sinks();

    let lines = read_lines(&events_path);
    assert_eq!(lines.len(), 400, "no event may be dropped");
    // Every sequence number appears exactly once.
    let mut seqs: Vec<u64> = lines.iter().map(|l| l["seq"].as_u64().unwrap()).collect();
    seqs.sort_unstable();
    assert_eq!(seqs, (0..400).collect::<Vec<_>>());
}

#[test]
fn sink_tees_metrics() {
    let dir = tempfile::tempdir().unwrap();
    let events_path = dir.path().join("events.jsonl");
    let metrics_path = dir.path().join("metrics.jsonl");
    let sink = JsonlEventSink::create(&events_path, Some(&metrics_path), 1).unwrap();

    sink.record(&step_event("r1", 0)); // not a metric — no tee
    sink.record(&Event::TrialMetric {
        study_id: "s1".into(),
        trial_id: "trial_0001".into(),
        metric: metric("val_f1", 0.83, 3),
    });
    sink.record(&Event::MetricReported {
        run_id: "r1".into(),
        metric: metric("loss", 0.5, 12),
        node_id: Some("encoder".into()),
        trial_id: None,
    });
    sink.flush();

    assert_eq!(read_lines(&events_path).len(), 3);
    let metrics = read_lines(&metrics_path);
    assert_eq!(metrics.len(), 2);
    assert_eq!(metrics[0]["name"], "val_f1");
    assert_eq!(metrics[0]["trial_id"], "trial_0001");
    assert_eq!(metrics[1]["name"], "loss");
    assert_eq!(metrics[1]["node_id"], "encoder");
    assert_eq!(metrics[1]["step"], 12);
}

#[test]
fn local_tracker_creates_valid_run_dir() {
    let root = tempfile::tempdir().unwrap();
    let tracker = LocalTracker::create(root.path(), RunKind::Train, "baseline").unwrap();

    assert!(tracker.run_id().starts_with("run_"));
    let manifest = load_manifest(tracker.run_dir()).unwrap();
    assert_eq!(manifest.name, "baseline");
    assert_eq!(
        manifest.schema_version,
        somatize_core::tracking::RUN_SCHEMA_VERSION
    );
    assert!(manifest.soma_version.is_some());
    assert!(!manifest.argv.is_empty());

    let status = load_status(tracker.run_dir()).unwrap();
    assert_eq!(status.state, RunState::Running);
    assert!(status.finished_at.is_none());

    tracker.sink().record(&step_event(tracker.run_id(), 0));
    tracker.finalize(RunState::Completed).unwrap();
    let status = load_status(tracker.run_dir()).unwrap();
    assert_eq!(status.state, RunState::Completed);
    assert!(status.finished_at.is_some());
    // finalize flushed the sink.
    assert_eq!(read_lines(&tracker.run_dir().join("events.jsonl")).len(), 1);
}

#[test]
fn save_artifact_creates_parent_dirs() {
    let root = tempfile::tempdir().unwrap();
    let tracker = LocalTracker::create(root.path(), RunKind::Train, "t").unwrap();
    tracker
        .save_artifact("diagnostics/channels/index.jsonl", b"{}\n")
        .unwrap();
    assert!(
        tracker
            .run_dir()
            .join("diagnostics/channels/index.jsonl")
            .exists()
    );
}

#[test]
fn save_study_is_atomic_and_readable() {
    let root = tempfile::tempdir().unwrap();
    let tracker = LocalTracker::create(root.path(), RunKind::Study, "grid").unwrap();
    assert!(tracker.run_id().starts_with("study_"));

    let study = Study::new(
        "grid",
        somatize_core::search::SearchSpace::new(),
        SearchStrategy::Random {
            n_trials: 4,
            seed: Some(1),
        },
        vec![Objective {
            metric: "f1".into(),
            direction: Direction::Maximize,
        }],
    );
    tracker.save_study(&study).unwrap();

    let bytes = std::fs::read(tracker.run_dir().join("study.json")).unwrap();
    let back: Study = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(back.name, "grid");
    // No stray tmp file left behind.
    assert!(!tracker.run_dir().join("study.json.tmp").exists());
}

#[test]
fn open_resumes_sequence_and_status() {
    let root = tempfile::tempdir().unwrap();
    let run_dir = {
        let tracker = LocalTracker::create(root.path(), RunKind::Train, "resumable").unwrap();
        for i in 0..5 {
            tracker.sink().record(&step_event(tracker.run_id(), i));
        }
        tracker.finalize(RunState::Failed).unwrap();
        tracker.run_dir().to_path_buf()
    };

    let tracker = LocalTracker::open(&run_dir).unwrap();
    assert_eq!(load_status(&run_dir).unwrap().state, RunState::Running);
    tracker.sink().record(&step_event(tracker.run_id(), 5));
    tracker.finalize(RunState::Completed).unwrap();

    let lines = read_lines(&run_dir.join("events.jsonl"));
    assert_eq!(lines.len(), 6);
    let seqs: Vec<u64> = lines.iter().map(|l| l["seq"].as_u64().unwrap()).collect();
    assert_eq!(
        seqs,
        vec![0, 1, 2, 3, 4, 5],
        "sequence continues after resume"
    );
}

#[test]
fn graph_fit_events_reach_the_run_dir() {
    use somatize_core::cache::CacheKey;
    use somatize_core::filter::{Filter, FilterKind, FilterMeta, StreamMode};
    use somatize_core::graph::{Edge, Graph, Node};
    use somatize_core::value::Value;
    use somatize_runtime::{FilterLibrary, GraphSession};

    struct Doubler;
    impl Filter for Doubler {
        fn config_hash(&self) -> CacheKey {
            CacheKey::from_parts(&[b"Doubler"])
        }
        fn fit(&self, _x: &Value, _y: Option<&Value>) -> somatize_core::Result<Value> {
            Ok(Value::json(serde_json::json!({})))
        }
        fn forward(&self, x: &Value, _state: &Value) -> somatize_core::Result<Value> {
            let (data, shape) = x.as_tensor().unwrap();
            Ok(Value::tensor(
                data.iter().map(|v| v * 2.0).collect(),
                shape.to_vec(),
            ))
        }
        fn meta(&self) -> FilterMeta {
            FilterMeta {
                name: "Doubler".into(),
                kind: FilterKind::Trainable,
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

    let root = tempfile::tempdir().unwrap();
    let tracker = LocalTracker::create(root.path(), RunKind::Fit, "fit-e2e").unwrap();

    let mut graph = Graph::new();
    graph.nodes.push(Node::new("a", "a", "a"));
    graph.nodes.push(Node::new("b", "b", "b"));
    graph.edges.push(Edge::data("e0", "a", "b"));
    let mut lib = FilterLibrary::new();
    lib.register("a", Box::new(Doubler));
    lib.register("b", Box::new(Doubler));

    let bus = Arc::new(EventBus::new(64));
    bus.add_sink(tracker.sink());
    let mut session = GraphSession::new(graph, lib).with_event_bus(bus);
    session
        .fit(&Value::tensor(vec![1.0, 2.0], vec![2]), None)
        .unwrap();
    tracker.finalize(RunState::Completed).unwrap();

    let lines = read_lines(&tracker.run_dir().join("events.jsonl"));
    let types: Vec<&str> = lines
        .iter()
        .map(|l| l["event_type"].as_str().unwrap())
        .collect();
    assert!(types.contains(&"NodeStarted"), "got {types:?}");
    assert!(types.contains(&"NodeCompleted"), "got {types:?}");
    assert_eq!(
        lines
            .iter()
            .filter(|l| l["event_type"] == "NodeStarted")
            .count(),
        2,
        "one NodeStarted per node"
    );
}

#[test]
fn heartbeat_updates_status() {
    let root = tempfile::tempdir().unwrap();
    let tracker = LocalTracker::create(root.path(), RunKind::Train, "hb").unwrap();
    let before = load_status(tracker.run_dir()).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(5));
    tracker.heartbeat().unwrap();
    let after = load_status(tracker.run_dir()).unwrap();
    assert!(after.heartbeat_at.unwrap() > before.heartbeat_at.unwrap());
    assert_eq!(after.state, RunState::Running);
}
