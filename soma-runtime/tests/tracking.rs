//! Integration tests for the local tracking backend: JSONL sink,
//! LocalTracker run directories, and EventBus sink wiring.

use somatize_core::optimizer::study::{Direction, Objective, SearchStrategy, Study};
use somatize_core::tracking::event::{Event, MetricRecord};
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
fn sink_flush_cadence_makes_lines_durable_without_manual_flush() {
    let dir = tempfile::tempdir().unwrap();
    let events_path = dir.path().join("events.jsonl");
    let sink = JsonlEventSink::create(&events_path, None, 5).unwrap();

    // Exactly flush_every events: the cadence flush must have fired.
    for i in 0..5 {
        sink.record(&step_event("r", i));
    }
    // No manual flush, sink still alive.
    assert_eq!(read_lines(&events_path).len(), 5);

    // flush_every=0 is clamped to 1: every record is durable.
    let path2 = dir.path().join("events2.jsonl");
    let sink2 = JsonlEventSink::create(&path2, None, 0).unwrap();
    sink2.record(&step_event("r", 0));
    assert_eq!(read_lines(&path2).len(), 1);
    drop((sink, sink2));
}

#[test]
fn sink_flushes_on_drop() {
    let dir = tempfile::tempdir().unwrap();
    let events_path = dir.path().join("events.jsonl");
    {
        let sink = JsonlEventSink::create(&events_path, None, 100).unwrap();
        for i in 0..3 {
            sink.record(&step_event("r", i));
        }
        // 3 < flush_every: nothing guaranteed on disk yet.
    } // drop flushes
    assert_eq!(read_lines(&events_path).len(), 3);
}

#[test]
fn sink_append_continues_without_truncating_and_create_truncates() {
    let dir = tempfile::tempdir().unwrap();
    let events_path = dir.path().join("events.jsonl");

    let sink = JsonlEventSink::create(&events_path, None, 1).unwrap();
    sink.record(&step_event("r", 0));
    drop(sink);

    let appended = JsonlEventSink::append(&events_path, None, 1, 5).unwrap();
    assert_eq!(appended.next_seq(), 5);
    appended.record(&step_event("r", 1));
    drop(appended);
    let lines = read_lines(&events_path);
    assert_eq!(lines.len(), 2, "append must not truncate");
    assert_eq!(lines[1]["seq"], 5);

    let fresh = JsonlEventSink::create(&events_path, None, 1).unwrap();
    assert_eq!(fresh.next_seq(), 0);
    drop(fresh);
    assert_eq!(read_lines(&events_path).len(), 0, "create truncates");
}

#[test]
fn sink_metric_tee_lines_are_typed_and_complete() {
    #[derive(serde::Deserialize)]
    struct MetricLine {
        ts: chrono::DateTime<chrono::Utc>,
        name: String,
        value: f64,
        step: usize,
        trial_id: Option<String>,
        node_id: Option<String>,
    }

    let dir = tempfile::tempdir().unwrap();
    let events_path = dir.path().join("events.jsonl");
    let metrics_path = dir.path().join("metrics.jsonl");
    let sink = JsonlEventSink::create(&events_path, Some(&metrics_path), 1).unwrap();

    sink.record(&Event::MetricReported {
        run_id: "r".into(),
        metric: metric("val_f1", 0.9, 4),
        node_id: Some("encoder".into()),
        trial_id: Some("trial_0002".into()),
    });
    sink.record(&Event::TrialMetric {
        study_id: "s".into(),
        trial_id: "trial_0003".into(),
        metric: metric("loss", 0.1, 9),
    });
    sink.flush();

    let raw = std::fs::read_to_string(&metrics_path).unwrap();
    let lines: Vec<MetricLine> = raw
        .lines()
        .map(|l| serde_json::from_str(l).expect("typed metric line"))
        .collect();
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[0].name, "val_f1");
    assert_eq!(lines[0].trial_id.as_deref(), Some("trial_0002"));
    assert_eq!(lines[0].node_id.as_deref(), Some("encoder"));
    assert_eq!(lines[1].trial_id.as_deref(), Some("trial_0003"));
    assert!(lines[1].node_id.is_none(), "TrialMetric has null node_id");
    assert_eq!(lines[1].step, 9);
    assert!(lines[0].ts.timestamp() > 0);
    assert!(lines[1].value < 1.0);
}

#[cfg(target_os = "linux")]
#[test]
fn sink_swallows_io_errors_without_panicking() {
    // /dev/full accepts opens but fails every write with ENOSPC — the
    // sink's headline contract is that tracking I/O failures must
    // never take down the training run.
    let sink = JsonlEventSink::append(std::path::Path::new("/dev/full"), None, 1, 0).unwrap();
    for i in 0..10 {
        sink.record(&step_event("r", i)); // must not panic
    }
    sink.flush(); // must not panic
    assert_eq!(
        sink.next_seq(),
        10,
        "sequence advances even when writes fail"
    );
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
        somatize_core::optimizer::search::SearchSpace::new(),
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
    use somatize_core::data::value::Value;
    use somatize_core::graph::filter::{Filter, FilterKind, FilterMeta, StreamMode};
    use somatize_core::graph::{Edge, Graph, Node};
    use somatize_runtime::{GraphSession, NodeCatalog};

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
                deterministic: true,
                stream_mode: StreamMode::FixedState,
                distribution: somatize_core::graph::filter::Distribution::Local,
                input_schema: None,
                output_schema: None,
            }
        }
    }

    let root = tempfile::tempdir().unwrap();
    let tracker = LocalTracker::create(root.path(), RunKind::Fit, "fit-e2e").unwrap();

    let mut graph = Graph::new();
    graph.nodes.push(Node::new("a", "a", "a"));
    graph.nodes.push(Node::new("b", "b", "b"));
    graph.edges.push(Edge::data("e0", "a", "b"));
    let mut lib = NodeCatalog::new();
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
fn open_repairs_torn_trailing_line() {
    // A crash mid-write leaves a torn last line in events.jsonl. On
    // reopen the tracker must not concatenate the next event onto it:
    // every line must parse and seq must stay contiguous.
    let root = tempfile::tempdir().unwrap();
    let run_dir = {
        let tracker = LocalTracker::create(root.path(), RunKind::Train, "torn").unwrap();
        for i in 0..3 {
            tracker.sink().record(&step_event(tracker.run_id(), i));
        }
        tracker.sink().flush();
        tracker.run_dir().to_path_buf()
    };
    // Simulate the crash: append half a line with no newline.
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .append(true)
        .open(run_dir.join("events.jsonl"))
        .unwrap();
    write!(f, "{{\"seq\":3,\"ts\":\"2026-").unwrap();
    drop(f);

    let tracker = LocalTracker::open(&run_dir).unwrap();
    tracker.sink().record(&step_event(tracker.run_id(), 99));
    tracker.sink().flush();

    let lines = read_lines(&run_dir.join("events.jsonl"));
    let seqs: Vec<u64> = lines.iter().map(|l| l["seq"].as_u64().unwrap()).collect();
    assert_eq!(seqs, vec![0, 1, 2, 3], "torn tail dropped, seq contiguous");
}

#[test]
fn open_without_events_file_starts_at_zero() {
    let root = tempfile::tempdir().unwrap();
    let run_dir = {
        let tracker = LocalTracker::create(root.path(), RunKind::Train, "bare").unwrap();
        tracker.run_dir().to_path_buf()
    };
    std::fs::remove_file(run_dir.join("events.jsonl")).unwrap();

    let tracker = LocalTracker::open(&run_dir).unwrap();
    tracker.sink().record(&step_event(tracker.run_id(), 0));
    tracker.sink().flush();
    let lines = read_lines(&run_dir.join("events.jsonl"));
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0]["seq"], 0);
}

#[test]
fn open_with_corrupt_manifest_errors_without_side_effects() {
    let root = tempfile::tempdir().unwrap();
    let run_dir = {
        let tracker = LocalTracker::create(root.path(), RunKind::Train, "c").unwrap();
        tracker.finalize(RunState::Completed).unwrap();
        tracker.run_dir().to_path_buf()
    };
    std::fs::write(run_dir.join("manifest.json"), "{broken").unwrap();

    assert!(LocalTracker::open(&run_dir).is_err());
    // The failed open must not have flipped the status back to running.
    assert_eq!(load_status(&run_dir).unwrap().state, RunState::Completed);
}

#[test]
fn save_manifest_roundtrips_updates() {
    let root = tempfile::tempdir().unwrap();
    let tracker = LocalTracker::create(root.path(), RunKind::Train, "m").unwrap();
    let mut manifest = load_manifest(tracker.run_dir()).unwrap();
    manifest.tags = vec!["updated".into()];
    manifest.python_version = Some("3.13.0".into());
    tracker.save_manifest(&manifest).unwrap();

    let back = load_manifest(tracker.run_dir()).unwrap();
    assert_eq!(back.tags, vec!["updated"]);
    assert_eq!(back.python_version.as_deref(), Some("3.13.0"));
    assert!(!tracker.run_dir().join("manifest.json.tmp").exists());
}

#[test]
fn run_id_prefixes_and_study_path_by_kind() {
    let root = tempfile::tempdir().unwrap();
    let trial = LocalTracker::create(root.path(), RunKind::Trial, "t").unwrap();
    assert!(trial.run_id().starts_with("trial_"));

    let train = LocalTracker::create(root.path(), RunKind::Train, "t").unwrap();
    assert!(train.run_id().starts_with("run_"));
    assert!(load_manifest(train.run_dir()).unwrap().study_path.is_none());

    let study = LocalTracker::create(root.path(), RunKind::Study, "s").unwrap();
    assert_eq!(
        load_manifest(study.run_dir())
            .unwrap()
            .study_path
            .as_deref(),
        Some("study.json")
    );

    // Ids from consecutive calls in the same second still differ.
    let a = LocalTracker::create(root.path(), RunKind::Train, "a").unwrap();
    let b = LocalTracker::create(root.path(), RunKind::Train, "b").unwrap();
    assert_ne!(a.run_id(), b.run_id());
}

#[test]
fn save_artifact_overwrites_and_roundtrips_binary() {
    let root = tempfile::tempdir().unwrap();
    let tracker = LocalTracker::create(root.path(), RunKind::Train, "a").unwrap();
    let bytes: Vec<u8> = (0u8..=255).collect();
    tracker.save_artifact("blob.bin", &bytes).unwrap();
    tracker.save_artifact("blob.bin", &bytes[..16]).unwrap(); // overwrite
    assert_eq!(
        std::fs::read(tracker.run_dir().join("blob.bin")).unwrap(),
        &bytes[..16]
    );
}

#[test]
fn collect_git_info_inside_and_outside_a_repo() {
    use somatize_runtime::collect_git_info;

    // This test file lives inside the soma-git repository.
    let inside = collect_git_info(std::path::Path::new(env!("CARGO_MANIFEST_DIR")));
    let sha = inside.sha.expect("sha inside a repo");
    assert_eq!(sha.len(), 40);
    assert!(sha.chars().all(|c| c.is_ascii_hexdigit()));
    assert!(inside.branch.is_some());
    assert!(inside.dirty.is_some());

    // A tempdir is outside any repository: everything is None.
    let dir = tempfile::tempdir().unwrap();
    let outside = collect_git_info(dir.path());
    assert!(outside.sha.is_none());
    assert!(outside.branch.is_none());
    assert!(outside.dirty.is_none());
}

#[test]
fn heartbeat_on_finalized_run_keeps_terminal_state() {
    // CONTRACT: heartbeat only refreshes timestamps; it never
    // resurrects a finished run.
    let root = tempfile::tempdir().unwrap();
    let tracker = LocalTracker::create(root.path(), RunKind::Train, "h").unwrap();
    tracker.finalize(RunState::Completed).unwrap();
    tracker.heartbeat().unwrap();
    assert_eq!(
        load_status(tracker.run_dir()).unwrap().state,
        RunState::Completed
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
    assert!(after.updated_at > before.updated_at);
    assert_eq!(after.state, RunState::Running);
}

// ── RunReader (read-side) ──

#[test]
fn run_reader_aggregates_a_tracked_run() {
    use somatize_core::cache::{CacheKey, CacheTier};
    use somatize_runtime::tracking::RunReader;

    let root = tempfile::tempdir().unwrap();
    let tracker = LocalTracker::create(root.path(), RunKind::Train, "agg").unwrap();
    let sink = tracker.sink();
    let rid = tracker.run_id().to_string();

    sink.record(&Event::RunStarted {
        run_id: rid.clone(),
        plan_summary: somatize_core::tracking::event::PlanSummary {
            total_nodes: 2,
            cached_nodes: 0,
            parallel_branches: 0,
        },
    });
    sink.record(&Event::NodeStarted {
        run_id: rid.clone(),
        node_id: "scaler".into(),
        kind: somatize_core::graph::filter::FilterKind::Trainable,
        effectful: false,
    });
    sink.record(&Event::NodeCompleted {
        run_id: rid.clone(),
        node_id: "scaler".into(),
        duration: std::time::Duration::from_millis(120),
        output_summary: "tensor".into(),
    });
    sink.record(&Event::NodeCacheHit {
        run_id: rid.clone(),
        node_id: "model".into(),
        key: CacheKey::from_parts(&[b"k"]),
        tier: CacheTier::Memory,
        load_time: std::time::Duration::from_millis(3),
    });
    sink.record(&Event::NodeCacheMiss {
        run_id: rid.clone(),
        node_id: "scaler".into(),
        key: CacheKey::from_parts(&[b"k2"]),
    });
    sink.record(&Event::MetricReported {
        run_id: rid.clone(),
        metric: metric("val_f1", 0.91, 3),
        node_id: None,
        trial_id: None,
    });
    sink.record(&Event::HealthFlag {
        run_id: rid.clone(),
        node_id: "model".into(),
        step: 7,
        flag: "LEAKAGE".into(),
        detail: "cka=0.99".into(),
    });
    sink.record(&Event::RunCompleted {
        run_id: rid.clone(),
        duration: std::time::Duration::from_millis(500),
    });
    tracker.finalize(RunState::Completed).unwrap();

    let reader = RunReader::open(tracker.run_dir()).unwrap();

    let events = reader.events().unwrap();
    assert_eq!(events.len(), 8);
    assert_eq!(events[0].seq, 0);

    let spans = reader.node_timings().unwrap();
    assert_eq!(spans.len(), 2);
    assert_eq!(spans[0].node_id, "scaler");
    assert_eq!(spans[0].outcome, "completed");
    assert_eq!(spans[0].duration_ms, Some(120));
    assert!(spans[0].started_ts.is_some());
    assert!(spans[0].finished_ts.is_some());
    assert_eq!(spans[1].node_id, "model");
    assert_eq!(spans[1].outcome, "cache_hit");
    assert_eq!(spans[1].cache_tier.as_deref(), Some("memory"));

    let cache = reader.cache_activity().unwrap();
    assert_eq!(cache.hits, 1);
    assert_eq!(cache.misses, 1);
    assert_eq!(cache.by_node["model"].hits, 1);
    assert_eq!(cache.by_node["scaler"].misses, 1);

    let series = reader.metric_series(Some("val_f1")).unwrap();
    assert_eq!(series.len(), 1);
    assert_eq!(series[0].value, 0.91);
    assert_eq!(series[0].step, 3);
    assert!(reader.metric_series(Some("missing")).unwrap().is_empty());

    let flags = reader.health_flags().unwrap();
    assert_eq!(flags.len(), 1);
    assert_eq!(flags[0].flag, "LEAKAGE");
    assert_eq!(flags[0].node_id, "model");

    let info = reader.info().unwrap();
    assert_eq!(info.state, "completed");
    assert_eq!(info.kind, "train");
    assert!(info.duration_ms.is_some());

    // Non-study run: empty trial timeline, no study.
    assert!(reader.trial_timeline().unwrap().is_empty());
    assert!(reader.study().unwrap().is_none());
}

/// The agent-level events a driver emits reach the reader as aggregates:
/// per-node activity with the suspend/complete accounting rules, and the
/// per-effect timeline. This is the read side of what
/// `agentic_step.rs::a_step_emits_agent_events` proves about emission.
#[test]
fn run_reader_aggregates_agent_events() {
    use somatize_runtime::tracking::RunReader;
    use std::time::Duration;

    let root = tempfile::tempdir().unwrap();
    let tracker = LocalTracker::create(root.path(), RunKind::Other, "agentic").unwrap();
    let sink = tracker.sink();
    let rid = tracker.run_id().to_string();

    // "planner": a full step — one LLM effect, one tool, a fan-out, done.
    sink.record(&Event::NodeStarted {
        run_id: rid.clone(),
        node_id: "planner".into(),
        kind: somatize_core::graph::filter::FilterKind::Opaque,
        effectful: true,
    });
    sink.record(&Event::AgentTurnStarted {
        run_id: rid.clone(),
        node_id: "planner".into(),
        turn: 0,
    });
    sink.record(&Event::EffectRequested {
        run_id: rid.clone(),
        node_id: "planner".into(),
        turn: 0,
        effect: "llm:m".into(),
    });
    sink.record(&Event::EffectCompleted {
        run_id: rid.clone(),
        node_id: "planner".into(),
        turn: 0,
        effect: "llm:m".into(),
        duration: Duration::from_millis(800),
        replayed: false,
        is_error: false,
    });
    sink.record(&Event::ToolCalled {
        run_id: rid.clone(),
        node_id: "planner".into(),
        tool: "search".into(),
        is_error: false,
    });
    sink.record(&Event::AgentSpawned {
        run_id: rid.clone(),
        node_id: "planner".into(),
        turn: 1,
        children: vec!["planner/w0".into(), "planner/w1".into()],
        join: "all".into(),
    });
    sink.record(&Event::AgentStepCompleted {
        run_id: rid.clone(),
        node_id: "planner/w0".into(),
        turns: 1,
        duration: Duration::from_millis(200),
        input_tokens: 10,
        output_tokens: 5,
        failed: false,
    });
    sink.record(&Event::AgentStepCompleted {
        run_id: rid.clone(),
        node_id: "planner".into(),
        turns: 2,
        duration: Duration::from_millis(1200),
        input_tokens: 100,
        output_tokens: 40,
        failed: false,
    });
    sink.record(&Event::NodeCompleted {
        run_id: rid.clone(),
        node_id: "planner".into(),
        duration: Duration::from_millis(1300),
        output_summary: "plan".into(),
    });

    // "approve": suspended and never resumed — its cost stands.
    sink.record(&Event::AgentTurnStarted {
        run_id: rid.clone(),
        node_id: "approve".into(),
        turn: 0,
    });
    sink.record(&Event::Suspended {
        run_id: rid.clone(),
        node_id: "approve".into(),
        reason: "human".into(),
        turns: 1,
        duration: Duration::from_millis(300),
        input_tokens: 7,
        output_tokens: 2,
    });

    // "zombie": died mid-effect — no accounting event at all, so its
    // turn count falls back to what was seen on the wire.
    sink.record(&Event::AgentTurnStarted {
        run_id: rid.clone(),
        node_id: "zombie".into(),
        turn: 0,
    });
    sink.record(&Event::AgentTurnStarted {
        run_id: rid.clone(),
        node_id: "zombie".into(),
        turn: 1,
    });
    sink.record(&Event::EffectRequested {
        run_id: rid.clone(),
        node_id: "zombie".into(),
        turn: 1,
        effect: "llm:m".into(),
    });
    tracker.finalize(RunState::Completed).unwrap();

    let reader = RunReader::open(tracker.run_dir()).unwrap();

    let activity = reader.agentic_activity().unwrap();
    let planner = &activity.by_node["planner"];
    assert_eq!(planner.turns, 2);
    assert_eq!(planner.effects, 1);
    assert_eq!(planner.effects_by_label["llm:m"], 1);
    assert_eq!(planner.tool_calls, 1);
    assert_eq!(planner.spawned, 2);
    assert_eq!(planner.completions, 1);
    assert_eq!((planner.input_tokens, planner.output_tokens), (100, 40));
    assert_eq!(planner.duration_ms, 1200);

    let approve = &activity.by_node["approve"];
    assert_eq!(approve.suspensions, 1);
    assert_eq!(
        (approve.turns, approve.input_tokens, approve.output_tokens),
        (1, 7, 2),
        "an unresumed suspension's cost must stand"
    );

    let zombie = &activity.by_node["zombie"];
    assert_eq!(
        zombie.turns, 2,
        "wire-observed turns are the fallback when nothing accounted"
    );
    assert_eq!(zombie.effects, 0);

    assert_eq!(activity.turns, 2 + 1 + 1 + 2);
    assert_eq!(activity.input_tokens, 100 + 10 + 7);
    assert_eq!(activity.steps_completed, 2);
    assert_eq!(activity.suspensions, 1);
    assert_eq!(activity.tool_calls, 1);

    let timeline = reader.agentic_timeline().unwrap();
    assert_eq!(timeline.len(), 2);
    assert_eq!(timeline[0].node_id, "planner");
    assert_eq!(timeline[0].outcome, "completed");
    assert_eq!(timeline[0].duration_ms, Some(800));
    assert!(!timeline[0].replayed);
    assert_eq!(timeline[1].node_id, "zombie");
    assert_eq!(
        timeline[1].outcome, "running",
        "an unclosed effect span means the run died mid-effect"
    );

    // The step's node span knows it was a step.
    let spans = reader.node_timings().unwrap();
    let planner_span = spans.iter().find(|s| s.node_id == "planner").unwrap();
    assert!(planner_span.effectful);

    // And the summary rolls the cost up for the pool and the headline.
    let summary = somatize_runtime::tracking::summarize(&reader).unwrap();
    let cost = summary.conclusion.agent_cost.as_ref().unwrap();
    assert_eq!(cost.turns, 6);
    assert_eq!(cost.input_tokens, 117);
    assert_eq!(cost.suspensions, 1);
    assert!(
        summary.conclusion.headline.contains("agent 6 turns"),
        "{}",
        summary.conclusion.headline
    );
}

/// A resumed run's final totals are cumulative; the earlier suspension
/// cost must be superseded, not added.
#[test]
fn a_resumed_completion_supersedes_the_suspension_cost() {
    use somatize_runtime::tracking::RunReader;
    use std::time::Duration;

    let root = tempfile::tempdir().unwrap();
    let tracker = LocalTracker::create(root.path(), RunKind::Other, "resumed").unwrap();
    let sink = tracker.sink();
    let rid = tracker.run_id().to_string();

    sink.record(&Event::Suspended {
        run_id: rid.clone(),
        node_id: "approve".into(),
        reason: "human".into(),
        turns: 1,
        duration: Duration::from_millis(300),
        input_tokens: 7,
        output_tokens: 2,
    });
    sink.record(&Event::Resumed {
        run_id: rid.clone(),
        node_id: "approve".into(),
        turn: 0,
    });
    sink.record(&Event::AgentStepCompleted {
        run_id: rid.clone(),
        node_id: "approve".into(),
        turns: 2,
        duration: Duration::from_millis(900),
        input_tokens: 15, // cumulative: replayed effects re-count
        output_tokens: 6,
        failed: false,
    });
    tracker.finalize(RunState::Completed).unwrap();

    let activity = RunReader::open(tracker.run_dir())
        .unwrap()
        .agentic_activity()
        .unwrap();
    let approve = &activity.by_node["approve"];
    assert_eq!(
        (approve.turns, approve.input_tokens, approve.output_tokens),
        (2, 15, 6),
        "the cumulative completion must supersede the suspension, not stack on it"
    );
    assert_eq!(approve.suspensions, 1, "the suspension itself still counts");
}

#[test]
fn run_reader_skips_torn_and_unknown_lines() {
    use somatize_runtime::tracking::RunReader;

    let root = tempfile::tempdir().unwrap();
    let tracker = LocalTracker::create(root.path(), RunKind::Train, "torn-read").unwrap();
    for i in 0..3 {
        tracker.sink().record(&step_event(tracker.run_id(), i));
    }
    tracker.sink().flush();

    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .append(true)
        .open(tracker.run_dir().join("events.jsonl"))
        .unwrap();
    // An event kind from a future soma, then a torn tail.
    writeln!(
        f,
        "{{\"seq\":3,\"ts\":\"2026-07-29T10:00:00Z\",\"event_type\":\"FutureThing\",\"x\":1}}"
    )
    .unwrap();
    write!(f, "{{\"seq\":4,\"ts\":\"2026-").unwrap();
    drop(f);

    let reader = RunReader::open(tracker.run_dir()).unwrap();
    let events = reader.events().unwrap();
    assert_eq!(events.len(), 3, "torn + unknown lines are skipped");
    let seqs: Vec<u64> = events.iter().map(|e| e.seq).collect();
    assert_eq!(seqs, vec![0, 1, 2]);
}

#[test]
fn list_runs_orders_and_detects_crashes() {
    use somatize_runtime::tracking::{RunReader, list_runs};

    let root = tempfile::tempdir().unwrap();

    let t1 = LocalTracker::create(root.path(), RunKind::Train, "first").unwrap();
    t1.finalize(RunState::Completed).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(1100)); // run ids have 1s resolution
    let t2 = LocalTracker::create(root.path(), RunKind::Fit, "second").unwrap();
    // Simulate a crash: running status whose heartbeat went stale.
    let stale = serde_json::json!({
        "state": "running",
        "updated_at": "2026-01-01T00:00:00Z",
        "heartbeat_at": "2026-01-01T00:00:00Z",
    });
    std::fs::write(
        t2.run_dir().join("status.json"),
        serde_json::to_vec_pretty(&stale).unwrap(),
    )
    .unwrap();

    // A stray non-run directory is skipped.
    std::fs::create_dir_all(root.path().join("runs/not-a-run")).unwrap();

    let runs = list_runs(root.path()).unwrap();
    assert_eq!(runs.len(), 2);
    assert_eq!(runs[0].name, "second", "newest first");
    assert_eq!(runs[0].state, "crashed");
    assert_eq!(runs[1].name, "first");
    assert_eq!(runs[1].state, "completed");

    let info = RunReader::open(&runs[0].dir).unwrap().info().unwrap();
    assert_eq!(info.state, "crashed");
}

#[test]
fn session_fit_and_run_emit_matching_run_bracket() {
    use somatize_core::cache::CacheKey;
    use somatize_core::data::value::Value;
    use somatize_core::graph::filter::{Filter, FilterKind, FilterMeta, StreamMode};
    use somatize_core::graph::{Edge, Graph, Node};
    use somatize_runtime::tracking::RunReader;
    use somatize_runtime::{GraphSession, NodeCatalog};

    struct Doubler;
    impl Filter for Doubler {
        fn config_hash(&self) -> CacheKey {
            CacheKey::from_parts(&[b"Doubler"])
        }
        fn fit(&self, _x: &Value, _y: Option<&Value>) -> somatize_core::Result<Value> {
            Ok(Value::json(serde_json::json!({})))
        }
        fn forward(&self, x: &Value, _state: &Value) -> somatize_core::Result<Value> {
            Ok(x.clone())
        }
        fn meta(&self) -> FilterMeta {
            FilterMeta {
                name: "Doubler".into(),
                kind: FilterKind::Trainable,
                cacheable: false,
                differentiable: false,
                deterministic: true,
                stream_mode: StreamMode::FixedState,
                distribution: somatize_core::graph::filter::Distribution::Local,
                input_schema: None,
                output_schema: None,
            }
        }
    }

    let root = tempfile::tempdir().unwrap();
    let tracker = LocalTracker::create(root.path(), RunKind::Fit, "bracket").unwrap();

    let mut graph = Graph::new();
    graph.nodes.push(Node::new("a", "a", "a"));
    graph.nodes.push(Node::new("b", "b", "b"));
    graph.edges.push(Edge::data("e0", "a", "b"));
    let mut lib = NodeCatalog::new();
    lib.register("a", Box::new(Doubler));
    lib.register("b", Box::new(Doubler));

    let bus = Arc::new(EventBus::new(64));
    bus.add_sink(tracker.sink());
    let mut session = GraphSession::new(graph, lib).with_event_bus(bus);
    session
        .fit(&Value::tensor(vec![1.0, 2.0], vec![2]), None)
        .unwrap();
    tracker.finalize(RunState::Completed).unwrap();

    let reader = RunReader::open(tracker.run_dir()).unwrap();
    let events = reader.events().unwrap();

    let started: Vec<_> = events
        .iter()
        .filter_map(|e| match &e.event {
            Event::RunStarted { run_id, .. } => Some(run_id.clone()),
            _ => None,
        })
        .collect();
    let completed: Vec<_> = events
        .iter()
        .filter_map(|e| match &e.event {
            Event::RunCompleted { run_id, .. } => Some(run_id.clone()),
            _ => None,
        })
        .collect();
    let node_run_ids: Vec<_> = events
        .iter()
        .filter_map(|e| match &e.event {
            Event::NodeStarted { run_id, .. } => Some(run_id.clone()),
            _ => None,
        })
        .collect();

    assert_eq!(started.len(), 1, "exactly one RunStarted");
    assert_eq!(completed.len(), 1, "exactly one RunCompleted");
    assert_eq!(started[0], completed[0], "bracket shares one run id");
    assert_eq!(node_run_ids.len(), 2);
    assert!(
        node_run_ids.iter().all(|id| *id == started[0]),
        "node events tagged with the bracket's run id: {node_run_ids:?} vs {}",
        started[0]
    );

    // The reader can now compute a total-run duration for a local fit.
    matches!(
        events.first().map(|e| &e.event),
        Some(Event::RunStarted { .. })
    );
}

#[test]
fn run_reader_overlay_and_annotated_mermaid() {
    use somatize_core::cache::{CacheKey, CacheTier};
    use somatize_core::graph::{Edge, Graph, Node};
    use somatize_core::viz::NodeStatus;
    use somatize_runtime::tracking::RunReader;

    let root = tempfile::tempdir().unwrap();
    let tracker = LocalTracker::create(root.path(), RunKind::Fit, "overlay").unwrap();
    let sink = tracker.sink();
    let rid = tracker.run_id().to_string();

    // Snapshot the graph the way `track_run` does.
    let mut graph = Graph::new();
    graph.nodes.push(Node::new("scaler", "scaler", "scaler"));
    graph.nodes.push(Node::new("model", "model", "model"));
    graph.edges.push(Edge::data("e0", "scaler", "model"));
    tracker
        .save_artifact(
            "graph.json",
            serde_json::to_string(&graph).unwrap().as_bytes(),
        )
        .unwrap();

    // scaler runs twice (e.g. two epochs); model is a cache hit and flagged.
    for _ in 0..2 {
        sink.record(&Event::NodeStarted {
            run_id: rid.clone(),
            node_id: "scaler".into(),
            kind: somatize_core::graph::filter::FilterKind::Trainable,
            effectful: false,
        });
        sink.record(&Event::NodeCompleted {
            run_id: rid.clone(),
            node_id: "scaler".into(),
            duration: std::time::Duration::from_millis(600),
            output_summary: String::new(),
        });
    }
    sink.record(&Event::NodeCacheHit {
        run_id: rid.clone(),
        node_id: "model".into(),
        key: CacheKey::from_parts(&[b"k"]),
        tier: CacheTier::Memory,
        load_time: std::time::Duration::from_millis(2),
    });
    sink.record(&Event::HealthFlag {
        run_id: rid.clone(),
        node_id: "model".into(),
        step: 1,
        flag: "LEAKAGE".into(),
        detail: String::new(),
    });
    sink.record(&Event::HealthFlag {
        run_id: rid.clone(),
        node_id: "model".into(),
        step: 2,
        flag: "LEAKAGE".into(), // duplicate — must dedupe
        detail: String::new(),
    });
    tracker.finalize(RunState::Completed).unwrap();

    let reader = RunReader::open(tracker.run_dir()).unwrap();
    let overlay = reader.overlay().unwrap();

    let scaler = &overlay.nodes["scaler"];
    assert_eq!(scaler.status, Some(NodeStatus::Completed));
    assert_eq!(scaler.duration_ms, Some(1200), "durations accumulate");
    assert_eq!(scaler.sublabel.as_deref(), Some("×2"));

    let model = &overlay.nodes["model"];
    assert_eq!(model.status, Some(NodeStatus::Cached));
    assert_eq!(model.cache_tier.as_deref(), Some("memory"));
    assert_eq!(model.flags, vec!["LEAKAGE".to_string()], "flags deduped");

    let mermaid = reader.to_mermaid().unwrap();
    assert!(
        mermaid.contains("scaler[\"scaler<br/>1.2s · ×2\"]"),
        "{mermaid}"
    );
    assert!(mermaid.contains("class scaler soma_completed"), "{mermaid}");
    assert!(mermaid.contains("class model soma_flagged"), "{mermaid}");
}

#[test]
fn run_reader_to_mermaid_without_graph_snapshot_errors() {
    use somatize_runtime::tracking::RunReader;

    let root = tempfile::tempdir().unwrap();
    let tracker = LocalTracker::create(root.path(), RunKind::Train, "no-graph").unwrap();
    tracker.finalize(RunState::Completed).unwrap();

    let reader = RunReader::open(tracker.run_dir()).unwrap();
    assert!(reader.graph().unwrap().is_none());
    assert!(reader.overlay().unwrap().nodes.is_empty());
    let err = reader.to_mermaid().unwrap_err();
    assert!(err.to_string().contains("graph.json"), "{err}");
}
