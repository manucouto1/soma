//! The join: one run directory → one self-describing summary.
//!
//! [`RunReader`] already computes every aggregate a reader could want,
//! but each in its own shape and each requiring the caller to know
//! which files exist. [`summarize`] folds all of them — manifest,
//! status, node timings, cache activity, health flags, metrics, study,
//! trial timeline, graph, `fingerprint.json`, `diagnostics/report.json`
//! — into a single [`RunSummary`].
//!
//! Only the reading lives here. The shapes it produces, and the
//! deterministic headline template, are pure data and live in
//! `somatize_core::summary` so consumers of the journal never have to
//! depend on the execution engine.
//!
//! Every input is optional. A run that crashed before writing anything
//! but its manifest still summarizes — it just collects warnings.

use serde::Deserialize;
use somatize_core::error::Result;
use somatize_core::fingerprint::{ArchitectureFingerprint, pipeline_summary};
use somatize_core::summary::{
    FlagCount, NodeCost, RunConclusion, RunOutcome, RunSummary, TrialSummary,
};
use std::collections::BTreeMap;
use std::fs;

use super::reader::RunReader;

/// Fold a run directory into a [`RunSummary`].
///
/// Fails only if the manifest is unreadable — every other missing or
/// corrupt file lands in `conclusion.warnings`.
pub fn summarize(reader: &RunReader) -> Result<RunSummary> {
    let manifest = reader.manifest()?;
    let info = reader.info()?;
    let mut warnings = Vec::new();

    let architecture = read_fingerprint(reader, &mut warnings);
    let pipeline = match reader.graph() {
        Ok(Some(graph)) => pipeline_summary(&graph),
        Ok(None) => String::new(),
        Err(e) => {
            warnings.push(format!("graph.json is unreadable: {e}"));
            String::new()
        }
    };

    let outcome = RunOutcome::from_state(&info.state);
    let (dominant_cost, node_error) = node_cost(reader, &mut warnings);
    let cache_hit_ratio = cache_ratio(reader, &mut warnings);
    let health_flags = health_flags(reader, &mut warnings);
    let audit_flags = audit_flags(reader, &mut warnings);
    let metrics = final_metrics(reader, &mut warnings);
    let trials = trial_summary(reader, &mut warnings);

    // A study run has no graph to describe, but it is not shapeless:
    // the sweep itself is the pipeline.
    let pipeline = match (&trials, pipeline.is_empty()) {
        (Some(trials), true) => format!("study over {} trials", trials.total),
        _ => pipeline,
    };

    if metrics.is_empty() && trials.is_none() && matches!(outcome, RunOutcome::Completed) {
        warnings.push("run completed without recording any metric".into());
    }

    let mut conclusion = RunConclusion {
        headline: String::new(),
        outcome: Some(outcome),
        dominant_cost,
        cache_hit_ratio,
        health_flags,
        audit_flags,
        trials,
        warnings,
    };
    conclusion.headline =
        conclusion.render_headline(info.duration_ms, &metrics, node_error.as_deref());

    Ok(RunSummary {
        run_id: manifest.run_id,
        run_dir: reader.dir().display().to_string(),
        name: manifest.name,
        kind: info.kind,
        created_at: manifest.created_at,
        finished_at: info.finished_at,
        duration_ms: info.duration_ms,
        tags: manifest.tags,
        git: manifest.git,
        seeds: manifest.seeds.into_iter().collect(),
        params: manifest.params.into_iter().collect(),
        hypothesis: manifest.hypothesis,
        parent_run_id: manifest.parent_run_id,
        architecture,
        pipeline_summary: pipeline,
        metrics,
        conclusion,
    })
}

fn read_fingerprint(
    reader: &RunReader,
    warnings: &mut Vec<String>,
) -> Option<ArchitectureFingerprint> {
    let path = reader.dir().join("fingerprint.json");
    if !path.exists() {
        return None;
    }
    match fs::read(&path).map(|b| serde_json::from_slice(&b)) {
        Ok(Ok(fingerprint)) => Some(fingerprint),
        Ok(Err(e)) => {
            warnings.push(format!("fingerprint.json is malformed: {e}"));
            None
        }
        Err(e) => {
            warnings.push(format!("fingerprint.json is unreadable: {e}"));
            None
        }
    }
}

/// Slowest node plus, when the run failed, the first node error —
/// which is the single most useful thing a failed run can tell you.
fn node_cost(reader: &RunReader, warnings: &mut Vec<String>) -> (Option<NodeCost>, Option<String>) {
    let spans = match reader.node_timings() {
        Ok(spans) => spans,
        Err(e) => {
            warnings.push(format!("event log is unreadable: {e}"));
            return (None, None);
        }
    };
    let error = spans.iter().find_map(|s| s.error.clone());
    let mut totals: BTreeMap<&str, u64> = BTreeMap::new();
    for span in &spans {
        if let Some(ms) = span.duration_ms {
            *totals.entry(span.node_id.as_str()).or_default() += ms;
        }
    }
    let total: u64 = totals.values().sum();
    // Ties break on node id (BTreeMap order), keeping this deterministic.
    let cost = totals
        .iter()
        .max_by_key(|(_, ms)| **ms)
        .filter(|_| total > 0)
        .map(|(node_id, ms)| NodeCost {
            node_id: (*node_id).to_string(),
            duration_ms: *ms,
            share: *ms as f64 / total as f64,
        });
    if !spans.is_empty() && spans.iter().any(|s| s.outcome == "running") {
        warnings.push("some nodes never reported completion".into());
    }
    (cost, error)
}

fn cache_ratio(reader: &RunReader, warnings: &mut Vec<String>) -> Option<f64> {
    let activity = match reader.cache_activity() {
        Ok(a) => a,
        Err(e) => {
            warnings.push(format!("cache activity is unreadable: {e}"));
            return None;
        }
    };
    let total = activity.hits + activity.misses;
    (total > 0).then(|| activity.hits as f64 / total as f64)
}

fn health_flags(reader: &RunReader, warnings: &mut Vec<String>) -> Vec<FlagCount> {
    let records = match reader.health_flags() {
        Ok(r) => r,
        Err(e) => {
            warnings.push(format!("health flags are unreadable: {e}"));
            return Vec::new();
        }
    };
    let mut grouped: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for record in records {
        grouped.entry(record.flag).or_default().push(record.node_id);
    }
    grouped
        .into_iter()
        .map(|(flag, nodes)| FlagCount::group(flag, nodes))
        .collect()
}

/// The audit's own view, from `diagnostics/report.json`.
///
/// That file is a serialized Python dataclass, not a Rust type, so it
/// is parsed structurally: anything that does not match the shape is a
/// warning, never an error.
fn audit_flags(reader: &RunReader, warnings: &mut Vec<String>) -> Vec<FlagCount> {
    #[derive(Deserialize)]
    struct AuditReport {
        #[serde(default)]
        filters: Vec<AuditFilter>,
    }
    #[derive(Deserialize)]
    struct AuditFilter {
        #[serde(rename = "filter")]
        filter_id: String,
        #[serde(default)]
        flags: Vec<String>,
    }

    let path = reader.dir().join("diagnostics").join("report.json");
    if !path.exists() {
        return Vec::new();
    }
    let report: AuditReport = match fs::read(&path).map(|b| serde_json::from_slice(&b)) {
        Ok(Ok(report)) => report,
        Ok(Err(e)) => {
            warnings.push(format!("diagnostics/report.json is malformed: {e}"));
            return Vec::new();
        }
        Err(e) => {
            warnings.push(format!("diagnostics/report.json is unreadable: {e}"));
            return Vec::new();
        }
    };
    let mut grouped: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for filter in report.filters {
        for flag in filter.flags {
            grouped
                .entry(flag)
                .or_default()
                .push(filter.filter_id.clone());
        }
    }
    grouped
        .into_iter()
        .map(|(flag, nodes)| FlagCount::group(flag, nodes))
        .collect()
}

/// Last value per metric name, in log order.
fn final_metrics(reader: &RunReader, warnings: &mut Vec<String>) -> BTreeMap<String, f64> {
    let points = match reader.metric_series(None) {
        Ok(points) => points,
        Err(e) => {
            warnings.push(format!("metrics are unreadable: {e}"));
            return BTreeMap::new();
        }
    };
    let mut latest: BTreeMap<String, (u64, f64)> = BTreeMap::new();
    for point in points {
        let entry = latest
            .entry(point.name)
            .or_insert((point.step, point.value));
        if point.step >= entry.0 {
            *entry = (point.step, point.value);
        }
    }
    latest
        .into_iter()
        .map(|(name, (_, value))| (name, value))
        .collect()
}

fn trial_summary(reader: &RunReader, warnings: &mut Vec<String>) -> Option<TrialSummary> {
    let study = match reader.study() {
        Ok(Some(study)) => study,
        Ok(None) => return None,
        Err(e) => {
            warnings.push(format!("study.json is unreadable: {e}"));
            return None;
        }
    };
    let mut summary = TrialSummary {
        total: study.trials.len(),
        objective: study
            .composite
            .as_ref()
            .map(|_| "composite".to_string())
            .or_else(|| study.objectives.first().map(|o| o.metric.clone())),
        best_value: study.best_value(),
        best_trial_id: study.best_trial().map(|t| t.id.clone()),
        ..TrialSummary::default()
    };
    for span in reader.trial_timeline().unwrap_or_default() {
        match span.state.as_str() {
            "completed" => summary.completed += 1,
            "pruned" => summary.pruned += 1,
            "failed" => summary.failed += 1,
            _ => {}
        }
    }
    Some(summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracking::LocalTracker;
    use chrono::Utc;
    use somatize_core::cache::CacheKey;
    use somatize_core::event::{Event, MetricRecord};
    use somatize_core::filter::FilterKind;
    use somatize_core::graph::{Node, linear_pipeline};
    use somatize_core::tracking::{RunKind, RunState, Tracker};
    use std::time::Duration;
    use tempfile::TempDir;

    /// A tracker over a fresh root, plus the root itself.
    fn tracker(kind: RunKind, name: &str) -> (TempDir, LocalTracker) {
        let root = TempDir::new().unwrap();
        let tracker = LocalTracker::create(root.path(), kind, name).unwrap();
        (root, tracker)
    }

    fn metric(name: &str, value: f64, step: usize) -> MetricRecord {
        MetricRecord {
            name: name.into(),
            value,
            step,
            timestamp: Utc::now(),
        }
    }

    #[test]
    fn summarizes_a_completed_run_end_to_end() {
        let (_root, tracker) = tracker(RunKind::Train, "baseline");
        let graph = linear_pipeline(vec![
            Node::new("a", "Scaler", "StandardScaler"),
            Node::new("b", "Model", "SVM"),
        ]);
        tracker
            .save_artifact("graph.json", &serde_json::to_vec(&graph).unwrap())
            .unwrap();
        let fingerprint = ArchitectureFingerprint::of(&graph).unwrap();
        tracker
            .save_artifact(
                "fingerprint.json",
                &serde_json::to_vec(&fingerprint).unwrap(),
            )
            .unwrap();

        let sink = tracker.sink();
        let run_id = tracker.run_id().to_string();
        for (node, ms) in [("a", 100u64), ("b", 900)] {
            sink.record(&Event::NodeStarted {
                run_id: run_id.clone(),
                node_id: node.into(),
                kind: FilterKind::Trainable,
                effectful: false,
            });
            sink.record(&Event::NodeCompleted {
                run_id: run_id.clone(),
                node_id: node.into(),
                duration: Duration::from_millis(ms),
                output_summary: "ok".into(),
            });
        }
        sink.record(&Event::MetricReported {
            run_id: run_id.clone(),
            metric: metric("val_f1", 0.9125, 3),
            node_id: None,
            trial_id: None,
        });
        sink.record(&Event::MetricReported {
            run_id: run_id.clone(),
            metric: metric("val_f1", 0.75, 1),
            node_id: None,
            trial_id: None,
        });
        sink.record(&Event::HealthFlag {
            run_id: run_id.clone(),
            node_id: "b".into(),
            step: 3,
            flag: "DEAD_CHANNELS".into(),
            detail: "12 dead".into(),
        });
        tracker.finalize(RunState::Completed).unwrap();

        let summary = summarize(&RunReader::open(tracker.run_dir()).unwrap()).unwrap();
        assert_eq!(summary.name, "baseline");
        assert_eq!(summary.kind, "train");
        assert_eq!(summary.pipeline_summary, "a(StandardScaler) → b(SVM)");
        assert_eq!(
            summary.architecture.as_ref().unwrap().digest,
            fingerprint.digest
        );
        // Last-by-step wins, not last-by-log-order.
        assert_eq!(summary.metrics["val_f1"], 0.9125);

        let c = &summary.conclusion;
        assert_eq!(c.outcome, Some(RunOutcome::Completed));
        let cost = c.dominant_cost.as_ref().unwrap();
        assert_eq!(cost.node_id, "b");
        assert_eq!(cost.duration_ms, 900);
        assert!((cost.share - 0.9).abs() < 1e-9);
        assert_eq!(c.health_flags[0].flag, "DEAD_CHANNELS");
        assert_eq!(c.health_flags[0].nodes, vec!["b"]);
        assert!(c.warnings.is_empty(), "{:?}", c.warnings);

        assert!(c.headline.starts_with("completed in "), "{}", c.headline);
        assert!(c.headline.contains("val_f1=0.9125"), "{}", c.headline);
        assert!(c.headline.contains("slowest b (900ms, 90% of compute)"));
        assert!(c.headline.contains("flags: DEAD_CHANNELS"));
    }

    #[test]
    fn headline_is_deterministic() {
        let (_root, tracker) = tracker(RunKind::Fit, "repeat");
        let sink = tracker.sink();
        for name in ["b_metric", "a_metric"] {
            sink.record(&Event::MetricReported {
                run_id: tracker.run_id().into(),
                metric: metric(name, 1.0, 0),
                node_id: None,
                trial_id: None,
            });
        }
        tracker.finalize(RunState::Completed).unwrap();
        let reader = RunReader::open(tracker.run_dir()).unwrap();
        let first = summarize(&reader).unwrap().conclusion.headline;
        for _ in 0..5 {
            assert_eq!(summarize(&reader).unwrap().conclusion.headline, first);
        }
        // Metric order comes from the name, not the log.
        assert!(first.contains("a_metric=1 b_metric=1"), "{first}");
    }

    #[test]
    fn a_failed_run_leads_with_its_error() {
        let (_root, tracker) = tracker(RunKind::Train, "boom");
        let sink = tracker.sink();
        sink.record(&Event::NodeStarted {
            run_id: tracker.run_id().into(),
            node_id: "encoder".into(),
            kind: FilterKind::Trainable,
            effectful: false,
        });
        sink.record(&Event::NodeFailed {
            run_id: tracker.run_id().into(),
            node_id: "encoder".into(),
            error: "shape mismatch: expected [32, 8]\ngot [32, 16]".into(),
        });
        tracker.finalize(RunState::Failed).unwrap();

        let summary = summarize(&RunReader::open(tracker.run_dir()).unwrap()).unwrap();
        assert_eq!(summary.conclusion.outcome, Some(RunOutcome::Failed));
        let headline = &summary.conclusion.headline;
        assert!(headline.starts_with("failed after "), "{headline}");
        assert!(headline.contains("error: shape mismatch"), "{headline}");
        // Newlines never leak into a one-line headline.
        assert!(!headline.contains('\n'));
    }

    #[test]
    fn a_bare_run_dir_summarizes_with_warnings() {
        let (_root, tracker) = tracker(RunKind::Other, "empty");
        tracker.finalize(RunState::Completed).unwrap();
        let summary = summarize(&RunReader::open(tracker.run_dir()).unwrap()).unwrap();
        assert_eq!(summary.pipeline_summary, "");
        assert!(summary.architecture.is_none());
        assert!(summary.metrics.is_empty());
        assert!(summary.conclusion.dominant_cost.is_none());
        assert!(summary.conclusion.cache_hit_ratio.is_none());
        assert!(
            summary
                .conclusion
                .warnings
                .iter()
                .any(|w| w.contains("without recording any metric"))
        );
        assert!(summary.conclusion.headline.starts_with("completed in "));
    }

    #[test]
    fn malformed_artifacts_warn_instead_of_failing() {
        let (_root, tracker) = tracker(RunKind::Train, "torn");
        tracker.save_artifact("graph.json", b"{not json").unwrap();
        tracker.save_artifact("fingerprint.json", b"[]").unwrap();
        tracker
            .save_artifact("diagnostics/report.json", b"{\"filters\": 3}")
            .unwrap();
        // A torn tail from a crash mid-write: skipped, never fatal.
        tracker
            .save_artifact("events.jsonl", b"{\"seq\":0,\"ts\":\"nope\"}\n{trunc")
            .unwrap();
        tracker.finalize(RunState::Completed).unwrap();

        let summary = summarize(&RunReader::open(tracker.run_dir()).unwrap()).unwrap();
        let warnings = summary.conclusion.warnings.join(" | ");
        assert!(warnings.contains("graph.json is unreadable"), "{warnings}");
        assert!(
            warnings.contains("fingerprint.json is malformed"),
            "{warnings}"
        );
        assert!(
            warnings.contains("diagnostics/report.json is malformed"),
            "{warnings}"
        );
        assert!(!summary.conclusion.headline.is_empty());
    }

    #[test]
    fn audit_report_flags_are_grouped_by_family() {
        let (_root, tracker) = tracker(RunKind::Train, "audited");
        let report = serde_json::json!({
            "n_steps": 30,
            "filters": [
                {"filter": "enc", "n_steps": 30, "metrics": {}, "flags": ["DEAD_CHANNELS"]},
                {"filter": "enc/layers.0", "n_steps": 30, "metrics": {},
                 "flags": ["DEAD_CHANNELS", "LEAKAGE"]},
            ],
        });
        tracker
            .save_artifact("diagnostics/report.json", report.to_string().as_bytes())
            .unwrap();
        tracker.finalize(RunState::Completed).unwrap();

        let summary = summarize(&RunReader::open(tracker.run_dir()).unwrap()).unwrap();
        let flags = &summary.conclusion.audit_flags;
        assert_eq!(flags.len(), 2);
        assert_eq!(flags[0].flag, "DEAD_CHANNELS");
        assert_eq!(flags[0].count, 2);
        assert_eq!(flags[0].nodes, vec!["enc", "enc/layers.0"]);
        assert_eq!(flags[1].flag, "LEAKAGE");
        assert!(
            summary
                .conclusion
                .headline
                .contains("flags: DEAD_CHANNELS×2, LEAKAGE")
        );
    }

    #[test]
    fn cache_ratio_counts_hits_over_attempts() {
        let (_root, tracker) = tracker(RunKind::Fit, "cached");
        let sink = tracker.sink();
        let run_id = tracker.run_id().to_string();
        sink.record(&Event::NodeCacheHit {
            run_id: run_id.clone(),
            node_id: "a".into(),
            key: CacheKey::hash_data(b"k"),
            tier: somatize_core::cache::CacheTier::Memory,
            load_time: Duration::from_millis(2),
        });
        for node in ["b", "c", "d"] {
            sink.record(&Event::NodeCacheMiss {
                run_id: run_id.clone(),
                node_id: node.into(),
                key: CacheKey::hash_data(b"k"),
            });
        }
        tracker.finalize(RunState::Completed).unwrap();

        let summary = summarize(&RunReader::open(tracker.run_dir()).unwrap()).unwrap();
        assert_eq!(summary.conclusion.cache_hit_ratio, Some(0.25));
        assert!(summary.conclusion.headline.contains("cache 25% hits"));
    }

    #[test]
    fn a_study_run_summarizes_its_trials() {
        use somatize_core::search::SearchSpace;
        use somatize_core::study::{
            Direction, Objective, SearchStrategy, Study, Trial, TrialState,
        };

        let (_root, tracker) = tracker(RunKind::Study, "sweep");
        let mut study = Study::new(
            "sweep",
            SearchSpace::new(),
            SearchStrategy::Random {
                n_trials: 4,
                seed: Some(0),
            },
            vec![Objective {
                metric: "val_f1".into(),
                direction: Direction::Maximize,
            }],
        );
        for (id, state, value) in [
            ("t0", TrialState::Completed, 0.80),
            ("t1", TrialState::Completed, 0.91),
            (
                "t2",
                TrialState::Pruned {
                    step: 2,
                    reason: "median".into(),
                },
                0.40,
            ),
            (
                "t3",
                TrialState::Failed {
                    error: "oom".into(),
                },
                0.0,
            ),
        ] {
            let mut trial = Trial::new(id, Default::default());
            trial.state = state;
            trial.metrics.push(metric("val_f1", value, 0));
            study.trials.push(trial);
        }
        tracker.save_study(&study).unwrap();
        tracker.finalize(RunState::Completed).unwrap();

        let summary = summarize(&RunReader::open(tracker.run_dir()).unwrap()).unwrap();
        let trials = summary.conclusion.trials.as_ref().unwrap();
        assert_eq!(trials.total, 4);
        assert_eq!(trials.completed, 2);
        assert_eq!(trials.pruned, 1);
        assert_eq!(trials.failed, 1);
        assert_eq!(trials.objective.as_deref(), Some("val_f1"));
        assert_eq!(trials.best_trial_id.as_deref(), Some("t1"));
        assert_eq!(trials.best_value, Some(0.91));
        assert!(
            summary
                .conclusion
                .headline
                .contains("4 trials (1 pruned, 1 failed), best val_f1=0.91"),
            "{}",
            summary.conclusion.headline
        );
        // A study run has no graph, but the sweep is its shape.
        assert_eq!(summary.pipeline_summary, "study over 4 trials");
        assert!(summary.conclusion.warnings.is_empty());
    }
}
