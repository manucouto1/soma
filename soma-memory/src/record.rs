//! Data types for experiment tracking: records, research lines, trends.
//!
//! [`ExperimentRecord`] is the line format of `experiments.jsonl` — the
//! contract between whatever produces runs and whatever reasons about
//! them. Every field added after the first release is `#[serde(default)]`
//! and every unknown field is ignored, so a journal written by a newer
//! soma still loads on an older one and vice versa. The back-compat
//! tests at the bottom of this file are that promise, in code.

use crate::derivation::DerivationMove;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use somatize_core::fingerprint::ArchitectureFingerprint;
use somatize_core::summary::{RunConclusion, RunSummary};
use somatize_core::tracking::GitInfo;
use std::collections::HashMap;
use std::time::Duration;

/// Current `ExperimentRecord` schema version.
pub const RECORD_SCHEMA_VERSION: u32 = 2;

/// What a journal line is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum RecordKind {
    /// A run that happened.
    #[default]
    Experiment,
    /// A later correction or addition to an earlier record, named by
    /// `amends`. The journal stays strictly append-only: nothing is
    /// ever rewritten, conclusions are layered on.
    Amendment,
    /// Written by a newer soma with a kind this version does not know.
    #[serde(other)]
    Other,
}

/// A dense vector attached to a record, with the identity of whatever
/// produced it — a vector from a different model is not comparable, and
/// silently mixing the two is worse than having none.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Embedding {
    pub embedder_id: String,
    pub vector: Vec<f32>,
}

/// A recorded experiment in the knowledge base.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentRecord {
    pub id: String,
    pub name: String,
    pub hypothesis: Option<String>,
    pub pipeline_summary: String,
    pub params: HashMap<String, serde_json::Value>,
    pub metrics: HashMap<String, f64>,
    pub timestamp: DateTime<Utc>,
    pub duration: Duration,
    pub parent: Option<String>,
    pub research_line: Option<String>,
    pub tags: Vec<String>,
    pub notes: Option<String>,

    // ── Added in schema version 2. All optional, all defaulted. ──
    #[serde(default = "legacy_schema_version")]
    pub schema_version: u32,
    #[serde(default)]
    pub kind: RecordKind,
    /// The run this record came from, and where its raw artifacts are —
    /// so a reader can go look at the events, diagnostics and figures
    /// this summary was distilled from.
    #[serde(default)]
    pub run_id: Option<String>,
    #[serde(default)]
    pub run_dir: Option<String>,
    #[serde(default)]
    pub architecture: Option<ArchitectureFingerprint>,
    /// Metric this experiment was optimizing, when it declared one.
    #[serde(default)]
    pub objective: Option<String>,
    #[serde(default)]
    pub conclusion: Option<RunConclusion>,
    /// The move that produced this experiment from its parent.
    #[serde(default)]
    pub derivation: Option<DerivationMove>,
    #[serde(default)]
    pub git: Option<GitInfo>,
    /// For `kind = Amendment`: the id of the record being amended.
    #[serde(default)]
    pub amends: Option<String>,
    #[serde(default)]
    pub embedding: Option<Embedding>,
}

/// Records written before `schema_version` existed are version 1.
fn legacy_schema_version() -> u32 {
    1
}

impl ExperimentRecord {
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            hypothesis: None,
            pipeline_summary: String::new(),
            params: HashMap::new(),
            metrics: HashMap::new(),
            timestamp: Utc::now(),
            duration: Duration::ZERO,
            parent: None,
            research_line: None,
            tags: Vec::new(),
            notes: None,
            schema_version: RECORD_SCHEMA_VERSION,
            kind: RecordKind::Experiment,
            run_id: None,
            run_dir: None,
            architecture: None,
            objective: None,
            conclusion: None,
            derivation: None,
            git: None,
            amends: None,
            embedding: None,
        }
    }

    /// Build a record from a finished run's summary.
    ///
    /// This is where `pipeline_summary` stops being the constant
    /// `"tracked run"`: everything here is read off the run directory.
    /// A record with no parent starts its own research line, named
    /// after the run — children inherit that name, which is what makes
    /// the line analytics work on real data.
    pub fn from_run(summary: &RunSummary) -> Self {
        let mut tags = summary.tags.clone();
        let run_tag = format!("run:{}", summary.run_id);
        if !tags.contains(&run_tag) {
            tags.push(run_tag);
        }
        Self {
            id: summary.run_id.clone(),
            name: summary.name.clone(),
            hypothesis: None,
            pipeline_summary: summary.pipeline_summary.clone(),
            params: summary
                .seeds
                .iter()
                .map(|(k, v)| (format!("seed.{k}"), serde_json::json!(v)))
                .chain(summary.params.iter().map(|(k, v)| (k.clone(), v.clone())))
                .collect(),
            metrics: summary.metrics.clone().into_iter().collect(),
            timestamp: summary.created_at,
            duration: Duration::from_millis(summary.duration_ms.unwrap_or(0)),
            parent: summary.parent_run_id.clone(),
            research_line: Some(slugify(&summary.name)),
            tags,
            notes: None,
            schema_version: RECORD_SCHEMA_VERSION,
            kind: RecordKind::Experiment,
            run_id: Some(summary.run_id.clone()),
            run_dir: Some(summary.run_dir.clone()),
            architecture: summary.architecture.clone(),
            objective: summary
                .conclusion
                .trials
                .as_ref()
                .and_then(|t| t.objective.clone()),
            conclusion: Some(summary.conclusion.clone()),
            derivation: None,
            git: Some(summary.git.clone()),
            amends: None,
            embedding: None,
        }
    }

    /// Attach this record to its parent: set `parent`, inherit the
    /// research line, and compute the derivation move between them.
    ///
    /// The line is inherited rather than recomputed so that every
    /// descendant of one root shares one name, however far it drifts.
    pub fn descended_from(mut self, parent: &ExperimentRecord) -> Self {
        self.parent = Some(parent.id.clone());
        self.research_line = parent
            .research_line
            .clone()
            .or_else(|| Some(slugify(&parent.name)));
        self.derivation = Some(crate::derivation::derive(parent, &self));
        self
    }

    /// Merge user-supplied parameters over whatever the run captured.
    pub fn with_extra_params(
        mut self,
        params: impl IntoIterator<Item = (String, serde_json::Value)>,
    ) -> Self {
        self.params.extend(params);
        self
    }

    /// Merge extra metrics over whatever the run directory recorded —
    /// a study's best-trial values, a training loop's summary metrics.
    pub fn with_extra_metrics(mut self, metrics: impl IntoIterator<Item = (String, f64)>) -> Self {
        self.metrics.extend(metrics);
        self
    }

    /// An amendment: a later note attached to an existing record,
    /// appended as its own line so the journal is never rewritten.
    pub fn amendment(
        id: impl Into<String>,
        amends: impl Into<String>,
        notes: impl Into<String>,
    ) -> Self {
        let amends = amends.into();
        Self {
            kind: RecordKind::Amendment,
            amends: Some(amends.clone()),
            notes: Some(notes.into()),
            ..Self::new(id, format!("amendment to {amends}"))
        }
    }

    pub fn with_hypothesis(mut self, h: impl Into<String>) -> Self {
        self.hypothesis = Some(h.into());
        self
    }

    pub fn with_pipeline(mut self, summary: impl Into<String>) -> Self {
        self.pipeline_summary = summary.into();
        self
    }

    pub fn with_params(mut self, params: HashMap<String, serde_json::Value>) -> Self {
        self.params = params;
        self
    }

    pub fn with_metrics(mut self, metrics: HashMap<String, f64>) -> Self {
        self.metrics = metrics;
        self
    }

    pub fn with_duration(mut self, d: Duration) -> Self {
        self.duration = d;
        self
    }

    pub fn with_parent(mut self, parent: impl Into<String>) -> Self {
        self.parent = Some(parent.into());
        self
    }

    pub fn with_research_line(mut self, line: impl Into<String>) -> Self {
        self.research_line = Some(line.into());
        self
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    pub fn with_notes(mut self, notes: impl Into<String>) -> Self {
        self.notes = Some(notes.into());
        self
    }

    /// Whether this record carries a conclusion worth retrieving.
    pub fn has_conclusion(&self) -> bool {
        self.conclusion.as_ref().is_some_and(|c| !c.is_empty())
            || self.notes.is_some()
            || self.hypothesis.is_some()
    }

    /// The record's headline, or the best one-liner available.
    pub fn headline(&self) -> &str {
        self.conclusion
            .as_ref()
            .map(|c| c.headline.as_str())
            .filter(|h| !h.is_empty())
            .unwrap_or(&self.pipeline_summary)
    }
}

/// Lowercase, hyphen-separated, alphanumerics only — a stable research
/// line name derived from a run name.
pub fn slugify(name: &str) -> String {
    let mut slug = String::with_capacity(name.len());
    let mut pending_dash = false;
    for ch in name.chars() {
        if ch.is_alphanumeric() {
            if pending_dash && !slug.is_empty() {
                slug.push('-');
            }
            pending_dash = false;
            slug.extend(ch.to_lowercase());
        } else {
            pending_dash = true;
        }
    }
    if slug.is_empty() {
        "unnamed".into()
    } else {
        slug
    }
}

/// A research line: a group of related experiments tracking evolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchLine {
    pub name: String,
    pub experiments: Vec<String>,
    pub trend: Trend,
    pub best_metric_value: Option<f64>,
    pub best_metric_name: Option<String>,
}

/// Trend direction of a research line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Trend {
    Improving,
    Plateaued,
    Declining,
    Unknown,
}

impl std::fmt::Display for Trend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Improving => write!(f, "improving"),
            Self::Plateaued => write!(f, "plateaued"),
            Self::Declining => write!(f, "declining"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

/// A point where experiment results changed significantly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangePoint {
    pub experiment_id: String,
    pub timestamp: DateTime<Utc>,
    pub metric_name: String,
    pub value_before: f64,
    pub value_after: f64,
    pub description: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use somatize_core::summary::{RunOutcome, TrialSummary};
    use std::collections::BTreeMap;

    /// A byte-exact line as soma 0.3.0 wrote it, before any of the
    /// experiment-pool fields existed. This must load forever: a
    /// journal is the one artifact a user cannot be asked to migrate.
    const LEGACY_LINE: &str = r#"{"id":"study_001","name":"mos-sweep","hypothesis":null,"pipeline_summary":"study over 40 trials","params":{"lr":0.01},"metrics":{"val_f1":0.87},"timestamp":"2026-07-26T10:00:00Z","duration":{"secs":420,"nanos":0},"parent":null,"research_line":null,"tags":["mos","run:run_x"],"notes":null}"#;

    fn summary() -> RunSummary {
        RunSummary {
            run_id: "run_42".into(),
            run_dir: "/proj/.soma/runs/run_42".into(),
            name: "MoS Baseline!".into(),
            kind: "train".into(),
            created_at: Utc::now(),
            finished_at: None,
            duration_ms: Some(2_000),
            tags: vec!["mos".into()],
            git: GitInfo::default(),
            seeds: BTreeMap::from([("torch".to_string(), 42)]),
            params: BTreeMap::from([("lr".to_string(), serde_json::json!(0.01))]),
            parent_run_id: None,
            architecture: None,
            pipeline_summary: "a(Scaler) → b(SVM)".into(),
            metrics: BTreeMap::from([("val_f1".to_string(), 0.9)]),
            conclusion: RunConclusion {
                headline: "completed in 2.0s".into(),
                outcome: Some(RunOutcome::Completed),
                ..RunConclusion::default()
            },
        }
    }

    #[test]
    fn a_legacy_line_still_loads_and_defaults_the_new_fields() {
        let record: ExperimentRecord = serde_json::from_str(LEGACY_LINE).unwrap();
        assert_eq!(record.id, "study_001");
        assert_eq!(record.pipeline_summary, "study over 40 trials");
        assert_eq!(record.metrics["val_f1"], 0.87);
        assert_eq!(record.duration, Duration::from_secs(420));

        // Every field added since is absent, not wrong.
        assert_eq!(record.schema_version, 1, "pre-versioning lines are v1");
        assert_eq!(record.kind, RecordKind::Experiment);
        assert!(record.run_id.is_none());
        assert!(record.run_dir.is_none());
        assert!(record.architecture.is_none());
        assert!(record.conclusion.is_none());
        assert!(record.derivation.is_none());
        assert!(record.git.is_none());
        assert!(record.embedding.is_none());
    }

    #[test]
    fn a_line_from_a_newer_soma_loads_on_this_one() {
        // Unknown fields are ignored, unknown kinds fall back to Other:
        // an old reader never chokes on a journal a new writer appended.
        let future = serde_json::json!({
            "id": "x", "name": "n", "hypothesis": null, "pipeline_summary": "p",
            "params": {}, "metrics": {}, "timestamp": "2026-07-30T10:00:00Z",
            "duration": {"secs": 1, "nanos": 0}, "parent": null,
            "research_line": null, "tags": [], "notes": null,
            "schema_version": 99,
            "kind": "retraction",
            "causal_graph": {"nested": ["anything"]},
        });
        let record: ExperimentRecord = serde_json::from_value(future).unwrap();
        assert_eq!(record.schema_version, 99);
        assert_eq!(record.kind, RecordKind::Other);
    }

    #[test]
    fn a_current_record_roundtrips_with_every_field_populated() {
        let mut record = ExperimentRecord::new("r", "run")
            .with_hypothesis("wider is better")
            .with_notes("looked good until epoch 20");
        record.conclusion = Some(RunConclusion {
            headline: "completed in 2.0s".into(),
            outcome: Some(RunOutcome::Completed),
            trials: Some(TrialSummary {
                total: 3,
                ..TrialSummary::default()
            }),
            ..RunConclusion::default()
        });
        record.architecture = Some(ArchitectureFingerprint {
            digest: "abc".into(),
            n_nodes: 1,
            ..ArchitectureFingerprint::default()
        });
        record.git = Some(GitInfo {
            sha: Some("deadbeef".into()),
            ..GitInfo::default()
        });
        record.embedding = Some(Embedding {
            embedder_id: "minilm-v2".into(),
            vector: vec![0.1, 0.2],
        });
        record.run_dir = Some("/tmp/r".into());

        let json = serde_json::to_string(&record).unwrap();
        let back: ExperimentRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(back.schema_version, RECORD_SCHEMA_VERSION);
        assert_eq!(back.headline(), "completed in 2.0s");
        assert!(back.has_conclusion());
        assert_eq!(back.architecture.as_ref().unwrap().digest, "abc");
        assert_eq!(back.embedding.unwrap().embedder_id, "minilm-v2");
    }

    #[test]
    fn from_run_replaces_the_tracked_run_placeholder() {
        let record = ExperimentRecord::from_run(&summary());
        assert_eq!(record.id, "run_42");
        assert_eq!(record.run_id.as_deref(), Some("run_42"));
        assert_eq!(record.run_dir.as_deref(), Some("/proj/.soma/runs/run_42"));
        assert_eq!(record.pipeline_summary, "a(Scaler) → b(SVM)");
        assert_ne!(record.pipeline_summary, "tracked run");
        assert_eq!(record.metrics["val_f1"], 0.9);
        assert_eq!(record.params["seed.torch"], serde_json::json!(42));
        assert_eq!(record.params["lr"], serde_json::json!(0.01));
        assert_eq!(record.duration, Duration::from_millis(2_000));
        assert_eq!(record.headline(), "completed in 2.0s");
        // A rootless run opens its own research line, named after itself.
        assert_eq!(record.research_line.as_deref(), Some("mos-baseline"));
        assert!(record.tags.contains(&"run:run_42".to_string()));
    }

    #[test]
    fn the_run_tag_is_not_appended_twice() {
        let mut s = summary();
        s.tags = vec!["run:run_42".into()];
        let record = ExperimentRecord::from_run(&s);
        assert_eq!(record.tags, vec!["run:run_42"]);
    }

    #[test]
    fn descending_inherits_the_line_and_computes_the_move() {
        let parent = ExperimentRecord::from_run(&summary());
        let mut variant = summary();
        variant.run_id = "run_43".into();
        variant.name = "MoS Wider".into();
        variant.metrics.insert("val_f1".into(), 0.95);

        let child = ExperimentRecord::from_run(&variant).descended_from(&parent);
        assert_eq!(child.parent.as_deref(), Some("run_42"));
        assert_eq!(
            child.research_line.as_deref(),
            Some("mos-baseline"),
            "a variant stays in its parent's line, however it is named"
        );
        let derivation = child.derivation.as_ref().unwrap();
        assert_eq!(derivation.from, "run_42");
        assert_eq!(derivation.to, "run_43");
        let delta = derivation.metric_delta["val_f1"];
        assert!((delta.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn an_amendment_points_at_what_it_amends() {
        let amendment =
            ExperimentRecord::amendment("amend_1", "run_42", "the val split was leaking");
        assert_eq!(amendment.kind, RecordKind::Amendment);
        assert_eq!(amendment.amends.as_deref(), Some("run_42"));
        assert!(amendment.has_conclusion());
        let json = serde_json::to_string(&amendment).unwrap();
        assert!(json.contains("\"kind\":\"amendment\""), "{json}");
    }

    #[test]
    fn slugs_are_stable_and_never_empty() {
        assert_eq!(slugify("MoS Baseline!"), "mos-baseline");
        assert_eq!(slugify("mos_baseline v2"), "mos-baseline-v2");
        assert_eq!(slugify("  spaced  out  "), "spaced-out");
        assert_eq!(slugify("已经"), "已经");
        assert_eq!(slugify("!!!"), "unnamed");
        assert_eq!(slugify(""), "unnamed");
    }
}
