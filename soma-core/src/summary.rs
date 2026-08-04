//! What a run says about itself — the data half.
//!
//! [`RunSummary`] and [`RunConclusion`] are pure, serializable facts:
//! how a run ended, what it cost, what it measured, what it flagged.
//! They live in `soma-core` (not next to the code that computes them)
//! for the same reason [`GraphOverlay`](crate::viz::GraphOverlay) does:
//! the producer is `soma-runtime`'s `summarize`, which reads a run
//! directory, but the *consumers* — the experiment journal, the MCP
//! tools, any front-end — must be able to name these types without
//! taking a dependency on the execution engine.
//!
//! [`RunConclusion::render_headline`] is a deterministic template: same
//! facts, same string, no model in the loop. That is what makes a
//! headline safe to hash, snapshot-test and index for retrieval.

use crate::fingerprint::ArchitectureFingerprint;
use crate::tracking::GitInfo;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt::Write as _;

/// How a run ended, independent of whether its results were any good.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum RunOutcome {
    Completed,
    Failed,
    /// Marked running, but the heartbeat went stale: the process died.
    Crashed,
    /// Still going.
    Running,
}

impl RunOutcome {
    /// Map a [`RunInfo`]-style state string onto an outcome. Anything
    /// unrecognized is treated as still running, never as success.
    ///
    /// [`RunInfo`]: https://docs.rs/somatize-runtime
    pub fn from_state(state: &str) -> Self {
        match state {
            "completed" => Self::Completed,
            "failed" => Self::Failed,
            "crashed" => Self::Crashed,
            _ => Self::Running,
        }
    }

    /// Past-tense verb used to open a headline.
    pub fn verb(&self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Crashed => "crashed",
            Self::Running => "running",
        }
    }
}

/// The node that dominated wall time, and by how much.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeCost {
    pub node_id: String,
    pub duration_ms: u64,
    /// This node's share of all node compute time, in `[0, 1]`.
    pub share: f64,
}

/// One flag family and where it fired.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlagCount {
    pub flag: String,
    pub count: usize,
    /// Node (or `node/module.path`) ids that raised it, sorted, deduped.
    pub nodes: Vec<String>,
}

impl FlagCount {
    /// Group one flag's occurrences: `count` is how many times it fired,
    /// `nodes` the distinct places it fired in.
    pub fn group(flag: impl Into<String>, mut nodes: Vec<String>) -> Self {
        let count = nodes.len();
        nodes.sort();
        nodes.dedup();
        Self {
            flag: flag.into(),
            count,
            nodes,
        }
    }

    /// Union of two flag groupings, summing counts and merging nodes.
    pub fn merge_all(a: &[FlagCount], b: &[FlagCount]) -> Vec<FlagCount> {
        let mut grouped: BTreeMap<&str, Vec<String>> = BTreeMap::new();
        for flag in a.iter().chain(b) {
            grouped
                .entry(flag.flag.as_str())
                .or_default()
                .extend(flag.nodes.iter().cloned());
        }
        grouped
            .into_iter()
            .map(|(flag, nodes)| FlagCount::group(flag, nodes))
            .collect()
    }
}

/// What a run's agent steps cost, absent for runs that had none.
///
/// Totals across every step node (spawned instances included). Token
/// counts are what the providers reported; a mock provider reports
/// whatever it likes, so zeros here mean "nothing reported", not free.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentCost {
    pub turns: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub tool_calls: u64,
    /// Steps that ended in an error — their cost is included above.
    pub steps_failed: u64,
    pub suspensions: u64,
}

/// Trial statistics for a study run.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TrialSummary {
    pub total: usize,
    pub completed: usize,
    pub pruned: usize,
    pub failed: usize,
    pub best_trial_id: Option<String>,
    pub best_value: Option<f64>,
    /// Metric the study optimized, when it declares one.
    pub objective: Option<String>,
}

/// What happened in a run, in fields and in one line.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunConclusion {
    /// Deterministic one-line rendering of everything below.
    #[serde(default)]
    pub headline: String,
    #[serde(default)]
    pub outcome: Option<RunOutcome>,
    /// The slowest node, when node timings exist.
    #[serde(default)]
    pub dominant_cost: Option<NodeCost>,
    /// `hits / (hits + misses)`, `None` when the run touched no cache.
    #[serde(default)]
    pub cache_hit_ratio: Option<f64>,
    /// `HealthFlag` events, grouped by flag.
    #[serde(default)]
    pub health_flags: Vec<FlagCount>,
    /// Flags from `diagnostics/report.json`, grouped by flag. Overlaps
    /// `health_flags` by construction (the audit both emits events and
    /// writes the report); kept separate because the report survives a
    /// truncated event log and carries the audited filter ids.
    #[serde(default)]
    pub audit_flags: Vec<FlagCount>,
    #[serde(default)]
    pub trials: Option<TrialSummary>,
    /// What the run's agent steps cost, `None` for runs with none.
    #[serde(default)]
    pub agent_cost: Option<AgentCost>,
    /// What the summarizer could not read, in the summarizer's words.
    /// Never an error — a half-written run is still worth recording.
    #[serde(default)]
    pub warnings: Vec<String>,
}

/// Number of metrics named in a headline before it says "+N more".
const HEADLINE_METRICS: usize = 3;

impl RunConclusion {
    /// Whether the conclusion carries any fact at all.
    pub fn is_empty(&self) -> bool {
        self.outcome.is_none()
            && self.dominant_cost.is_none()
            && self.cache_hit_ratio.is_none()
            && self.health_flags.is_empty()
            && self.audit_flags.is_empty()
            && self.trials.is_none()
            && self.agent_cost.is_none()
    }

    /// Render this conclusion as one deterministic line.
    ///
    /// Section order is fixed — outcome, error, trials, metrics, cost,
    /// cache, flags — so two runs' headlines are comparable by eye and
    /// diffable in tests. `error` is the run's first node error, which
    /// for a failed run is the single most useful thing it can say.
    pub fn render_headline(
        &self,
        duration_ms: Option<u64>,
        metrics: &BTreeMap<String, f64>,
        error: Option<&str>,
    ) -> String {
        let mut parts: Vec<String> = Vec::new();

        let outcome = self.outcome.unwrap_or(RunOutcome::Running);
        let preposition = match outcome {
            RunOutcome::Completed => "in",
            RunOutcome::Running => "for",
            _ => "after",
        };
        parts.push(match duration_ms {
            Some(ms) => format!("{} {preposition} {}", outcome.verb(), human_duration(ms)),
            None => outcome.verb().to_string(),
        });

        if let Some(error) = error {
            parts.push(format!("error: {}", one_line(error, 120)));
        }

        if let Some(trials) = &self.trials {
            let mut line = format!("{} trials", trials.total);
            let mut lost = Vec::new();
            if trials.pruned > 0 {
                lost.push(format!("{} pruned", trials.pruned));
            }
            if trials.failed > 0 {
                lost.push(format!("{} failed", trials.failed));
            }
            if !lost.is_empty() {
                let _ = write!(line, " ({})", lost.join(", "));
            }
            match (&trials.objective, trials.best_value) {
                (Some(objective), Some(best)) => {
                    let _ = write!(line, ", best {objective}={}", round4(best));
                }
                // A study that produced no scorable trial is a dead end
                // worth remembering, so say so instead of staying quiet.
                _ if trials.total > 0 => line.push_str(", no scorable trial"),
                _ => {}
            }
            parts.push(line);
        }

        if !metrics.is_empty() {
            let mut named: Vec<String> = metrics
                .iter()
                .take(HEADLINE_METRICS)
                .map(|(name, value)| format!("{name}={}", round4(*value)))
                .collect();
            if metrics.len() > HEADLINE_METRICS {
                named.push(format!("+{} more", metrics.len() - HEADLINE_METRICS));
            }
            parts.push(named.join(" "));
        }

        if let Some(cost) = &self.dominant_cost {
            parts.push(format!(
                "slowest {} ({}, {}% of compute)",
                cost.node_id,
                human_duration(cost.duration_ms),
                (cost.share * 100.0).round() as i64
            ));
        }

        if let Some(ratio) = self.cache_hit_ratio {
            parts.push(format!("cache {}% hits", (ratio * 100.0).round() as i64));
        }

        if let Some(agent) = &self.agent_cost {
            let mut line = format!("agent {} turns", agent.turns);
            if agent.input_tokens + agent.output_tokens > 0 {
                let _ = write!(
                    line,
                    ", {}→{} tokens",
                    human_count(agent.input_tokens),
                    human_count(agent.output_tokens)
                );
            }
            if agent.tool_calls > 0 {
                let _ = write!(line, ", {} tool calls", agent.tool_calls);
            }
            if agent.steps_failed > 0 {
                let _ = write!(line, ", {} steps failed", agent.steps_failed);
            }
            if agent.suspensions > 0 {
                let _ = write!(line, ", {} suspended", agent.suspensions);
            }
            parts.push(line);
        }

        let flags = FlagCount::merge_all(&self.health_flags, &self.audit_flags);
        if !flags.is_empty() {
            let rendered: Vec<String> = flags
                .iter()
                .map(|f| {
                    if f.count > 1 {
                        format!("{}×{}", f.flag, f.count)
                    } else {
                        f.flag.clone()
                    }
                })
                .collect();
            parts.push(format!("flags: {}", rendered.join(", ")));
        }

        parts.join(" · ")
    }
}

/// Everything one run directory says about itself.
///
/// Produced by `somatize_runtime::tracking::summarize`; consumed by the
/// experiment journal and anything that wants run facts without opening
/// five files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSummary {
    pub run_id: String,
    /// Absolute path, so a consumer can go read raw artifacts.
    pub run_dir: String,
    pub name: String,
    /// Manifest kind as its snake_case string (`fit`, `train`, `study`).
    pub kind: String,
    pub created_at: DateTime<Utc>,
    #[serde(default)]
    pub finished_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub duration_ms: Option<u64>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub git: GitInfo,
    #[serde(default)]
    pub seeds: BTreeMap<String, i64>,
    /// Hyperparameters declared at run start (`manifest.params`).
    #[serde(default)]
    pub params: BTreeMap<String, serde_json::Value>,
    /// What the run was expected to show, declared before it ran.
    #[serde(default)]
    pub hypothesis: Option<String>,
    #[serde(default)]
    pub parent_run_id: Option<String>,
    /// From `fingerprint.json`, absent for runs started before it
    /// existed or for graph-less runs (a study run has no graph).
    #[serde(default)]
    pub architecture: Option<ArchitectureFingerprint>,
    /// Human topology line, empty when the run has no `graph.json`.
    #[serde(default)]
    pub pipeline_summary: String,
    /// Last recorded value per metric name.
    #[serde(default)]
    pub metrics: BTreeMap<String, f64>,
    #[serde(default)]
    pub conclusion: RunConclusion,
}

/// Compact count rendering: `842`, `12.3k`, `1.2M`.
pub fn human_count(n: u64) -> String {
    if n < 1_000 {
        return n.to_string();
    }
    if n < 1_000_000 {
        return format!("{:.1}k", n as f64 / 1_000.0);
    }
    format!("{:.1}M", n as f64 / 1_000_000.0)
}

/// Compact wall-clock rendering: `840ms`, `2.4s`, `3m 07s`, `1h 12m`.
pub fn human_duration(ms: u64) -> String {
    if ms < 1_000 {
        return format!("{ms}ms");
    }
    let secs = ms as f64 / 1000.0;
    if secs < 60.0 {
        return format!("{secs:.1}s");
    }
    let total = ms / 1000;
    let (h, m, s) = (total / 3600, (total % 3600) / 60, total % 60);
    if h > 0 {
        format!("{h}h {m:02}m")
    } else {
        format!("{m}m {s:02}s")
    }
}

/// Four decimals, trailing zeros trimmed — stable across platforms.
pub fn round4(value: f64) -> String {
    if !value.is_finite() {
        return format!("{value}");
    }
    let text = format!("{value:.4}");
    let trimmed = text.trim_end_matches('0').trim_end_matches('.');
    if trimmed.is_empty() { "0" } else { trimmed }.to_string()
}

/// Collapse to a single line and cap the length — a headline is one
/// line by contract, and an error message is not.
pub fn one_line(text: &str, max: usize) -> String {
    let text = text.replace('\n', " ");
    if text.chars().count() <= max {
        return text;
    }
    let head: String = text.chars().take(max).collect();
    format!("{head}…")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metrics(pairs: &[(&str, f64)]) -> BTreeMap<String, f64> {
        pairs.iter().map(|(k, v)| ((*k).to_string(), *v)).collect()
    }

    #[test]
    fn headline_sections_appear_in_a_fixed_order() {
        let conclusion = RunConclusion {
            outcome: Some(RunOutcome::Completed),
            dominant_cost: Some(NodeCost {
                node_id: "encoder".into(),
                duration_ms: 9_000,
                share: 0.75,
            }),
            cache_hit_ratio: Some(0.5),
            health_flags: vec![FlagCount::group("LEAKAGE", vec!["a".into()])],
            audit_flags: vec![FlagCount::group("LEAKAGE", vec!["b".into()])],
            trials: Some(TrialSummary {
                total: 12,
                pruned: 4,
                objective: Some("val_f1".into()),
                best_value: Some(0.9),
                ..TrialSummary::default()
            }),
            ..RunConclusion::default()
        };
        let headline = conclusion.render_headline(Some(12_000), &metrics(&[("loss", 0.25)]), None);
        assert_eq!(
            headline,
            "completed in 12.0s · 12 trials (4 pruned), best val_f1=0.9 · loss=0.25 · \
             slowest encoder (9.0s, 75% of compute) · cache 50% hits · flags: LEAKAGE×2"
        );
    }

    #[test]
    fn headline_is_stable_across_renderings() {
        let conclusion = RunConclusion {
            outcome: Some(RunOutcome::Completed),
            ..RunConclusion::default()
        };
        let m = metrics(&[("b", 1.0), ("a", 2.0), ("d", 3.0), ("c", 4.0)]);
        let first = conclusion.render_headline(Some(1_000), &m, None);
        for _ in 0..5 {
            assert_eq!(conclusion.render_headline(Some(1_000), &m, None), first);
        }
        // Alphabetical, capped, with an honest count of the remainder.
        assert!(first.contains("a=2 b=1 c=4 +1 more"), "{first}");
    }

    #[test]
    fn an_error_never_breaks_the_single_line_contract() {
        let conclusion = RunConclusion {
            outcome: Some(RunOutcome::Failed),
            ..RunConclusion::default()
        };
        let headline = conclusion.render_headline(
            Some(500),
            &BTreeMap::new(),
            Some("shape mismatch\nexpected [32, 8]\ngot [32, 16]"),
        );
        assert_eq!(
            headline,
            "failed after 500ms · error: shape mismatch expected [32, 8] got [32, 16]"
        );
        assert!(!headline.contains('\n'));
    }

    #[test]
    fn a_headline_without_a_duration_still_names_the_outcome() {
        let conclusion = RunConclusion {
            outcome: Some(RunOutcome::Running),
            ..RunConclusion::default()
        };
        assert_eq!(
            conclusion.render_headline(None, &BTreeMap::new(), None),
            "running"
        );
        assert_eq!(
            conclusion.render_headline(Some(30_000), &BTreeMap::new(), None),
            "running for 30.0s"
        );
    }

    #[test]
    fn flag_grouping_counts_occurrences_and_dedupes_places() {
        let flag = FlagCount::group("DEAD_CHANNELS", vec!["b".into(), "a".into(), "a".into()]);
        assert_eq!(flag.count, 3);
        assert_eq!(flag.nodes, vec!["a", "b"]);

        let merged = FlagCount::merge_all(
            &[FlagCount::group("X", vec!["n1".into()])],
            &[
                FlagCount::group("X", vec!["n2".into()]),
                FlagCount::group("A", vec!["n3".into()]),
            ],
        );
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].flag, "A", "merged flags sort by name");
        assert_eq!(merged[1].flag, "X");
        assert_eq!(merged[1].count, 2);
        assert_eq!(merged[1].nodes, vec!["n1", "n2"]);
    }

    #[test]
    fn conclusion_emptiness_ignores_the_headline() {
        assert!(RunConclusion::default().is_empty());
        let only_text = RunConclusion {
            headline: "something".into(),
            ..RunConclusion::default()
        };
        assert!(only_text.is_empty(), "prose alone is not a fact");
        let with_outcome = RunConclusion {
            outcome: Some(RunOutcome::Failed),
            ..RunConclusion::default()
        };
        assert!(!with_outcome.is_empty());
    }

    #[test]
    fn summary_roundtrips_and_tolerates_a_minimal_record() {
        let summary = RunSummary {
            run_id: "r1".into(),
            run_dir: "/tmp/r1".into(),
            name: "baseline".into(),
            kind: "train".into(),
            created_at: Utc::now(),
            finished_at: None,
            duration_ms: Some(10),
            tags: vec!["mos".into()],
            git: GitInfo::default(),
            seeds: BTreeMap::from([("torch".into(), 42)]),
            params: BTreeMap::from([("lr".into(), serde_json::json!(0.01))]),
            hypothesis: Some("wider is better".into()),
            parent_run_id: Some("r0".into()),
            architecture: None,
            pipeline_summary: "a → b".into(),
            metrics: metrics(&[("f1", 0.5)]),
            conclusion: RunConclusion::default(),
        };
        let json = serde_json::to_string(&summary).unwrap();
        let back: RunSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(back.run_id, "r1");
        assert_eq!(back.seeds["torch"], 42);
        assert_eq!(back.params["lr"], serde_json::json!(0.01));
        assert_eq!(back.hypothesis.as_deref(), Some("wider is better"));

        let minimal = serde_json::json!({
            "run_id": "r", "run_dir": "/tmp/r", "name": "n", "kind": "fit",
            "created_at": "2026-07-30T10:00:00Z",
        });
        let back: RunSummary = serde_json::from_value(minimal).unwrap();
        assert!(back.metrics.is_empty());
        assert!(back.params.is_empty());
        assert!(back.conclusion.is_empty());
    }

    #[test]
    fn unknown_outcome_reads_as_running_not_success() {
        assert_eq!(RunOutcome::from_state("completed"), RunOutcome::Completed);
        assert_eq!(RunOutcome::from_state("crashed"), RunOutcome::Crashed);
        assert_eq!(RunOutcome::from_state("teleported"), RunOutcome::Running);
    }

    #[test]
    fn human_duration_scales() {
        assert_eq!(human_duration(0), "0ms");
        assert_eq!(human_duration(840), "840ms");
        assert_eq!(human_duration(2_400), "2.4s");
        assert_eq!(human_duration(59_900), "59.9s");
        assert_eq!(human_duration(187_000), "3m 07s");
        assert_eq!(human_duration(4_320_000), "1h 12m");
    }

    #[test]
    fn round4_trims_without_losing_precision() {
        assert_eq!(round4(1.0), "1");
        assert_eq!(round4(0.9125), "0.9125");
        assert_eq!(round4(0.912_549), "0.9125");
        assert_eq!(round4(-0.5), "-0.5");
        assert_eq!(round4(f64::NAN), "NaN");
    }

    #[test]
    fn one_line_truncates_on_characters_not_bytes() {
        assert_eq!(one_line("abc", 10), "abc");
        assert_eq!(one_line("a\nb", 10), "a b");
        assert_eq!(one_line("ααααα", 3), "ααα…");
    }
}
