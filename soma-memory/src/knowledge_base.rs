//! KnowledgeBase trait and in-memory implementation.
//!
//! The trait returns **owned** records, not references. That costs a
//! clone per hit and buys two things a reference-returning trait cannot
//! have: a backend that pages, streams or queries a remote store (a
//! reference has to point at something the base already holds in
//! memory), and an implementation that does not have to keep a
//! duplicate `Vec` alive purely to have something to lend out.
//!
//! Only [`record`](KnowledgeBase::record), [`all`](KnowledgeBase::all)
//! and [`len`](KnowledgeBase::len) are required. Every query — search,
//! research lines, trends, change points, lineage, retrieval — has a
//! default implementation over `all()`, so a new backend gets the whole
//! analytics surface by implementing three methods, and the analytics
//! themselves live in one place instead of once per backend.

use crate::record::{ChangePoint, ExperimentRecord, ResearchLine, Trend};
use crate::retrieval::{RetrievalQuery, ScoredRecord, rank};
use somatize_core::error::Result;
use std::collections::HashMap;

/// One node of a lineage tree, with its distance from the focus.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LineageNode {
    /// The descendant record itself.
    pub record: ExperimentRecord,
    /// Generations below the focus (1 = direct child).
    pub depth: usize,
}

/// An experiment in context: where it came from and what came of it.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Lineage {
    /// The experiment the lineage was asked about.
    pub focus: ExperimentRecord,
    /// Ancestors from the root down to the direct parent.
    pub ancestors: Vec<ExperimentRecord>,
    /// Descendants in pre-order, so the list reads as an indented tree.
    pub descendants: Vec<LineageNode>,
}

impl Lineage {
    /// The root of the focus's line.
    pub fn root(&self) -> &ExperimentRecord {
        self.ancestors.first().unwrap_or(&self.focus)
    }
}

/// Trait for knowledge base backends.
pub trait KnowledgeBase: Send + Sync {
    /// Store a new experiment record.
    fn record(&mut self, experiment: ExperimentRecord) -> Result<()>;

    /// Every record this base holds, in insertion order.
    fn all(&self) -> Result<Vec<ExperimentRecord>>;

    /// Total number of experiments.
    fn len(&self) -> usize;

    /// Pick up records another process appended since this handle was
    /// opened. Returns how many were newly loaded.
    ///
    /// Defaults to a no-op for bases with nothing behind them. A
    /// long-lived reader (the MCP server) must call this before a read,
    /// or it answers from a snapshot that gets staler by the hour.
    fn refresh(&mut self) -> Result<usize> {
        Ok(0)
    }

    /// Whether the base holds no records at all.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get an experiment by ID.
    fn get(&self, id: &str) -> Result<Option<ExperimentRecord>> {
        Ok(self.all()?.into_iter().find(|e| e.id == id))
    }

    /// Substring search over name, hypothesis, notes, tags and pipeline.
    ///
    /// Kept as a cheap literal filter; [`retrieve`](Self::retrieve) is
    /// the ranked one.
    fn search(&self, query: &str, max_results: usize) -> Result<Vec<ExperimentRecord>> {
        let q = query.to_lowercase();
        Ok(self
            .all()?
            .into_iter()
            .filter(|e| matches_literal(e, &q))
            .take(max_results)
            .collect())
    }

    /// Rank records against a query by relevance, structure, recency
    /// and importance. See [`crate::retrieval`] for the formula.
    fn retrieve(&self, query: &RetrievalQuery) -> Result<Vec<ScoredRecord>> {
        Ok(rank(&self.all()?, query))
    }

    /// An experiment with its ancestors and descendants.
    fn lineage(&self, id: &str) -> Result<Option<Lineage>> {
        Ok(build_lineage(&self.all()?, id))
    }

    /// List experiments in a research line, ordered chronologically.
    fn experiments_in_line(&self, line: &str) -> Result<Vec<ExperimentRecord>> {
        let mut exps: Vec<ExperimentRecord> = self
            .all()?
            .into_iter()
            .filter(|e| e.research_line.as_deref() == Some(line))
            .collect();
        exps.sort_by_key(|e| e.timestamp);
        Ok(exps)
    }

    /// Get all research lines with trend analysis.
    fn research_lines(&self) -> Result<Vec<ResearchLine>> {
        Ok(research_lines_of(&self.all()?))
    }

    /// Find promising research lines (improving trend, high metric values).
    fn promising_lines(&self, metric: &str) -> Result<Vec<ResearchLine>> {
        Ok(self
            .research_lines()?
            .into_iter()
            .filter(|l| {
                l.trend == Trend::Improving || l.best_metric_name.as_deref() == Some(metric)
            })
            .collect())
    }

    /// Get the metric trajectory for a research line.
    fn trajectory(&self, line: &str, metric: &str) -> Result<Vec<(String, f64)>> {
        Ok(self
            .experiments_in_line(line)?
            .into_iter()
            .filter_map(|e| e.metrics.get(metric).map(|&v| (e.id.clone(), v)))
            .collect())
    }

    /// Detect change points where metrics shifted significantly.
    fn change_points(&self, line: &str, metric: &str, threshold: f64) -> Result<Vec<ChangePoint>> {
        let experiments = self.experiments_in_line(line)?;
        let traj: Vec<(&ExperimentRecord, f64)> = experiments
            .iter()
            .filter_map(|e| e.metrics.get(metric).map(|&v| (e, v)))
            .collect();
        let mut points = Vec::new();
        for window in traj.windows(2) {
            let (_, val_before) = window[0];
            let (exp_after, val_after) = window[1];
            let delta = (val_after - val_before).abs();
            if delta >= threshold {
                points.push(ChangePoint {
                    experiment_id: exp_after.id.clone(),
                    timestamp: exp_after.timestamp,
                    metric_name: metric.to_string(),
                    value_before: val_before,
                    value_after: val_after,
                    description: format!(
                        "{metric} changed from {val_before:.4} to {val_after:.4} (delta={delta:.4})"
                    ),
                });
            }
        }
        Ok(points)
    }

    /// Direct children of an experiment.
    fn children(&self, experiment_id: &str) -> Result<Vec<ExperimentRecord>> {
        Ok(self
            .all()?
            .into_iter()
            .filter(|e| e.parent.as_deref() == Some(experiment_id))
            .collect())
    }
}

/// Whether a lowercased query appears anywhere a human would look.
fn matches_literal(e: &ExperimentRecord, q: &str) -> bool {
    let contains = |s: &str| s.to_lowercase().contains(q);
    contains(&e.name)
        || contains(&e.pipeline_summary)
        || e.hypothesis.as_deref().is_some_and(contains)
        || e.notes.as_deref().is_some_and(contains)
        || e.tags.iter().any(|t| contains(t))
        || e.conclusion.as_ref().is_some_and(|c| contains(&c.headline))
}

/// Group records into research lines with a trend per line.
pub fn research_lines_of(records: &[ExperimentRecord]) -> Vec<ResearchLine> {
    let mut lines: HashMap<&str, Vec<&ExperimentRecord>> = HashMap::new();
    for record in records {
        if let Some(line) = record.research_line.as_deref() {
            lines.entry(line).or_default().push(record);
        }
    }

    let mut result: Vec<ResearchLine> = lines
        .into_iter()
        .map(|(name, members)| {
            let mut best_value = None;
            let mut best_name = None;
            for record in &members {
                for (metric, &value) in &record.metrics {
                    if best_value.is_none_or(|best| value > best) {
                        best_value = Some(value);
                        best_name = Some(metric.clone());
                    }
                }
            }
            ResearchLine {
                name: name.to_string(),
                experiments: members.iter().map(|e| e.id.clone()).collect(),
                trend: compute_trend(&members),
                best_metric_value: best_value,
                best_metric_name: best_name,
            }
        })
        .collect();
    result.sort_by(|a, b| a.name.cmp(&b.name));
    result
}

/// Direction of the last few values of the line's first metric.
fn compute_trend(members: &[&ExperimentRecord]) -> Trend {
    if members.len() < 2 {
        return Trend::Unknown;
    }
    let Some(key) = members[0].metrics.keys().next() else {
        return Trend::Unknown;
    };
    let values: Vec<f64> = members
        .iter()
        .filter_map(|e| e.metrics.get(key).copied())
        .collect();
    if values.len() < 2 {
        return Trend::Unknown;
    }
    let recent = &values[values.len().saturating_sub(3)..];
    let diffs: Vec<f64> = recent.windows(2).map(|w| w[1] - w[0]).collect();
    if diffs.is_empty() {
        Trend::Unknown
    } else if diffs.iter().all(|&d| d > 0.001) {
        Trend::Improving
    } else if diffs.iter().all(|&d| d < -0.001) {
        Trend::Declining
    } else {
        Trend::Plateaued
    }
}

/// Walk parents up and children down from `id`.
///
/// A parent cycle (only reachable from a hand-edited journal) stops the
/// upward walk instead of hanging.
pub fn build_lineage(records: &[ExperimentRecord], id: &str) -> Option<Lineage> {
    let by_id: HashMap<&str, &ExperimentRecord> =
        records.iter().map(|r| (r.id.as_str(), r)).collect();
    let focus = *by_id.get(id)?;

    let mut ancestors = Vec::new();
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::from([id]);
    let mut cursor = focus.parent.as_deref();
    while let Some(parent_id) = cursor {
        if !seen.insert(parent_id) {
            break;
        }
        let Some(parent) = by_id.get(parent_id) else {
            break;
        };
        ancestors.push((*parent).clone());
        cursor = parent.parent.as_deref();
    }
    ancestors.reverse();

    let mut descendants = Vec::new();
    collect_descendants(records, id, 1, &mut descendants);

    Some(Lineage {
        focus: focus.clone(),
        ancestors,
        descendants,
    })
}

/// Pre-order walk so the flat list reads as an indented tree.
fn collect_descendants(
    records: &[ExperimentRecord],
    parent_id: &str,
    depth: usize,
    out: &mut Vec<LineageNode>,
) {
    // A malformed journal must not recurse forever.
    if depth > 64 {
        return;
    }
    let mut children: Vec<&ExperimentRecord> = records
        .iter()
        .filter(|r| r.parent.as_deref() == Some(parent_id))
        .collect();
    children.sort_by_key(|r| r.timestamp);
    for child in children {
        out.push(LineageNode {
            record: child.clone(),
            depth,
        });
        collect_descendants(records, &child.id, depth + 1, out);
    }
}

/// In-memory knowledge base implementation.
///
/// The reference backend and the storage half of [`FileKnowledgeBase`].
///
/// [`FileKnowledgeBase`]: crate::FileKnowledgeBase
#[derive(Default)]
pub struct MemoryKnowledgeBase {
    experiments: Vec<ExperimentRecord>,
    index: HashMap<String, usize>,
}

impl MemoryKnowledgeBase {
    /// An empty base.
    pub fn new() -> Self {
        Self::default()
    }

    /// Borrowed view, for callers inside this crate that would rather
    /// not clone (the trait itself hands out owned records on purpose).
    pub fn records(&self) -> &[ExperimentRecord] {
        &self.experiments
    }
}

impl KnowledgeBase for MemoryKnowledgeBase {
    fn record(&mut self, experiment: ExperimentRecord) -> Result<()> {
        // Last write wins on the id index; the log keeps both.
        self.index
            .insert(experiment.id.clone(), self.experiments.len());
        self.experiments.push(experiment);
        Ok(())
    }

    fn all(&self) -> Result<Vec<ExperimentRecord>> {
        Ok(self.experiments.clone())
    }

    fn len(&self) -> usize {
        self.experiments.len()
    }

    /// Indexed, unlike the linear default.
    fn get(&self, id: &str) -> Result<Option<ExperimentRecord>> {
        Ok(self.index.get(id).map(|&idx| self.experiments[idx].clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn make_experiment(id: &str, line: &str, metric_val: f64) -> ExperimentRecord {
        let mut metrics = BTreeMap::new();
        metrics.insert("f1".to_string(), metric_val);
        ExperimentRecord::new(id, format!("Experiment {id}"))
            .with_research_line(line)
            .with_metrics(metrics)
            .with_pipeline("Pipeline([Scaler, Classifier])")
    }

    #[test]
    fn record_and_get() {
        let mut kb = MemoryKnowledgeBase::new();
        kb.record(make_experiment("exp_001", "line_a", 0.85))
            .unwrap();

        assert_eq!(kb.len(), 1);
        assert_eq!(kb.get("exp_001").unwrap().unwrap().id, "exp_001");
        assert_eq!(kb.all().unwrap().len(), 1);
        assert_eq!(kb.refresh().unwrap(), 0, "an in-memory base has no backing");
    }

    #[test]
    fn get_missing_returns_none() {
        let kb = MemoryKnowledgeBase::new();
        assert!(kb.get("nonexistent").unwrap().is_none());
    }

    #[test]
    fn search_by_name() {
        let mut kb = MemoryKnowledgeBase::new();
        kb.record(make_experiment("exp_001", "line_a", 0.8))
            .unwrap();
        kb.record(ExperimentRecord::new("exp_002", "SVM experiment").with_research_line("line_b"))
            .unwrap();

        let results = kb.search("SVM", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "exp_002");
    }

    #[test]
    fn search_by_tag() {
        let mut kb = MemoryKnowledgeBase::new();
        kb.record(
            make_experiment("exp_001", "line_a", 0.8)
                .with_tags(vec!["normalization".into(), "time-series".into()]),
        )
        .unwrap();

        assert_eq!(kb.search("time-series", 10).unwrap().len(), 1);
    }

    #[test]
    fn search_reaches_the_conclusion_headline() {
        let mut kb = MemoryKnowledgeBase::new();
        let mut rec = make_experiment("e1", "line_a", 0.8);
        rec.conclusion = Some(somatize_core::tracking::summary::RunConclusion {
            headline: "completed in 2m · flags: DEAD_CHANNELS".into(),
            ..Default::default()
        });
        kb.record(rec).unwrap();
        assert_eq!(kb.search("dead_channels", 10).unwrap().len(), 1);
    }

    #[test]
    fn experiments_in_line_ordered() {
        let mut kb = MemoryKnowledgeBase::new();
        kb.record(make_experiment("exp_003", "line_a", 0.9))
            .unwrap();
        kb.record(make_experiment("exp_001", "line_a", 0.7))
            .unwrap();
        kb.record(make_experiment("exp_002", "line_b", 0.8))
            .unwrap();

        assert_eq!(kb.experiments_in_line("line_a").unwrap().len(), 2);
    }

    #[test]
    fn research_lines_detected() {
        let mut kb = MemoryKnowledgeBase::new();
        for (i, v) in [0.7, 0.8, 0.85].iter().enumerate() {
            kb.record(make_experiment(&format!("e{i}"), "rocket_znorm", *v))
                .unwrap();
        }
        kb.record(make_experiment("e4", "inception_minmax", 0.6))
            .unwrap();

        let lines = kb.research_lines().unwrap();
        assert_eq!(lines.len(), 2);
        let rocket = lines.iter().find(|l| l.name == "rocket_znorm").unwrap();
        assert_eq!(rocket.experiments.len(), 3);
        assert_eq!(rocket.trend, Trend::Improving);
    }

    #[test]
    fn trajectory_returns_metric_values() {
        let mut kb = MemoryKnowledgeBase::new();
        kb.record(make_experiment("e1", "line_a", 0.7)).unwrap();
        kb.record(make_experiment("e2", "line_a", 0.8)).unwrap();
        kb.record(make_experiment("e3", "line_a", 0.85)).unwrap();

        let traj = kb.trajectory("line_a", "f1").unwrap();
        assert_eq!(traj.len(), 3);
        assert!((traj[0].1 - 0.7).abs() < 0.001);
        assert!((traj[2].1 - 0.85).abs() < 0.001);
    }

    #[test]
    fn change_points_detected() {
        let mut kb = MemoryKnowledgeBase::new();
        for (id, v) in [("e1", 0.50), ("e2", 0.51), ("e3", 0.80), ("e4", 0.82)] {
            kb.record(make_experiment(id, "line_a", v)).unwrap();
        }

        let points = kb.change_points("line_a", "f1", 0.1).unwrap();
        assert_eq!(points.len(), 1);
        assert_eq!(points[0].experiment_id, "e3");
        assert!((points[0].value_before - 0.51).abs() < 0.001);
        assert!((points[0].value_after - 0.80).abs() < 0.001);
    }

    #[test]
    fn change_points_skip_experiments_missing_the_metric() {
        let mut kb = MemoryKnowledgeBase::new();
        kb.record(make_experiment("e1", "line_a", 0.5)).unwrap();
        // No metrics at all: it must not be treated as a zero.
        kb.record(ExperimentRecord::new("e2", "gap").with_research_line("line_a"))
            .unwrap();
        kb.record(make_experiment("e3", "line_a", 0.9)).unwrap();

        let points = kb.change_points("line_a", "f1", 0.1).unwrap();
        assert_eq!(points.len(), 1);
        assert_eq!(points[0].experiment_id, "e3");
    }

    #[test]
    fn children_found() {
        let mut kb = MemoryKnowledgeBase::new();
        kb.record(make_experiment("parent", "line_a", 0.7)).unwrap();
        kb.record(make_experiment("child_1", "line_a", 0.8).with_parent("parent"))
            .unwrap();
        kb.record(make_experiment("child_2", "line_a", 0.75).with_parent("parent"))
            .unwrap();
        kb.record(make_experiment("unrelated", "line_b", 0.6))
            .unwrap();

        assert_eq!(kb.children("parent").unwrap().len(), 2);
    }

    #[test]
    fn lineage_walks_up_and_down() {
        let mut kb = MemoryKnowledgeBase::new();
        kb.record(make_experiment("root", "l", 0.5)).unwrap();
        kb.record(make_experiment("mid", "l", 0.6).with_parent("root"))
            .unwrap();
        kb.record(make_experiment("leaf_a", "l", 0.7).with_parent("mid"))
            .unwrap();
        kb.record(make_experiment("leaf_b", "l", 0.8).with_parent("mid"))
            .unwrap();
        kb.record(make_experiment("elsewhere", "l", 0.9)).unwrap();

        let lineage = kb.lineage("mid").unwrap().unwrap();
        assert_eq!(lineage.focus.id, "mid");
        assert_eq!(
            lineage.ancestors.iter().map(|a| &a.id).collect::<Vec<_>>(),
            vec!["root"]
        );
        assert_eq!(lineage.root().id, "root");
        assert_eq!(
            lineage
                .descendants
                .iter()
                .map(|d| (d.record.id.as_str(), d.depth))
                .collect::<Vec<_>>(),
            vec![("leaf_a", 1), ("leaf_b", 1)]
        );

        // From the root, the whole tree in pre-order.
        let from_root = kb.lineage("root").unwrap().unwrap();
        assert!(from_root.ancestors.is_empty());
        assert_eq!(from_root.root().id, "root");
        assert_eq!(
            from_root
                .descendants
                .iter()
                .map(|d| (d.record.id.as_str(), d.depth))
                .collect::<Vec<_>>(),
            vec![("mid", 1), ("leaf_a", 2), ("leaf_b", 2)]
        );

        assert!(kb.lineage("does-not-exist").unwrap().is_none());
    }

    #[test]
    fn lineage_survives_a_hand_edited_parent_cycle() {
        let records = vec![
            make_experiment("a", "l", 0.1).with_parent("b"),
            make_experiment("b", "l", 0.2).with_parent("a"),
        ];
        let lineage = build_lineage(&records, "a").unwrap();
        // Walks up once, then refuses to loop.
        assert_eq!(lineage.ancestors.len(), 1);
        assert_eq!(lineage.ancestors[0].id, "b");
    }

    #[test]
    fn promising_lines() {
        let mut kb = MemoryKnowledgeBase::new();
        for (i, v) in [0.7, 0.8, 0.9].iter().enumerate() {
            kb.record(make_experiment(&format!("g{i}"), "good_line", *v))
                .unwrap();
        }
        for (i, v) in [0.9, 0.8, 0.7].iter().enumerate() {
            kb.record(make_experiment(&format!("b{i}"), "bad_line", *v))
                .unwrap();
        }

        let promising = kb.promising_lines("f1").unwrap();
        assert!(promising.iter().any(|l| l.name == "good_line"));
    }

    #[test]
    fn trend_detection() {
        let mut kb = MemoryKnowledgeBase::new();
        for (line, values) in [
            ("improving", [0.5, 0.7, 0.9]),
            ("declining", [0.9, 0.7, 0.5]),
            ("plateau", [0.8, 0.8, 0.8]),
        ] {
            for (i, v) in values.iter().enumerate() {
                kb.record(make_experiment(&format!("{line}{i}"), line, *v))
                    .unwrap();
            }
        }

        let lines = kb.research_lines().unwrap();
        let of = |name: &str| lines.iter().find(|l| l.name == name).unwrap().trend;
        assert_eq!(of("improving"), Trend::Improving);
        assert_eq!(of("declining"), Trend::Declining);
        assert_eq!(of("plateau"), Trend::Plateaued);
    }

    #[test]
    fn empty_kb() {
        let kb = MemoryKnowledgeBase::new();
        assert!(kb.is_empty());
        assert!(kb.research_lines().unwrap().is_empty());
        assert!(kb.search("anything", 10).unwrap().is_empty());
        assert!(kb.all().unwrap().is_empty());
    }

    #[test]
    fn serde_roundtrip() {
        let exp = make_experiment("e1", "line_a", 0.85)
            .with_hypothesis("Z-norm improves accuracy")
            .with_notes("First attempt with z-normalization")
            .with_tags(vec!["normalization".into()]);

        let json = serde_json::to_string(&exp).unwrap();
        let deserialized: ExperimentRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, "e1");
        assert_eq!(deserialized.hypothesis.unwrap(), "Z-norm improves accuracy");
    }
}
