//! KnowledgeBase trait and in-memory implementation.
//!
//! Stores experiments, detects research lines, computes trajectories
//! and change points, and identifies promising directions.

use crate::record::{ChangePoint, ExperimentRecord, ResearchLine, Trend};
use somatize_core::error::Result;
use std::collections::HashMap;

/// Trait for knowledge base backends.
///
/// Implementations may use in-memory storage, SQLite, or ChronosVector
/// for temporal-aware experiment tracking.
pub trait KnowledgeBase: Send + Sync {
    /// Store a new experiment record.
    fn record(&mut self, experiment: ExperimentRecord) -> Result<()>;

    /// Get an experiment by ID.
    fn get(&self, id: &str) -> Result<Option<&ExperimentRecord>>;

    /// Search experiments by text query (matches name, hypothesis, notes, tags).
    fn search(&self, query: &str, max_results: usize) -> Result<Vec<&ExperimentRecord>>;

    /// List experiments in a research line, ordered chronologically.
    fn experiments_in_line(&self, line: &str) -> Result<Vec<&ExperimentRecord>>;

    /// Get all research lines with trend analysis.
    fn research_lines(&self) -> Result<Vec<ResearchLine>>;

    /// Find promising research lines (improving trend, high metric values).
    fn promising_lines(&self, metric: &str) -> Result<Vec<ResearchLine>>;

    /// Get the metric trajectory for a research line.
    fn trajectory(&self, line: &str, metric: &str) -> Result<Vec<(String, f64)>>;

    /// Detect change points where metrics shifted significantly.
    fn change_points(&self, line: &str, metric: &str, threshold: f64) -> Result<Vec<ChangePoint>>;

    /// Get the experiment tree (parent-child relationships).
    fn children(&self, experiment_id: &str) -> Result<Vec<&ExperimentRecord>>;

    /// Total number of experiments.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// In-memory knowledge base implementation.
///
/// Suitable for development and testing. For production, use the
/// ChronosVector-backed implementation for temporal analysis.
pub struct MemoryKnowledgeBase {
    experiments: Vec<ExperimentRecord>,
    index: HashMap<String, usize>,
}

impl MemoryKnowledgeBase {
    pub fn new() -> Self {
        Self {
            experiments: Vec::new(),
            index: HashMap::new(),
        }
    }
}

impl Default for MemoryKnowledgeBase {
    fn default() -> Self {
        Self::new()
    }
}

impl KnowledgeBase for MemoryKnowledgeBase {
    fn record(&mut self, experiment: ExperimentRecord) -> Result<()> {
        let idx = self.experiments.len();
        self.index.insert(experiment.id.clone(), idx);
        self.experiments.push(experiment);
        Ok(())
    }

    fn get(&self, id: &str) -> Result<Option<&ExperimentRecord>> {
        Ok(self.index.get(id).map(|&idx| &self.experiments[idx]))
    }

    fn search(&self, query: &str, max_results: usize) -> Result<Vec<&ExperimentRecord>> {
        let q = query.to_lowercase();
        let results: Vec<&ExperimentRecord> = self
            .experiments
            .iter()
            .filter(|e| {
                e.name.to_lowercase().contains(&q)
                    || e.hypothesis
                        .as_deref()
                        .unwrap_or("")
                        .to_lowercase()
                        .contains(&q)
                    || e.notes.as_deref().unwrap_or("").to_lowercase().contains(&q)
                    || e.tags.iter().any(|t| t.to_lowercase().contains(&q))
                    || e.pipeline_summary.to_lowercase().contains(&q)
            })
            .take(max_results)
            .collect();
        Ok(results)
    }

    fn experiments_in_line(&self, line: &str) -> Result<Vec<&ExperimentRecord>> {
        let mut exps: Vec<&ExperimentRecord> = self
            .experiments
            .iter()
            .filter(|e| e.research_line.as_deref() == Some(line))
            .collect();
        exps.sort_by_key(|e| e.timestamp);
        Ok(exps)
    }

    fn research_lines(&self) -> Result<Vec<ResearchLine>> {
        let mut lines: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, exp) in self.experiments.iter().enumerate() {
            if let Some(ref line) = exp.research_line {
                lines.entry(line.clone()).or_default().push(i);
            }
        }

        let mut result = Vec::new();
        for (name, indices) in &lines {
            let experiments: Vec<String> = indices
                .iter()
                .map(|&i| self.experiments[i].id.clone())
                .collect();

            // Find best metric across any metric
            let mut best_value = None;
            let mut best_name = None;
            for &i in indices {
                for (k, &v) in &self.experiments[i].metrics {
                    if best_value.is_none() || v > best_value.unwrap() {
                        best_value = Some(v);
                        best_name = Some(k.clone());
                    }
                }
            }

            let trend = self.compute_trend(indices);

            result.push(ResearchLine {
                name: name.clone(),
                experiments,
                trend,
                best_metric_value: best_value,
                best_metric_name: best_name,
            });
        }

        result.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(result)
    }

    fn promising_lines(&self, metric: &str) -> Result<Vec<ResearchLine>> {
        let all_lines = self.research_lines()?;
        let promising: Vec<ResearchLine> = all_lines
            .into_iter()
            .filter(|l| {
                l.trend == Trend::Improving || l.best_metric_name.as_deref() == Some(metric)
            })
            .collect();
        Ok(promising)
    }

    fn trajectory(&self, line: &str, metric: &str) -> Result<Vec<(String, f64)>> {
        let exps = self.experiments_in_line(line)?;
        let traj: Vec<(String, f64)> = exps
            .iter()
            .filter_map(|e| e.metrics.get(metric).map(|&v| (e.id.clone(), v)))
            .collect();
        Ok(traj)
    }

    fn change_points(&self, line: &str, metric: &str, threshold: f64) -> Result<Vec<ChangePoint>> {
        let traj = self.trajectory(line, metric)?;
        let mut points = Vec::new();

        for window in traj.windows(2) {
            let (ref _id_before, val_before) = window[0];
            let (ref id_after, val_after) = window[1];
            let delta = (val_after - val_before).abs();

            if delta >= threshold {
                let exp = self.get(id_after)?.unwrap();
                points.push(ChangePoint {
                    experiment_id: id_after.clone(),
                    timestamp: exp.timestamp,
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

    fn children(&self, experiment_id: &str) -> Result<Vec<&ExperimentRecord>> {
        let children: Vec<&ExperimentRecord> = self
            .experiments
            .iter()
            .filter(|e| e.parent.as_deref() == Some(experiment_id))
            .collect();
        Ok(children)
    }

    fn len(&self) -> usize {
        self.experiments.len()
    }
}

impl MemoryKnowledgeBase {
    fn compute_trend(&self, indices: &[usize]) -> Trend {
        if indices.len() < 2 {
            return Trend::Unknown;
        }

        // Look at the most common metric across experiments
        let mut all_values: Vec<f64> = Vec::new();
        let first_metrics = &self.experiments[indices[0]].metrics;
        if let Some(first_key) = first_metrics.keys().next() {
            for &i in indices {
                if let Some(&v) = self.experiments[i].metrics.get(first_key) {
                    all_values.push(v);
                }
            }
        }

        if all_values.len() < 2 {
            return Trend::Unknown;
        }

        // Simple trend: compare last 3 values
        let n = all_values.len();
        let recent = &all_values[n.saturating_sub(3)..];
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_experiment(id: &str, line: &str, metric_val: f64) -> ExperimentRecord {
        let mut metrics = HashMap::new();
        metrics.insert("f1".to_string(), metric_val);
        ExperimentRecord::new(id, format!("Experiment {id}"))
            .with_research_line(line)
            .with_metrics(metrics)
            .with_pipeline("Pipeline([Scaler, Classifier])")
    }

    #[test]
    fn record_and_get() {
        let mut kb = MemoryKnowledgeBase::new();
        let exp = make_experiment("exp_001", "line_a", 0.85);
        kb.record(exp).unwrap();

        assert_eq!(kb.len(), 1);
        let retrieved = kb.get("exp_001").unwrap().unwrap();
        assert_eq!(retrieved.id, "exp_001");
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

        let results = kb.search("time-series", 10).unwrap();
        assert_eq!(results.len(), 1);
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

        let line_a = kb.experiments_in_line("line_a").unwrap();
        assert_eq!(line_a.len(), 2);
    }

    #[test]
    fn research_lines_detected() {
        let mut kb = MemoryKnowledgeBase::new();
        kb.record(make_experiment("e1", "rocket_znorm", 0.7))
            .unwrap();
        kb.record(make_experiment("e2", "rocket_znorm", 0.8))
            .unwrap();
        kb.record(make_experiment("e3", "rocket_znorm", 0.85))
            .unwrap();
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
        kb.record(make_experiment("e1", "line_a", 0.50)).unwrap();
        kb.record(make_experiment("e2", "line_a", 0.51)).unwrap();
        kb.record(make_experiment("e3", "line_a", 0.80)).unwrap(); // big jump
        kb.record(make_experiment("e4", "line_a", 0.82)).unwrap();

        let points = kb.change_points("line_a", "f1", 0.1).unwrap();
        assert_eq!(points.len(), 1);
        assert_eq!(points[0].experiment_id, "e3");
        assert!((points[0].value_before - 0.51).abs() < 0.001);
        assert!((points[0].value_after - 0.80).abs() < 0.001);
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

        let kids = kb.children("parent").unwrap();
        assert_eq!(kids.len(), 2);
    }

    #[test]
    fn promising_lines() {
        let mut kb = MemoryKnowledgeBase::new();
        kb.record(make_experiment("e1", "good_line", 0.7)).unwrap();
        kb.record(make_experiment("e2", "good_line", 0.8)).unwrap();
        kb.record(make_experiment("e3", "good_line", 0.9)).unwrap();
        kb.record(make_experiment("e4", "bad_line", 0.9)).unwrap();
        kb.record(make_experiment("e5", "bad_line", 0.8)).unwrap();
        kb.record(make_experiment("e6", "bad_line", 0.7)).unwrap();

        let promising = kb.promising_lines("f1").unwrap();
        // good_line is improving, bad_line is declining
        assert!(promising.iter().any(|l| l.name == "good_line"));
    }

    #[test]
    fn trend_detection() {
        let mut kb = MemoryKnowledgeBase::new();

        // Improving
        kb.record(make_experiment("i1", "improving", 0.5)).unwrap();
        kb.record(make_experiment("i2", "improving", 0.7)).unwrap();
        kb.record(make_experiment("i3", "improving", 0.9)).unwrap();

        // Declining
        kb.record(make_experiment("d1", "declining", 0.9)).unwrap();
        kb.record(make_experiment("d2", "declining", 0.7)).unwrap();
        kb.record(make_experiment("d3", "declining", 0.5)).unwrap();

        // Plateaued
        kb.record(make_experiment("p1", "plateau", 0.8)).unwrap();
        kb.record(make_experiment("p2", "plateau", 0.8)).unwrap();
        kb.record(make_experiment("p3", "plateau", 0.8)).unwrap();

        let lines = kb.research_lines().unwrap();
        let improving = lines.iter().find(|l| l.name == "improving").unwrap();
        let declining = lines.iter().find(|l| l.name == "declining").unwrap();
        let plateau = lines.iter().find(|l| l.name == "plateau").unwrap();

        assert_eq!(improving.trend, Trend::Improving);
        assert_eq!(declining.trend, Trend::Declining);
        assert_eq!(plateau.trend, Trend::Plateaued);
    }

    #[test]
    fn empty_kb() {
        let kb = MemoryKnowledgeBase::new();
        assert!(kb.is_empty());
        assert!(kb.research_lines().unwrap().is_empty());
        assert!(kb.search("anything", 10).unwrap().is_empty());
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
