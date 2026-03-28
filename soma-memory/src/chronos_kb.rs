//! ChronosVector-backed KnowledgeBase implementation.
//!
//! Uses TemporalHnsw for semantic + temporal experiment search.
//! Experiments are stored as temporal points where:
//! - entity_id = hash of experiment ID
//! - timestamp = experiment timestamp (microseconds)
//! - vector = embedding of experiment description + params + metrics
//! - metadata = serialized ExperimentRecord fields

use crate::knowledge_base::KnowledgeBase;
use crate::record::{ChangePoint, ExperimentRecord, ResearchLine, Trend};
use chronos_vector::core::TemporalFilter;
use chronos_vector::index::hnsw::HnswConfig;
use chronos_vector::index::hnsw::temporal::TemporalHnsw;
use chronos_vector::index::metrics::L2Distance;
use soma_core::error::Result;
use std::collections::HashMap;

/// KnowledgeBase backed by ChronosVector's TemporalHnsw.
///
/// Provides semantic search (find similar experiments),
/// temporal trajectories (how experiments evolved),
/// and change point detection (where breakthroughs happened).
pub struct ChronosKnowledgeBase {
    /// The temporal index for semantic + temporal search.
    index: TemporalHnsw<L2Distance>,
    /// Full experiment records (for retrieval after index lookup).
    records: Vec<ExperimentRecord>,
    /// Map from experiment ID to record index.
    id_to_idx: HashMap<String, usize>,
    /// Map from experiment ID to entity_id in the index.
    id_to_entity: HashMap<String, u64>,
    /// Embedding dimension.
    dim: usize,
    /// Next entity ID.
    next_entity: u64,
}

impl ChronosKnowledgeBase {
    /// Create a new ChronosVector-backed knowledge base.
    ///
    /// `dim` is the embedding dimension. Experiments will be encoded
    /// as vectors of this size using a simple feature hashing scheme.
    pub fn new(dim: usize) -> Self {
        let config = HnswConfig {
            m: 16,
            ef_construction: 100,
            ef_search: 50,
            ..HnswConfig::default()
        };
        Self {
            index: TemporalHnsw::new(config, L2Distance),
            records: Vec::new(),
            id_to_idx: HashMap::new(),
            id_to_entity: HashMap::new(),
            dim,
            next_entity: 1,
        }
    }

    /// Encode an experiment as a fixed-size vector.
    ///
    /// Uses a simple feature hashing scheme:
    /// - First half: hash of hypothesis + pipeline + tags
    /// - Second half: normalized metric values
    fn encode(&self, record: &ExperimentRecord) -> Vec<f32> {
        let mut vec = vec![0.0f32; self.dim];
        let half = self.dim / 2;

        // Text features: hash words into first half
        let text = format!(
            "{} {} {} {}",
            record.hypothesis.as_deref().unwrap_or(""),
            record.pipeline_summary,
            record.tags.join(" "),
            record.notes.as_deref().unwrap_or(""),
        );
        for word in text.split_whitespace() {
            let h = simple_hash(word) as usize % half;
            vec[h] += 1.0;
        }

        // Metric features: place in second half
        for (i, (_, &val)) in record.metrics.iter().enumerate() {
            let idx = half + (i % half);
            vec[idx] = val as f32;
        }

        // Param features: hash into first half with offset
        for (key, val) in &record.params {
            let h = simple_hash(key) as usize % half;
            let v = match val {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.0) as f32,
                serde_json::Value::String(s) => simple_hash(s) as f32 / u64::MAX as f32,
                _ => 0.0,
            };
            vec[h] += v;
        }

        // L2 normalize
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut vec {
                *v /= norm;
            }
        }

        vec
    }

    /// Semantic search: find experiments similar to a query string.
    pub fn semantic_search(
        &self,
        query: &str,
        k: usize,
        alpha: f32,
    ) -> Result<Vec<&ExperimentRecord>> {
        // Encode query as vector
        let mut query_vec = vec![0.0f32; self.dim];
        let half = self.dim / 2;
        for word in query.split_whitespace() {
            let h = simple_hash(word) as usize % half;
            query_vec[h] += 1.0;
        }
        let norm: f32 = query_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut query_vec {
                *v /= norm;
            }
        }

        let results = self.index.search(
            &query_vec,
            k,
            TemporalFilter::All,
            alpha,
            0, // no temporal bias
        );

        let experiments: Vec<&ExperimentRecord> = results
            .iter()
            .filter_map(|(node_id, _score)| {
                let entity_id = self.index.entity_id(*node_id);
                self.id_to_entity
                    .iter()
                    .find(|(_, eid)| **eid == entity_id)
                    .and_then(|(id, _)| self.id_to_idx.get(id))
                    .map(|&idx| &self.records[idx])
            })
            .collect();

        Ok(experiments)
    }
}

impl KnowledgeBase for ChronosKnowledgeBase {
    fn record(&mut self, experiment: ExperimentRecord) -> Result<()> {
        let entity_id = self.next_entity;
        self.next_entity += 1;

        let vector = self.encode(&experiment);
        let timestamp = experiment.timestamp.timestamp_micros();

        self.index.insert(entity_id, timestamp, &vector);

        let idx = self.records.len();
        self.id_to_idx.insert(experiment.id.clone(), idx);
        self.id_to_entity.insert(experiment.id.clone(), entity_id);
        self.records.push(experiment);

        Ok(())
    }

    fn get(&self, id: &str) -> Result<Option<&ExperimentRecord>> {
        Ok(self.id_to_idx.get(id).map(|&idx| &self.records[idx]))
    }

    fn search(&self, query: &str, max_results: usize) -> Result<Vec<&ExperimentRecord>> {
        // Use semantic search with pure semantic weight (alpha=1.0)
        if self.records.is_empty() {
            return Ok(Vec::new());
        }
        self.semantic_search(query, max_results, 1.0)
    }

    fn experiments_in_line(&self, line: &str) -> Result<Vec<&ExperimentRecord>> {
        let mut exps: Vec<&ExperimentRecord> = self
            .records
            .iter()
            .filter(|e| e.research_line.as_deref() == Some(line))
            .collect();
        exps.sort_by_key(|e| e.timestamp);
        Ok(exps)
    }

    fn research_lines(&self) -> Result<Vec<ResearchLine>> {
        let mut lines: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, exp) in self.records.iter().enumerate() {
            if let Some(ref line) = exp.research_line {
                lines.entry(line.clone()).or_default().push(i);
            }
        }

        let mut result = Vec::new();
        for (name, indices) in &lines {
            let experiments: Vec<String> = indices
                .iter()
                .map(|&i| self.records[i].id.clone())
                .collect();

            let mut best_value = None;
            let mut best_name = None;
            for &i in indices {
                for (k, &v) in &self.records[i].metrics {
                    if best_value.is_none() || v > best_value.unwrap() {
                        best_value = Some(v);
                        best_name = Some(k.clone());
                    }
                }
            }

            let trend = compute_trend(&self.records, indices);

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
        let all = self.research_lines()?;
        Ok(all
            .into_iter()
            .filter(|l| {
                l.trend == Trend::Improving || l.best_metric_name.as_deref() == Some(metric)
            })
            .collect())
    }

    fn trajectory(&self, line: &str, metric: &str) -> Result<Vec<(String, f64)>> {
        let exps = self.experiments_in_line(line)?;
        Ok(exps
            .iter()
            .filter_map(|e| e.metrics.get(metric).map(|&v| (e.id.clone(), v)))
            .collect())
    }

    fn change_points(&self, line: &str, metric: &str, threshold: f64) -> Result<Vec<ChangePoint>> {
        let traj = self.trajectory(line, metric)?;
        let mut points = Vec::new();
        for window in traj.windows(2) {
            let (_, val_before) = &window[0];
            let (ref id_after, val_after) = window[1];
            let delta = (val_after - val_before).abs();
            if delta >= threshold {
                let exp = self.get(id_after)?.unwrap();
                points.push(ChangePoint {
                    experiment_id: id_after.clone(),
                    timestamp: exp.timestamp,
                    metric_name: metric.to_string(),
                    value_before: *val_before,
                    value_after: val_after,
                    description: format!("{metric} changed from {val_before:.4} to {val_after:.4}"),
                });
            }
        }
        Ok(points)
    }

    fn children(&self, experiment_id: &str) -> Result<Vec<&ExperimentRecord>> {
        Ok(self
            .records
            .iter()
            .filter(|e| e.parent.as_deref() == Some(experiment_id))
            .collect())
    }

    fn len(&self) -> usize {
        self.records.len()
    }
}

fn compute_trend(records: &[ExperimentRecord], indices: &[usize]) -> Trend {
    if indices.len() < 2 {
        return Trend::Unknown;
    }
    let first_metrics = &records[indices[0]].metrics;
    let Some(first_key) = first_metrics.keys().next() else {
        return Trend::Unknown;
    };

    let values: Vec<f64> = indices
        .iter()
        .filter_map(|&i| records[i].metrics.get(first_key).copied())
        .collect();

    if values.len() < 2 {
        return Trend::Unknown;
    }

    let n = values.len();
    let recent = &values[n.saturating_sub(3)..];
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

fn simple_hash(s: &str) -> u64 {
    let mut h: u64 = 5381;
    for b in s.bytes() {
        h = h.wrapping_mul(33).wrapping_add(b as u64);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_exp(id: &str, line: &str, f1: f64) -> ExperimentRecord {
        let mut metrics = HashMap::new();
        metrics.insert("f1".to_string(), f1);
        ExperimentRecord::new(id, format!("Experiment {id}"))
            .with_research_line(line)
            .with_metrics(metrics)
            .with_pipeline("Pipeline([Scaler, SVM])")
            .with_hypothesis(format!("Test hypothesis for {id}"))
    }

    #[test]
    fn chronos_record_and_get() {
        let mut kb = ChronosKnowledgeBase::new(32);
        kb.record(make_exp("e1", "line_a", 0.85)).unwrap();

        assert_eq!(kb.len(), 1);
        let exp = kb.get("e1").unwrap().unwrap();
        assert_eq!(exp.id, "e1");
    }

    #[test]
    fn chronos_semantic_search() {
        let mut kb = ChronosKnowledgeBase::new(32);
        kb.record(
            make_exp("e1", "line_a", 0.8)
                .with_hypothesis("SVM classification with z-normalization")
                .with_tags(vec!["svm".into(), "normalization".into()]),
        )
        .unwrap();
        kb.record(
            make_exp("e2", "line_b", 0.7)
                .with_hypothesis("Random forest on raw features")
                .with_tags(vec!["rf".into()]),
        )
        .unwrap();
        kb.record(
            make_exp("e3", "line_a", 0.9)
                .with_hypothesis("SVM with robust scaling")
                .with_tags(vec!["svm".into(), "scaling".into()]),
        )
        .unwrap();

        // Search for SVM-related experiments
        let results = kb.search("SVM normalization", 10).unwrap();
        assert!(!results.is_empty());
        // SVM experiments should rank higher
        assert!(
            results.iter().any(|r| r.id == "e1" || r.id == "e3"),
            "SVM experiments should appear in results"
        );
    }

    #[test]
    fn chronos_research_lines() {
        let mut kb = ChronosKnowledgeBase::new(16);
        kb.record(make_exp("e1", "rocket", 0.7)).unwrap();
        kb.record(make_exp("e2", "rocket", 0.8)).unwrap();
        kb.record(make_exp("e3", "rocket", 0.9)).unwrap();
        kb.record(make_exp("e4", "inception", 0.6)).unwrap();

        let lines = kb.research_lines().unwrap();
        assert_eq!(lines.len(), 2);

        let rocket = lines.iter().find(|l| l.name == "rocket").unwrap();
        assert_eq!(rocket.trend, Trend::Improving);
        assert_eq!(rocket.experiments.len(), 3);
    }

    #[test]
    fn chronos_trajectory() {
        let mut kb = ChronosKnowledgeBase::new(16);
        kb.record(make_exp("e1", "line_a", 0.5)).unwrap();
        kb.record(make_exp("e2", "line_a", 0.7)).unwrap();
        kb.record(make_exp("e3", "line_a", 0.9)).unwrap();

        let traj = kb.trajectory("line_a", "f1").unwrap();
        assert_eq!(traj.len(), 3);
        assert!((traj[0].1 - 0.5).abs() < 0.001);
        assert!((traj[2].1 - 0.9).abs() < 0.001);
    }

    #[test]
    fn chronos_change_points() {
        let mut kb = ChronosKnowledgeBase::new(16);
        kb.record(make_exp("e1", "line_a", 0.50)).unwrap();
        kb.record(make_exp("e2", "line_a", 0.52)).unwrap();
        kb.record(make_exp("e3", "line_a", 0.85)).unwrap(); // jump

        let cps = kb.change_points("line_a", "f1", 0.1).unwrap();
        assert_eq!(cps.len(), 1);
        assert_eq!(cps[0].experiment_id, "e3");
    }

    #[test]
    fn chronos_children() {
        let mut kb = ChronosKnowledgeBase::new(16);
        kb.record(make_exp("parent", "line_a", 0.7)).unwrap();
        kb.record(make_exp("child", "line_a", 0.8).with_parent("parent"))
            .unwrap();

        let kids = kb.children("parent").unwrap();
        assert_eq!(kids.len(), 1);
        assert_eq!(kids[0].id, "child");
    }

    #[test]
    fn chronos_empty() {
        let kb = ChronosKnowledgeBase::new(16);
        assert!(kb.is_empty());
        assert!(kb.search("anything", 10).unwrap().is_empty());
    }

    #[test]
    fn chronos_promising_lines() {
        let mut kb = ChronosKnowledgeBase::new(16);
        kb.record(make_exp("e1", "good", 0.5)).unwrap();
        kb.record(make_exp("e2", "good", 0.7)).unwrap();
        kb.record(make_exp("e3", "good", 0.9)).unwrap();
        kb.record(make_exp("e4", "bad", 0.9)).unwrap();
        kb.record(make_exp("e5", "bad", 0.7)).unwrap();
        kb.record(make_exp("e6", "bad", 0.5)).unwrap();

        let promising = kb.promising_lines("f1").unwrap();
        assert!(promising.iter().any(|l| l.name == "good"));
    }
}
