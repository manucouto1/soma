//! Data types for experiment tracking: records, research lines, trends.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

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
