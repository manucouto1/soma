use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// An action the agent decides to take.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Action {
    /// Run an experiment with specific parameters.
    RunExperiment {
        /// Name for this experiment.
        name: String,
        /// Which research line this belongs to.
        research_line: String,
        /// Hypothesis being tested.
        hypothesis: String,
        /// Pipeline configuration (filter names + params).
        pipeline_config: HashMap<String, serde_json::Value>,
        /// Parent experiment (if refining).
        parent: Option<String>,
    },

    /// Conclude the research (no more iterations).
    Conclude {
        reason: String,
    },
}
