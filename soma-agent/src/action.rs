//! What a research agent can decide to do.
//!
//! Two things, and deliberately no more. An agent that can run an experiment
//! and an agent that can stop covers the whole loop; every other verb people
//! reach for ("analyze", "summarize", "compare") is the model thinking, and
//! thinking does not need a protocol.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// An action the agent decided to take.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Action {
    /// Run an experiment with specific parameters.
    RunExperiment {
        /// Name for this experiment.
        name: String,
        /// Which research line this belongs to.
        research_line: String,
        /// What this experiment is meant to settle.
        ///
        /// Required, and required to be falsifiable — an experiment run
        /// without one is a number nobody can interpret later, and the pool
        /// exists to be read later.
        hypothesis: String,
        /// Parameters to apply to the pipeline, as `"<node>.<param>"`.
        params: HashMap<String, serde_json::Value>,
        /// The run this refines, when it refines one.
        #[serde(default)]
        parent: Option<String>,
    },

    /// Stop: the objective is met, or nothing left is worth trying.
    Conclude {
        /// What was concluded, in the agent's own words.
        reason: String,
    },
}

impl Action {
    /// The JSON Schema a model is asked to answer in.
    ///
    /// Constraining the reply is what turns "the model suggested something"
    /// into "the agent decided something" — a free-text plan has to be
    /// parsed, and a parser for prose is a source of silent misreadings.
    pub fn response_schema() -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["run_experiment", "conclude"]
                },
                "name": {"type": "string"},
                "research_line": {"type": "string"},
                "hypothesis": {"type": "string"},
                "params": {"type": "object"},
                "parent": {"type": ["string", "null"]},
                "reason": {"type": "string"}
            },
            "required": ["action"]
        })
    }

    /// Whether this ends the loop.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Conclude { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn an_experiment_round_trips() {
        let action: Action = serde_json::from_value(json!({
            "action": "run_experiment",
            "name": "exp_0001",
            "research_line": "regularization",
            "hypothesis": "stronger L2 lifts held-out F1 above 0.8",
            "params": {"classifier.C": 0.1}
        }))
        .unwrap();

        match &action {
            Action::RunExperiment {
                name,
                params,
                parent,
                ..
            } => {
                assert_eq!(name, "exp_0001");
                assert_eq!(params["classifier.C"], json!(0.1));
                assert!(parent.is_none(), "parent is optional");
            }
            other => panic!("{other:?}"),
        }
        assert!(!action.is_terminal());
    }

    #[test]
    fn a_conclusion_is_terminal() {
        let action: Action =
            serde_json::from_value(json!({"action": "conclude", "reason": "plateaued"})).unwrap();
        assert!(action.is_terminal());
    }

    /// An experiment without a hypothesis is a number nobody can read later.
    #[test]
    fn an_experiment_without_a_hypothesis_is_rejected() {
        let result: std::result::Result<Action, _> = serde_json::from_value(json!({
            "action": "run_experiment",
            "name": "exp",
            "research_line": "l",
            "params": {}
        }));
        assert!(result.is_err());
    }
}
