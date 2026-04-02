//! Research planner — generates actions based on experiment history.

use crate::action::Action;
use somatize_core::error::Result;
use somatize_memory::ResearchLine;
use std::collections::HashMap;

/// Trait for generating research actions.
///
/// In a full system, this would be backed by an LLM that reasons about
/// what experiments to run. For now, implementors provide domain-specific
/// logic for generating experiments.
pub trait ResearchPlan: Send + Sync {
    /// Generate the first action for a new research objective.
    fn initial_action(&self, objective: &str) -> Result<Action>;

    /// Generate an action to refine an existing promising line.
    fn refine_action(&self, line_name: &str, objective: &str) -> Result<Action>;

    /// Generate an action to explore a new direction.
    fn explore_action(&self, objective: &str, existing_lines: &[ResearchLine]) -> Result<Action>;
}

/// A simple rule-based research planner for testing.
///
/// Generates experiments by varying a single parameter at a time.
pub struct SimpleResearchPlan {
    /// Base pipeline config.
    pub base_config: HashMap<String, serde_json::Value>,
    /// Parameter to vary.
    pub vary_param: String,
    /// Values to try.
    pub values: Vec<serde_json::Value>,
    /// Counter for generating unique names.
    counter: std::sync::atomic::AtomicUsize,
}

impl SimpleResearchPlan {
    pub fn new(
        base_config: HashMap<String, serde_json::Value>,
        vary_param: impl Into<String>,
        values: Vec<serde_json::Value>,
    ) -> Self {
        Self {
            base_config,
            vary_param: vary_param.into(),
            values,
            counter: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn next_id(&self) -> usize {
        self.counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }
}

impl ResearchPlan for SimpleResearchPlan {
    fn initial_action(&self, objective: &str) -> Result<Action> {
        let idx = self.next_id();
        let value = self
            .values
            .get(idx % self.values.len())
            .cloned()
            .unwrap_or(serde_json::Value::Null);

        let mut config = self.base_config.clone();
        config.insert(self.vary_param.clone(), value.clone());

        Ok(Action::RunExperiment {
            name: format!("exp_{idx:04}"),
            research_line: format!("{}_exploration", self.vary_param),
            hypothesis: format!("Testing {objective} with {} = {value}", self.vary_param),
            pipeline_config: config,
            parent: None,
        })
    }

    fn refine_action(&self, line_name: &str, _objective: &str) -> Result<Action> {
        let idx = self.next_id();
        let value = self
            .values
            .get(idx % self.values.len())
            .cloned()
            .unwrap_or(serde_json::Value::Null);

        let mut config = self.base_config.clone();
        config.insert(self.vary_param.clone(), value.clone());

        Ok(Action::RunExperiment {
            name: format!("exp_{idx:04}"),
            research_line: line_name.to_string(),
            hypothesis: format!("Refining {line_name}: {} = {value}", self.vary_param),
            pipeline_config: config,
            parent: None,
        })
    }

    fn explore_action(&self, objective: &str, _lines: &[ResearchLine]) -> Result<Action> {
        self.initial_action(objective)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn simple_plan_generates_actions() {
        let plan = SimpleResearchPlan::new(
            HashMap::from([("model".into(), json!("svm"))]),
            "C",
            vec![json!(0.1), json!(1.0), json!(10.0)],
        );

        let action = plan.initial_action("classify iris").unwrap();
        if let Action::RunExperiment {
            name,
            pipeline_config,
            hypothesis,
            ..
        } = action
        {
            assert!(name.starts_with("exp_"));
            assert!(pipeline_config.contains_key("C"));
            assert!(pipeline_config.contains_key("model"));
            assert!(hypothesis.contains("classify iris"));
        } else {
            panic!("expected RunExperiment");
        }
    }

    #[test]
    fn simple_plan_cycles_values() {
        let plan = SimpleResearchPlan::new(HashMap::new(), "x", vec![json!(1), json!(2), json!(3)]);

        let a1 = plan.initial_action("test").unwrap();
        let _a2 = plan.initial_action("test").unwrap();
        let _a3 = plan.initial_action("test").unwrap();
        let a4 = plan.initial_action("test").unwrap(); // wraps around

        // All should be RunExperiment with different values
        if let (
            Action::RunExperiment {
                pipeline_config: c1,
                ..
            },
            Action::RunExperiment {
                pipeline_config: c4,
                ..
            },
        ) = (a1, a4)
        {
            // a4 wraps to same value as a1
            assert_eq!(c1["x"], c4["x"]);
        }
    }
}
