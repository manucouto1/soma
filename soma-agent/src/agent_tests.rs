#[cfg(test)]
mod tests {
    use crate::action::Action;
    use crate::agent::Agent;
    use crate::planner::SimpleResearchPlan;
    #[allow(unused_imports)]
    use somatize_memory::{ExperimentRecord, KnowledgeBase, MemoryKnowledgeBase};
    use std::collections::HashMap;

    fn make_plan() -> SimpleResearchPlan {
        SimpleResearchPlan::new(
            HashMap::from([("model".into(), serde_json::json!("knn"))]),
            "k",
            vec![
                serde_json::json!(1),
                serde_json::json!(3),
                serde_json::json!(5),
                serde_json::json!(7),
            ],
        )
    }

    fn make_agent() -> Agent {
        let kb = Box::new(MemoryKnowledgeBase::new());
        Agent::new("test-agent", "maximize accuracy on iris", kb)
    }

    fn make_record(name: &str, line: &str, accuracy: f64) -> ExperimentRecord {
        ExperimentRecord::new(name, name)
            .with_hypothesis(format!("test hypothesis for {name}"))
            .with_research_line(line)
            .with_metrics(HashMap::from([("accuracy".into(), accuracy)]))
            .with_params(HashMap::from([("k".into(), serde_json::json!(3))]))
    }

    #[test]
    fn agent_first_step_explores() {
        let mut agent = make_agent();
        let plan = make_plan();
        let action = agent.step(&plan).unwrap();

        match action {
            Action::RunExperiment { hypothesis, .. } => {
                assert!(hypothesis.contains("maximize accuracy"));
            }
            _ => panic!("expected RunExperiment on first step"),
        }

        assert_eq!(agent.iteration(), 1);
        assert_eq!(agent.decisions().len(), 1);
    }

    #[test]
    fn agent_concludes_at_max_iterations() {
        let mut agent = make_agent().with_max_iterations(2);
        let plan = make_plan();

        agent.step(&plan).unwrap();
        agent.step(&plan).unwrap();

        let action = agent.step(&plan).unwrap();
        assert!(matches!(action, Action::Conclude { .. }));
    }

    #[test]
    fn agent_records_result() {
        let mut agent = make_agent();
        let record = make_record("exp_001", "k_exploration", 0.95);
        agent.record_result(record).unwrap();
        assert_eq!(agent.knowledge_base().len(), 1);
    }

    #[test]
    fn agent_step_after_recording() {
        let mut agent = make_agent().with_max_iterations(10);
        let plan = make_plan();

        // First step
        agent.step(&plan).unwrap();

        // Record results
        for (i, acc) in [0.85, 0.88, 0.91].iter().enumerate() {
            agent
                .record_result(make_record(&format!("exp_{i:04}"), "k_exploration", *acc))
                .unwrap();
        }

        // Should be able to step again
        let action = agent.step(&plan).unwrap();
        assert!(matches!(action, Action::RunExperiment { .. }));
    }

    #[test]
    fn agent_decisions_tracked() {
        let mut agent = make_agent().with_max_iterations(5);
        let plan = make_plan();

        for _ in 0..3 {
            agent.step(&plan).unwrap();
        }

        assert_eq!(agent.decisions().len(), 3);
        for (i, decision) in agent.decisions().iter().enumerate() {
            assert_eq!(decision.iteration, i + 1);
            assert!(!decision.reasoning.is_empty());
        }
    }

    #[test]
    fn action_serde_roundtrip() {
        let action = Action::RunExperiment {
            name: "exp_001".into(),
            research_line: "test_line".into(),
            hypothesis: "testing".into(),
            pipeline_config: HashMap::from([("k".into(), serde_json::json!(5))]),
            parent: Some("exp_000".into()),
        };

        let json = serde_json::to_string(&action).unwrap();
        let deserialized: Action = serde_json::from_str(&json).unwrap();

        match deserialized {
            Action::RunExperiment { name, parent, .. } => {
                assert_eq!(name, "exp_001");
                assert_eq!(parent.as_deref(), Some("exp_000"));
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn conclude_action_serde() {
        let action = Action::Conclude {
            reason: "objective reached".into(),
        };
        let json = serde_json::to_string(&action).unwrap();
        let deserialized: Action = serde_json::from_str(&json).unwrap();
        assert!(matches!(deserialized, Action::Conclude { .. }));
    }

    #[test]
    fn agent_zero_max_iterations() {
        let mut agent = make_agent().with_max_iterations(0);
        let plan = make_plan();
        let action = agent.step(&plan).unwrap();
        assert!(matches!(action, Action::Conclude { .. }));
    }
}
