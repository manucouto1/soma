//! Integration test: full autonomous research loop.

use serde_json::json;
use somatize_agent::{Action, Agent, SimpleResearchPlan};
use somatize_memory::{ExperimentRecord, MemoryKnowledgeBase};
use std::collections::HashMap;

/// Simulate executing an experiment: returns a metric based on params.
fn simulate_experiment(config: &HashMap<String, serde_json::Value>) -> f64 {
    let c = config.get("C").and_then(|v| v.as_f64()).unwrap_or(1.0);
    // Simulated metric: peaks at C=1.0
    1.0 - (c - 1.0).abs() * 0.1
}

#[test]
fn agent_runs_research_loop() {
    let kb = Box::new(MemoryKnowledgeBase::new());
    let mut agent = Agent::new("test_agent", "find best C for SVM", kb).with_max_iterations(5);

    let plan = SimpleResearchPlan::new(
        HashMap::from([("model".into(), json!("svm"))]),
        "C",
        vec![json!(0.1), json!(0.5), json!(1.0), json!(5.0), json!(10.0)],
    );

    for _ in 0..5 {
        let action = agent.step(&plan).unwrap();

        match action {
            Action::RunExperiment {
                name,
                research_line,
                hypothesis,
                pipeline_config,
                parent,
            } => {
                // Simulate execution
                let metric = simulate_experiment(&pipeline_config);

                // Record result
                let mut metrics = HashMap::new();
                metrics.insert("accuracy".to_string(), metric);

                let record = ExperimentRecord::new(&name, &name)
                    .with_hypothesis(&hypothesis)
                    .with_research_line(&research_line)
                    .with_params(pipeline_config)
                    .with_metrics(metrics);

                if let Some(p) = parent {
                    agent.record_result(record.with_parent(&p)).unwrap();
                } else {
                    agent.record_result(record).unwrap();
                }
            }
            Action::Conclude { .. } => break,
            _ => {}
        }
    }

    // Agent should have run experiments
    assert!(agent.iteration() > 0);
    assert!(!agent.decisions().is_empty());

    // Knowledge base should have records
    let kb = agent.knowledge_base();
    assert!(!kb.is_empty());
    assert!(!kb.research_lines().unwrap().is_empty());
}

#[test]
fn agent_stops_at_max_iterations() {
    let kb = Box::new(MemoryKnowledgeBase::new());
    let mut agent = Agent::new("test", "objective", kb).with_max_iterations(3);

    let plan = SimpleResearchPlan::new(HashMap::new(), "x", vec![json!(1)]);

    let mut actions = Vec::new();
    for _ in 0..10 {
        let action = agent.step(&plan).unwrap();
        let is_conclude = matches!(&action, Action::Conclude { .. });
        actions.push(action);
        if is_conclude {
            break;
        }
    }

    // Should conclude at iteration 4 (3 experiments + 1 conclude)
    assert!(actions.len() <= 4);
    assert!(matches!(actions.last().unwrap(), Action::Conclude { .. }));
}

#[test]
fn agent_tracks_decisions() {
    let kb = Box::new(MemoryKnowledgeBase::new());
    let mut agent = Agent::new("test", "objective", kb).with_max_iterations(3);

    let plan = SimpleResearchPlan::new(HashMap::new(), "x", vec![json!(1)]);

    agent.step(&plan).unwrap();
    agent.step(&plan).unwrap();

    let decisions = agent.decisions();
    assert_eq!(decisions.len(), 2);
    assert_eq!(decisions[0].iteration, 1);
    assert_eq!(decisions[1].iteration, 2);
}
