use crate::action::Action;
use crate::planner::ResearchPlan;
use soma_core::error::Result;
use soma_memory::{ExperimentRecord, KnowledgeBase, ResearchLine, Trend};

/// An autonomous research agent that plans, executes, and analyzes experiments.
///
/// The agent follows a loop:
/// 1. Analyze current knowledge (what's been tried, what works)
/// 2. Generate hypotheses (what to try next)
/// 3. Plan experiments (build pipelines with params)
/// 4. Execute and record results
/// 5. Decide: continue, refine, or conclude
pub struct Agent {
    /// Agent identity and instructions.
    pub name: String,
    /// Research objective.
    pub objective: String,
    /// Knowledge base for experiment tracking.
    kb: Box<dyn KnowledgeBase>,
    /// Maximum iterations before stopping.
    pub max_iterations: usize,
    /// Current iteration.
    iteration: usize,
    /// Decision history.
    decisions: Vec<Decision>,
}

/// A decision made by the agent at each iteration.
#[derive(Debug, Clone)]
pub struct Decision {
    pub iteration: usize,
    pub action: Action,
    pub reasoning: String,
}

impl Agent {
    pub fn new(
        name: impl Into<String>,
        objective: impl Into<String>,
        kb: Box<dyn KnowledgeBase>,
    ) -> Self {
        Self {
            name: name.into(),
            objective: objective.into(),
            kb,
            max_iterations: 20,
            iteration: 0,
            decisions: Vec::new(),
        }
    }

    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    /// Run a single research iteration.
    ///
    /// Returns the action taken. The caller is responsible for executing
    /// the action (running a pipeline, etc.) and recording the result
    /// via `record_result()`.
    pub fn step(&mut self, plan: &dyn ResearchPlan) -> Result<Action> {
        if self.iteration >= self.max_iterations {
            let action = Action::Conclude {
                reason: "max iterations reached".into(),
            };
            self.record_decision(action.clone(), "iteration limit");
            return Ok(action);
        }

        self.iteration += 1;

        // Analyze current state
        let lines = self.kb.research_lines().unwrap_or_default();
        let n_experiments = self.kb.len();

        // First iteration: explore
        if n_experiments == 0 {
            let action = plan.initial_action(&self.objective)?;
            self.record_decision(action.clone(), "initial exploration");
            return Ok(action);
        }

        // Check for promising lines
        let promising: Vec<&ResearchLine> = lines
            .iter()
            .filter(|l| l.trend == Trend::Improving)
            .collect();

        // Check for plateaued lines
        let plateaued: Vec<&ResearchLine> = lines
            .iter()
            .filter(|l| l.trend == Trend::Plateaued)
            .collect();

        // Decision logic
        let action = if !promising.is_empty() {
            // Exploit: refine the best promising line
            let best_line = promising
                .iter()
                .max_by(|a, b| {
                    a.best_metric_value
                        .partial_cmp(&b.best_metric_value)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();

            plan.refine_action(&best_line.name, &self.objective)?
        } else if !plateaued.is_empty() {
            // Pivot: try a new approach since current ones plateaued
            plan.explore_action(&self.objective, &lines)?
        } else {
            // Explore: try something new
            plan.explore_action(&self.objective, &lines)?
        };

        let reasoning = match &action {
            Action::RunExperiment { .. } => {
                if !promising.is_empty() {
                    format!("refining promising line (iter {})", self.iteration)
                } else {
                    format!("exploring new direction (iter {})", self.iteration)
                }
            }
            Action::Conclude { reason } => reason.clone(),
        };

        self.record_decision(action.clone(), &reasoning);
        Ok(action)
    }

    /// Record an experiment result in the knowledge base.
    pub fn record_result(&mut self, record: ExperimentRecord) -> Result<()> {
        self.kb.record(record)
    }

    /// Get the knowledge base (for querying).
    pub fn knowledge_base(&self) -> &dyn KnowledgeBase {
        self.kb.as_ref()
    }

    /// Get all decisions made so far.
    pub fn decisions(&self) -> &[Decision] {
        &self.decisions
    }

    /// Current iteration number.
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    fn record_decision(&mut self, action: Action, reasoning: &str) {
        self.decisions.push(Decision {
            iteration: self.iteration,
            action,
            reasoning: reasoning.to_string(),
        });
    }
}
