//! The research loop, as a [`Step`].
//!
//! An autonomous researcher is not a special kind of runtime — it is a node
//! that thinks between experiments. Writing it as a `Step` means it gets the
//! same journal, cache, events and replay every other node gets, and it
//! means an agent can be *part of* a pipeline rather than something that
//! wraps one.
//!
//! One turn of the loop:
//!
//! 1. Ask the model what to try next, given the objective and what the pool
//!    already knows.
//! 2. Read an [`Action`] out of the reply — a constrained schema, not prose.
//! 3. `RunExperiment` → [`Effect::Graph`], which runs the pipeline with the
//!    proposed parameters and hands back its metrics.
//! 4. `Conclude` → done.
//!
//! The previous version of this file generated experiments by cycling a
//! list of values through a rule-based planner, with a comment saying an LLM
//! would go here. This is that.

use crate::action::Action;
use somatize_core::effect::{Effect, EffectResult, GraphEffectMode, LlmRequest};
use somatize_core::error::{Result, SomaError};
use somatize_core::graph::Graph;
use somatize_core::message::{Message, Messages};
use somatize_core::step::{Step, StepCtx, StepMeta, Transition};
use somatize_core::util::{extract_json, truncate};
use somatize_core::value::Value;
use somatize_memory::ExperimentRecord;

/// How much of the pool to put in front of the model each turn.
const HISTORY_LIMIT: usize = 20;

/// An agent that proposes experiments, runs them, and decides when to stop.
#[derive(serde::Serialize, somatize_core::SomaStep)]
#[soma(cache_version = "soma-research-step-v1")]
pub struct ResearchStep {
    model: String,
    objective: String,
    /// The pipeline being investigated. Its parameters are what the agent
    /// proposes values for.
    pipeline: Graph,
    max_iterations: usize,
    /// What was already known when the loop started. Everything the loop
    /// itself learns is reconstructed from `ctx.history` instead of kept
    /// here — see [`ResearchStep::completed`].
    seed: Vec<ExperimentRecord>,
}

impl ResearchStep {
    /// A researcher that pursues `objective` by experimenting on
    /// `pipeline`, asking `model` what to try next. Starts with an empty
    /// seed and 20 iterations; adjust with
    /// [`with_history`](Self::with_history) and
    /// [`with_max_iterations`](Self::with_max_iterations).
    pub fn new(model: impl Into<String>, objective: impl Into<String>, pipeline: Graph) -> Self {
        Self {
            model: model.into(),
            objective: objective.into(),
            pipeline,
            max_iterations: 20,
            seed: Vec::new(),
        }
    }

    /// Seed the loop with what is already known. An agent that starts from
    /// an empty pool repeats work someone already did.
    pub fn with_history(mut self, records: Vec<ExperimentRecord>) -> Self {
        self.seed = records;
        self
    }

    /// Cap the loop at `n` experiments. The cap is the budget, not the
    /// goal — the agent may `Conclude` well before reaching it.
    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    /// Every experiment behind this run, seed first, newest last.
    ///
    /// Rebuilt from the turns rather than accumulated in a field. A step is
    /// a description of behaviour, not a place to keep state: derive the
    /// history and a replay reconstructs exactly the history the original
    /// run had, because it replays exactly the same results.
    pub fn completed(&self, ctx: &StepCtx<'_>) -> Vec<ExperimentRecord> {
        let mut records = self.seed.clone();
        let mut proposed: Option<Action> = None;

        for turn in ctx.history {
            match turn.first() {
                // A reply: whatever it proposed is what the next result
                // belongs to.
                Some(EffectResult::Llm(response)) => {
                    proposed = self.parse_action(&response.message.text()).ok();
                }
                // An experiment came back. Pair it with what asked for it.
                Some(outcome) => {
                    if let Some(action) = proposed.take()
                        && let Some(record) = self.record(&action, outcome)
                    {
                        records.push(record);
                    }
                }
                None => {}
            }
        }
        records
    }

    /// Ask the model what to do next.
    fn ask(&self, ctx: &StepCtx<'_>) -> Effect {
        LlmRequest::new(
            &self.model,
            Messages::from(vec![Message::user(
                self.history_prompt(&self.completed(ctx)),
            )]),
        )
        .with_system(self.system())
        // The shape is declared once, as a schema. It used to be written
        // out a second time as prose inside `system()` while
        // `Action::response_schema` sat unused — two descriptions of one
        // contract, and only the prose could drift. Declaring it here
        // also lets an endpoint that supports constrained decoding
        // *enforce* it rather than be asked nicely.
        .with_schema(crate::action::Action::response_schema())
        .into_effect()
    }

    fn system(&self) -> String {
        format!(
            "You are running an experimental campaign on a Soma pipeline.\n\n\
             Objective: {}\n\n\
             Each turn, propose ONE experiment or conclude; the reply must \
             match the declared schema. Use `params` keys of the form \
             `<node>.<param>`.\n\n\
             Every experiment needs a falsifiable hypothesis — a result \
             nobody can interpret later is a result nobody will read. Vary \
             one thing at a time so the comparison means something, and \
             conclude when the objective is met or the line has stopped \
             paying.\n\n\
             Pipeline nodes: {}",
            self.objective,
            self.pipeline.node_ids().join(", ")
        )
    }

    /// What is known so far, as a compact table.
    fn history_prompt(&self, history: &[ExperimentRecord]) -> String {
        if history.is_empty() {
            return "No experiments yet. Propose the first one.".into();
        }

        let mut lines = vec![format!("{} experiments so far:", history.len())];
        for record in history.iter().rev().take(HISTORY_LIMIT).rev() {
            let mut metrics: Vec<String> = record
                .metrics
                .iter()
                .map(|(k, v)| format!("{k}={v:.4}"))
                .collect();
            metrics.sort();
            lines.push(format!(
                "- {} [{}] {} → {}",
                record.name,
                record.research_line.as_deref().unwrap_or("unfiled"),
                serde_json::to_string(&record.params).unwrap_or_default(),
                if metrics.is_empty() {
                    "no metrics".to_string()
                } else {
                    metrics.join(" ")
                }
            ));
        }
        lines.push("\nWhat next?".into());
        lines.join("\n")
    }

    /// What the loop produced: the conclusion, and every experiment behind
    /// it. The records are the point — a conclusion nobody can trace back
    /// to the runs that support it is an opinion.
    fn report(&self, reason: &str, done: &[ExperimentRecord]) -> Value {
        Value::json(serde_json::json!({
            "concluded": reason,
            "objective": self.objective,
            "experiments": done.len(),
            "records": done,
        }))
    }

    /// Read an action out of a reply, tolerating the ways models wrap JSON.
    fn parse_action(&self, text: &str) -> Result<Action> {
        let json = extract_json(text).ok_or_else(|| {
            SomaError::Other(format!(
                "the model replied with no JSON object, so there is no action \
                 to take: {}",
                truncate(text, 200)
            ))
        })?;

        serde_json::from_value(json).map_err(|e| {
            SomaError::Other(format!(
                "the model's reply is not an action ({e}): {}",
                truncate(text, 200)
            ))
        })
    }

    /// The graph to run for an experiment.
    ///
    /// Parameters travel as the effect's *input*, not baked into the graph:
    /// the pipeline is the thing being studied and must stay identical
    /// across experiments, or the comparison is between two different
    /// pipelines and means nothing.
    fn experiment_effect(&self, params: &serde_json::Map<String, serde_json::Value>) -> Effect {
        Effect::Graph {
            graph: Box::new(self.pipeline.clone()),
            input: Value::json(serde_json::Value::Object(params.clone())),
            mode: GraphEffectMode::Fit,
        }
    }

    fn record(&self, action: &Action, result: &EffectResult) -> Option<ExperimentRecord> {
        let Action::RunExperiment {
            name,
            research_line,
            hypothesis,
            params,
            parent,
        } = action
        else {
            return None;
        };

        let mut record = ExperimentRecord::new(name.clone(), name.clone());
        record.hypothesis = Some(hypothesis.clone());
        record.research_line = Some(research_line.clone());
        record.parent = parent.clone();
        record.params = params.clone();
        record.tags = vec!["agent".into()];
        record.pipeline_summary = self.pipeline.node_ids().join(" → ");

        match result {
            EffectResult::Graph(value) => {
                record.metrics = read_metrics(value);
            }
            // A failed experiment is a finding. Recording it is what stops
            // the agent proposing the same broken configuration next turn.
            EffectResult::Failed { message } => {
                record.notes = Some(format!("failed: {message}"));
            }
            other => {
                record.notes = Some(format!("unexpected result: {other:?}"));
            }
        }
        Some(record)
    }
}

impl Step for ResearchStep {
    /// Derived: the pipeline under investigation is part of the key too,
    /// which the hand-written version left out — two research loops over
    /// different graphs shared a journal.
    fn config_hash(&self) -> somatize_core::cache::CacheKey {
        ResearchStep::config_hash(self)
    }

    fn meta(&self) -> StepMeta {
        StepMeta::new("research")
            .with_max_turns(self.max_iterations * 2 + 2)
            .with_output_schema(somatize_core::schema::Schema::json())
    }

    fn poll(&self, ctx: &StepCtx<'_>) -> Result<Transition> {
        // Turn 0, and every turn after an experiment: ask what to do next.
        let last = ctx.results.first();

        match last {
            None => Ok(Transition::Await(vec![self.ask(ctx)])),

            // The model answered. Either run what it proposed, or stop.
            Some(EffectResult::Llm(response)) => {
                // A cut-off proposal is not a proposal. Parsing it fails
                // with "no JSON action found", which reads like the model
                // misbehaving when it actually ran out of tokens — and the
                // fix for each is different.
                response.reject_non_answers(ctx.node_id)?;
                let action = self.parse_action(&response.message.text())?;
                let done = self.completed(ctx);
                match &action {
                    Action::Conclude { reason } => Ok(Transition::Done(self.report(reason, &done))),
                    Action::RunExperiment { params, .. } => {
                        if done.len() >= self.max_iterations {
                            return Ok(Transition::Done(
                                self.report("iteration budget exhausted", &done),
                            ));
                        }
                        Ok(Transition::Await(vec![
                            self.experiment_effect(&to_object(params)),
                        ]))
                    }
                }
            }

            // An experiment finished. `completed` folds it into the
            // history the next question is asked against.
            Some(_) => Ok(Transition::Await(vec![self.ask(ctx)])),
        }
    }
}

fn to_object(
    params: &std::collections::BTreeMap<String, serde_json::Value>,
) -> serde_json::Map<String, serde_json::Value> {
    params.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
}

/// Metrics out of whatever the pipeline produced.
///
/// Every number in the result is a metric, named by where it sits:
/// `{"classifier": {"f1": 0.9}}` gives `classifier.f1`. Qualifying by node
/// is what keeps two nodes reporting `loss` from being the same series when
/// they are compared across experiments later.
fn read_metrics(value: &Value) -> std::collections::BTreeMap<String, f64> {
    let mut metrics = std::collections::BTreeMap::new();
    collect_numbers(&value.to_plain_json(), "", &mut metrics);
    metrics
}

fn collect_numbers(
    json: &serde_json::Value,
    path: &str,
    out: &mut std::collections::BTreeMap<String, f64>,
) {
    match json {
        serde_json::Value::Number(n) => {
            if let Some(f) = n.as_f64() {
                let name = if path.is_empty() { "value" } else { path };
                out.insert(name.to_string(), f);
            }
        }
        serde_json::Value::Object(map) => {
            for (key, val) in map {
                let child = if path.is_empty() {
                    key.clone()
                } else {
                    format!("{path}.{key}")
                };
                collect_numbers(val, &child, out);
            }
        }
        // An array is data, not a metric. A learned weight vector is not a
        // number anyone wants to compare experiments on.
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn step() -> ResearchStep {
        let mut graph = Graph::new();
        graph.add_node(somatize_core::graph::Node::filter_with_id(
            "classifier",
            "svm",
        ));
        ResearchStep::new("mock/model", "beat 0.8 F1", graph)
    }

    #[test]
    fn the_history_prompt_starts_empty() {
        assert!(step().history_prompt(&[]).contains("No experiments yet"));
    }

    #[test]
    fn the_history_prompt_lists_what_ran() {
        let mut record = ExperimentRecord::new("exp_1", "exp_1");
        record.research_line = Some("regularization".into());
        record.params = [("classifier.C".to_string(), json!(0.1))]
            .into_iter()
            .collect();
        record.metrics = [("f1".to_string(), 0.72)].into_iter().collect();

        let prompt = step().history_prompt(&[record]);
        assert!(prompt.contains("exp_1"), "{prompt}");
        assert!(prompt.contains("regularization"), "{prompt}");
        assert!(prompt.contains("f1=0.7200"), "{prompt}");
    }

    #[test]
    fn an_action_is_read_out_of_a_fenced_reply() {
        let action = step()
            .parse_action(
                "Here is my plan:\n```json\n{\"action\": \"conclude\", \
                 \"reason\": \"plateaued\"}\n```",
            )
            .unwrap();
        assert!(action.is_terminal());
    }

    /// Prose is not an action. Guessing one would run an experiment nobody
    /// asked for, on a budget somebody is paying.
    #[test]
    fn prose_is_not_an_action() {
        let err = step().parse_action("I think we should try more C values.");
        assert!(err.is_err());
    }

    #[test]
    fn a_reply_that_is_not_an_action_is_refused() {
        let err = step().parse_action("{\"thoughts\": \"hmm\"}");
        assert!(err.is_err());
    }

    #[test]
    fn metrics_are_read_from_a_mapping() {
        let value = Value::json(json!({"f1": 0.9, "notes": "fine", "loss": 0.1}));
        let metrics = read_metrics(&value);
        assert_eq!(metrics.len(), 2);
        assert_eq!(metrics["f1"], 0.9);
    }

    /// A metric is named by where it sits, so two nodes both reporting
    /// `loss` stay two series.
    #[test]
    fn nested_metrics_are_qualified_by_node() {
        let value = Value::json(json!({
            "encoder": {"loss": 0.3},
            "classifier": {"loss": 0.1, "f1": 0.9, "weights": [1.0, 2.0]}
        }));
        let metrics = read_metrics(&value);
        assert_eq!(metrics["encoder.loss"], 0.3);
        assert_eq!(metrics["classifier.loss"], 0.1);
        assert_eq!(metrics["classifier.f1"], 0.9);
        assert_eq!(metrics.len(), 3, "an array is data, not a metric");
    }

    #[test]
    fn a_failed_experiment_is_still_recorded() {
        let action: Action = serde_json::from_value(json!({
            "action": "run_experiment",
            "name": "exp_bad",
            "research_line": "l",
            "hypothesis": "h",
            "params": {}
        }))
        .unwrap();

        let record = step()
            .record(
                &action,
                &EffectResult::Failed {
                    message: "no such filter".into(),
                },
            )
            .unwrap();

        assert!(record.metrics.is_empty());
        assert!(record.notes.unwrap().contains("no such filter"));
    }
}
