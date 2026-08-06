//! The research loop, driven end to end.
//!
//! A scripted model stands in for a real one and a one-node pipeline stands
//! in for real work, so what this exercises is the wiring: propose → run →
//! read metrics → propose again → conclude, over the same effect driver and
//! journal every other step uses.

use somatize_agent::ResearchStep;
use somatize_core::agentic::effect::{Effect, EffectResult, LlmResponse, StopReason};
use somatize_core::agentic::message::Message;
use somatize_core::cache::CacheKey;
use somatize_core::data::value::Value;
use somatize_core::error::{Result, SomaError};
use somatize_core::graph::filter::{Distribution, Filter, FilterKind, FilterMeta, StreamMode};
use somatize_core::graph::{Graph, Node};
use somatize_runtime::cache::FsActionStore;
use somatize_runtime::effects::{
    EffectDriver, EffectHandler, EffectJournal, GraphHandler, NodeOutcome,
};
use somatize_runtime::node_catalog::NodeCatalog;
use std::sync::{Arc, Mutex};

// ── A model that answers from a script ──

struct ScriptedModel {
    replies: Mutex<Vec<String>>,
    asked: Mutex<Vec<String>>,
}

impl ScriptedModel {
    fn new(replies: Vec<String>) -> Arc<Self> {
        Arc::new(Self {
            replies: Mutex::new(replies),
            asked: Mutex::new(Vec::new()),
        })
    }
}

impl EffectHandler for ScriptedModel {
    fn handles(&self, effect: &Effect) -> bool {
        matches!(effect, Effect::Llm(_))
    }

    fn perform(&self, effect: &Effect) -> Result<EffectResult> {
        let Effect::Llm(request) = effect else {
            return Err(SomaError::Other("not an llm effect".into()));
        };
        self.asked.lock().unwrap().push(
            request
                .messages
                .0
                .last()
                .map(|m| m.text())
                .unwrap_or_default(),
        );

        let mut replies = self.replies.lock().unwrap();
        let text = if replies.is_empty() {
            r#"{"action": "conclude", "reason": "script exhausted"}"#.to_string()
        } else {
            replies.remove(0)
        };

        Ok(EffectResult::Llm(LlmResponse {
            message: Message::assistant(text),
            stop_reason: StopReason::EndTurn,
            usage: Default::default(),
            model: None,
        }))
    }
}

// ── A pipeline whose score depends on a parameter ──

struct Scored;

impl Filter for Scored {
    fn meta(&self) -> FilterMeta {
        FilterMeta {
            name: "scored".into(),
            kind: FilterKind::Trainable,
            cacheable: false,
            differentiable: false,
            deterministic: true,
            stream_mode: StreamMode::FixedState,
            distribution: Distribution::Local,
            input_schema: None,
            output_schema: None,
        }
    }

    fn fit(&self, _x: &Value, _y: Option<&Value>) -> Result<Value> {
        Ok(Value::Empty)
    }

    /// The "experiment": bigger C, better f1, with a ceiling. A pipeline
    /// reports its metrics as a node's output, the same way a `Study`
    /// objective reads them.
    fn forward(&self, x: &Value, _state: &Value) -> Result<Value> {
        let c = x.to_plain_json()["classifier.C"].as_f64().unwrap_or(0.0);
        Ok(Value::json(
            serde_json::json!({ "f1": (0.5 + c / 10.0).min(0.95) }),
        ))
    }

    fn config_hash(&self) -> CacheKey {
        CacheKey::from_parts(&[b"scored"])
    }
}

fn pipeline() -> Graph {
    let mut graph = Graph::new();
    graph.add_node(Node::filter_with_id("classifier", "scored"));
    graph
}

fn library() -> NodeCatalog {
    let mut library = NodeCatalog::new();
    library.register("classifier", Box::new(Scored));
    library
}

fn driver(model: Arc<ScriptedModel>) -> (EffectDriver, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(FsActionStore::new(dir.path()).unwrap());
    let driver = EffectDriver::new(EffectJournal::new(store.clone(), store))
        .with_handler(model)
        .with_handler(Arc::new(GraphHandler::new(library())));
    // The tempdir travels with the driver: dropping it would delete the
    // journal out from under the run.
    (driver, dir)
}

fn experiment(name: &str, c: f64) -> String {
    format!(
        r#"{{"action": "run_experiment", "name": "{name}",
            "research_line": "regularization",
            "hypothesis": "C={c} lifts f1",
            "params": {{"classifier.C": {c}}}}}"#
    )
}

fn run(step: ResearchStep, model: Arc<ScriptedModel>) -> Value {
    let (driver, _dir) = driver(model);
    match driver
        .run(&step, "run_1", "researcher", &Value::Empty)
        .unwrap()
    {
        NodeOutcome::Produced(value) => value,
        other => panic!("expected the loop to finish, got {other:?}"),
    }
}

#[test]
fn the_loop_runs_experiments_and_concludes() {
    let model = ScriptedModel::new(vec![
        experiment("exp_1", 1.0),
        experiment("exp_2", 4.0),
        r#"{"action": "conclude", "reason": "f1 above target"}"#.into(),
    ]);

    let report = run(
        ResearchStep::new("mock/model", "beat 0.8 f1", pipeline()),
        model,
    )
    .to_plain_json();

    assert_eq!(report["concluded"], "f1 above target");
    assert_eq!(report["experiments"], 2);

    // Each experiment carries the metrics its pipeline produced, so a
    // conclusion can be traced back to the runs supporting it.
    let records = report["records"].as_array().unwrap();
    assert_eq!(records[0]["metrics"]["classifier.f1"], 0.6);
    assert_eq!(records[1]["metrics"]["classifier.f1"], 0.9);
    assert_eq!(records[1]["hypothesis"], "C=4 lifts f1");
}

#[test]
fn the_model_is_shown_what_already_ran() {
    let model = ScriptedModel::new(vec![
        experiment("exp_1", 1.0),
        r#"{"action": "conclude", "reason": "enough"}"#.into(),
    ]);

    run(
        ResearchStep::new("mock/model", "beat 0.8 f1", pipeline()),
        model.clone(),
    );

    let asked = model.asked.lock().unwrap();
    assert!(asked[0].contains("No experiments yet"), "{}", asked[0]);
    // The second question carries the first result — an agent that cannot
    // see what it already tried proposes it again.
    assert!(asked[1].contains("exp_1"), "{}", asked[1]);
    assert!(asked[1].contains("classifier.f1=0.6000"), "{}", asked[1]);
}

#[test]
fn a_failed_experiment_is_reported_back_not_fatal() {
    let mut broken = Graph::new();
    broken.add_node(Node::filter_with_id("nowhere", "missing"));

    let model = ScriptedModel::new(vec![
        experiment("exp_bad", 1.0),
        r#"{"action": "conclude", "reason": "the pipeline is broken"}"#.into(),
    ]);

    let report = run(
        ResearchStep::new("mock/model", "find out", broken),
        model.clone(),
    )
    .to_plain_json();

    // A configuration that will not run is a finding, and one of the more
    // valuable kinds — it belongs in the record, not in a stack trace.
    assert_eq!(report["experiments"], 1);
    assert!(
        report["records"][0]["notes"]
            .as_str()
            .unwrap()
            .contains("failed")
    );
    assert!(model.asked.lock().unwrap()[1].contains("exp_bad"));
}

#[test]
fn the_iteration_budget_stops_a_loop_that_will_not_stop_itself() {
    // A model that only ever proposes more work.
    let model = ScriptedModel::new(
        (0..20)
            .map(|i| experiment(&format!("exp_{i}"), 1.0))
            .collect(),
    );

    let report = run(
        ResearchStep::new("mock/model", "never satisfied", pipeline()).with_max_iterations(3),
        model,
    )
    .to_plain_json();

    assert_eq!(report["concluded"], "iteration budget exhausted");
    assert_eq!(report["experiments"], 3);
}

#[test]
fn a_seeded_history_is_in_the_first_question() {
    let mut record = somatize_memory::ExperimentRecord::new("earlier", "earlier");
    record.research_line = Some("regularization".into());
    record.metrics = [("f1".to_string(), 0.55)].into_iter().collect();

    let model = ScriptedModel::new(vec![r#"{"action": "conclude", "reason": "known"}"#.into()]);
    run(
        ResearchStep::new("mock/model", "beat 0.8 f1", pipeline()).with_history(vec![record]),
        model.clone(),
    );

    // Starting from an empty pool means repeating work somebody already did.
    assert!(model.asked.lock().unwrap()[0].contains("earlier"));
}

#[test]
fn prose_instead_of_an_action_ends_the_run_rather_than_guessing() {
    let model = ScriptedModel::new(vec!["I think we should try more values of C.".into()]);
    let step = ResearchStep::new("mock/model", "beat 0.8 f1", pipeline());

    let (driver, _dir) = driver(model);
    let err = driver
        .run(&step, "run_1", "researcher", &Value::Empty)
        .unwrap_err();
    assert!(err.to_string().contains("no JSON object"), "{err}");
}
