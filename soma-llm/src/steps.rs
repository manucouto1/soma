//! Steps you can use without writing one.
//!
//! Everything below the surface — effects, the journal, the driver — is
//! substrate. These are the pieces that make it usable: a ReAct loop, a
//! single call, and an LLM judge. Between them they cover most of what an
//! agentic node does, and they are the building blocks the Python
//! combinators (`route`, `refine`, `orchestrate`, …) assemble.
//!
//! They are deliberately plain `Step` implementations with no privileged
//! access. Anything they do, a user's own step can do.

use somatize_core::cache::CacheKey;
use somatize_core::effect::{Effect, EffectResult, LlmRequest, StopReason, ToolSpec};
use somatize_core::error::Result;
use somatize_core::message::{ContentBlock, Message, Messages, Role};
use somatize_core::schema::Schema;
use somatize_core::step::{Step, StepCtx, StepMeta, Transition};
use somatize_core::value::Value;

/// Ask the model; run whatever tools it asks for; repeat until it stops.
///
/// The loop Yao et al. named ReAct, and the shape most single-agent nodes
/// take. It is short because the substrate does the work: tool calls become
/// [`Effect::Tool`]s the driver runs concurrently and journals, so a
/// crashed run resumes mid-loop and a replay costs nothing.
///
/// Input may be a bare prompt or a whole conversation — see
/// [`Messages::from_value`]. Output is the conversation, so a downstream
/// node sees the reasoning and not only the conclusion.
pub struct ReactStep {
    model: String,
    system: Option<String>,
    tools: Vec<ToolSpec>,
    max_turns: usize,
    max_tokens: Option<u32>,
    effort: Option<String>,
    /// Return only the final prose rather than the whole conversation.
    text_only: bool,
}

impl ReactStep {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            system: None,
            tools: Vec::new(),
            max_turns: 12,
            max_tokens: None,
            effort: None,
            text_only: false,
        }
    }

    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Tools the model may call. Their results come back as
    /// [`ContentBlock::ToolResult`] paired to the call that asked.
    pub fn with_tools(mut self, tools: Vec<ToolSpec>) -> Self {
        self.tools = tools;
        self
    }

    /// Cap on model calls. Each turn is one call plus its tools.
    pub fn with_max_turns(mut self, n: usize) -> Self {
        self.max_turns = n;
        self
    }

    pub fn with_max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = Some(n);
        self
    }

    pub fn with_effort(mut self, effort: impl Into<String>) -> Self {
        self.effort = Some(effort.into());
        self
    }

    /// Emit just the final text instead of the conversation.
    pub fn text_only(mut self) -> Self {
        self.text_only = true;
        self
    }

    fn ask(&self, messages: Messages) -> Effect {
        let mut req = LlmRequest::new(&self.model, messages);
        if let Some(system) = &self.system {
            req = req.with_system(system);
        }
        if let Some(max) = self.max_tokens {
            req = req.with_max_tokens(max);
        }
        if let Some(effort) = &self.effort {
            req = req.with_effort(effort);
        }
        req.with_tools(self.tools.clone()).into_effect()
    }

    fn finish(&self, conversation: &Messages) -> Value {
        if self.text_only {
            Value::text(conversation.last().map(Message::text).unwrap_or_default())
        } else {
            conversation.to_value()
        }
    }
}

impl Step for ReactStep {
    fn config_hash(&self) -> CacheKey {
        // Everything that changes what this step would do. Tools included:
        // the same prompt with a different toolset is a different node.
        let tools = serde_json::to_vec(&self.tools).unwrap_or_default();
        CacheKey::from_parts(&[
            b"ReactStep",
            self.model.as_bytes(),
            self.system.as_deref().unwrap_or("").as_bytes(),
            &tools,
            self.effort.as_deref().unwrap_or("").as_bytes(),
        ])
    }

    fn meta(&self) -> StepMeta {
        StepMeta::new("ReactStep")
            // Each ReAct round is two turns: ask, then hand back tool
            // results. Plus one to finish.
            .with_max_turns(self.max_turns * 2 + 1)
            .with_input_schema(Schema::messages())
            .with_output_schema(if self.text_only {
                Schema::text()
            } else {
                Schema::messages()
            })
    }

    fn poll(&self, ctx: &StepCtx<'_>) -> Result<Transition> {
        // Turn 0: nothing has happened yet.
        if ctx.results.is_empty() {
            let conversation = Messages::from_value(ctx.input)?;
            return Ok(Transition::Await(vec![self.ask(conversation)]));
        }

        // The conversation is rebuilt from the input plus *every* turn so
        // far, rather than held in `self`. Folding only the current turn
        // would drop the assistant message that asked for a tool, leaving
        // its result paired to nothing. And rebuilding from the replayed
        // history is what makes a resumed run reach the same decisions.
        let mut conversation = Messages::from_value(ctx.input)?;
        for turn in ctx.history {
            replay_into(&mut conversation, turn);
        }

        // Were we waiting on tools, or on the model? The conversation says:
        // tool results only ever follow an assistant turn that asked for
        // them. Getting this wrong is how a failed *model* call turns into
        // twenty-five retries of a request that can never succeed.
        let awaiting_tools = conversation
            .last()
            .is_some_and(|m| m.role == Role::Assistant && m.tool_uses().next().is_some());

        if !awaiting_tools && let Some(EffectResult::Failed { message }) = ctx.results.first() {
            return Err(somatize_core::error::SomaError::Execution {
                node_id: ctx.node_id.to_string(),
                message: format!("the model call failed: {message}"),
            });
        }

        match ctx.results.first() {
            // A model reply: either it wants tools, or it is done.
            Some(EffectResult::Llm(response)) => {
                if response.stop_reason != StopReason::ToolUse {
                    return Ok(Transition::Done(self.finish(&conversation)));
                }
                let calls: Vec<Effect> = response
                    .message
                    .tool_uses()
                    .map(|(_, name, input)| Effect::Tool {
                        name: name.to_string(),
                        args: Value::json(input.clone()),
                    })
                    .collect();

                // `stop_reason: tool_use` with no calls happens; treating it
                // as "done" beats awaiting nothing and erroring.
                if calls.is_empty() {
                    return Ok(Transition::Done(self.finish(&conversation)));
                }
                Ok(Transition::Await(calls))
            }

            // Tool results: hand them back and let the model continue.
            Some(EffectResult::Tool { .. }) | Some(EffectResult::Failed { .. }) => {
                Ok(Transition::Await(vec![self.ask(conversation)]))
            }

            other => Ok(Transition::Done(Value::text(format!(
                "unexpected effect result: {other:?}"
            )))),
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Fold a turn's effect results back into the conversation.
///
/// A model reply appends its assistant turn verbatim — prose and tool calls
/// together. Tool results append as a user turn whose blocks carry the id of
/// the call each answers, which is what keeps a call and its result from
/// drifting apart.
fn replay_into(conversation: &mut Messages, results: &[EffectResult]) {
    let mut tool_blocks: Vec<ContentBlock> = Vec::new();

    for result in results {
        match result {
            EffectResult::Llm(response) => conversation.push(response.message.clone()),
            EffectResult::Tool { output, is_error } => {
                let text = output
                    .as_text()
                    .map(str::to_string)
                    .unwrap_or_else(|| output.to_plain_json().to_string());
                // The id is filled in below, once we know which call this
                // answers — position in the batch matches request order.
                tool_blocks.push(if *is_error {
                    ContentBlock::tool_error("", text)
                } else {
                    ContentBlock::tool_result("", text)
                });
            }
            // A failed tool is reported to the model, not hidden: it can
            // rephrase, try another tool, or explain that it cannot proceed.
            EffectResult::Failed { message } => {
                tool_blocks.push(ContentBlock::tool_error("", message.clone()));
            }
            _ => {}
        }
    }

    if tool_blocks.is_empty() {
        return;
    }

    // Pair each result with the call it answers, by position.
    let ids: Vec<String> = conversation
        .last()
        .map(|m| m.tool_uses().map(|(id, _, _)| id.to_string()).collect())
        .unwrap_or_default();

    for (block, id) in tool_blocks.iter_mut().zip(ids) {
        if let ContentBlock::ToolResult { tool_use_id, .. } = block {
            *tool_use_id = id;
        }
    }
    conversation.push(Message::new(Role::User, tool_blocks));
}

/// One model call, no tools. The cheapest useful node.
pub struct LlmStep {
    inner: ReactStep,
}

impl LlmStep {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            inner: ReactStep::new(model).text_only(),
        }
    }

    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.inner = self.inner.with_system(system);
        self
    }

    pub fn with_max_tokens(mut self, n: u32) -> Self {
        self.inner = self.inner.with_max_tokens(n);
        self
    }

    /// Return the whole conversation rather than just the reply.
    pub fn keep_conversation(mut self) -> Self {
        self.inner.text_only = false;
        self
    }
}

impl Step for LlmStep {
    fn config_hash(&self) -> CacheKey {
        CacheKey::from_parts(&[b"LlmStep", &self.inner.config_hash().0])
    }
    fn meta(&self) -> StepMeta {
        StepMeta {
            name: "LlmStep".into(),
            ..self.inner.meta()
        }
    }
    fn poll(&self, ctx: &StepCtx<'_>) -> Result<Transition> {
        self.inner.poll(ctx)
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// A verdict a judge returns.
#[derive(Debug, Clone, PartialEq)]
pub struct Verdict {
    pub score: f64,
    pub passed: bool,
    pub reason: String,
}

/// Score something with a model against a rubric.
///
/// The evaluator half of evaluator-optimizer, and — once a study wraps an
/// agentic graph — the objective function. Output is
/// `{"score", "passed", "reason"}`, so a loop can read `passed` as its
/// termination signal and a study can read `score` as its metric, with no
/// glue between them.
pub struct JudgeStep {
    model: String,
    rubric: String,
    threshold: f64,
}

impl JudgeStep {
    /// A rubric should be explicitly gradeable — "the CSV has a numeric
    /// `price` column per SKU", not "the data looks good". Vague criteria
    /// produce noisy scores, and a noisy objective is worse than none.
    pub fn new(model: impl Into<String>, rubric: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            rubric: rubric.into(),
            threshold: 0.8,
        }
    }

    /// Score at or above which `passed` is true.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    fn system(&self) -> String {
        format!(
            "You grade work against a rubric. Reply with JSON only, in the form \
             {{\"score\": <0.0-1.0>, \"reason\": \"<one sentence>\"}}. \
             Grade each criterion independently and score the whole.\n\n\
             Rubric:\n{}",
            self.rubric
        )
    }

    /// Read a verdict out of a reply, tolerating the ways models wrap JSON.
    fn parse(&self, text: &str) -> Verdict {
        let json = extract_json(text);
        let score = json
            .as_ref()
            .and_then(|j| j.get("score"))
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);
        let reason = json
            .as_ref()
            .and_then(|j| j.get("reason"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or(text)
            .trim()
            .to_string();

        Verdict {
            score,
            passed: score >= self.threshold,
            reason,
        }
    }
}

impl Step for JudgeStep {
    fn config_hash(&self) -> CacheKey {
        CacheKey::from_parts(&[
            b"JudgeStep",
            self.model.as_bytes(),
            self.rubric.as_bytes(),
            &self.threshold.to_le_bytes(),
        ])
    }

    fn meta(&self) -> StepMeta {
        StepMeta::new("JudgeStep")
            .with_max_turns(2)
            .with_output_schema(Schema::json())
    }

    fn poll(&self, ctx: &StepCtx<'_>) -> Result<Transition> {
        match ctx.result() {
            None => {
                let subject = ctx
                    .input
                    .as_text()
                    .map(str::to_string)
                    .unwrap_or_else(|| ctx.input.to_plain_json().to_string());

                Ok(Transition::Await(vec![
                    LlmRequest::new(
                        &self.model,
                        vec![Message::user(format!("Grade this:\n\n{subject}"))].into(),
                    )
                    .with_system(self.system())
                    .into_effect(),
                ]))
            }
            Some(EffectResult::Llm(response)) => {
                let verdict = self.parse(&response.message.text());
                Ok(Transition::Done(Value::json(serde_json::json!({
                    "score": verdict.score,
                    // A loop reads this as its termination signal; the name
                    // is the one `read_loop_signal` recognises.
                    "done": verdict.passed,
                    "passed": verdict.passed,
                    "reason": verdict.reason,
                    // What was judged, echoed back. In a refine loop the
                    // verdict is what the next round reads, so dropping the
                    // artifact here would ask the worker to improve something
                    // it can no longer see.
                    "value": ctx.input.to_plain_json(),
                }))))
            }
            // A judge that cannot reach a model must not silently pass.
            Some(other) => Ok(Transition::Done(Value::json(serde_json::json!({
                "score": 0.0,
                "done": false,
                "passed": false,
                "reason": format!("judging failed: {other:?}"),
            })))),
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Pull a JSON object out of a reply that may be fenced or prefaced.
///
/// Models wrap JSON in ```json fences, or preface it with a sentence, often
/// enough that requiring a bare object would fail on working output.
fn extract_json(text: &str) -> Option<serde_json::Value> {
    if let Ok(value) = serde_json::from_str(text.trim()) {
        return Some(value);
    }
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    if end <= start {
        return None;
    }
    serde_json::from_str(&text[start..=end]).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use somatize_core::effect::{LlmResponse, Usage};

    fn reply(message: Message, stop: StopReason) -> EffectResult {
        EffectResult::Llm(LlmResponse {
            message,
            stop_reason: stop,
            usage: Usage::default(),
            model: None,
        })
    }

    // ── ReactStep ──

    #[test]
    fn the_first_turn_asks_the_model() {
        let step = ReactStep::new("kimi/kimi-k2").with_system("be terse");
        let input = Value::text("what is soma?");
        let ctx = StepCtx::new("n", "r", &input, 0);

        match step.poll(&ctx).unwrap() {
            Transition::Await(effects) => {
                let Effect::Llm(req) = &effects[0] else {
                    panic!("expected an llm effect")
                };
                assert_eq!(req.model, "kimi/kimi-k2");
                assert_eq!(req.system.as_deref(), Some("be terse"));
                // A bare prompt is promoted to a user turn.
                assert_eq!(req.messages.len(), 1);
                assert_eq!(req.messages.0[0].text(), "what is soma?");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_reply_without_tool_use_finishes() {
        let step = ReactStep::new("m").text_only();
        let input = Value::text("hi");
        let results = [reply(Message::assistant("hello"), StopReason::EndTurn)];
        let history = [results.to_vec()];
        let ctx = StepCtx::new("n", "r", &input, 1).with_history(&history);

        match step.poll(&ctx).unwrap() {
            Transition::Done(v) => assert_eq!(v.as_text(), Some("hello")),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_tool_request_becomes_tool_effects() {
        let step = ReactStep::new("m");
        let input = Value::text("weather?");
        let asked = Message::new(
            Role::Assistant,
            vec![
                ContentBlock::text("checking"),
                ContentBlock::tool_use("c1", "weather", serde_json::json!({"city": "Vigo"})),
                ContentBlock::tool_use("c2", "clock", serde_json::json!({})),
            ],
        );
        let results = [reply(asked, StopReason::ToolUse)];
        let ctx = StepCtx::new("n", "r", &input, 1).with_results(&results);

        match step.poll(&ctx).unwrap() {
            Transition::Await(effects) => {
                assert_eq!(effects.len(), 2);
                let Effect::Tool { name, args } = &effects[0] else {
                    panic!("expected a tool effect")
                };
                assert_eq!(name, "weather");
                assert_eq!(args.to_plain_json()["city"], "Vigo");
            }
            other => panic!("{other:?}"),
        }
    }

    /// `tool_use` with no calls is a real provider behaviour; finishing beats
    /// awaiting nothing and erroring out.
    #[test]
    fn a_tool_stop_with_no_calls_finishes() {
        let step = ReactStep::new("m").text_only();
        let input = Value::text("hi");
        let results = [reply(
            Message::assistant("done anyway"),
            StopReason::ToolUse,
        )];
        let ctx = StepCtx::new("n", "r", &input, 1).with_results(&results);
        assert!(matches!(step.poll(&ctx).unwrap(), Transition::Done(_)));
    }

    /// Tool results go back paired to the call that asked for them — the
    /// pairing a handoff most often loses.
    #[test]
    fn tool_results_are_paired_to_their_calls() {
        let mut conversation = Messages::new();
        conversation.push(Message::user("weather?"));
        conversation.push(Message::new(
            Role::Assistant,
            vec![
                ContentBlock::tool_use("c1", "weather", serde_json::json!({})),
                ContentBlock::tool_use("c2", "clock", serde_json::json!({})),
            ],
        ));

        replay_into(
            &mut conversation,
            &[
                EffectResult::Tool {
                    output: Value::text("sunny"),
                    is_error: false,
                },
                EffectResult::Tool {
                    output: Value::text("12:00"),
                    is_error: false,
                },
            ],
        );

        let last = conversation.last().unwrap();
        assert_eq!(last.role, Role::User);
        match (&last.content[0], &last.content[1]) {
            (
                ContentBlock::ToolResult {
                    tool_use_id: a,
                    content: ca,
                    ..
                },
                ContentBlock::ToolResult {
                    tool_use_id: b,
                    content: cb,
                    ..
                },
            ) => {
                assert_eq!((a.as_str(), ca.as_str()), ("c1", "sunny"));
                assert_eq!((b.as_str(), cb.as_str()), ("c2", "12:00"));
            }
            other => panic!("{other:?}"),
        }
    }

    /// A failed tool is reported to the model, not hidden — it can adapt.
    #[test]
    fn a_failed_tool_is_reported_to_the_model() {
        let mut conversation = Messages::new();
        conversation.push(Message::new(
            Role::Assistant,
            vec![ContentBlock::tool_use(
                "c1",
                "search",
                serde_json::json!({}),
            )],
        ));
        replay_into(
            &mut conversation,
            &[EffectResult::Failed {
                message: "rate limited".into(),
            }],
        );

        let last = conversation.last().unwrap();
        match &last.content[0] {
            ContentBlock::ToolResult {
                is_error, content, ..
            } => {
                assert!(is_error);
                assert!(content.contains("rate limited"));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn tool_results_send_the_conversation_back_to_the_model() {
        let step = ReactStep::new("m");
        let input = Value::text("hi");
        let results = [EffectResult::Tool {
            output: Value::text("sunny"),
            is_error: false,
        }];
        let ctx = StepCtx::new("n", "r", &input, 2).with_results(&results);
        assert!(matches!(step.poll(&ctx).unwrap(), Transition::Await(_)));
    }

    /// Changing the toolset changes the node's identity: the same prompt
    /// with different tools is a different computation.
    #[test]
    fn the_toolset_is_part_of_the_config_hash() {
        let bare = ReactStep::new("m");
        let armed = ReactStep::new("m").with_tools(vec![ToolSpec::no_args("search", "d")]);
        assert_ne!(bare.config_hash(), armed.config_hash());

        assert_ne!(
            ReactStep::new("m").with_system("a").config_hash(),
            ReactStep::new("m").with_system("b").config_hash()
        );
        assert_eq!(
            ReactStep::new("m").config_hash(),
            ReactStep::new("m").config_hash()
        );
    }

    /// The turn cap has to allow for the ask/answer pairs a ReAct round
    /// takes, or a tool-using agent would be cut off mid-loop.
    #[test]
    fn the_turn_cap_accounts_for_tool_rounds() {
        assert!(ReactStep::new("m").with_max_turns(3).meta().max_turns >= 7);
    }

    #[test]
    fn schemas_reflect_the_output_shape() {
        assert_eq!(
            ReactStep::new("m").meta().output_schema,
            Some(Schema::messages())
        );
        assert_eq!(
            ReactStep::new("m").text_only().meta().output_schema,
            Some(Schema::text())
        );
    }

    // ── JudgeStep ──

    #[test]
    fn a_judge_scores_and_signals_a_loop() {
        let judge = JudgeStep::new("m", "Has a numeric price column").with_threshold(0.7);
        let input = Value::text("the artifact");
        let results = [reply(
            Message::assistant(r#"{"score": 0.9, "reason": "price column present"}"#),
            StopReason::EndTurn,
        )];
        let ctx = StepCtx::new("n", "r", &input, 1).with_results(&results);

        match judge.poll(&ctx).unwrap() {
            Transition::Done(v) => {
                let json = v.to_plain_json();
                assert_eq!(json["score"], 0.9);
                assert_eq!(json["passed"], true);
                // `done` is the key `read_loop_signal` looks for, so a
                // refine loop terminates on a passing grade with no glue.
                assert_eq!(json["done"], true);
                assert_eq!(
                    somatize_core::control::read_loop_signal(&v),
                    Some(somatize_core::control::LoopSignal::Stop)
                );
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_failing_grade_keeps_a_loop_going() {
        let judge = JudgeStep::new("m", "r").with_threshold(0.8);
        let input = Value::text("x");
        let results = [reply(
            Message::assistant(r#"{"score": 0.3, "reason": "missing column"}"#),
            StopReason::EndTurn,
        )];
        let ctx = StepCtx::new("n", "r", &input, 1).with_results(&results);

        match judge.poll(&ctx).unwrap() {
            Transition::Done(v) => {
                assert_eq!(
                    somatize_core::control::read_loop_signal(&v),
                    Some(somatize_core::control::LoopSignal::Continue)
                );
            }
            other => panic!("{other:?}"),
        }
    }

    /// Models fence and preface their JSON; requiring a bare object would
    /// fail on output that is otherwise fine.
    #[test]
    fn a_verdict_survives_fences_and_prefaces() {
        let judge = JudgeStep::new("m", "r");
        for text in [
            r#"{"score": 0.5, "reason": "ok"}"#,
            "```json\n{\"score\": 0.5, \"reason\": \"ok\"}\n```",
            "Here is my grade:\n{\"score\": 0.5, \"reason\": \"ok\"}\nHope that helps.",
        ] {
            let v = judge.parse(text);
            assert_eq!(v.score, 0.5, "failed on: {text}");
            assert_eq!(v.reason, "ok", "failed on: {text}");
        }
    }

    /// An unreadable verdict scores zero rather than passing by accident.
    #[test]
    fn an_unreadable_verdict_does_not_pass() {
        let judge = JudgeStep::new("m", "r");
        let v = judge.parse("I think it's pretty good honestly");
        assert_eq!(v.score, 0.0);
        assert!(!v.passed);
    }

    #[test]
    fn scores_are_clamped() {
        let judge = JudgeStep::new("m", "r");
        assert_eq!(judge.parse(r#"{"score": 5}"#).score, 1.0);
        assert_eq!(judge.parse(r#"{"score": -2}"#).score, 0.0);
    }

    /// A judge that could not reach a model must fail closed.
    #[test]
    fn a_judge_that_cannot_grade_fails_closed() {
        let judge = JudgeStep::new("m", "r");
        let input = Value::text("x");
        let results = [EffectResult::Failed {
            message: "no provider".into(),
        }];
        let ctx = StepCtx::new("n", "r", &input, 1).with_results(&results);

        match judge.poll(&ctx).unwrap() {
            Transition::Done(v) => {
                let json = v.to_plain_json();
                assert_eq!(json["passed"], false);
                assert_eq!(json["score"], 0.0);
            }
            other => panic!("{other:?}"),
        }
    }

    // ── LlmStep ──

    #[test]
    fn a_single_call_step_returns_text() {
        let step = LlmStep::new("m").with_system("be brief");
        let input = Value::text("hi");
        let history = [vec![reply(
            Message::assistant("hello"),
            StopReason::EndTurn,
        )]];
        let ctx = StepCtx::new("n", "r", &input, 1).with_history(&history);

        assert_eq!(step.meta().name, "LlmStep");
        match step.poll(&ctx).unwrap() {
            Transition::Done(v) => assert_eq!(v.as_text(), Some("hello")),
            other => panic!("{other:?}"),
        }
    }
}
