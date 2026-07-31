//! Record-once, replay-forever — the durability half of effectful execution.
//!
//! A run that calls a model is not reproducible by re-running it. The answer
//! differs, so a crash halfway through means starting over, and an experiment
//! record of "what this agent did" cannot be checked against anything.
//!
//! Durable-execution engines solve this by journaling: perform each effect
//! once, write the result down, and on replay serve the recorded result
//! instead of performing it again. The orchestration is deterministic; only
//! the recorded edges are not.
//!
//! Soma already has the storage for that — the same two-table action store
//! the filter cache uses ([`ActionCache`] for small records, [`BlobStore`]
//! for payloads). What this module adds is the *keying*, and the distinction
//! the filter cache cannot express:
//!
//! | Effect kind | Key includes | Reused |
//! |---|---|---|
//! | Pure (a graph run) | the effect's content | across every run, like a filter |
//! | Impure (a model call, a tool) | run, node, turn, effect | only when replaying *that* run |
//!
//! That second row is the point. Asking a model the same question twice is
//! genuinely two events, so memoizing by content would freeze the first
//! answer forever — the `_deterministic = false` foot-gun wearing a hat.
//! Scoping the key to `(run, node, turn)` means a resumed run sees exactly
//! what the original saw, while a fresh run asks afresh.

use somatize_core::action::{ActionCache, ActionResult, BlobStore, ContentHash};
use somatize_core::cache::{CacheKey, Origin};
use somatize_core::effect::{Effect, EffectResult};
use somatize_core::error::{Result, SomaError};
use std::sync::Arc;

/// Where an effect happened, for keying its record.
#[derive(Debug, Clone, Copy)]
pub struct EffectSite<'a> {
    pub run_id: &'a str,
    pub node_id: &'a str,
    pub turn: usize,
    /// Position within the turn, since one turn may await several effects.
    pub index: usize,
}

/// Reads and writes effect results.
///
/// Cloning is cheap; the stores are shared.
#[derive(Clone)]
pub struct EffectJournal {
    actions: Arc<dyn ActionCache>,
    blobs: Arc<dyn BlobStore>,
    /// When false, nothing is written or read. Set per step by
    /// [`somatize_core::step::StepMeta::journal`], for work whose payloads
    /// must not reach disk.
    enabled: bool,
}

impl EffectJournal {
    pub fn new(actions: Arc<dyn ActionCache>, blobs: Arc<dyn BlobStore>) -> Self {
        Self {
            actions,
            blobs,
            enabled: true,
        }
    }

    /// A journal that records nothing. The run still works; it just cannot
    /// be replayed.
    pub fn disabled(actions: Arc<dyn ActionCache>, blobs: Arc<dyn BlobStore>) -> Self {
        Self {
            actions,
            blobs,
            enabled: false,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Toggle recording, e.g. per step.
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// The record key for an effect at a site.
    ///
    /// Pure effects key on content alone, so any run may reuse them. Impure
    /// effects additionally key on the site, which is what confines their
    /// reuse to a replay of the same run.
    pub fn key(&self, site: EffectSite<'_>, effect: &Effect) -> CacheKey {
        let effect_key = effect.cache_key();
        if effect.is_pure() {
            CacheKey::from_parts(&[b"soma-journal-v1", b"pure", &effect_key.0])
        } else {
            CacheKey::from_parts(&[
                b"soma-journal-v1",
                b"sited",
                site.run_id.as_bytes(),
                site.node_id.as_bytes(),
                &site.turn.to_le_bytes(),
                &site.index.to_le_bytes(),
                &effect_key.0,
            ])
        }
    }

    /// Fetch a recorded result, if there is one.
    ///
    /// A record whose blob has been evicted reads as absent: the effect is
    /// performed again. For a pure effect that is merely slower; for an
    /// impure one it means a replay diverges, which is why
    /// [`Self::record`] marks impure blobs as expensive so GC keeps them.
    pub fn lookup(&self, site: EffectSite<'_>, effect: &Effect) -> Result<Option<EffectResult>> {
        if !self.enabled {
            return Ok(None);
        }
        let key = self.key(site, effect);
        let Some(record) = self.actions.get_action(&key)? else {
            return Ok(None);
        };
        let Some(hash) = record.outputs.get("effect_result") else {
            return Ok(None);
        };
        let Some(bytes) = self.blobs.get_bytes(hash)? else {
            tracing::warn!(
                node = site.node_id,
                turn = site.turn,
                "journal record present but its blob is gone; performing the effect again"
            );
            return Ok(None);
        };
        let result: EffectResult = serde_json::from_slice(&bytes)
            .map_err(|e| SomaError::Cache(format!("journal: decoding effect result: {e}")))?;
        Ok(Some(result))
    }

    /// Write down what an effect produced.
    pub fn record(
        &self,
        site: EffectSite<'_>,
        effect: &Effect,
        result: &EffectResult,
        compute_ms: u64,
    ) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        // A failure is not worth pinning: on replay we would rather retry it
        // than faithfully reproduce the outage that caused it.
        if matches!(result, EffectResult::Failed { .. }) {
            return Ok(());
        }

        let bytes = serde_json::to_vec(result)
            .map_err(|e| SomaError::Cache(format!("journal: encoding effect result: {e}")))?;
        let hash: ContentHash = self.blobs.put_bytes(&bytes)?;

        let now = chrono::Utc::now();
        let record = ActionResult {
            key: self.key(site, effect),
            outputs: [("effect_result".to_string(), hash)].into_iter().collect(),
            output_bytes: bytes.len() as u64,
            compute_ms,
            // An impure effect's record is the *only* copy of what happened.
            // Marking it non-deterministic tells GC and any future reader
            // that recomputing it would not reproduce this value.
            deterministic: effect.is_pure(),
            origin: Origin::Computed {
                node_id: site.node_id.to_string(),
                run_id: site.run_id.to_string(),
            },
            created_at: now,
            last_accessed: now,
        };
        self.actions.put_action(&record)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::fs_store::FsActionStore;
    use somatize_core::effect::{LlmRequest, LlmResponse, StopReason, Usage};
    use somatize_core::message::Message;
    use somatize_core::value::Value;

    fn store() -> (Arc<FsActionStore>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(FsActionStore::new(dir.path()).unwrap());
        (store, dir)
    }

    fn journal(store: Arc<FsActionStore>) -> EffectJournal {
        EffectJournal::new(store.clone(), store)
    }

    fn site<'a>(run: &'a str, node: &'a str, turn: usize) -> EffectSite<'a> {
        EffectSite {
            run_id: run,
            node_id: node,
            turn,
            index: 0,
        }
    }

    fn llm(prompt: &str) -> Effect {
        Effect::Llm(LlmRequest::new(
            "claude-opus-5",
            vec![Message::user(prompt)].into(),
        ))
    }

    fn reply(text: &str) -> EffectResult {
        EffectResult::Llm(LlmResponse {
            message: Message::assistant(text),
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
            model: None,
        })
    }

    #[test]
    fn records_then_replays() {
        let (s, _d) = store();
        let j = journal(s);
        let effect = llm("hello");

        assert!(j.lookup(site("r1", "n", 0), &effect).unwrap().is_none());

        j.record(site("r1", "n", 0), &effect, &reply("hi"), 1200)
            .unwrap();

        let got = j.lookup(site("r1", "n", 0), &effect).unwrap().unwrap();
        match got {
            EffectResult::Llm(r) => assert_eq!(r.message.text(), "hi"),
            other => panic!("wrong result: {other:?}"),
        }
    }

    /// The load-bearing property: a *different* run asking the identical
    /// question must not be served the first run's answer.
    #[test]
    fn impure_effects_do_not_leak_across_runs() {
        let (s, _d) = store();
        let j = journal(s);
        let effect = llm("hello");

        j.record(site("r1", "n", 0), &effect, &reply("hi"), 10)
            .unwrap();

        assert!(
            j.lookup(site("r2", "n", 0), &effect).unwrap().is_none(),
            "a fresh run reused a recorded model answer"
        );
        assert!(
            j.lookup(site("r1", "other", 0), &effect).unwrap().is_none(),
            "another node reused a recorded model answer"
        );
        assert!(
            j.lookup(site("r1", "n", 1), &effect).unwrap().is_none(),
            "a later turn reused an earlier turn's answer"
        );
    }

    /// Two effects in the same turn are distinct records, or a parallel
    /// fan-out would replay one answer for all of them.
    #[test]
    fn effects_within_a_turn_are_distinct() {
        let (s, _d) = store();
        let j = journal(s);
        let effect = llm("hello");

        let first = EffectSite {
            run_id: "r",
            node_id: "n",
            turn: 0,
            index: 0,
        };
        let second = EffectSite { index: 1, ..first };

        j.record(first, &effect, &reply("one"), 1).unwrap();
        assert!(j.lookup(second, &effect).unwrap().is_none());
    }

    /// Pure effects are ordinary content-addressed cache entries: any run,
    /// any node, same answer.
    #[test]
    fn pure_effects_are_shared() {
        let (s, _d) = store();
        let j = journal(s);
        let effect = Effect::Graph {
            graph: Box::new(somatize_core::graph::Graph::new()),
            input: Value::tensor(vec![1.0], vec![1]),
            mode: somatize_core::effect::GraphEffectMode::Forward,
        };
        assert!(effect.is_pure());

        j.record(
            site("r1", "n", 0),
            &effect,
            &EffectResult::Graph(Value::tensor(vec![2.0], vec![1])),
            5,
        )
        .unwrap();

        let got = j.lookup(site("r2", "elsewhere", 7), &effect).unwrap();
        assert!(got.is_some(), "a pure effect should be reusable anywhere");
    }

    /// Different questions must not collide, however alike their sites.
    #[test]
    fn different_effects_at_the_same_site_differ() {
        let (s, _d) = store();
        let j = journal(s);

        j.record(site("r", "n", 0), &llm("first"), &reply("A"), 1)
            .unwrap();
        assert!(
            j.lookup(site("r", "n", 0), &llm("second"))
                .unwrap()
                .is_none(),
            "two different prompts shared a journal record"
        );
    }

    /// A step that opts out leaves no trace on disk — the escape hatch for
    /// prompts that must not be persisted.
    #[test]
    fn a_disabled_journal_records_nothing() {
        let (s, _d) = store();
        let j = journal(s).with_enabled(false);
        let effect = llm("something sensitive");

        j.record(site("r", "n", 0), &effect, &reply("x"), 1)
            .unwrap();
        assert!(j.lookup(site("r", "n", 0), &effect).unwrap().is_none());
    }

    /// Failures are retried on replay, not faithfully reproduced.
    #[test]
    fn failures_are_not_recorded() {
        let (s, _d) = store();
        let j = journal(s);
        let effect = llm("hello");

        j.record(
            site("r", "n", 0),
            &effect,
            &EffectResult::Failed {
                message: "connection reset".into(),
            },
            1,
        )
        .unwrap();

        assert!(j.lookup(site("r", "n", 0), &effect).unwrap().is_none());
    }
}
