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
    pub fn key(&self, site: EffectSite<'_>, effect: &Effect) -> Result<CacheKey> {
        let effect_key = effect.cache_key()?;
        Ok(if effect.is_pure() {
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
        })
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
        let key = self.key(site, effect)?;
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
            key: self.key(site, effect)?,
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

    /// Delete every file under `dir`, keeping the directory tree — what GC
    /// eviction does to CAS blobs (records are retained, blobs go).
    fn delete_files_under(dir: &std::path::Path) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                delete_files_under(&path);
            } else {
                std::fs::remove_file(&path).unwrap();
            }
        }
    }

    /// An action record whose blob was evicted must read as *absent*, so
    /// the effect is performed again — not as an error, and never as a
    /// half-answer. This is the fallback in `lookup`; without it a GC pass
    /// over the shared store would turn every replay into a decode failure.
    #[test]
    fn an_evicted_blob_reads_as_absent() {
        let (s, dir) = store();
        let j = journal(s);
        let effect = llm("hello");

        j.record(site("r", "n", 0), &effect, &reply("hi"), 1)
            .unwrap();
        assert!(j.lookup(site("r", "n", 0), &effect).unwrap().is_some());

        // Evict the blob; the action record stays where it is.
        delete_files_under(&dir.path().join("cas"));

        assert!(
            j.lookup(site("r", "n", 0), &effect).unwrap().is_none(),
            "a record without its blob must be treated as a miss"
        );
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

    // ── Keying properties ──
    //
    // The key is the whole safety story: a pure record shared where it must
    // not be, or two sites colliding, silently serves one run another run's
    // model answer. These pin the key function itself, over arbitrary sites.
    mod keying {
        use super::*;
        use proptest::prelude::*;

        fn pure_effect() -> Effect {
            Effect::Graph {
                graph: Box::new(somatize_core::graph::Graph::new()),
                input: Value::tensor(vec![1.0], vec![1]),
                mode: somatize_core::effect::GraphEffectMode::Forward,
            }
        }

        fn journal() -> (EffectJournal, tempfile::TempDir) {
            let (s, d) = store();
            (super::journal(s), d)
        }

        proptest! {
            /// The `pure`/`sited` namespaces are disjoint: whatever the
            /// site, a content-keyed record can never shadow a site-keyed
            /// one, or vice versa.
            #[test]
            fn pure_and_sited_keys_never_collide(
                run in "[a-z0-9]{0,8}",
                node in "[a-z0-9/]{0,8}",
                turn in 0usize..64,
                index in 0usize..8,
            ) {
                let (j, _d) = journal();
                let s = EffectSite { run_id: &run, node_id: &node, turn, index };
                prop_assert_ne!(j.key(s, &llm("q")).unwrap(), j.key(s, &pure_effect()).unwrap());
            }

            /// A sited key is a function of exactly (site, effect): change
            /// any site component and the key changes; change nothing and
            /// it is bit-identical. The first half is what confines an
            /// impure record to its own run/node/turn/index; the second is
            /// what lets a replay find it at all.
            #[test]
            fn a_sited_key_is_exactly_its_site_and_effect(
                a in ("[a-z]{0,4}", "[a-z]{0,4}", 0usize..4, 0usize..4),
                b in ("[a-z]{0,4}", "[a-z]{0,4}", 0usize..4, 0usize..4),
            ) {
                let (j, _d) = journal();
                let sa = EffectSite { run_id: &a.0, node_id: &a.1, turn: a.2, index: a.3 };
                let sb = EffectSite { run_id: &b.0, node_id: &b.1, turn: b.2, index: b.3 };
                let effect = llm("same question");

                prop_assert_eq!(j.key(sa, &effect).unwrap(), j.key(sa, &effect).unwrap());
                if a == b {
                    prop_assert_eq!(j.key(sa, &effect).unwrap(), j.key(sb, &effect).unwrap());
                } else {
                    prop_assert_ne!(j.key(sa, &effect).unwrap(), j.key(sb, &effect).unwrap());
                }
            }

            /// Same site, different questions: distinct keys, always.
            #[test]
            fn a_sited_key_separates_effects(
                run in "[a-z]{0,6}",
                node in "[a-z]{0,6}",
                turn in 0usize..8,
            ) {
                let (j, _d) = journal();
                let s = EffectSite { run_id: &run, node_id: &node, turn, index: 0 };
                prop_assert_ne!(j.key(s, &llm("one")).unwrap(), j.key(s, &llm("two")).unwrap());
            }
        }

        /// The classic concatenation collision: `("ab", "c")` and
        /// `("a", "bc")` flatten to the same bytes unless each part is
        /// length-prefixed. `CacheKey::from_parts` prefixes; this is the
        /// regression test that notices if that ever changes.
        #[test]
        fn adjacent_site_fields_do_not_blur_together() {
            let (j, _d) = journal();
            let effect = llm("q");
            let key = |run: &str, node: &str| {
                j.key(
                    EffectSite {
                        run_id: run,
                        node_id: node,
                        turn: 0,
                        index: 0,
                    },
                    &effect,
                )
                .unwrap()
            };
            assert_ne!(
                key("ab", "c"),
                key("a", "bc"),
                "site fields concatenated without length prefixes"
            );
        }
    }
}
