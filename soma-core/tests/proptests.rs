//! Property-based tests for the tracking/study schema: serde
//! roundtrips must be lossless for arbitrary values, and the composite
//! objective must satisfy its algebraic invariants.

use chrono::{DateTime, TimeZone, Utc};
use proptest::prelude::*;
use somatize_core::event::{Event, MetricRecord};
use somatize_core::study::{CompositeObjective, Direction, Scalarizer, Trial, TrialState};
use somatize_core::tracking::EventEnvelope;
use std::collections::HashMap;

fn arb_timestamp() -> impl Strategy<Value = DateTime<Utc>> {
    // Bounded, second-precision timestamps roundtrip exactly through
    // RFC3339 (sub-second precision is preserved too, but this keeps
    // shrinking readable).
    (0i64..4_000_000_000).prop_map(|secs| Utc.timestamp_opt(secs, 0).unwrap())
}

fn arb_metric() -> impl Strategy<Value = MetricRecord> {
    // Any finite f64 must roundtrip exactly: this property is what
    // forced enabling serde_json's `float_roundtrip` feature (the
    // default parser loses ULPs), which matters because events.jsonl
    // is the canonical store.
    (
        "[a-z_]{1,12}",
        prop_oneof![prop::num::f64::NORMAL, Just(0.0)],
        0usize..10_000,
        arb_timestamp(),
    )
        .prop_map(|(name, value, step, timestamp)| MetricRecord {
            name,
            value,
            step,
            timestamp,
        })
}

fn arb_level5_event() -> impl Strategy<Value = Event> {
    let run_id = "[a-z0-9_]{1,16}";
    prop_oneof![
        (run_id, 0usize..100, prop::option::of(1usize..100)).prop_map(
            |(run_id, epoch, total_epochs)| Event::EpochStarted {
                run_id,
                epoch,
                total_epochs,
            }
        ),
        (
            run_id,
            0usize..100,
            prop::collection::vec(arb_metric(), 0..4)
        )
            .prop_map(|(run_id, epoch, metrics)| Event::EpochCompleted {
                run_id,
                epoch,
                metrics,
            }),
        (run_id, 0usize..10_000, prop::option::of(0usize..100)).prop_map(
            |(run_id, step, epoch)| Event::StepCompleted {
                run_id,
                step,
                epoch,
            }
        ),
        (
            run_id,
            arb_metric(),
            prop::option::of("[a-z]{1,8}".prop_map(String::from)),
            prop::option::of("trial_[0-9]{4}".prop_map(String::from)),
        )
            .prop_map(
                |(run_id, metric, node_id, trial_id)| Event::MetricReported {
                    run_id,
                    metric,
                    node_id,
                    trial_id,
                }
            ),
        (
            run_id,
            "[a-z]{1,8}",
            0usize..1000,
            "[A-Z_()0-9]{1,24}",
            ".{0,64}"
        )
            .prop_map(|(run_id, node_id, step, flag, detail)| Event::HealthFlag {
                run_id,
                node_id,
                step,
                flag,
                detail,
            }),
    ]
}

fn arb_completed_trial(metric_names: Vec<String>) -> impl Strategy<Value = Trial> {
    // Bounded values: the algebraic invariants below are about
    // linearity, not float overflow (1e308 × weight → inf).
    prop::collection::vec(-1e12f64..1e12, metric_names.len()).prop_map(move |values| {
        let mut t = Trial::new("t", HashMap::new());
        t.state = TrialState::Completed;
        for (name, value) in metric_names.iter().zip(values) {
            t.metrics.push(MetricRecord {
                name: name.clone(),
                value,
                step: 0,
                timestamp: Utc::now(),
            });
        }
        t
    })
}

proptest! {
    #[test]
    fn level5_events_roundtrip(event in arb_level5_event()) {
        let json = serde_json::to_string(&event).unwrap();
        let back: Event = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(
            serde_json::to_value(&back).unwrap(),
            serde_json::to_value(&event).unwrap()
        );
    }

    #[test]
    fn envelope_preserves_seq_and_ts(
        seq in any::<u64>(),
        ts in arb_timestamp(),
        event in arb_level5_event(),
    ) {
        let envelope = EventEnvelope { seq, ts, event };
        let json = serde_json::to_string(&envelope).unwrap();
        let back: EventEnvelope = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(back.seq, seq);
        prop_assert_eq!(back.ts, ts);
    }

    /// WeightedSum is linear: scaling every weight by k scales the
    /// composite by k.
    #[test]
    fn weighted_sum_is_linear_in_weights(
        weights in prop::collection::vec(-10.0f64..10.0, 1..4),
        k in 0.5f64..4.0,
        trial in arb_completed_trial(vec!["m0".into(), "m1".into(), "m2".into()]),
    ) {
        let terms: Vec<(String, f64)> = weights
            .iter()
            .enumerate()
            .map(|(i, w)| (format!("m{i}"), *w))
            .collect();
        let base = CompositeObjective {
            terms: terms.clone(),
            direction: Direction::Maximize,
            scalarizer: Scalarizer::WeightedSum,
        };
        let scaled = CompositeObjective {
            terms: terms.into_iter().map(|(n, w)| (n, w * k)).collect(),
            direction: Direction::Maximize,
            scalarizer: Scalarizer::WeightedSum,
        };
        let a = base.evaluate(&trial).unwrap();
        let b = scaled.evaluate(&trial).unwrap();
        prop_assert!((b - k * a).abs() <= 1e-9 * (1.0 + a.abs().max(b.abs())));
    }

    /// evaluate() is Some iff terms is non-empty and every term's
    /// metric exists on the trial.
    #[test]
    fn composite_someness_invariant(
        present in prop::collection::vec("[ab]", 0..3),
        trial in arb_completed_trial(vec!["a".into(), "b".into()]),
        use_missing in any::<bool>(),
    ) {
        let mut terms: Vec<(String, f64)> =
            present.iter().map(|n| (n.clone(), 1.0)).collect();
        if use_missing {
            terms.push(("nope".into(), 1.0));
        }
        let composite = CompositeObjective {
            terms: terms.clone(),
            direction: Direction::Maximize,
            scalarizer: Scalarizer::WeightedSum,
        };
        let expect_some = !terms.is_empty() && !use_missing;
        prop_assert_eq!(composite.evaluate(&trial).is_some(), expect_some);
    }

    /// normalize is an order isomorphism: it preserves order for
    /// Maximize and reverses it for Minimize.
    #[test]
    fn normalize_is_order_isomorphism(a in prop::num::f64::NORMAL, b in prop::num::f64::NORMAL) {
        prop_assert_eq!(
            a < b,
            Direction::Maximize.normalize(a) < Direction::Maximize.normalize(b)
        );
        prop_assert_eq!(
            a < b,
            Direction::Minimize.normalize(a) > Direction::Minimize.normalize(b)
        );
    }
}
