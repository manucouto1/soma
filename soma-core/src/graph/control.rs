//! Data-dependent control flow: how a loop decides to stop and how a branch
//! picks an arm.
//!
//! Both decisions read a node's output `Value`. The contract lives here, in
//! one place, because the compiler resolves it and the executor applies it —
//! and because it is the surface a Python filter has to satisfy.
//!
//! The rule throughout: **an unreadable signal is an error, never a default.**
//! Guessing (continue looping, take the first arm) turns a typo into a silent
//! wrong answer that surfaces hours later as a bad result rather than a
//! stack trace.

use crate::data::value::Value;
use crate::graph::NodeId;
use serde::{Deserialize, Serialize};

/// What ends a loop.
///
/// The compiler resolves this to a concrete form before the executor sees it,
/// so at runtime there is never a question of *which* node decides.
// Adjacently tagged, not internally: serde cannot put an internal tag on a
// newtype variant wrapping a string, so `#[serde(tag = "type")]` made every
// graph containing a resolved loop unserializable — including the
// `graph.json` snapshot the experiment pool writes for it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "node")]
#[non_exhaustive]
pub enum LoopCondition {
    /// Resolve at compile time to the body's single terminal node — the one
    /// no other body node depends on. Compilation fails if the body has more
    /// than one, rather than picking whichever finished last.
    ///
    /// This is the default for `Node::loop_node`.
    #[default]
    BodyTerminal,

    /// Stop when the named node's output signals completion.
    WhenSignaled(NodeId),

    /// Ignore signals; run the body exactly `max_iterations` times.
    Exhaust,
}

/// A loop body's verdict on whether to go round again.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopSignal {
    /// Run the body again.
    Continue,
    /// The loop is done; do not run the body again.
    Stop,
}

/// Read a termination signal out of a node's output.
///
/// Recognized as **stop**: `true`, `"done"`, `"stop"`, `{"done": true}`,
/// and `Value::Empty` (nothing left to produce).
/// Recognized as **continue**: `false`, `{"done": false}`.
///
/// Returns `None` for anything else — including tensors. A body that only
/// produces tensors cannot express termination, and the honest response is to
/// say so rather than silently run to `max_iterations`.
pub fn read_loop_signal(value: &Value) -> Option<LoopSignal> {
    use LoopSignal::{Continue, Stop};

    match value {
        Value::Empty => Some(Stop),
        Value::Text(s) => match s.as_ref() {
            "done" | "stop" => Some(Stop),
            "continue" => Some(Continue),
            _ => None,
        },
        Value::Json(j) => {
            if let Some(b) = j.as_bool() {
                return Some(if b { Stop } else { Continue });
            }
            if let Some(s) = j.as_str() {
                return match s {
                    "done" | "stop" => Some(Stop),
                    "continue" => Some(Continue),
                    _ => None,
                };
            }
            j.get("done")
                .and_then(|d| d.as_bool())
                .map(|b| if b { Stop } else { Continue })
        }
        _ => None,
    }
}

/// Read the arm label a branch condition selected.
///
/// Recognized: a string (`"billing"`), a bool (`"true"` / `"false"`), or an
/// object with a `"branch"` field. Returns `None` for anything else, so the
/// executor can report an unusable condition instead of running arm zero.
pub fn read_arm_selector(value: &Value) -> Option<String> {
    match value {
        Value::Text(s) => Some(s.to_string()),
        Value::Json(j) => j
            .as_str()
            .map(String::from)
            .or_else(|| j.as_bool().map(|b| b.to_string()))
            .or_else(|| j.get("branch").and_then(|b| b.as_str()).map(String::from)),
        _ => None,
    }
}

/// Labels treated as the catch-all arm when no label matches the selector.
pub const DEFAULT_ARM_LABELS: [&str; 2] = ["default", "else"];

/// Is this label a catch-all arm?
pub fn is_default_arm(label: &str) -> bool {
    DEFAULT_ARM_LABELS.contains(&label)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every variant has to survive JSON, or the graph containing it cannot
    /// be written to a run directory, sent to a worker, or read by anything
    /// outside this process. `WhenSignaled` is the one that broke: an
    /// internal tag cannot be applied to a newtype variant wrapping a string.
    #[test]
    fn every_condition_round_trips_through_json() {
        for condition in [
            LoopCondition::BodyTerminal,
            LoopCondition::WhenSignaled("critic".into()),
            LoopCondition::Exhaust,
        ] {
            let text = serde_json::to_string(&condition).expect("serializable");
            let back: LoopCondition = serde_json::from_str(&text).expect("readable");
            assert_eq!(back, condition, "{text}");
        }
    }

    #[test]
    fn stop_signals() {
        for v in [
            Value::Empty,
            Value::json(serde_json::json!(true)),
            Value::json(serde_json::json!("done")),
            Value::json(serde_json::json!("stop")),
            Value::json(serde_json::json!({"done": true})),
            Value::text("done"),
            Value::text("stop"),
        ] {
            assert_eq!(read_loop_signal(&v), Some(LoopSignal::Stop), "{v:?}");
        }
    }

    #[test]
    fn continue_signals() {
        for v in [
            Value::json(serde_json::json!(false)),
            Value::json(serde_json::json!({"done": false})),
            Value::text("continue"),
        ] {
            assert_eq!(read_loop_signal(&v), Some(LoopSignal::Continue), "{v:?}");
        }
    }

    /// A tensor carries no termination signal. Reporting that is the whole
    /// point — the old executor read `_ => false` and burned 100 iterations.
    #[test]
    fn tensors_are_not_signals() {
        assert_eq!(read_loop_signal(&Value::tensor(vec![1.0], vec![1])), None);
        assert_eq!(read_loop_signal(&Value::json(serde_json::json!(42))), None);
        assert_eq!(
            read_loop_signal(&Value::json(serde_json::json!({"score": 0.9}))),
            None
        );
    }

    #[test]
    fn arm_selectors() {
        assert_eq!(
            read_arm_selector(&Value::json(serde_json::json!("billing"))),
            Some("billing".into())
        );
        assert_eq!(
            read_arm_selector(&Value::json(serde_json::json!(true))),
            Some("true".into())
        );
        assert_eq!(
            read_arm_selector(&Value::json(serde_json::json!({"branch": "retry"}))),
            Some("retry".into())
        );
        assert_eq!(read_arm_selector(&Value::text("tech")), Some("tech".into()));
    }

    /// No selector must mean "error", not "arm zero".
    #[test]
    fn unusable_selectors_are_none() {
        assert_eq!(read_arm_selector(&Value::tensor(vec![1.0], vec![1])), None);
        assert_eq!(read_arm_selector(&Value::Empty), None);
        assert_eq!(
            read_arm_selector(&Value::json(serde_json::json!({"score": 1}))),
            None
        );
    }

    #[test]
    fn default_arm_labels() {
        assert!(is_default_arm("default"));
        assert!(is_default_arm("else"));
        assert!(!is_default_arm("billing"));
    }
}
