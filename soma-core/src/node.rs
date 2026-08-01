//! What the compiler and executor need to know about a node, whichever
//! kind it is.
//!
//! A graph has two sorts of node. A [`Filter`](crate::filter::Filter) is a
//! function: same config, same state, same input, same output, which is
//! what makes content-addressed caching sound. A [`Step`](crate::step::Step)
//! calls models, reads the world and may pause for a person.
//!
//! That difference matters to *them*. It should not reach the machinery
//! around them: resolving inputs from predecessors, catching a panic,
//! emitting a start and a completion, deciding whether an output may be
//! cached — none of it depends on which kind ran. So the two metadata
//! types collapse into one here, and the distinction survives as **data**
//! on it rather than as a second code path.
//!
//! The load-bearing part is [`NodeMeta::cacheable`] and
//! [`NodeMeta::deterministic`]: `From<StepMeta>` sets both to `false`, so
//! the cache guard the executor already runs skips a step without anyone
//! writing `if is_step`.

use crate::effect::SuspendReason;
use crate::filter::{Distribution, FilterKind, FilterMeta};
use crate::graph::NodeId;
use crate::schema::Schema;
use crate::step::StepMeta;
use crate::value::Value;
use serde::{Deserialize, Serialize};

/// How a node finished.
///
/// The three ways execution can leave a node, whichever kind it was. A
/// filter only ever produces; a step can also hand control on or stop and
/// wait. The runtime used to carry this as `StepOutcome` and translate it
/// three times on the way out — into control flow, then into an error
/// with the reason flattened to a JSON string that nothing parsed back.
///
/// Deliberately *not* `#[non_exhaustive]`, against the convention for
/// public enums here. Every consumer of this type is deciding control
/// flow, and a wildcard arm in that position is a silent wrong answer —
/// the pattern that let an unhandled plan variant become a successful
/// no-op. If a fourth way to finish is ever added, the places that must
/// think about it should stop compiling.
#[derive(Debug)]
pub enum NodeOutcome {
    /// A value, which the node's successors read as its output.
    Produced(Value),

    /// Control passes to another node, carrying a value.
    ///
    /// The carry is stored under the *handing* node, so the target reads
    /// it as an ordinary predecessor output rather than through a special
    /// path.
    HandOff { target: NodeId, carry: Value },

    /// The run stopped, pending something outside it. `turn` is where to
    /// deliver the answer when resuming.
    Paused { turn: usize, reason: SuspendReason },
}

/// A node's contract, independent of whether it computes or acts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMeta {
    /// The name/type identifier of the implementation behind this node.
    pub name: String,

    /// Does it reach outside the graph — models, tools, people?
    ///
    /// The one thing downstream genuinely needs to tell apart: an
    /// effectful node is journaled rather than memoized, and it is
    /// reported as itself in events instead of borrowing a
    /// [`FilterKind`].
    pub effectful: bool,

    /// Classification of a computational node's behaviour.
    ///
    /// [`FilterKind::Opaque`] for an effectful node, which has no trained
    /// state to speak of.
    pub kind: FilterKind,

    /// May this node's output be cached?
    pub cacheable: bool,

    /// Same inputs, same output? See [`FilterMeta::deterministic`].
    pub deterministic: bool,

    /// Does `forward()` keep a differentiable graph?
    pub differentiable: bool,

    /// Where it may run.
    pub distribution: Distribution,

    /// What it accepts (`None` = anything).
    pub input_schema: Option<Schema>,

    /// What it produces (`None` = unknown).
    pub output_schema: Option<Schema>,
}

impl NodeMeta {
    /// Is there trained state to learn and cache?
    pub fn trainable(&self) -> bool {
        !self.effectful && self.kind == FilterKind::Trainable
    }
}

impl From<FilterMeta> for NodeMeta {
    fn from(m: FilterMeta) -> Self {
        Self {
            name: m.name,
            effectful: false,
            kind: m.kind,
            cacheable: m.cacheable,
            deterministic: m.deterministic,
            differentiable: m.differentiable,
            distribution: m.distribution,
            input_schema: m.input_schema,
            output_schema: m.output_schema,
        }
    }
}

impl From<StepMeta> for NodeMeta {
    fn from(m: StepMeta) -> Self {
        Self {
            name: m.name,
            effectful: true,
            // A step has no state to fit, so nothing about it is trainable
            // or differentiable.
            kind: FilterKind::Opaque,
            // The two fields that replace an `if is_step` in the executor.
            // A step's output is not a function of its input — the model
            // is on the other end of it — so serving a recorded one would
            // be a lie. Its *effects* are still journaled, which is the
            // replay mechanism that actually fits an effectful node.
            cacheable: false,
            deterministic: false,
            differentiable: false,
            distribution: m.distribution,
            input_schema: m.input_schema,
            output_schema: m.output_schema,
        }
    }
}

impl NodeMeta {
    /// The computational half, for consumers that only speak `FilterMeta`.
    ///
    /// Lossy by design: an effectful node has no honest `FilterMeta`, and
    /// callers should ask [`NodeMeta::effectful`] before reaching for this.
    pub fn as_filter_meta(&self) -> FilterMeta {
        FilterMeta {
            name: self.name.clone(),
            kind: self.kind,
            cacheable: self.cacheable,
            differentiable: self.differentiable,
            deterministic: self.deterministic,
            stream_mode: crate::filter::StreamMode::FixedState,
            distribution: self.distribution.clone(),
            input_schema: self.input_schema.clone(),
            output_schema: self.output_schema.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filter_meta() -> FilterMeta {
        FilterMeta {
            name: "Scaler".into(),
            kind: FilterKind::Trainable,
            cacheable: true,
            differentiable: false,
            deterministic: true,
            stream_mode: crate::filter::StreamMode::FixedState,
            distribution: Distribution::Local,
            input_schema: None,
            output_schema: None,
        }
    }

    #[test]
    fn a_filter_keeps_its_caching_contract() {
        let meta = NodeMeta::from(filter_meta());
        assert!(!meta.effectful);
        assert!(meta.cacheable);
        assert!(meta.deterministic);
        assert!(meta.trainable());
    }

    /// The whole point of the type: "a step is not output-cacheable" is a
    /// pair of fields, so the executor's existing guard handles it and no
    /// one writes `if is_step`.
    #[test]
    fn a_step_is_not_output_cacheable() {
        let meta = NodeMeta::from(StepMeta::new("ReactStep"));
        assert!(meta.effectful);
        assert!(!meta.cacheable);
        assert!(!meta.deterministic);
        assert!(!meta.differentiable);
        assert!(!meta.trainable());
    }

    #[test]
    fn schemas_survive_both_directions() {
        let mut sm = StepMeta::new("Judge");
        sm.input_schema = Some(Schema::text());
        sm.output_schema = Some(Schema::messages());
        let meta = NodeMeta::from(sm);
        assert_eq!(meta.input_schema, Some(Schema::text()));
        assert_eq!(meta.output_schema, Some(Schema::messages()));
    }
}
