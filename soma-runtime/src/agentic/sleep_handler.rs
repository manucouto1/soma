//! Waiting, as an effect.

use somatize_core::agentic::effect::{Effect, EffectHandler, EffectResult};
use somatize_core::data::value::Value;
use somatize_core::error::Result;

/// Performs [`Effect::Sleep`].
///
/// `Effect::Sleep` was declared, documented as "journaled like anything
/// else, so a replay does not sleep again", and handled by nobody: a step
/// that awaited one got "no handler for effect `sleep:1ms`". The variant
/// existed; the four lines that make it work did not.
///
/// And the journaling is the point. A step that backs off for thirty
/// seconds pays that once — a replay reads the recorded result and moves
/// on, which is what makes a resumed agentic run cheap.
#[derive(Debug, Default, Clone, Copy)]
pub struct SleepHandler;

impl EffectHandler for SleepHandler {
    fn handles(&self, effect: &Effect) -> bool {
        matches!(effect, Effect::Sleep(_))
    }

    fn perform(&self, effect: &Effect) -> Result<EffectResult> {
        let Effect::Sleep(duration) = effect else {
            return Err(somatize_core::error::SomaError::Other(
                "not a sleep effect".into(),
            ));
        };
        std::thread::sleep(*duration);
        Ok(EffectResult::Node(Value::Empty))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_sleeps_and_claims_only_sleeps() {
        let h = SleepHandler;
        let effect = Effect::Sleep(std::time::Duration::from_millis(5));
        assert!(h.handles(&effect));
        assert!(!h.handles(&Effect::Tool {
            name: "t".into(),
            args: Value::Empty
        }));

        let start = std::time::Instant::now();
        h.perform(&effect).unwrap();
        assert!(start.elapsed() >= std::time::Duration::from_millis(5));
    }
}
