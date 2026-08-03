//! Reserved keys in a run's output store.
//!
//! A run threads node outputs through one `HashMap<String, Value>`, and a
//! few entries in it are not node outputs: the graph's input, a specific
//! node's input, and the state a trainable node learned. They are
//! distinguished by a `__` prefix.
//!
//! The prefixes used to be spelled inline — written with `format!` in the
//! runner and read back with `strip_prefix` in three different crates. This
//! module is the only place that knows how the key is spelled, so a change
//! to it cannot leave one reader behind.
//!
//! The prefix is a convention, not a guarantee: a node whose id is literally
//! `__state_x` would collide. [`is_reserved`] is what a caller uses to keep
//! these out of a list of node outputs.

/// Prefix for a state a trainable node learned during a fit.
const STATE: &str = "__state_";
/// Prefix for the input handed to a specific node.
const INPUT: &str = "__input_";
/// The graph's own input, available to every root node.
pub const GRAPH_INPUT: &str = "__input__";

/// Where the state fitted for `node_id` is stored.
pub fn state_key(node_id: &str) -> String {
    format!("{STATE}{node_id}")
}

/// Where the input handed to `node_id` is stored.
pub fn input_key(node_id: &str) -> String {
    format!("{INPUT}{node_id}")
}

/// The node whose state this key holds, or `None` if it is not a state key.
pub fn node_of_state_key(key: &str) -> Option<&str> {
    key.strip_prefix(STATE)
}

/// Does this key hold an input rather than an output or a state?
pub fn is_input_key(key: &str) -> bool {
    key == GRAPH_INPUT || key.starts_with(INPUT)
}

/// Is this key one of the run's own entries rather than a node's output?
///
/// Callers that report "what did each node produce" filter with this;
/// without it a fit's answer included its own inputs and states as though
/// nodes had produced them.
pub fn is_reserved(key: &str) -> bool {
    key == GRAPH_INPUT || key.starts_with(STATE) || key.starts_with(INPUT)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_keys_round_trip() {
        let key = state_key("scaler");
        assert_eq!(node_of_state_key(&key), Some("scaler"));
        assert!(is_reserved(&key));
    }

    #[test]
    fn an_ordinary_node_id_is_not_a_state_key() {
        assert_eq!(node_of_state_key("scaler"), None);
        assert!(!is_reserved("scaler"));
    }

    #[test]
    fn every_reserved_shape_is_recognised() {
        assert!(is_reserved(GRAPH_INPUT));
        assert!(is_reserved(&input_key("model")));
        assert!(is_reserved(&state_key("model")));
    }
}
