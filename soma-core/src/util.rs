//! Shared utility functions.

use std::time::{SystemTime, UNIX_EPOCH};

/// Generate a hex-encoded nanosecond timestamp ID with a prefix.
///
/// Used for unique run IDs, plan IDs, etc.
pub fn timestamp_id(prefix: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{prefix}_{nanos:x}")
}
