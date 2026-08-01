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

/// Pull a JSON object out of a reply that may be fenced or prefaced.
///
/// Models wrap JSON in ```json fences, or preface it with a sentence,
/// often enough that requiring a bare object would fail on working
/// output.
///
/// Two implementations of this used to live in two crates. The naive one
/// took everything between the first `{` and the *last* `}`, so a reply
/// holding an object followed by any later brace — a second example, a
/// closing fence with a brace in it, prose about `{}` — parsed as
/// nothing. This is the scanner: the first *balanced* object, respecting
/// strings and escapes.
pub fn extract_json(text: &str) -> Option<serde_json::Value> {
    // The whole reply, when the model simply answered with JSON.
    if let Ok(value) = serde_json::from_str(text.trim()) {
        return Some(value);
    }

    let start = text.find('{')?;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    for (i, &b) in text.as_bytes().iter().enumerate().skip(start) {
        if escaped {
            escaped = false;
            continue;
        }
        match b {
            b'\\' if in_string => escaped = true,
            b'"' => in_string = !in_string,
            b'{' if !in_string => depth += 1,
            b'}' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    return serde_json::from_str(&text[start..=i]).ok();
                }
            }
            _ => {}
        }
    }
    None
}

/// Enough of a long string to recognise, without pasting the whole thing
/// into an error message.
pub fn truncate(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    text.chars().take(max).collect::<String>() + "…"
}

#[cfg(test)]
mod util_tests {
    use super::*;

    /// The naive version took the first `{` to the *last* `}`, which is
    /// not the same object as soon as anything follows it.
    #[test]
    fn extract_json_takes_the_first_balanced_object() {
        let reply = "Here you go:\n```json\n{\"a\": 1}\n```\nand another: {\"b\": 2}";
        let got = extract_json(reply).expect("should find the first object");
        assert_eq!(got, serde_json::json!({"a": 1}));
    }

    #[test]
    fn extract_json_is_not_fooled_by_braces_in_strings() {
        let reply = r#"prose {"note": "a } inside", "n": 3} trailing"#;
        let got = extract_json(reply).expect("should find the object");
        assert_eq!(got, serde_json::json!({"note": "a } inside", "n": 3}));
    }

    #[test]
    fn extract_json_accepts_a_bare_object() {
        assert_eq!(
            extract_json("  {\"a\": 1}  "),
            Some(serde_json::json!({"a": 1}))
        );
    }

    #[test]
    fn truncate_keeps_short_text_whole() {
        assert_eq!(truncate("hola", 10), "hola");
        assert_eq!(truncate("hola", 2), "ho…");
    }
}
