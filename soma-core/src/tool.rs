//! How a tool describes itself.
//!
//! Soma sits on both sides of this: it *publishes* tools over MCP
//! (`soma-mcp`), and it *calls* tools on a model's behalf (`soma-llm`).
//! Those are the same description, so they are the same type — describe a
//! tool once and it works in either direction.
//!
//! The wire form is MCP's (`inputSchema`, camelCase), because that is the
//! one with a specification. Providers that want a different envelope build
//! it at their own edge, which is where provider shape belongs.

use serde::{Deserialize, Serialize};

/// A tool's name, purpose, and argument schema.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,

    /// What it does — this is the text a model reads to decide whether to
    /// call it, so it is prompt, not documentation. Say *when* to use the
    /// tool, not only what it does: trigger conditions measurably raise the
    /// rate at which a model reaches for the right one.
    pub description: String,

    /// JSON Schema for the arguments.
    ///
    /// `inputSchema` on the wire (MCP's spelling), `input_schema` also
    /// accepted so hand-written JSON in either convention loads.
    #[serde(rename = "inputSchema", alias = "input_schema")]
    pub input_schema: serde_json::Value,
}

impl ToolSpec {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
        }
    }

    /// A tool taking no arguments.
    pub fn no_args(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self::new(
            name,
            description,
            serde_json::json!({"type": "object", "properties": {}}),
        )
    }

    /// The argument names the schema marks required.
    pub fn required_args(&self) -> Vec<&str> {
        self.input_schema
            .get("required")
            .and_then(|r| r.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// MCP's spelling is the one that goes out; both come in.
    #[test]
    fn the_wire_form_is_mcp_camel_case() {
        let spec = ToolSpec::new(
            "search",
            "Search the web. Call this when the answer depends on current information.",
            serde_json::json!({"type": "object", "required": ["q"]}),
        );

        let json = serde_json::to_value(&spec).unwrap();
        assert!(json.get("inputSchema").is_some(), "{json}");
        assert!(json.get("input_schema").is_none(), "{json}");

        // And a hand-written snake_case definition still loads.
        let snake: ToolSpec = serde_json::from_value(serde_json::json!({
            "name": "search",
            "description": "d",
            "input_schema": {"type": "object"}
        }))
        .unwrap();
        assert_eq!(snake.name, "search");

        assert_eq!(
            serde_json::from_value::<ToolSpec>(json).unwrap(),
            spec,
            "the wire form should round-trip"
        );
    }

    #[test]
    fn required_arguments_are_readable() {
        let spec = ToolSpec::new(
            "f",
            "d",
            serde_json::json!({"type": "object", "required": ["a", "b"]}),
        );
        assert_eq!(spec.required_args(), vec!["a", "b"]);
        assert!(ToolSpec::no_args("g", "d").required_args().is_empty());
    }
}
