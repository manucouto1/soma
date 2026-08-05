//! Tools a model can call.
//!
//! Two ways in, one type out:
//!
//! - **Native** — a Rust closure registered with [`Toolbox::add_fn`].
//! - **MCP** — every tool a Model Context Protocol server publishes,
//!   discovered at registration with [`Toolbox::add_mcp_server`].
//!
//! The second is why there is no plugin system here. MCP is the protocol the
//! ecosystem settled on, and a server that speaks it is already a tool
//! provider — including Soma's own `soma-mcp`, which means an agent can
//! query the experiment pool through the same door anyone else does.
//! (Compare the alternative: four bespoke backends, one of which was never
//! implemented.)

use crate::mcp_client::McpClient;
use somatize_core::effect::EffectHandler;
use somatize_core::effect::{Effect, EffectResult};
use somatize_core::error::Result;
use somatize_core::tool::ToolSpec;
use somatize_core::value::Value;
use std::collections::BTreeMap;
use std::sync::Arc;

/// What a tool produced.
pub struct ToolOutcome {
    /// The result the model reads — text for prose, JSON for structure.
    pub output: Value,
    /// A failure the *model* should see and adapt to, rather than one that
    /// stops the run. A search returning nothing is information.
    pub is_error: bool,
}

impl ToolOutcome {
    /// A successful result.
    pub fn ok(output: Value) -> Self {
        Self {
            output,
            is_error: false,
        }
    }

    /// A failure to report to the model, phrased so it can adapt.
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            output: Value::text(message.into()),
            is_error: true,
        }
    }
}

/// Something a model can call.
pub trait Tool: Send + Sync {
    /// Name, description and argument schema — what the model is shown.
    fn spec(&self) -> ToolSpec;

    /// Run it. Blocking; the driver calls these on threads.
    fn call(&self, args: &Value) -> Result<ToolOutcome>;
}

/// A tool backed by a closure.
pub struct FnTool<F> {
    spec: ToolSpec,
    f: F,
}

impl<F> FnTool<F>
where
    F: Fn(&Value) -> Result<ToolOutcome> + Send + Sync,
{
    /// Pair a spec with the closure that fulfils it. Usually reached via
    /// [`Toolbox::add_fn`].
    pub fn new(spec: ToolSpec, f: F) -> Self {
        Self { spec, f }
    }
}

impl<F> Tool for FnTool<F>
where
    F: Fn(&Value) -> Result<ToolOutcome> + Send + Sync,
{
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }
    fn call(&self, args: &Value) -> Result<ToolOutcome> {
        (self.f)(args)
    }
}

/// The tools available to a run, and the handler that runs them.
#[derive(Default)]
pub struct Toolbox {
    tools: BTreeMap<String, Arc<dyn Tool>>,
}

impl Toolbox {
    /// An empty toolbox.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tool under its spec's name, replacing any tool that
    /// already held it.
    pub fn add(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.spec().name, tool);
    }

    /// Register a closure as a tool.
    pub fn add_fn<F>(&mut self, spec: ToolSpec, f: F)
    where
        F: Fn(&Value) -> Result<ToolOutcome> + Send + Sync + 'static,
    {
        self.add(Arc::new(FnTool::new(spec, f)));
    }

    /// Start an MCP server and register everything it publishes.
    ///
    /// Discovery happens now, so a misconfigured server fails at setup
    /// rather than mid-run when a model first reaches for one of its tools.
    pub fn add_mcp_server(&mut self, command: &str, args: &[&str]) -> Result<usize> {
        let client = Arc::new(McpClient::start(command, args)?);
        let specs = client.list_tools()?;
        let count = specs.len();
        for spec in specs {
            self.add(Arc::new(McpTool {
                client: client.clone(),
                spec,
            }));
        }
        Ok(count)
    }

    /// The tool registered under `name`, if any.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    /// Take every tool from another toolbox. Used to fold a per-MCP-server
    /// set into the one the driver actually consults.
    pub fn merge_from(&mut self, other: &Toolbox) {
        for (name, tool) in &other.tools {
            self.tools.insert(name.clone(), tool.clone());
        }
    }

    /// What to advertise to a model. Sorted, so the prompt prefix a provider
    /// caches stays byte-identical between runs.
    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools.values().map(|t| t.spec()).collect()
    }

    /// Registered tool names, sorted.
    pub fn names(&self) -> Vec<&str> {
        self.tools.keys().map(String::as_str).collect()
    }

    /// How many tools are registered.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether no tools are registered.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

impl EffectHandler for Toolbox {
    fn handles(&self, effect: &Effect) -> bool {
        matches!(effect, Effect::Tool { .. })
    }

    fn perform(&self, effect: &Effect) -> Result<EffectResult> {
        let Effect::Tool { name, args } = effect else {
            return Err(
                crate::error::LlmError::UnexpectedEffect("not a tool effect".into()).into(),
            );
        };

        // A model asking for a tool that does not exist is a routine
        // mistake, not a crash: tell it so, and it picks another.
        let Some(tool) = self.tools.get(name) else {
            return Ok(EffectResult::Tool {
                output: Value::text(format!(
                    "no tool named `{name}`. Available: {}",
                    self.names().join(", ")
                )),
                is_error: true,
            });
        };

        match tool.call(args) {
            Ok(outcome) => Ok(EffectResult::Tool {
                output: outcome.output,
                is_error: outcome.is_error,
            }),
            // Likewise a tool that blew up: report it to the model.
            Err(e) => Ok(EffectResult::Tool {
                output: Value::text(e.to_string()),
                is_error: true,
            }),
        }
    }
}

/// One tool published by an MCP server.
struct McpTool {
    client: Arc<McpClient>,
    spec: ToolSpec,
}

impl Tool for McpTool {
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }
    /// The seam: `Tool` is a `soma-core` trait, so the MCP error becomes
    /// a `SomaError` here, keeping the `mcp server \`name\`:` prefix.
    fn call(&self, args: &Value) -> Result<ToolOutcome> {
        Ok(self.client.call_tool(&self.spec.name, args)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn echo_box() -> Toolbox {
        let mut box_ = Toolbox::new();
        box_.add_fn(
            ToolSpec::new(
                "echo",
                "Echo the `text` argument back.",
                serde_json::json!({
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"]
                }),
            ),
            |args| {
                let text = args.to_plain_json()["text"]
                    .as_str()
                    .unwrap_or("")
                    .to_string();
                Ok(ToolOutcome::ok(Value::text(text)))
            },
        );
        box_
    }

    #[test]
    fn a_registered_tool_runs() {
        let box_ = echo_box();
        let result = box_
            .perform(&Effect::Tool {
                name: "echo".into(),
                args: Value::json(serde_json::json!({"text": "hi"})),
            })
            .unwrap();

        match result {
            EffectResult::Tool { output, is_error } => {
                assert_eq!(output.as_text(), Some("hi"));
                assert!(!is_error);
            }
            other => panic!("{other:?}"),
        }
    }

    /// Asking for a tool that does not exist is a routine model mistake —
    /// answer it so the model can pick another, rather than failing the run.
    #[test]
    fn an_unknown_tool_is_answered_not_raised() {
        let box_ = echo_box();
        let result = box_
            .perform(&Effect::Tool {
                name: "teleport".into(),
                args: Value::Empty,
            })
            .unwrap();

        match result {
            EffectResult::Tool { output, is_error } => {
                assert!(is_error);
                let text = output.as_text().unwrap();
                assert!(text.contains("teleport"), "{text}");
                assert!(text.contains("echo"), "should list what exists: {text}");
            }
            other => panic!("{other:?}"),
        }
    }

    /// A tool that errors reports back to the model too.
    #[test]
    fn a_failing_tool_reports_to_the_model() {
        let mut box_ = Toolbox::new();
        box_.add_fn(ToolSpec::no_args("boom", "always fails"), |_| {
            Err(somatize_core::error::SomaError::Other(
                "disk on fire".into(),
            ))
        });

        match box_
            .perform(&Effect::Tool {
                name: "boom".into(),
                args: Value::Empty,
            })
            .unwrap()
        {
            EffectResult::Tool { output, is_error } => {
                assert!(is_error);
                assert!(output.as_text().unwrap().contains("disk on fire"));
            }
            other => panic!("{other:?}"),
        }
    }

    /// Specs come out sorted, so the tool list a provider caches as part of
    /// the prompt prefix is byte-identical between runs.
    #[test]
    fn specs_are_ordered() {
        let mut box_ = Toolbox::new();
        for name in ["zebra", "alpha", "middle"] {
            box_.add_fn(ToolSpec::no_args(name, "d"), |_| {
                Ok(ToolOutcome::ok(Value::Empty))
            });
        }
        let names: Vec<String> = box_.specs().into_iter().map(|s| s.name).collect();
        assert_eq!(names, ["alpha", "middle", "zebra"]);
    }

    #[test]
    fn the_toolbox_claims_only_tool_effects() {
        let box_ = echo_box();
        assert!(box_.handles(&Effect::Tool {
            name: "echo".into(),
            args: Value::Empty
        }));
        assert!(!box_.handles(&Effect::Sleep(std::time::Duration::from_secs(1))));
    }
}
