//! Provider-agnostic model access.
//!
//! Two ideas carry this crate:
//!
//! 1. **A provider is not a model.** [`LlmProvider`] answers whatever
//!    [`LlmRequest::model`] names, so one Ollama instance serves every model
//!    you have pulled, and one NVIDIA NIM instance serves its whole catalog.
//!    Providers keyed per model — one object per name — is how a registry
//!    turns into a maintenance problem.
//!
//! 2. **A provider is configuration, not code.** Nearly the whole ecosystem
//!    speaks OpenAI's `/chat/completions`: Ollama, the Hugging Face router,
//!    NVIDIA NIM, Moonshot (Kimi), Z.ai (GLM), DeepSeek, Mistral, Groq,
//!    Together, OpenRouter, vLLM, llama.cpp. [`OpenAiCompatible`] is the one
//!    client for all of them; what varies lives in [`catalog::ProviderConfig`],
//!    and a new endpoint is a TOML entry rather than a release.
//!
//! Endpoints that genuinely differ implement [`LlmProvider`] directly. That
//! escape hatch exists precisely so it stays rare.
//!
//! ```ignore
//! let router = Router::from_catalog(Catalog::load()?)?.with_default("ollama");
//! let driver = EffectDriver::new(journal)
//!     .with_handler(Arc::new(LlmHandler::new(router)));
//!
//! // Models route by prefix; unprefixed ones go to the default.
//! LlmRequest::new("kimi/kimi-k2", messages);
//! LlmRequest::new("llama3.2", messages);          // → ollama
//! ```

pub mod catalog;
pub mod error;
pub mod mcp_client;
pub mod openai_compat;
pub mod steps;
pub mod tools;

pub use catalog::{Auth, Catalog, ProviderConfig, Quirks};
pub use error::LlmError;
pub use mcp_client::McpClient;
pub use openai_compat::OpenAiCompatible;
pub use steps::{JudgeStep, LlmStep, ReactStep, Verdict};
pub use tools::{FnTool, Tool, ToolOutcome, Toolbox};

use somatize_core::effect::EffectHandler;
use somatize_core::effect::{Effect, EffectResult, LlmRequest, LlmResponse};
use somatize_core::error::Result;
use std::collections::BTreeMap;
use std::sync::Arc;

/// A model offered by a provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInfo {
    pub id: String,
    pub provider: String,
}

impl ModelInfo {
    /// The name that routes back here: `provider/model`.
    pub fn qualified(&self) -> String {
        format!("{}/{}", self.provider, self.id)
    }
}

/// Something that can answer a completion request.
pub trait LlmProvider: Send + Sync {
    /// Short identifier, used as the routing prefix.
    fn id(&self) -> &str;

    /// Answer `req`. Blocking — the effect driver runs these on threads.
    fn complete(&self, req: &LlmRequest) -> Result<LlmResponse>;

    /// What this provider offers. Endpoints without a listing return empty
    /// rather than erroring, since discovery is a convenience, not a
    /// prerequisite for calling a model you already know the name of.
    fn models(&self) -> Result<Vec<ModelInfo>> {
        Ok(Vec::new())
    }
}

/// Sends each request to the provider its model names.
///
/// Routing is by `provider/model` prefix — the convention the Hugging Face
/// router and OpenRouter already use — falling back to a default provider
/// for bare model names. A prefix that names no known provider is left
/// alone, so `meta-llama/Llama-3-70B` stays one model name.
pub struct Router {
    providers: BTreeMap<String, Arc<dyn LlmProvider>>,
    default: Option<String>,
}

impl Router {
    pub fn new() -> Self {
        Self {
            providers: BTreeMap::new(),
            default: None,
        }
    }

    /// Build providers for every entry in a catalog.
    ///
    /// Entries are constructed, not contacted: nothing here reaches the
    /// network, and a provider whose key is unset fails only when used.
    pub fn from_catalog(catalog: Catalog) -> Result<Self> {
        let mut router = Self::new();
        for (id, config) in catalog.providers {
            router.register(Arc::new(OpenAiCompatible::new(id, config)?));
        }
        Ok(router)
    }

    pub fn register(&mut self, provider: Arc<dyn LlmProvider>) {
        self.providers.insert(provider.id().to_string(), provider);
    }

    /// Where bare model names go.
    pub fn with_default(mut self, provider_id: impl Into<String>) -> Self {
        self.default = Some(provider_id.into());
        self
    }

    pub fn get(&self, id: &str) -> Option<&Arc<dyn LlmProvider>> {
        self.providers.get(id)
    }

    pub fn ids(&self) -> Vec<&str> {
        self.providers.keys().map(String::as_str).collect()
    }

    /// Resolve `model` to a provider and the name that provider expects.
    pub fn route<'a>(&self, model: &'a str) -> Result<(&Arc<dyn LlmProvider>, &'a str)> {
        let (prefix, rest) = catalog::split_model(model, &|p| self.providers.contains_key(p));

        if let Some(prefix) = prefix {
            // `known` already checked membership.
            return Ok((&self.providers[prefix], rest));
        }

        let default = self.default.as_deref().ok_or_else(|| {
            LlmError::Config(format!(
                "model `{model}` names no provider and no default is set. \
                 Qualify it as `<provider>/{model}`, or set a default. \
                 Known providers: {}",
                self.ids().join(", ")
            ))
        })?;

        let provider = self.providers.get(default).ok_or_else(|| {
            LlmError::Config(format!("default provider `{default}` is not registered"))
        })?;
        Ok((provider, model))
    }

    /// Route and call.
    pub fn complete(&self, req: &LlmRequest) -> Result<LlmResponse> {
        let (provider, model) = self.route(&req.model)?;
        // The provider is told the unqualified name; the prefix was routing.
        let mut routed = req.clone();
        routed.model = model.to_string();
        provider.complete(&routed)
    }
}

impl Default for Router {
    fn default() -> Self {
        Self::new()
    }
}

/// Serves [`Effect::Llm`] from a [`Router`].
pub struct LlmHandler {
    router: Router,
}

impl LlmHandler {
    pub fn new(router: Router) -> Self {
        Self { router }
    }

    /// Every configured provider in the built-in catalog, plus any from
    /// `$SOMA_PROVIDERS` / `~/.soma/providers.toml`.
    pub fn from_env() -> Result<Self> {
        Ok(Self::new(Router::from_catalog(Catalog::load()?)?))
    }

    pub fn router(&self) -> &Router {
        &self.router
    }
}

impl EffectHandler for LlmHandler {
    fn handles(&self, effect: &Effect) -> bool {
        matches!(effect, Effect::Llm(_))
    }

    fn perform(&self, effect: &Effect) -> Result<EffectResult> {
        let Effect::Llm(req) = effect else {
            return Err(LlmError::UnexpectedEffect("not an llm effect".into()).into());
        };
        // A provider being down is something the step can act on — retry,
        // fall back to another model, give up deliberately. Raising here
        // would take that decision away from it.
        match self.router.complete(req) {
            Ok(response) => Ok(EffectResult::Llm(response)),
            Err(e) => Ok(EffectResult::Failed {
                message: e.to_string(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use somatize_core::effect::{StopReason, Usage};
    use somatize_core::message::Message;
    use std::sync::Mutex;

    /// Records what it was asked, answers with its own id.
    struct Spy {
        id: String,
        seen: Mutex<Vec<String>>,
    }

    impl Spy {
        fn new(id: &str) -> Arc<Self> {
            Arc::new(Self {
                id: id.into(),
                seen: Mutex::new(Vec::new()),
            })
        }
    }

    impl LlmProvider for Spy {
        fn id(&self) -> &str {
            &self.id
        }
        fn complete(&self, req: &LlmRequest) -> Result<LlmResponse> {
            self.seen.lock().unwrap().push(req.model.clone());
            Ok(LlmResponse {
                message: Message::assistant(format!("from {}", self.id)),
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
                model: Some(req.model.clone()),
            })
        }
    }

    fn ask(model: &str) -> LlmRequest {
        LlmRequest::new(model, vec![Message::user("hi")].into())
    }

    #[test]
    fn a_prefix_routes_to_its_provider() {
        let kimi = Spy::new("kimi");
        let ollama = Spy::new("ollama");
        let mut router = Router::new();
        router.register(kimi.clone());
        router.register(ollama.clone());

        let response = router.complete(&ask("kimi/kimi-k2")).unwrap();
        assert_eq!(response.message.text(), "from kimi");
        // The provider receives the *unqualified* name.
        assert_eq!(kimi.seen.lock().unwrap().as_slice(), ["kimi-k2"]);
        assert!(ollama.seen.lock().unwrap().is_empty());
    }

    #[test]
    fn a_bare_name_goes_to_the_default() {
        let ollama = Spy::new("ollama");
        let mut router = Router::new();
        router.register(ollama.clone());
        let router = router.with_default("ollama");

        router.complete(&ask("llama3.2")).unwrap();
        assert_eq!(ollama.seen.lock().unwrap().as_slice(), ["llama3.2"]);
    }

    /// A slash in a model name is not automatically a provider.
    #[test]
    fn a_slash_that_is_part_of_the_model_name_survives() {
        let hf = Spy::new("hf");
        let mut router = Router::new();
        router.register(hf.clone());
        let router = router.with_default("hf");

        router.complete(&ask("meta-llama/Llama-3-70B")).unwrap();
        assert_eq!(
            hf.seen.lock().unwrap().as_slice(),
            ["meta-llama/Llama-3-70B"],
            "the model name was mistaken for a provider prefix"
        );
    }

    #[test]
    fn an_unroutable_name_lists_what_is_available() {
        let mut router = Router::new();
        router.register(Spy::new("kimi"));

        let err = router
            .complete(&ask("mystery-model"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("mystery-model"), "{err}");
        assert!(err.contains("kimi"), "should list known providers: {err}");
    }

    /// A provider failure reaches the step as a result, not an error — it is
    /// the step's call whether to retry, fall back, or stop.
    #[test]
    fn a_provider_failure_becomes_a_failed_result() {
        struct Broken;
        impl LlmProvider for Broken {
            fn id(&self) -> &str {
                "broken"
            }
            fn complete(&self, _req: &LlmRequest) -> Result<LlmResponse> {
                Err(somatize_core::error::SomaError::Other(
                    "connection reset".into(),
                ))
            }
        }

        let mut router = Router::new();
        router.register(Arc::new(Broken));
        let handler = LlmHandler::new(router.with_default("broken"));

        let result = handler.perform(&Effect::Llm(ask("m"))).unwrap();
        match result {
            EffectResult::Failed { message } => assert!(message.contains("connection reset")),
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    #[test]
    fn the_handler_claims_only_llm_effects() {
        let handler = LlmHandler::new(Router::new());
        assert!(handler.handles(&Effect::Llm(ask("m"))));
        assert!(!handler.handles(&Effect::Tool {
            name: "search".into(),
            args: somatize_core::value::Value::Empty,
        }));
    }

    /// Building a router must not touch the network or require any key.
    #[test]
    fn the_builtin_catalog_builds_without_credentials() {
        let router = Router::from_catalog(Catalog::builtin()).unwrap();
        for expected in ["ollama", "hf", "nvidia", "kimi", "glm", "deepseek"] {
            assert!(router.get(expected).is_some(), "missing `{expected}`");
        }
    }

    #[test]
    fn qualified_names_round_trip() {
        let info = ModelInfo {
            id: "kimi-k2".into(),
            provider: "kimi".into(),
        };
        assert_eq!(info.qualified(), "kimi/kimi-k2");

        let router = Router::from_catalog(Catalog::builtin()).unwrap();
        let qualified = info.qualified();
        let (provider, model) = router.route(&qualified).unwrap();
        assert_eq!(provider.id(), "kimi");
        assert_eq!(model, "kimi-k2");
    }
}
