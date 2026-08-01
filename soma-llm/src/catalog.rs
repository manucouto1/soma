//! Which endpoints exist, and how each one deviates.
//!
//! The ecosystem converged: Ollama, the Hugging Face router, NVIDIA NIM,
//! Moonshot (Kimi), Z.ai (GLM), DeepSeek, Mistral, Groq, Together,
//! OpenRouter, vLLM and llama.cpp all speak `POST /chat/completions` in
//! OpenAI's shape. What differs between them is *configuration*, not code:
//! a base URL, how the key is presented, and a handful of quirks.
//!
//! So a provider here is **data**, not a trait implementation. Adding one is
//! a table entry — or a line in a TOML file, with no rebuild — which is the
//! only way this survives an ecosystem that adds an endpoint a month.
//!
//! ```toml
//! # ~/.soma/providers.toml
//! [providers.my-vllm]
//! base_url = "http://gpu-box.local:8000/v1"
//! auth = { type = "none" }
//! ```
//!
//! Endpoints that genuinely are not OpenAI-shaped (Anthropic's native API,
//! Gemini's) get their own [`crate::LlmProvider`] implementation. That is the
//! escape hatch, and it should stay rarely used.

use serde::{Deserialize, Serialize};
use somatize_core::error::{Result, SomaError};
use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;

/// How a provider wants credentials presented.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Auth {
    /// No credentials — a local Ollama, vLLM or llama.cpp.
    None,
    /// `Authorization: Bearer <value of env var>`. The common case.
    Bearer { env: String },
    /// A custom header, for endpoints that do not use `Authorization`.
    Header { name: String, env: String },
}

impl Auth {
    pub fn bearer(env: impl Into<String>) -> Self {
        Self::Bearer { env: env.into() }
    }

    /// The environment variable this auth reads, if any.
    pub fn env_var(&self) -> Option<&str> {
        match self {
            Self::None => None,
            Self::Bearer { env } | Self::Header { env, .. } => Some(env),
        }
    }

    /// Read the credential. `None` when the provider needs none.
    ///
    /// A missing variable is an error rather than an empty header: sending
    /// `Bearer ` and reading the 401 that comes back wastes a round trip and
    /// reports the wrong problem.
    pub fn resolve(&self, provider: &str) -> Result<Option<(String, String)>> {
        let (header, env) = match self {
            Self::None => return Ok(None),
            Self::Bearer { env } => ("Authorization".to_string(), env),
            Self::Header { name, env } => (name.clone(), env),
        };
        let value = std::env::var(env).map_err(|_| {
            SomaError::Other(format!(
                "provider `{provider}` needs the environment variable `{env}`, which is not set"
            ))
        })?;
        let value = match self {
            Self::Bearer { .. } => format!("Bearer {value}"),
            _ => value,
        };
        Ok(Some((header, value)))
    }
}

/// Where an endpoint departs from the shape everyone claims to implement.
///
/// Every one of these exists because some real endpoint needs it. Keeping
/// them as declarative flags rather than per-provider code is what stops the
/// catalog turning back into a pile of near-duplicate clients.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Quirks {
    /// Does it accept a `tools` array at all? Several open-weight endpoints
    /// reject the field outright rather than ignoring it.
    pub supports_tools: bool,

    /// Does it accept `response_format: {type: "json_schema", ...}`?
    /// When false, a schema is appended to the prompt instead.
    pub supports_json_schema: bool,

    /// Some newer endpoints renamed `max_tokens` to `max_completion_tokens`
    /// and reject the old name.
    pub max_tokens_field: String,

    /// Omit `tools` when the list is empty rather than sending `[]`, which a
    /// few endpoints treat as "tools required" and then fail on.
    pub omit_empty_tools: bool,

    /// Send the system prompt as a `system`-role message (the norm) rather
    /// than folding it into the first user turn.
    pub system_as_message: bool,
}

impl Default for Quirks {
    fn default() -> Self {
        Self {
            supports_tools: true,
            supports_json_schema: true,
            max_tokens_field: "max_tokens".into(),
            omit_empty_tools: true,
            system_as_message: true,
        }
    }
}

/// How hard to try when an endpoint pushes back.
///
/// Policy, not mechanism — it lives here with the rest of the provider data
/// so a rate-limited endpoint is tuned in `providers.toml` without a
/// rebuild. A local Ollama wants almost none of this; a free-tier hosted
/// endpoint wants all of it.
///
/// ```toml
/// [providers.nvidia.retry]
/// max_attempts = 6
/// budget_secs = 900
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct RetryPolicy {
    /// Total tries, not retries: 1 means never retry.
    pub max_attempts: u32,

    /// First backoff, doubling from there.
    pub base_ms: u64,

    /// Ceiling on a single wait. A `Retry-After` longer than this is
    /// honoured up to here and no further — an endpoint asking for an hour
    /// is telling you to come back later, not to block a thread.
    pub max_ms: u64,

    /// Wall clock across every attempt, checked *before* starting another
    /// one. The last attempt can still overshoot by up to `timeout_secs`,
    /// which is the honest bound: this budget decides whether to try again,
    /// it cannot interrupt a request already in flight.
    pub budget_secs: u64,

    /// Spread concurrent retries out. On by default, and off in tests —
    /// a study firing twenty trials at once would otherwise synchronise
    /// their retries into the same thundering wave that caused the 429.
    pub jitter: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 4,
            base_ms: 500,
            max_ms: 30_000,
            budget_secs: 600,
            jitter: true,
        }
    }
}

impl RetryPolicy {
    /// Never retry. For a caller that wants to see the first failure.
    pub fn none() -> Self {
        Self {
            max_attempts: 1,
            ..Self::default()
        }
    }

    /// How long to wait before attempt `attempt` (0-based), given what the
    /// endpoint asked for, if anything.
    ///
    /// A `Retry-After` is an instruction and wins outright: the endpoint
    /// knows when its window resets and guessing shorter just burns another
    /// rejection.
    pub fn backoff(&self, attempt: u32, retry_after: Option<Duration>) -> Duration {
        if let Some(asked) = retry_after {
            return asked.min(Duration::from_millis(self.max_ms));
        }
        let exponential = self
            .base_ms
            .saturating_mul(1u64 << attempt.min(20))
            .min(self.max_ms);
        Duration::from_millis(if self.jitter {
            // Full jitter: uniform in [0, exponential]. Cheaper and better
            // behaved under load than jittering around the midpoint.
            jitter_below(exponential)
        } else {
            exponential
        })
    }
}

/// A pseudo-random value in `[0, ceiling]`, from the clock.
///
/// Deliberately not a real RNG: spreading retries needs unpredictability
/// between processes, not statistical quality, and it is not worth a
/// dependency.
fn jitter_below(ceiling: u64) -> u64 {
    if ceiling == 0 {
        return 0;
    }
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64)
        .unwrap_or(0);
    nanos % (ceiling + 1)
}

/// Everything needed to talk to one endpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Base URL, without a trailing slash and without `/chat/completions`.
    ///
    /// Usually ends in `/v1`, but not always — Z.ai's GLM sits at
    /// `/api/paas/v4`, which is exactly why this is a full URL and not a
    /// hostname with `/v1` bolted on.
    pub base_url: String,

    #[serde(default = "default_auth")]
    pub auth: Auth,

    /// Prepended to the model name on the wire. The Hugging Face router
    /// wants `provider/model`; most want nothing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_prefix: Option<String>,

    /// Extra headers sent with every request (referer/title for OpenRouter,
    /// tenant ids, and so on).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub headers: BTreeMap<String, String>,

    #[serde(default)]
    pub quirks: Quirks,

    /// Request timeout in seconds. Local models on cold start, and reasoning
    /// models at high effort, both routinely exceed a default of 30.
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,

    /// What to do when this endpoint pushes back.
    #[serde(default)]
    pub retry: RetryPolicy,

    /// Human-readable note, surfaced in listings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

fn default_auth() -> Auth {
    Auth::None
}

fn default_timeout() -> u64 {
    300
}

impl ProviderConfig {
    /// A local, unauthenticated, OpenAI-compatible server.
    pub fn local(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            auth: Auth::None,
            model_prefix: None,
            headers: BTreeMap::new(),
            quirks: Quirks::default(),
            timeout_secs: default_timeout(),
            retry: RetryPolicy::default(),
            note: None,
        }
    }

    pub fn with_retry(mut self, retry: RetryPolicy) -> Self {
        self.retry = retry;
        self
    }

    /// Complain about a configuration that cannot mean what it says.
    ///
    /// A retry budget shorter than one request's timeout promises retries
    /// that can never happen. Warn rather than correct: silently rewriting
    /// somebody's TOML is worse than telling them it is wrong.
    pub fn warnings(&self, id: &str) -> Vec<String> {
        let mut out = Vec::new();
        if self.retry.max_attempts > 1 && self.retry.budget_secs < self.timeout_secs {
            out.push(format!(
                "provider `{id}`: retry.budget_secs ({}) is below timeout_secs ({}), so a \
                 request that times out leaves no budget to retry. Raise the budget to at \
                 least the timeout.",
                self.retry.budget_secs, self.timeout_secs
            ));
        }
        out
    }

    /// A hosted endpoint reading its key from `env`.
    pub fn hosted(base_url: impl Into<String>, env: impl Into<String>) -> Self {
        Self {
            auth: Auth::bearer(env),
            ..Self::local(base_url)
        }
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.note = Some(note.into());
        self
    }

    pub fn with_quirks(mut self, quirks: Quirks) -> Self {
        self.quirks = quirks;
        self
    }

    pub fn with_model_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.model_prefix = Some(prefix.into());
        self
    }

    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(name.into(), value.into());
        self
    }

    /// Full URL of the chat-completions endpoint.
    pub fn chat_url(&self) -> String {
        format!("{}/chat/completions", self.base_url.trim_end_matches('/'))
    }

    /// Full URL of the model-listing endpoint.
    pub fn models_url(&self) -> String {
        format!("{}/models", self.base_url.trim_end_matches('/'))
    }

    /// The model name as this endpoint wants it.
    pub fn wire_model(&self, model: &str) -> String {
        match &self.model_prefix {
            Some(prefix) if !model.starts_with(prefix.as_str()) => format!("{prefix}{model}"),
            _ => model.to_string(),
        }
    }
}

/// The providers known to a process.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Catalog {
    #[serde(default)]
    pub providers: BTreeMap<String, ProviderConfig>,
}

impl Catalog {
    /// The endpoints that ship with Soma.
    ///
    /// Entries are configuration, not endorsement — none is reachable until
    /// its key is set (or, for local ones, until something is listening).
    /// Base URLs verified against each provider's own documentation.
    pub fn builtin() -> Self {
        let mut providers = BTreeMap::new();

        // ── Local, no key ──
        providers.insert(
            "ollama".to_string(),
            ProviderConfig::local(
                std::env::var("OLLAMA_HOST")
                    .map(|h| format!("{}/v1", h.trim_end_matches('/')))
                    .unwrap_or_else(|_| "http://localhost:11434/v1".into()),
            )
            .with_note("local Ollama; honours OLLAMA_HOST"),
        );
        providers.insert(
            "vllm".to_string(),
            ProviderConfig::local(
                std::env::var("VLLM_URL").unwrap_or_else(|_| "http://localhost:8000/v1".into()),
            )
            .with_note("local vLLM / llama.cpp server; honours VLLM_URL"),
        );

        // ── Open-model aggregators ──
        providers.insert(
            "hf".to_string(),
            ProviderConfig::hosted("https://router.huggingface.co/v1", "HF_TOKEN")
                .with_note("Hugging Face Inference Providers router; free monthly credit"),
        );
        providers.insert(
            "nvidia".to_string(),
            ProviderConfig::hosted("https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY")
                .with_note("NVIDIA NIM (build.nvidia.com); free starter credits"),
        );
        providers.insert(
            "openrouter".to_string(),
            ProviderConfig::hosted("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY")
                .with_note("OpenRouter; routes to many providers"),
        );
        providers.insert(
            "together".to_string(),
            ProviderConfig::hosted("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
        );
        providers.insert(
            "groq".to_string(),
            ProviderConfig::hosted("https://api.groq.com/openai/v1", "GROQ_API_KEY")
                .with_note("fast inference for open-weight models"),
        );

        // ── Direct vendor endpoints ──
        providers.insert(
            "kimi".to_string(),
            ProviderConfig::hosted("https://api.moonshot.ai/v1", "MOONSHOT_API_KEY")
                .with_note("Moonshot AI (Kimi)"),
        );
        providers.insert(
            "glm".to_string(),
            // Not `/v1`: Z.ai serves the OpenAI-compatible surface under
            // `/api/paas/v4`. This is why `base_url` is a whole URL.
            ProviderConfig::hosted("https://api.z.ai/api/paas/v4", "ZHIPU_API_KEY")
                .with_note("Z.ai / Zhipu (GLM)"),
        );
        providers.insert(
            "deepseek".to_string(),
            ProviderConfig::hosted("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
        );
        providers.insert(
            "mistral".to_string(),
            ProviderConfig::hosted("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
        );
        providers.insert(
            "openai".to_string(),
            ProviderConfig::hosted("https://api.openai.com/v1", "OPENAI_API_KEY"),
        );

        Self { providers }
    }

    /// Built-ins, then anything in `path` layered on top.
    ///
    /// A file entry with a built-in's name replaces it wholesale, so a user
    /// can repoint `ollama` at another host without editing Soma.
    pub fn with_overrides_from(path: impl AsRef<Path>) -> Result<Self> {
        let mut catalog = Self::builtin();
        let path = path.as_ref();
        if !path.exists() {
            return Ok(catalog);
        }
        let raw = std::fs::read_to_string(path)?;
        let extra: Catalog = toml::from_str(&raw).map_err(|e| {
            SomaError::Other(format!(
                "provider catalog {}: {e}. Expected `[providers.<name>]` tables",
                path.display()
            ))
        })?;
        catalog.providers.extend(extra.providers);
        // Overridden settings can contradict each other; say so at load,
        // when the file is in front of the reader, rather than at the point
        // some retry silently never happens.
        for warning in catalog.warnings() {
            tracing::warn!("{warning}");
        }
        Ok(catalog)
    }

    /// Configurations that cannot mean what they say, across every provider.
    pub fn warnings(&self) -> Vec<String> {
        self.providers
            .iter()
            .flat_map(|(id, config)| config.warnings(id))
            .collect()
    }

    /// Built-ins plus `$SOMA_PROVIDERS`, else `~/.soma/providers.toml`.
    pub fn load() -> Result<Self> {
        let path = std::env::var("SOMA_PROVIDERS")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                dirs_home()
                    .unwrap_or_default()
                    .join(".soma")
                    .join("providers.toml")
            });
        Self::with_overrides_from(path)
    }

    pub fn get(&self, id: &str) -> Option<&ProviderConfig> {
        self.providers.get(id)
    }

    pub fn insert(&mut self, id: impl Into<String>, config: ProviderConfig) {
        self.providers.insert(id.into(), config);
    }

    /// Provider ids, sorted.
    pub fn ids(&self) -> Vec<&str> {
        self.providers.keys().map(String::as_str).collect()
    }

    /// Providers whose credentials are actually present — the ones a call
    /// could reach right now.
    pub fn configured(&self) -> Vec<&str> {
        self.providers
            .iter()
            .filter(|(_, c)| match c.auth.env_var() {
                None => true,
                Some(env) => std::env::var(env).is_ok(),
            })
            .map(|(id, _)| id.as_str())
            .collect()
    }
}

fn dirs_home() -> Option<std::path::PathBuf> {
    std::env::var_os("HOME").map(std::path::PathBuf::from)
}

/// Split `"kimi/kimi-k2"` into `(Some("kimi"), "kimi-k2")`.
///
/// The `provider/model` convention is what the Hugging Face router and
/// OpenRouter already use, so it reads naturally to anyone who has used
/// either. An unprefixed name goes to the router's default provider.
///
/// Only the *first* segment is treated as a provider, and only when it names
/// one — `"meta-llama/Llama-3-70B"` stays whole unless a provider is
/// literally called `meta-llama`.
pub fn split_model<'a>(model: &'a str, known: &dyn Fn(&str) -> bool) -> (Option<&'a str>, &'a str) {
    match model.split_once('/') {
        Some((prefix, rest)) if known(prefix) && !rest.is_empty() => (Some(prefix), rest),
        _ => (None, model),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_urls_are_whole_urls_not_host_plus_v1() {
        let c = Catalog::builtin();
        // Z.ai is the reason: its OpenAI-compatible surface is not under /v1.
        assert_eq!(
            c.get("glm").unwrap().chat_url(),
            "https://api.z.ai/api/paas/v4/chat/completions"
        );
        assert_eq!(
            c.get("kimi").unwrap().chat_url(),
            "https://api.moonshot.ai/v1/chat/completions"
        );
        assert_eq!(
            c.get("nvidia").unwrap().models_url(),
            "https://integrate.api.nvidia.com/v1/models"
        );
    }

    #[test]
    fn the_local_providers_need_no_key() {
        let c = Catalog::builtin();
        assert_eq!(c.get("ollama").unwrap().auth, Auth::None);
        assert!(
            c.get("ollama")
                .unwrap()
                .auth
                .resolve("ollama")
                .unwrap()
                .is_none()
        );
    }

    /// A missing key is reported as a missing key, not as a 401 later.
    #[test]
    fn a_missing_key_names_the_variable() {
        let auth = Auth::bearer("DEFINITELY_NOT_SET_ANYWHERE_12345");
        let err = auth.resolve("kimi").unwrap_err().to_string();
        assert!(err.contains("kimi"), "{err}");
        assert!(err.contains("DEFINITELY_NOT_SET_ANYWHERE_12345"), "{err}");
    }

    #[test]
    fn a_bearer_key_is_prefixed_once() {
        // SAFETY: single-threaded test process; no other thread reads env here.
        unsafe { std::env::set_var("SOMA_TEST_KEY", "sk-abc") };
        let (header, value) = Auth::bearer("SOMA_TEST_KEY")
            .resolve("x")
            .unwrap()
            .expect("a credential");
        assert_eq!(header, "Authorization");
        assert_eq!(value, "Bearer sk-abc");
        unsafe { std::env::remove_var("SOMA_TEST_KEY") };
    }

    #[test]
    fn model_prefixes_are_not_doubled() {
        let c = ProviderConfig::local("http://x/v1").with_model_prefix("meta-llama/");
        assert_eq!(c.wire_model("Llama-3"), "meta-llama/Llama-3");
        assert_eq!(c.wire_model("meta-llama/Llama-3"), "meta-llama/Llama-3");
    }

    #[test]
    fn model_names_split_on_a_known_provider_only() {
        let known = |p: &str| matches!(p, "kimi" | "ollama");

        assert_eq!(
            split_model("kimi/kimi-k2", &known),
            (Some("kimi"), "kimi-k2")
        );
        assert_eq!(split_model("gpt-4o", &known), (None, "gpt-4o"));
        // A slash that is part of the model name, not a provider.
        assert_eq!(
            split_model("meta-llama/Llama-3-70B", &known),
            (None, "meta-llama/Llama-3-70B")
        );
        assert_eq!(split_model("kimi/", &known), (None, "kimi/"));
    }

    /// Adding a provider must not require touching Soma.
    #[test]
    fn a_toml_file_can_add_and_override_providers() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("providers.toml");
        std::fs::write(
            &path,
            r#"
            [providers.my-vllm]
            base_url = "http://gpu-box.local:8000/v1"
            auth = { type = "none" }

            [providers.ollama]
            base_url = "http://otherbox:11434/v1"
            auth = { type = "none" }
            "#,
        )
        .unwrap();

        let c = Catalog::with_overrides_from(&path).unwrap();
        assert_eq!(
            c.get("my-vllm").unwrap().base_url,
            "http://gpu-box.local:8000/v1"
        );
        assert_eq!(
            c.get("ollama").unwrap().base_url,
            "http://otherbox:11434/v1",
            "a file entry should replace the built-in of the same name"
        );
        // Untouched built-ins survive.
        assert!(c.get("kimi").is_some());
    }

    #[test]
    fn a_malformed_catalog_says_where_and_what() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("providers.toml");
        std::fs::write(&path, "this is not toml [[[").unwrap();
        let err = Catalog::with_overrides_from(&path).unwrap_err().to_string();
        assert!(err.contains("providers.toml"), "{err}");
        assert!(
            err.contains("[providers."),
            "should say the expected shape: {err}"
        );
    }

    #[test]
    fn a_missing_catalog_file_is_not_an_error() {
        let c = Catalog::with_overrides_from("/nonexistent/providers.toml").unwrap();
        assert!(!c.providers.is_empty());
    }
}
