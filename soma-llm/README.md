# somatize-llm

LLM providers, tools and MCP clients: the library layer of the agentic side.

An OpenAI-compatible client that speaks to ollama, HuggingFace, NVIDIA,
Kimi, GLM, DeepSeek, Groq, vLLM and friends, with the provider catalog as
TOML **data** rather than code — including each one's retry policy and
quirks.

Retries live in the HTTP client, not the step: a 429 is transport, not
domain. 408/425/429/5xx and transport errors retry; everything else is
fatal. `Retry-After` is honoured in both RFC forms, and the wall-clock
budget is checked *before* sleeping.

Structured output uses `response_format` when the endpoint can enforce a
schema and a system-prompt append when it cannot. Validation is
deliberately structural and permissive: an invented violation costs a real
model call to "fix" a correct answer.

Also here: the `Toolbox`, an MCP client, and `ReactStep`/`JudgeStep`.

---

Part of [**Soma**](https://github.com/manucouto1/soma), a computational
graph runtime for research pipelines, agent orchestration and data
virtualization. Most users want the [`somatize`](https://crates.io/crates/somatize)
facade or the [Python package](https://pypi.org/project/somatize/) rather
than this crate on its own.

- Guides and design notes: <https://manucouto1.github.io/soma/>
- API documentation: <https://docs.rs/somatize-llm>

Licensed under the Elastic License 2.0.
