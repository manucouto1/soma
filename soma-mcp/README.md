# somatize-mcp

An MCP server exposing a Soma project to an agent: 20 tools over code, runs and the experiment pool.

The rendered text **is** the API — there is no structured result a client
will render — so every result ends with a `next:` line and a `run_dir:`,
and absence is stated rather than faked.

`run_pipeline` and `run_study` execute: a model describes a graph out of
the project's own filters and it runs in a Python subprocess rooted at the
project directory. A config value written `{"__search__": {...}}` becomes a
search dimension, which is the only difference between running a graph and
searching it.

The other tools read and write filter source, query the knowledge base,
and walk the experiment pool: `kb_find_similar`, `kb_lineage`, `kb_diff`,
`kb_record_conclusion`, `kb_branch_from`, `kb_summarize_run`, `kb_stats`.

**Not published to crates.io**, deliberately: it is a binary you run
against a project, not a library to depend on.

---

Part of [**Soma**](https://github.com/manucouto1/soma), a computational
graph runtime for research pipelines, agent orchestration and data
virtualization. Most users want the [`somatize`](https://crates.io/crates/somatize)
facade or the [Python package](https://pypi.org/project/somatize/) rather
than this crate on its own.

- Guides and design notes: <https://manucouto1.github.io/soma/>
- API documentation: <https://docs.rs/somatize-mcp>

Licensed under the Elastic License 2.0.
