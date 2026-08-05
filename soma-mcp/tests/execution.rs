//! `run_pipeline` and `run_study`, against a project on disk.
//!
//! These build a small project — three filters in a file — and ask the
//! server to run graphs made of them, exactly as a model would after
//! `list_filters` and `read_filter_source`.
//!
//! They need a Python that can `import soma`, which a bare CI runner has
//! no reason to have, so they skip when there is none rather than fail.
//! A skip prints why: a test that quietly does nothing is worse than one
//! that is not there.

use serde_json::json;
use somatize_mcp::context::SomaContext;

const FILTERS: &str = r#"
from soma import Filter


class Scale(Filter):
    """Multiplies by a factor."""

    _cache_version = "mcp-scale-v1"

    def __init__(self, factor=2.0):
        super().__init__(factor=factor)
        self.factor = factor

    def forward(self, x, state):
        return [v * self.factor for v in x]


class Center(Filter):
    """Subtracts the mean it learned, so the graph has something to fit."""

    _cache_version = "mcp-center-v1"

    def fit(self, x, y=None):
        return {"mu": sum(x) / len(x)}

    def forward(self, x, state):
        return [v - state["mu"] for v in x]


class Distance(Filter):
    """Mean squared distance to a target — a study's objective."""

    _cache_version = "mcp-distance-v1"

    def __init__(self, target=0.0):
        super().__init__(target=target)
        self.target = target

    def forward(self, x, state):
        return {"score": sum((v - self.target) ** 2 for v in x) / len(x)}
"#;

/// A project directory holding `my_filters.py`, plus its context.
fn project(name: &str) -> Option<(tempfile::TempDir, SomaContext)> {
    if !python_has_soma() {
        eprintln!("skipping {name}: no python3 that can `import soma`");
        return None;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("my_filters.py"), FILTERS).expect("write filters");
    // The pool writes here; without it the run is untracked and the
    // `run_dir` assertions below would be testing nothing.
    std::fs::create_dir_all(dir.path().join(".soma")).expect("mkdir .soma");
    let ctx = SomaContext::new(dir.path());
    Some((dir, ctx))
}

fn python_has_soma() -> bool {
    let python = std::env::var("SOMA_PYTHON").unwrap_or_else(|_| "python3".into());
    std::process::Command::new(python)
        .args(["-c", "import soma"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[test]
fn run_pipeline_fits_and_forwards_a_two_node_graph() {
    let Some((_dir, ctx)) = project("run_pipeline_fits_and_forwards_a_two_node_graph") else {
        return;
    };
    let result = ctx.run_pipeline(&json!({
        "nodes": [
            {"id": "scale", "filter": "my_filters.Scale", "config": {"factor": 3.0}},
            {"id": "center", "filter": "my_filters.Center"},
        ],
        "edges": [["scale", "center"]],
        "input": [1.0, 2.0, 3.0, 4.0],
        "name": "mcp-two-node",
    }));
    let text = result.content_text();
    assert_ne!(result.is_error, Some(true), "{text}");

    // 3x of 1..4 is 3,6,9,12, mean 7.5 — centering gives -4.5..4.5. The
    // numbers matter: they are the proof that both nodes ran, in order,
    // and that `fit` happened before `forward`.
    assert!(text.contains("-4.5"), "the scaled, centered output: {text}");
    assert!(text.contains("4.5"), "the scaled, centered output: {text}");
    assert!(
        text.contains("state learned by: center"),
        "the fit state should be reported: {text}"
    );
    assert!(
        text.contains("run_dir: "),
        "a tracked run has a dir: {text}"
    );
    assert_navigable(&text);
}

#[test]
fn run_pipeline_reports_a_filter_that_raises() {
    let Some((dir, ctx)) = project("run_pipeline_reports_a_filter_that_raises") else {
        return;
    };
    std::fs::write(
        dir.path().join("broken.py"),
        "from soma import Filter\n\
         class Boom(Filter):\n    \
             _cache_version = 'boom-v1'\n    \
             def forward(self, x, state):\n        \
                 raise ValueError('the filter said no')\n",
    )
    .expect("write");

    let result = ctx.run_pipeline(&json!({
        "nodes": [{"id": "b", "filter": "broken.Boom"}],
        "input": [1.0],
    }));
    let text = result.content_text();

    // The model is debugging its own graph: it needs the message and the
    // traceback, not a tidy "execution failed".
    assert!(text.contains("the filter said no"), "{text}");
    assert!(text.contains("traceback:"), "{text}");
    assert_navigable(&text);
}

#[test]
fn an_unresolvable_filter_says_how_to_name_one() {
    let Some((_dir, ctx)) = project("an_unresolvable_filter_says_how_to_name_one") else {
        return;
    };
    let result = ctx.run_pipeline(&json!({
        "nodes": [{"id": "nope", "filter": "no_such_module.Nothing"}],
        "input": [1.0],
    }));
    let text = result.content_text();
    assert!(text.contains("cannot resolve filter"), "{text}");
    assert!(text.contains("module.Class"), "{text}");
}

#[test]
fn run_study_searches_a_marked_config_value() {
    let Some((_dir, ctx)) = project("run_study_searches_a_marked_config_value") else {
        return;
    };
    let result = ctx.run_study(&json!({
        "nodes": [
            {"id": "scale", "filter": "my_filters.Scale",
             "config": {"factor": {"__search__": {"low": 0.1, "high": 4.0}}}},
            {"id": "dist", "filter": "my_filters.Distance", "config": {"target": 6.0}},
        ],
        "edges": [["scale", "dist"]],
        "input": [1.0, 2.0, 3.0],
        "name": "mcp-search",
        "strategy": "random",
        "n_trials": 8,
        "metric": "score",
        "direction": "minimize",
        "seed": 7,
    }));
    let text = result.content_text();
    assert_ne!(result.is_error, Some(true), "{text}");

    assert!(text.contains("run_study: 8 trials"), "{text}");
    assert!(text.contains("optimizing: score (minimize)"), "{text}");
    assert!(
        text.contains("scale.factor"),
        "the searched dimension: {text}"
    );
    assert_navigable(&text);
}

#[test]
fn a_study_over_nothing_says_so() {
    let Some((_dir, ctx)) = project("a_study_over_nothing_says_so") else {
        return;
    };
    // No `__search__` anywhere: every trial would be the same run, and a
    // search that cannot vary anything is a mistake worth naming.
    let result = ctx.run_study(&json!({
        "nodes": [{"id": "scale", "filter": "my_filters.Scale", "config": {"factor": 2.0}}],
        "input": [1.0, 2.0],
        "n_trials": 3,
    }));
    let text = result.content_text();
    assert!(text.contains("no search space"), "{text}");
    assert!(
        text.contains("__search__"),
        "it should say how to fix it: {text}"
    );
}

#[test]
fn an_empty_graph_is_refused_before_python_starts() {
    // No interpreter needed: this one never reaches the driver.
    let dir = tempfile::tempdir().expect("tempdir");
    let ctx = SomaContext::new(dir.path());
    let result = ctx.run_pipeline(&json!({"nodes": [], "input": [1.0]}));
    assert_eq!(result.is_error, Some(true));
    assert!(result.content_text().contains("no graph to run"));

    let result = ctx.run_pipeline(&json!({"input": [1.0]}));
    assert_eq!(result.is_error, Some(true));
    assert!(result.content_text().contains("nodes"));
}

/// Every result a model reads ends with a callable follow-up.
fn assert_navigable(text: &str) {
    let last = text.trim_end().lines().last().unwrap_or_default();
    assert!(last.starts_with("next: "), "no follow-up line: {last:?}");
    assert!(last.contains('('), "follow-ups must be callable: {last:?}");
}
