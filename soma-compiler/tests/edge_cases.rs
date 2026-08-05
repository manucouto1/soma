//! Compiler edge cases: empty graphs, cycles, unregistered nodes.

use somatize_compiler::{CompileMode, DiagnosticLevel, SimpleNodeRegistry, compile};
use somatize_core::cache::{CacheKey, CacheStore, EntryMeta};
use somatize_core::error::Result;
use somatize_core::filter::{FilterKind, FilterMeta, StreamMode};
use somatize_core::graph::{Edge, Graph, Node, linear_pipeline};
use somatize_core::schema::{DataType, Schema};
use somatize_core::value::Value;
use std::collections::HashSet;
use std::sync::Mutex;

fn make_meta(kind: FilterKind, differentiable: bool) -> FilterMeta {
    FilterMeta {
        name: "test".into(),
        kind,
        cacheable: true,
        differentiable,
        deterministic: true,
        stream_mode: StreamMode::FixedState,
        distribution: somatize_core::filter::Distribution::Local,
        input_schema: None,
        output_schema: None,
    }
}

struct MockCache {
    entries: Mutex<HashSet<CacheKey>>,
}

impl MockCache {
    fn new() -> Self {
        Self {
            entries: Mutex::new(HashSet::new()),
        }
    }
    fn insert(&self, key: CacheKey) {
        self.entries.lock().unwrap().insert(key);
    }
}

impl CacheStore for MockCache {
    fn get(&self, _: &CacheKey) -> Result<Option<Value>> {
        Ok(None)
    }
    fn put(&self, _: &CacheKey, _: &Value) -> Result<()> {
        Ok(())
    }
    fn exists(&self, key: &CacheKey) -> Result<bool> {
        Ok(self.entries.lock().unwrap().contains(key))
    }
    fn remove(&self, _: &CacheKey) -> Result<()> {
        Ok(())
    }
    fn metadata(&self, _: &CacheKey) -> Result<Option<EntryMeta>> {
        Ok(None)
    }
}

// ── Gradient flow edge cases ──

#[test]
fn gradient_multiple_interruptions() {
    // D → O → D → O → D  (D=differentiable, O=opaque)
    let graph = linear_pipeline(vec![
        Node::new("d1", "D1", "F"),
        Node::new("o1", "O1", "F"),
        Node::new("d2", "D2", "F"),
        Node::new("o2", "O2", "F"),
        Node::new("d3", "D3", "F"),
    ]);

    let mut reg = SimpleNodeRegistry::new();
    reg.register_meta(
        "d1",
        make_meta(FilterKind::Trainable, true),
        CacheKey::hash_data(b"d1"),
    );
    reg.register_meta(
        "o1",
        make_meta(FilterKind::Opaque, false),
        CacheKey::hash_data(b"o1"),
    );
    reg.register_meta(
        "d2",
        make_meta(FilterKind::Trainable, true),
        CacheKey::hash_data(b"d2"),
    );
    reg.register_meta(
        "o2",
        make_meta(FilterKind::Opaque, false),
        CacheKey::hash_data(b"o2"),
    );
    reg.register_meta(
        "d3",
        make_meta(FilterKind::Trainable, true),
        CacheKey::hash_data(b"d3"),
    );

    let result = compile(&graph, &reg, CompileMode::Inference, None).unwrap();

    // Should have 2 warnings: one for o1, one for o2
    assert_eq!(
        result.diagnostics.len(),
        2,
        "expected 2 gradient warnings, got: {:?}",
        result.diagnostics
    );
    assert_eq!(result.diagnostics[0].node_id, "o1");
    assert_eq!(result.diagnostics[1].node_id, "o2");
}

#[test]
fn gradient_all_opaque_single_warning() {
    let graph = linear_pipeline(vec![
        Node::new("o1", "O1", "F"),
        Node::new("o2", "O2", "F"),
        Node::new("o3", "O3", "F"),
    ]);

    let mut reg = SimpleNodeRegistry::new();
    reg.register_meta(
        "o1",
        make_meta(FilterKind::Opaque, false),
        CacheKey::hash_data(b"o1"),
    );
    reg.register_meta(
        "o2",
        make_meta(FilterKind::Opaque, false),
        CacheKey::hash_data(b"o2"),
    );
    reg.register_meta(
        "o3",
        make_meta(FilterKind::Opaque, false),
        CacheKey::hash_data(b"o3"),
    );

    let result = compile(&graph, &reg, CompileMode::Inference, None).unwrap();

    // Was: exactly one warning, on `o1`. The intent then was to avoid
    // warning once per opaque node; the intent now is not to warn at all,
    // because there is no gradient anywhere in this graph to interrupt.
    // Saying "gradients from upstream will not reach downstream filters"
    // about the first node of an all-opaque pipeline is simply false, and
    // it fired on every ordinary preprocessing graph in existence.
    assert!(
        result.diagnostics.is_empty(),
        "a graph with no differentiable node has no gradient to interrupt: {:?}",
        result.diagnostics
    );
}

// ── Cache cascade with diamond ──

#[test]
fn cache_diamond_cascade() {
    // root → b1 → merge
    // root → b2 → merge
    let mut graph = Graph::new();
    graph.add_node(Node::new("root", "Root", "F"));
    graph.add_node(Node::new("b1", "B1", "F"));
    graph.add_node(Node::new("b2", "B2", "F"));
    graph.add_node(Node::new("merge", "Merge", "F"));
    graph.add_edge(Edge::data("e1", "root", "b1"));
    graph.add_edge(Edge::data("e2", "root", "b2"));
    graph.add_edge(Edge::data("e3", "b1", "merge"));
    graph.add_edge(Edge::data("e4", "b2", "merge"));

    let mut reg = SimpleNodeRegistry::new();
    reg.register_meta(
        "root",
        make_meta(FilterKind::Trainable, true),
        CacheKey::hash_data(b"root"),
    );
    reg.register_meta(
        "b1",
        make_meta(FilterKind::Trainable, true),
        CacheKey::hash_data(b"b1"),
    );
    reg.register_meta(
        "b2",
        make_meta(FilterKind::Trainable, true),
        CacheKey::hash_data(b"b2"),
    );
    reg.register_meta(
        "merge",
        make_meta(FilterKind::Trainable, true),
        CacheKey::hash_data(b"merge"),
    );

    // Even with a fully-populated cache (old compile-time key scheme),
    // the compiler must emit no Cached nodes: its keys cannot include the
    // input data, so cache resolution happens at runtime per node.
    let cache = MockCache::new();
    let root_key = CacheKey::from_parts(&[&CacheKey::hash_data(b"root").0]);
    cache.insert(root_key.clone());
    let b1_key = CacheKey::from_parts(&[&CacheKey::hash_data(b"b1").0, &root_key.0]);
    cache.insert(b1_key.clone());
    let b2_key = CacheKey::from_parts(&[&CacheKey::hash_data(b"b2").0, &root_key.0]);
    cache.insert(b2_key.clone());
    let merge_key = CacheKey::from_parts(&[&CacheKey::hash_data(b"merge").0, &b1_key.0, &b2_key.0]);
    cache.insert(merge_key);

    let result = compile(&graph, &reg, CompileMode::Inference, Some(&cache)).unwrap();

    assert!(
        !format!("{:?}", result.plan).contains("Cached"),
        "cache resolution is deferred to runtime; the plan must contain no Cached nodes"
    );
    // The full diamond still executes.
    assert_eq!(result.plan.node_count(), 4);
}

// ── Unregistered node handling ──

#[test]
fn compile_with_unregistered_node() {
    let graph = linear_pipeline(vec![Node::new("a", "A", "F"), Node::new("b", "B", "F")]);

    // Only register "a", not "b"
    let mut reg = SimpleNodeRegistry::new();
    reg.register_meta(
        "a",
        make_meta(FilterKind::Trainable, true),
        CacheKey::hash_data(b"a"),
    );

    // Should still compile (unregistered nodes get Execute, no caching)
    let result = compile(&graph, &reg, CompileMode::Inference, None).unwrap();
    assert_eq!(result.plan.node_count(), 2);
}

// ── Deep chain ──

#[test]
fn compile_deep_chain() {
    let nodes: Vec<Node> = (0..20)
        .map(|i| Node::new(format!("n{i}"), format!("N{i}"), "F"))
        .collect();
    let graph = linear_pipeline(nodes);

    let mut reg = SimpleNodeRegistry::new();
    for i in 0..20 {
        reg.register_meta(
            format!("n{i}"),
            make_meta(FilterKind::Trainable, true),
            CacheKey::hash_data(format!("config_{i}").as_bytes()),
        );
    }

    let result = compile(&graph, &reg, CompileMode::Inference, None).unwrap();
    assert_eq!(result.plan.node_count(), 20);
}

// ── All compile modes on same graph ──

#[test]
fn all_compile_modes() {
    let graph = linear_pipeline(vec![Node::new("a", "A", "F"), Node::new("b", "B", "F")]);

    let mut reg = SimpleNodeRegistry::new();
    reg.register_meta(
        "a",
        make_meta(FilterKind::Trainable, true),
        CacheKey::hash_data(b"a"),
    );
    reg.register_meta(
        "b",
        make_meta(FilterKind::Trainable, true),
        CacheKey::hash_data(b"b"),
    );

    let cache = MockCache::new();
    let a_key = CacheKey::from_parts(&[&CacheKey::hash_data(b"a").0]);
    cache.insert(a_key);

    // No mode emits compile-time Cached nodes — resolution is at runtime.
    for mode in [
        CompileMode::Inference,
        CompileMode::Differentiable,
        CompileMode::NoCache,
    ] {
        let r = compile(&graph, &reg, mode, Some(&cache)).unwrap();
        assert!(!format!("{:?}", r.plan).contains("Cached"));
    }
}

// ── Schema validation ──

fn meta_with_schemas(output: Option<Schema>, input: Option<Schema>) -> FilterMeta {
    FilterMeta {
        name: "typed".into(),
        kind: FilterKind::Trainable,
        cacheable: true,
        differentiable: true,
        deterministic: true,
        stream_mode: StreamMode::FixedState,
        distribution: somatize_core::filter::Distribution::Local,
        input_schema: input,
        output_schema: output,
    }
}

#[test]
fn schema_compatible_no_warnings() {
    let graph = linear_pipeline(vec![Node::new("a", "A", "F"), Node::new("b", "B", "F")]);

    let mut reg = SimpleNodeRegistry::new();
    // A outputs f64[128], B expects f64[128] → compatible
    reg.register_meta(
        "a",
        meta_with_schemas(Some(Schema::vector(DataType::Float64, 128)), None),
        CacheKey::hash_data(b"a"),
    );
    reg.register_meta(
        "b",
        meta_with_schemas(None, Some(Schema::vector(DataType::Float64, 128))),
        CacheKey::hash_data(b"b"),
    );

    let result = compile(&graph, &reg, CompileMode::Inference, None).unwrap();
    let schema_warnings: Vec<_> = result
        .diagnostics
        .iter()
        .filter(|d| d.message.contains("schema mismatch"))
        .collect();
    assert!(schema_warnings.is_empty(), "should have no schema warnings");
}

#[test]
fn schema_incompatible_dtype_warns() {
    let graph = linear_pipeline(vec![Node::new("a", "A", "F"), Node::new("b", "B", "F")]);

    let mut reg = SimpleNodeRegistry::new();
    // A outputs f64, B expects i64 → incompatible
    reg.register_meta(
        "a",
        meta_with_schemas(Some(Schema::vector(DataType::Float64, 128)), None),
        CacheKey::hash_data(b"a"),
    );
    reg.register_meta(
        "b",
        meta_with_schemas(None, Some(Schema::vector(DataType::Int64, 128))),
        CacheKey::hash_data(b"b"),
    );

    let result = compile(&graph, &reg, CompileMode::Inference, None).unwrap();
    let schema_warnings: Vec<_> = result
        .diagnostics
        .iter()
        .filter(|d| d.message.contains("schema mismatch"))
        .collect();
    assert_eq!(schema_warnings.len(), 1);
    assert!(schema_warnings[0].message.contains("f64"));
    assert!(schema_warnings[0].message.contains("i64"));
}

#[test]
fn schema_incompatible_shape_warns() {
    let graph = linear_pipeline(vec![Node::new("a", "A", "F"), Node::new("b", "B", "F")]);

    let mut reg = SimpleNodeRegistry::new();
    // A outputs f64[128], B expects f64[256] → shape mismatch
    reg.register_meta(
        "a",
        meta_with_schemas(Some(Schema::vector(DataType::Float64, 128)), None),
        CacheKey::hash_data(b"a"),
    );
    reg.register_meta(
        "b",
        meta_with_schemas(None, Some(Schema::vector(DataType::Float64, 256))),
        CacheKey::hash_data(b"b"),
    );

    let result = compile(&graph, &reg, CompileMode::Inference, None).unwrap();
    let schema_warnings: Vec<_> = result
        .diagnostics
        .iter()
        .filter(|d| d.message.contains("schema mismatch"))
        .collect();
    assert_eq!(schema_warnings.len(), 1);
}

#[test]
fn schema_dynamic_compatible_with_fixed() {
    let graph = linear_pipeline(vec![Node::new("a", "A", "F"), Node::new("b", "B", "F")]);

    let mut reg = SimpleNodeRegistry::new();
    // A outputs f64[batch, 128], B expects f64[32, 128] → compatible (dynamic batch)
    reg.register_meta(
        "a",
        meta_with_schemas(Some(Schema::batched(DataType::Float64, &[128])), None),
        CacheKey::hash_data(b"a"),
    );
    reg.register_meta(
        "b",
        meta_with_schemas(None, Some(Schema::matrix(DataType::Float64, 32, 128))),
        CacheKey::hash_data(b"b"),
    );

    let result = compile(&graph, &reg, CompileMode::Inference, None).unwrap();
    let schema_warnings: Vec<_> = result
        .diagnostics
        .iter()
        .filter(|d| d.message.contains("schema mismatch"))
        .collect();
    assert!(schema_warnings.is_empty());
}

#[test]
fn schema_none_skips_validation() {
    let graph = linear_pipeline(vec![Node::new("a", "A", "F"), Node::new("b", "B", "F")]);

    let mut reg = SimpleNodeRegistry::new();
    // Both have None schemas → no validation, no warnings
    reg.register_meta(
        "a",
        make_meta(FilterKind::Trainable, true),
        CacheKey::hash_data(b"a"),
    );
    reg.register_meta(
        "b",
        make_meta(FilterKind::Trainable, true),
        CacheKey::hash_data(b"b"),
    );

    let result = compile(&graph, &reg, CompileMode::Inference, None).unwrap();
    assert!(result.diagnostics.is_empty());
}

// ── Architecture that is not wired into itself ──
//
// These exist because the author of the examples repository hit the DSL
// precedence trap on his first afternoon: `(A() | B()) >> C()` written
// without the parentheses leaves `A` dangling, and every existing check
// passed it — not a cycle, not a duplicate id, not a dangling endpoint,
// and a node nobody feeds satisfies its schemas trivially.

fn plain_graph(ids: &[&str], edges: &[(&str, &str)]) -> (Graph, SimpleNodeRegistry) {
    let mut graph = Graph::new();
    let mut reg = SimpleNodeRegistry::new();
    for id in ids {
        graph.add_node(Node::new(*id, *id, "P"));
        reg.register_meta(
            *id,
            make_meta(FilterKind::Stateless, false),
            CacheKey::hash_data(id.as_bytes()),
        );
    }
    for (from, to) in edges {
        graph.add_edge(Edge::data(format!("{}->{}", *from, *to), *from, *to));
    }
    (graph, reg)
}

#[test]
fn a_node_with_no_edges_is_reported() {
    let (graph, reg) = plain_graph(&["a", "b", "orphan"], &[("a", "b")]);
    let result = compile(&graph, &reg, CompileMode::Inference, None).unwrap();

    let orphan: Vec<_> = result
        .diagnostics
        .iter()
        .filter(|d| d.node_id == "orphan" && d.level == DiagnosticLevel::Warning)
        .collect();
    assert_eq!(orphan.len(), 1, "{:?}", result.diagnostics);
    assert!(orphan[0].message.contains("no edges"), "{orphan:?}");
}

#[test]
fn a_connected_graph_reports_nothing() {
    let (graph, reg) = plain_graph(&["a", "b"], &[("a", "b")]);
    let result = compile(&graph, &reg, CompileMode::Inference, None).unwrap();
    assert!(
        result.diagnostics.is_empty(),
        "a healthy linear graph must compile silently, got {:?}",
        result.diagnostics
    );
}

#[test]
fn a_fork_is_info_not_a_warning() {
    let (graph, reg) = plain_graph(&["a", "left", "right"], &[("a", "left"), ("a", "right")]);
    let result = compile(&graph, &reg, CompileMode::Inference, None).unwrap();

    assert!(
        !result
            .diagnostics
            .iter()
            .any(|d| d.level == DiagnosticLevel::Warning),
        "a fan-out is a legitimate shape: {:?}",
        result.diagnostics
    );
    assert!(
        result
            .diagnostics
            .iter()
            .any(|d| d.level == DiagnosticLevel::Info && d.message.contains("nobody consumes")),
        "{:?}",
        result.diagnostics
    );
}

#[test]
fn a_single_node_graph_is_not_an_orphan() {
    let (graph, reg) = plain_graph(&["only"], &[]);
    let result = compile(&graph, &reg, CompileMode::Inference, None).unwrap();
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
}

// The false positive that made the warning worthless: a graph of ordinary
// filters has no gradient anywhere, so nothing is interrupted at its first
// node. This used to warn on every preprocessing pipeline ever compiled.
#[test]
fn plain_filters_do_not_warn_about_gradient_flow() {
    let (graph, reg) = plain_graph(&["a", "b", "c"], &[("a", "b"), ("b", "c")]);
    let result = compile(&graph, &reg, CompileMode::Inference, None).unwrap();
    assert!(
        !result
            .diagnostics
            .iter()
            .any(|d| d.message.contains("gradient flow interrupted")),
        "{:?}",
        result.diagnostics
    );
}

// But a real interruption still reports: differentiable, then not, and the
// gradient from the first cannot reach anything past the second.
#[test]
fn a_real_gradient_interruption_still_warns() {
    let mut graph = Graph::new();
    let mut reg = SimpleNodeRegistry::new();
    for (id, diff) in [("enc", true), ("wall", false), ("head", true)] {
        graph.add_node(Node::new(id, id, "P"));
        reg.register_meta(
            id,
            make_meta(FilterKind::Trainable, diff),
            CacheKey::hash_data(id.as_bytes()),
        );
    }
    graph.add_edge(Edge::data(format!("{}->{}", "enc", "wall"), "enc", "wall"));
    graph.add_edge(Edge::data(
        format!("{}->{}", "wall", "head"),
        "wall",
        "head",
    ));

    let result = compile(&graph, &reg, CompileMode::Inference, None).unwrap();
    assert!(
        result
            .diagnostics
            .iter()
            .any(|d| d.node_id == "wall" && d.message.contains("gradient flow interrupted")),
        "{:?}",
        result.diagnostics
    );
}
