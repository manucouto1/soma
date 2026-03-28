use soma_compiler::{
    compile, CompileMode, ExecutionPlan, SimpleFilterRegistry,
};
use soma_core::cache::{CacheKey, CacheStore, EntryMeta};
use soma_core::error::{Result, SomaError};
use soma_core::filter::{FilterKind, FilterMeta, StreamMode};
use soma_core::schema::{DataType, Schema};
use soma_core::graph::{linear_pipeline, Edge, Graph, Node};
use soma_core::value::Value;
use std::collections::HashSet;
use std::sync::Mutex;

fn make_meta(kind: FilterKind, differentiable: bool) -> FilterMeta {
    FilterMeta {
        name: "test".into(),
        kind,
        cacheable: true,
        differentiable,
        stream_mode: StreamMode::FixedState,
        distribution: soma_core::filter::Distribution::Local,
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
    fn get(&self, _: &CacheKey) -> Result<Option<Value>> { Ok(None) }
    fn put(&self, _: &CacheKey, _: &Value) -> Result<()> { Ok(()) }
    fn exists(&self, key: &CacheKey) -> Result<bool> {
        Ok(self.entries.lock().unwrap().contains(key))
    }
    fn remove(&self, _: &CacheKey) -> Result<()> { Ok(()) }
    fn metadata(&self, _: &CacheKey) -> Result<Option<EntryMeta>> { Ok(None) }
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

    let mut reg = SimpleFilterRegistry::new();
    reg.register_meta("d1", make_meta(FilterKind::Trainable, true), CacheKey::hash_data(b"d1"));
    reg.register_meta("o1", make_meta(FilterKind::Opaque, false), CacheKey::hash_data(b"o1"));
    reg.register_meta("d2", make_meta(FilterKind::Trainable, true), CacheKey::hash_data(b"d2"));
    reg.register_meta("o2", make_meta(FilterKind::Opaque, false), CacheKey::hash_data(b"o2"));
    reg.register_meta("d3", make_meta(FilterKind::Trainable, true), CacheKey::hash_data(b"d3"));

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

    let mut reg = SimpleFilterRegistry::new();
    reg.register_meta("o1", make_meta(FilterKind::Opaque, false), CacheKey::hash_data(b"o1"));
    reg.register_meta("o2", make_meta(FilterKind::Opaque, false), CacheKey::hash_data(b"o2"));
    reg.register_meta("o3", make_meta(FilterKind::Opaque, false), CacheKey::hash_data(b"o3"));

    let result = compile(&graph, &reg, CompileMode::Inference, None).unwrap();

    // Only first opaque triggers warning (flow is already interrupted)
    assert_eq!(result.diagnostics.len(), 1);
    assert_eq!(result.diagnostics[0].node_id, "o1");
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

    let mut reg = SimpleFilterRegistry::new();
    reg.register_meta("root", make_meta(FilterKind::Trainable, true), CacheKey::hash_data(b"root"));
    reg.register_meta("b1", make_meta(FilterKind::Trainable, true), CacheKey::hash_data(b"b1"));
    reg.register_meta("b2", make_meta(FilterKind::Trainable, true), CacheKey::hash_data(b"b2"));
    reg.register_meta("merge", make_meta(FilterKind::Trainable, true), CacheKey::hash_data(b"merge"));

    // Cache root's output
    let cache = MockCache::new();
    let root_key = CacheKey::from_parts(&[&CacheKey::hash_data(b"root").0]);
    cache.insert(root_key.clone());

    // Cache b1 (depends on root)
    let b1_key = CacheKey::from_parts(&[&CacheKey::hash_data(b"b1").0, &root_key.0]);
    cache.insert(b1_key.clone());

    // Cache b2 (depends on root)
    let b2_key = CacheKey::from_parts(&[&CacheKey::hash_data(b"b2").0, &root_key.0]);
    cache.insert(b2_key.clone());

    // Cache merge (depends on b1 AND b2)
    let merge_key = CacheKey::from_parts(&[
        &CacheKey::hash_data(b"merge").0,
        &b1_key.0,
        &b2_key.0,
    ]);
    cache.insert(merge_key);

    let result = compile(&graph, &reg, CompileMode::Inference, Some(&cache)).unwrap();

    // Everything should be cached
    assert_eq!(result.plan.cached_count(), 4, "all 4 nodes should be cached");
}

// ── Unregistered node handling ──

#[test]
fn compile_with_unregistered_node() {
    let graph = linear_pipeline(vec![
        Node::new("a", "A", "F"),
        Node::new("b", "B", "F"),
    ]);

    // Only register "a", not "b"
    let mut reg = SimpleFilterRegistry::new();
    reg.register_meta("a", make_meta(FilterKind::Trainable, true), CacheKey::hash_data(b"a"));

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

    let mut reg = SimpleFilterRegistry::new();
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
    let graph = linear_pipeline(vec![
        Node::new("a", "A", "F"),
        Node::new("b", "B", "F"),
    ]);

    let mut reg = SimpleFilterRegistry::new();
    reg.register_meta("a", make_meta(FilterKind::Trainable, true), CacheKey::hash_data(b"a"));
    reg.register_meta("b", make_meta(FilterKind::Trainable, true), CacheKey::hash_data(b"b"));

    let cache = MockCache::new();
    let a_key = CacheKey::from_parts(&[&CacheKey::hash_data(b"a").0]);
    cache.insert(a_key);

    // Inference: should cache "a"
    let r1 = compile(&graph, &reg, CompileMode::Inference, Some(&cache)).unwrap();
    assert_eq!(r1.plan.cached_count(), 1);

    // Differentiable: no caching
    let r2 = compile(&graph, &reg, CompileMode::Differentiable, Some(&cache)).unwrap();
    assert_eq!(r2.plan.cached_count(), 0);

    // NoCache: no caching
    let r3 = compile(&graph, &reg, CompileMode::NoCache, Some(&cache)).unwrap();
    assert_eq!(r3.plan.cached_count(), 0);
}

// ── Schema validation ──

fn meta_with_schemas(
    output: Option<Schema>,
    input: Option<Schema>,
) -> FilterMeta {
    FilterMeta {
        name: "typed".into(),
        kind: FilterKind::Trainable,
        cacheable: true,
        differentiable: true,
        stream_mode: StreamMode::FixedState,
        distribution: soma_core::filter::Distribution::Local,
        input_schema: input,
        output_schema: output,
    }
}

#[test]
fn schema_compatible_no_warnings() {
    let graph = linear_pipeline(vec![
        Node::new("a", "A", "F"),
        Node::new("b", "B", "F"),
    ]);

    let mut reg = SimpleFilterRegistry::new();
    // A outputs f64[128], B expects f64[128] → compatible
    reg.register_meta(
        "a",
        meta_with_schemas(
            Some(Schema::vector(DataType::Float64, 128)),
            None,
        ),
        CacheKey::hash_data(b"a"),
    );
    reg.register_meta(
        "b",
        meta_with_schemas(
            None,
            Some(Schema::vector(DataType::Float64, 128)),
        ),
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
    let graph = linear_pipeline(vec![
        Node::new("a", "A", "F"),
        Node::new("b", "B", "F"),
    ]);

    let mut reg = SimpleFilterRegistry::new();
    // A outputs f64, B expects i64 → incompatible
    reg.register_meta(
        "a",
        meta_with_schemas(
            Some(Schema::vector(DataType::Float64, 128)),
            None,
        ),
        CacheKey::hash_data(b"a"),
    );
    reg.register_meta(
        "b",
        meta_with_schemas(
            None,
            Some(Schema::vector(DataType::Int64, 128)),
        ),
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
    let graph = linear_pipeline(vec![
        Node::new("a", "A", "F"),
        Node::new("b", "B", "F"),
    ]);

    let mut reg = SimpleFilterRegistry::new();
    // A outputs f64[128], B expects f64[256] → shape mismatch
    reg.register_meta(
        "a",
        meta_with_schemas(
            Some(Schema::vector(DataType::Float64, 128)),
            None,
        ),
        CacheKey::hash_data(b"a"),
    );
    reg.register_meta(
        "b",
        meta_with_schemas(
            None,
            Some(Schema::vector(DataType::Float64, 256)),
        ),
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
    let graph = linear_pipeline(vec![
        Node::new("a", "A", "F"),
        Node::new("b", "B", "F"),
    ]);

    let mut reg = SimpleFilterRegistry::new();
    // A outputs f64[batch, 128], B expects f64[32, 128] → compatible (dynamic batch)
    reg.register_meta(
        "a",
        meta_with_schemas(
            Some(Schema::batched(DataType::Float64, &[128])),
            None,
        ),
        CacheKey::hash_data(b"a"),
    );
    reg.register_meta(
        "b",
        meta_with_schemas(
            None,
            Some(Schema::matrix(DataType::Float64, 32, 128)),
        ),
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
    let graph = linear_pipeline(vec![
        Node::new("a", "A", "F"),
        Node::new("b", "B", "F"),
    ]);

    let mut reg = SimpleFilterRegistry::new();
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
