use crate::plan::ExecutionPlan;
use soma_core::cache::{CacheKey, CacheStore};
use soma_core::error::Result;
use soma_core::filter::{Filter, FilterMeta};
use soma_core::graph::{Graph, NodeId};
use std::collections::{HashMap, HashSet};

/// Compilation mode affects caching behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompileMode {
    /// Full caching: skip nodes whose outputs are cached.
    Inference,
    /// Cache states only: re-execute forwards for gradient flow.
    Differentiable,
    /// No caching at all: force re-execution of everything.
    NoCache,
}

/// Diagnostic message emitted during compilation.
#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub node_id: NodeId,
    pub level: DiagnosticLevel,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticLevel {
    Warning,
    Info,
}

/// Compiled result: the plan plus any diagnostics.
pub struct CompileResult {
    pub plan: ExecutionPlan,
    pub diagnostics: Vec<Diagnostic>,
}

/// Registry that maps node IDs to their filter metadata.
/// The compiler needs metadata (cacheable, differentiable, etc.)
/// but doesn't need the actual filter implementations.
pub trait FilterRegistry: Send + Sync {
    fn meta(&self, node_id: &str) -> Option<FilterMeta>;
    fn config_hash(&self, node_id: &str) -> Option<CacheKey>;
}

/// Simple in-memory filter registry for compilation.
pub struct SimpleFilterRegistry {
    entries: HashMap<String, (FilterMeta, CacheKey)>,
}

impl SimpleFilterRegistry {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    pub fn register(&mut self, node_id: impl Into<String>, filter: &dyn Filter) {
        let id = node_id.into();
        self.entries.insert(id, (filter.meta(), filter.config_hash()));
    }

    pub fn register_meta(
        &mut self,
        node_id: impl Into<String>,
        meta: FilterMeta,
        config_hash: CacheKey,
    ) {
        self.entries.insert(node_id.into(), (meta, config_hash));
    }
}

impl Default for SimpleFilterRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl FilterRegistry for SimpleFilterRegistry {
    fn meta(&self, node_id: &str) -> Option<FilterMeta> {
        self.entries.get(node_id).map(|(m, _)| m.clone())
    }

    fn config_hash(&self, node_id: &str) -> Option<CacheKey> {
        self.entries.get(node_id).map(|(_, h)| h.clone())
    }
}

/// Compiles a Graph into an ExecutionPlan.
pub struct Compiler<'a> {
    graph: &'a Graph,
    registry: &'a dyn FilterRegistry,
    mode: CompileMode,
    diagnostics: Vec<Diagnostic>,
}

impl<'a> Compiler<'a> {
    pub fn new(
        graph: &'a Graph,
        registry: &'a dyn FilterRegistry,
        mode: CompileMode,
    ) -> Self {
        Self {
            graph,
            registry,
            mode,
            diagnostics: Vec::new(),
        }
    }

    /// Compile the graph into an execution plan.
    pub fn compile(mut self, cache: Option<&dyn CacheStore>) -> Result<CompileResult> {
        self.graph.validate()?;

        let sorted = self.graph.topological_sort()?;

        if sorted.is_empty() {
            return Ok(CompileResult {
                plan: ExecutionPlan::Empty,
                diagnostics: self.diagnostics,
            });
        }

        // Check gradient flow
        self.check_gradient_flow(&sorted);

        // Build the structural plan (detect parallelism)
        let plan = self.build_plan(&sorted);

        // Resolve caching if applicable
        let plan = if let Some(cache) = cache {
            self.resolve_cache(plan, cache, &sorted)?
        } else {
            plan
        };

        // Resolve distribution (wrap Remote nodes)
        let plan = self.resolve_distribution(plan);

        let plan = plan.simplify();

        Ok(CompileResult {
            plan,
            diagnostics: self.diagnostics,
        })
    }

    /// Build a plan from topologically sorted nodes, detecting parallelism.
    fn build_plan(&self, sorted: &[&str]) -> ExecutionPlan {
        // Compute topological levels (nodes at the same level can run in parallel)
        let levels = self.compute_levels(sorted);

        let mut plan_steps: Vec<ExecutionPlan> = Vec::new();

        for level in &levels {
            if level.len() == 1 {
                plan_steps.push(ExecutionPlan::Execute {
                    node_id: level[0].to_string(),
                });
            } else {
                // Multiple nodes at same level → parallel
                let branches: Vec<ExecutionPlan> = level
                    .iter()
                    .map(|id| ExecutionPlan::Execute {
                        node_id: id.to_string(),
                    })
                    .collect();
                plan_steps.push(ExecutionPlan::Parallel(branches));
            }
        }

        if plan_steps.len() == 1 {
            plan_steps.into_iter().next().unwrap()
        } else {
            ExecutionPlan::Sequence(plan_steps)
        }
    }

    /// Compute topological levels: groups of nodes that can execute concurrently.
    /// Each node's level = max(predecessor levels) + 1.
    fn compute_levels<'b>(&self, sorted: &[&'b str]) -> Vec<Vec<&'b str>> {
        let mut node_level: HashMap<&str, usize> = HashMap::new();
        let mut max_level: usize = 0;

        for &node in sorted {
            let preds = self.graph.predecessors(node);
            let level = if preds.is_empty() {
                0
            } else {
                preds
                    .iter()
                    .map(|p| node_level.get(p).copied().unwrap_or(0) + 1)
                    .max()
                    .unwrap_or(0)
            };
            node_level.insert(node, level);
            if level > max_level {
                max_level = level;
            }
        }

        let mut levels: Vec<Vec<&str>> = vec![Vec::new(); max_level + 1];
        for &node in sorted {
            let level = node_level[node];
            levels[level].push(node);
        }

        // Remove empty levels (shouldn't happen but defensive)
        levels.retain(|l| !l.is_empty());
        levels
    }

    /// Resolve caching: replace Execute nodes with Cached when possible.
    /// Implements cascade invalidation.
    fn resolve_cache(
        &self,
        plan: ExecutionPlan,
        cache: &dyn CacheStore,
        sorted: &[&str],
    ) -> Result<ExecutionPlan> {
        if self.mode == CompileMode::NoCache {
            return Ok(plan);
        }

        // Compute cache keys for all nodes in topological order.
        // A node's key depends on its config + its predecessors' keys.
        let mut node_keys: HashMap<String, CacheKey> = HashMap::new();
        let mut cached_nodes: HashSet<String> = HashSet::new();

        for &node_id in sorted {
            let config_hash = match self.registry.config_hash(node_id) {
                Some(h) => h,
                None => continue, // no filter registered, can't cache
            };

            let meta = self.registry.meta(node_id);
            let cacheable = meta.as_ref().is_some_and(|m| m.cacheable);

            // In differentiable mode, only cache states (not forward outputs)
            // For simplicity at this stage, we skip caching in differentiable mode
            let can_cache = cacheable && self.mode == CompileMode::Inference;

            // Build the cache key from config + predecessor output keys
            let pred_ids = self.graph.predecessors(node_id);
            let mut key_parts: Vec<Vec<u8>> = vec![config_hash.0.to_vec()];
            for pred in &pred_ids {
                if let Some(pred_key) = node_keys.get(*pred) {
                    key_parts.push(pred_key.0.to_vec());
                }
            }
            let parts_refs: Vec<&[u8]> = key_parts.iter().map(|p| p.as_slice()).collect();
            let key = CacheKey::from_parts(&parts_refs);
            node_keys.insert(node_id.to_string(), key.clone());

            // Check if this node's output exists in cache
            if can_cache {
                // Only use cache if ALL predecessors are also cached (or roots).
                // This ensures cascade invalidation: if any upstream re-executed,
                // the key is already different (since it includes predecessor keys).
                if cache.exists(&key)? {
                    cached_nodes.insert(node_id.to_string());
                }
            }
        }

        // Replace Execute nodes with Cached where applicable
        Ok(self.apply_cache_to_plan(plan, &cached_nodes, &node_keys))
    }

    fn apply_cache_to_plan(
        &self,
        plan: ExecutionPlan,
        cached: &HashSet<String>,
        keys: &HashMap<String, CacheKey>,
    ) -> ExecutionPlan {
        match plan {
            ExecutionPlan::Execute { ref node_id } => {
                if cached.contains(node_id)
                    && let Some(key) = keys.get(node_id)
                {
                    return ExecutionPlan::Cached {
                        node_id: node_id.clone(),
                        key: key.clone(),
                    };
                }
                plan
            }
            ExecutionPlan::Sequence(steps) => ExecutionPlan::Sequence(
                steps
                    .into_iter()
                    .map(|s| self.apply_cache_to_plan(s, cached, keys))
                    .collect(),
            ),
            ExecutionPlan::Parallel(branches) => ExecutionPlan::Parallel(
                branches
                    .into_iter()
                    .map(|b| self.apply_cache_to_plan(b, cached, keys))
                    .collect(),
            ),
            other => other,
        }
    }

    /// Wrap nodes with Remote distribution in ExecutionPlan::Remote.
    fn resolve_distribution(&self, plan: ExecutionPlan) -> ExecutionPlan {
        match plan {
            ExecutionPlan::Execute { ref node_id } => {
                if let Some(meta) = self.registry.meta(node_id) {
                    match &meta.distribution {
                        soma_core::filter::Distribution::Remote(target) => {
                            ExecutionPlan::Remote {
                                node_id: node_id.clone(),
                                target: target.clone(),
                                plan: Box::new(plan),
                            }
                        }
                        _ => plan,
                    }
                } else {
                    plan
                }
            }
            ExecutionPlan::Sequence(steps) => ExecutionPlan::Sequence(
                steps.into_iter().map(|s| self.resolve_distribution(s)).collect(),
            ),
            ExecutionPlan::Parallel(branches) => ExecutionPlan::Parallel(
                branches.into_iter().map(|b| self.resolve_distribution(b)).collect(),
            ),
            other => other,
        }
    }

    /// Check gradient flow and emit warnings for interruptions.
    fn check_gradient_flow(&mut self, sorted: &[&str]) {
        let mut gradient_flows = true;

        for &node_id in sorted {
            if let Some(meta) = self.registry.meta(node_id) {
                if gradient_flows && !meta.differentiable {
                    self.diagnostics.push(Diagnostic {
                        node_id: node_id.to_string(),
                        level: DiagnosticLevel::Warning,
                        message: format!(
                            "gradient flow interrupted at `{}` ({:?}). \
                             Downstream filters will not receive upstream gradients.",
                            node_id, meta.kind,
                        ),
                    });
                    gradient_flows = false;
                } else if !gradient_flows && meta.differentiable {
                    gradient_flows = true;
                }
            }
        }
    }
}

/// Convenience function: compile a graph with default settings.
pub fn compile(
    graph: &Graph,
    registry: &dyn FilterRegistry,
    mode: CompileMode,
    cache: Option<&dyn CacheStore>,
) -> Result<CompileResult> {
    Compiler::new(graph, registry, mode).compile(cache)
}

#[cfg(test)]
mod tests {
    use super::*;
    use soma_core::cache::EntryMeta;
    use soma_core::error::SomaError;
    use soma_core::filter::{FilterKind, StreamMode};
    use soma_core::graph::{linear_pipeline, Edge, Graph, Node};
    use soma_core::value::Value;
    use std::sync::Mutex;

    // ── Mock cache store ──

    struct MockCacheStore {
        entries: Mutex<HashSet<CacheKey>>,
    }

    impl MockCacheStore {
        fn new() -> Self {
            Self {
                entries: Mutex::new(HashSet::new()),
            }
        }

        fn insert(&self, key: CacheKey) {
            self.entries.lock().unwrap().insert(key);
        }
    }

    impl CacheStore for MockCacheStore {
        fn get(&self, _key: &CacheKey) -> Result<Option<Value>> {
            Ok(None)
        }
        fn put(&self, _key: &CacheKey, _value: &Value) -> Result<()> {
            Ok(())
        }
        fn exists(&self, key: &CacheKey) -> Result<bool> {
            Ok(self.entries.lock().unwrap().contains(key))
        }
        fn remove(&self, _key: &CacheKey) -> Result<()> {
            Ok(())
        }
        fn metadata(&self, _key: &CacheKey) -> Result<Option<EntryMeta>> {
            Ok(None)
        }
    }

    // ── Helpers ──

    fn make_meta(kind: FilterKind, differentiable: bool) -> FilterMeta {
        FilterMeta {
            name: "test".into(),
            kind,
            cacheable: true,
            differentiable,
            stream_mode: StreamMode::FixedState,
            distribution: soma_core::filter::Distribution::Local,
        }
    }

    fn register_nodes(registry: &mut SimpleFilterRegistry, ids: &[&str], meta: FilterMeta) {
        for (i, id) in ids.iter().enumerate() {
            let hash = CacheKey::from_parts(&[id.as_bytes(), &[i as u8]]);
            registry.register_meta(*id, meta.clone(), hash);
        }
    }

    // ── Tests ──

    #[test]
    fn compile_empty_graph() {
        let graph = Graph::new();
        let registry = SimpleFilterRegistry::new();
        let result = compile(&graph, &registry, CompileMode::Inference, None).unwrap();
        assert!(matches!(result.plan, ExecutionPlan::Empty));
    }

    #[test]
    fn compile_single_node() {
        let mut graph = Graph::new();
        graph.add_node(Node::new("a", "A", "F"));
        let mut registry = SimpleFilterRegistry::new();
        register_nodes(
            &mut registry,
            &["a"],
            make_meta(FilterKind::Trainable, true),
        );

        let result = compile(&graph, &registry, CompileMode::Inference, None).unwrap();
        assert!(matches!(result.plan, ExecutionPlan::Execute { .. }));
    }

    #[test]
    fn compile_linear_pipeline_produces_sequence() {
        let graph = linear_pipeline(vec![
            Node::new("a", "Scaler", "F"),
            Node::new("b", "PCA", "F"),
            Node::new("c", "SVM", "F"),
        ]);
        let mut registry = SimpleFilterRegistry::new();
        register_nodes(
            &mut registry,
            &["a", "b", "c"],
            make_meta(FilterKind::Trainable, true),
        );

        let result = compile(&graph, &registry, CompileMode::Inference, None).unwrap();

        if let ExecutionPlan::Sequence(steps) = &result.plan {
            assert_eq!(steps.len(), 3);
            assert!(steps.iter().all(|s| matches!(s, ExecutionPlan::Execute { .. })));
        } else {
            panic!("expected Sequence, got: {:?}", result.plan);
        }
    }

    #[test]
    fn compile_diamond_detects_parallelism() {
        let mut graph = Graph::new();
        graph.add_node(Node::new("root", "Root", "F"));
        graph.add_node(Node::new("b1", "B1", "F"));
        graph.add_node(Node::new("b2", "B2", "F"));
        graph.add_node(Node::new("merge", "Merge", "F"));
        graph.add_edge(Edge::data("e1", "root", "b1"));
        graph.add_edge(Edge::data("e2", "root", "b2"));
        graph.add_edge(Edge::data("e3", "b1", "merge"));
        graph.add_edge(Edge::data("e4", "b2", "merge"));

        let mut registry = SimpleFilterRegistry::new();
        register_nodes(
            &mut registry,
            &["root", "b1", "b2", "merge"],
            make_meta(FilterKind::Trainable, true),
        );

        let result = compile(&graph, &registry, CompileMode::Inference, None).unwrap();

        // Should be: Sequence(Execute(root), Parallel(Execute(b1), Execute(b2)), Execute(merge))
        if let ExecutionPlan::Sequence(steps) = &result.plan {
            assert_eq!(steps.len(), 3);
            assert!(matches!(&steps[0], ExecutionPlan::Execute { node_id } if node_id == "root"));
            assert!(matches!(&steps[1], ExecutionPlan::Parallel(branches) if branches.len() == 2));
            assert!(matches!(&steps[2], ExecutionPlan::Execute { node_id } if node_id == "merge"));
        } else {
            panic!("expected Sequence, got: {:?}", result.plan);
        }
    }

    #[test]
    fn compile_independent_roots_parallel() {
        let mut graph = Graph::new();
        graph.add_node(Node::new("a", "A", "F"));
        graph.add_node(Node::new("b", "B", "F"));
        // No edges: fully independent

        let mut registry = SimpleFilterRegistry::new();
        register_nodes(
            &mut registry,
            &["a", "b"],
            make_meta(FilterKind::Trainable, true),
        );

        let result = compile(&graph, &registry, CompileMode::Inference, None).unwrap();

        // Both at level 0 → Parallel
        assert!(matches!(result.plan, ExecutionPlan::Parallel(_)));
    }

    #[test]
    fn cache_resolution_replaces_cached_nodes() {
        let graph = linear_pipeline(vec![
            Node::new("a", "Scaler", "F"),
            Node::new("b", "PCA", "F"),
            Node::new("c", "SVM", "F"),
        ]);

        let mut registry = SimpleFilterRegistry::new();
        register_nodes(
            &mut registry,
            &["a", "b", "c"],
            make_meta(FilterKind::Trainable, true),
        );

        // Pre-compute the cache key for node "a" (same logic as compiler)
        let a_config = registry.config_hash("a").unwrap();
        let a_cache_key = CacheKey::from_parts(&[&a_config.0]);

        let cache = MockCacheStore::new();
        cache.insert(a_cache_key);

        let result = compile(&graph, &registry, CompileMode::Inference, Some(&cache)).unwrap();

        if let ExecutionPlan::Sequence(steps) = &result.plan {
            assert!(
                matches!(&steps[0], ExecutionPlan::Cached { node_id, .. } if node_id == "a"),
                "first node should be cached, got: {:?}",
                steps[0]
            );
            assert!(matches!(&steps[1], ExecutionPlan::Execute { .. }));
            assert!(matches!(&steps[2], ExecutionPlan::Execute { .. }));
        } else {
            panic!("expected Sequence, got: {:?}", result.plan);
        }
    }

    #[test]
    fn cascade_invalidation_different_config_changes_keys() {
        // Register with config hash "v1"
        let mut reg1 = SimpleFilterRegistry::new();
        reg1.register_meta(
            "a",
            make_meta(FilterKind::Trainable, true),
            CacheKey::hash_data(b"scaler_v1"),
        );
        reg1.register_meta(
            "b",
            make_meta(FilterKind::Trainable, true),
            CacheKey::hash_data(b"pca_v1"),
        );

        // Register with config hash "v2" for node "a"
        let mut reg2 = SimpleFilterRegistry::new();
        reg2.register_meta(
            "a",
            make_meta(FilterKind::Trainable, true),
            CacheKey::hash_data(b"scaler_v2"), // changed!
        );
        reg2.register_meta(
            "b",
            make_meta(FilterKind::Trainable, true),
            CacheKey::hash_data(b"pca_v1"), // same
        );

        // Compute keys for both configurations
        // The plans have same structure but when cache keys are computed,
        // changing "a" config changes "b"'s key too (cascade).
        // We verify this by computing keys manually:
        let a_key_v1 = CacheKey::from_parts(&[&CacheKey::hash_data(b"scaler_v1").0]);
        let b_key_v1 = CacheKey::from_parts(&[&CacheKey::hash_data(b"pca_v1").0, &a_key_v1.0]);

        let a_key_v2 = CacheKey::from_parts(&[&CacheKey::hash_data(b"scaler_v2").0]);
        let b_key_v2 = CacheKey::from_parts(&[&CacheKey::hash_data(b"pca_v1").0, &a_key_v2.0]);

        // a changed → a's key changed
        assert_ne!(a_key_v1, a_key_v2);
        // b's config didn't change but a's key is in b's key → b's key also changed
        assert_ne!(b_key_v1, b_key_v2);
    }

    #[test]
    fn no_cache_mode_skips_all_caching() {
        let graph = linear_pipeline(vec![
            Node::new("a", "A", "F"),
            Node::new("b", "B", "F"),
        ]);

        let mut registry = SimpleFilterRegistry::new();
        register_nodes(
            &mut registry,
            &["a", "b"],
            make_meta(FilterKind::Trainable, true),
        );

        // Put everything in cache
        let a_config = registry.config_hash("a").unwrap();
        let a_key = CacheKey::from_parts(&[&a_config.0]);
        let cache = MockCacheStore::new();
        cache.insert(a_key);

        let result = compile(&graph, &registry, CompileMode::NoCache, Some(&cache)).unwrap();

        // Nothing should be cached
        assert_eq!(result.plan.cached_count(), 0);
    }

    #[test]
    fn differentiable_mode_skips_output_caching() {
        let graph = linear_pipeline(vec![
            Node::new("a", "A", "F"),
            Node::new("b", "B", "F"),
        ]);

        let mut registry = SimpleFilterRegistry::new();
        register_nodes(
            &mut registry,
            &["a", "b"],
            make_meta(FilterKind::Trainable, true),
        );

        let a_config = registry.config_hash("a").unwrap();
        let a_key = CacheKey::from_parts(&[&a_config.0]);
        let cache = MockCacheStore::new();
        cache.insert(a_key);

        let result =
            compile(&graph, &registry, CompileMode::Differentiable, Some(&cache)).unwrap();

        // Differentiable mode should not cache forward outputs
        assert_eq!(result.plan.cached_count(), 0);
    }

    #[test]
    fn gradient_flow_diagnostic_on_opaque() {
        let graph = linear_pipeline(vec![
            Node::new("scaler", "Scaler", "F"),
            Node::new("tree", "DecisionTree", "F"),
            Node::new("linear", "Linear", "F"),
        ]);

        let mut registry = SimpleFilterRegistry::new();
        registry.register_meta(
            "scaler",
            make_meta(FilterKind::Trainable, true),
            CacheKey::hash_data(b"s"),
        );
        registry.register_meta(
            "tree",
            make_meta(FilterKind::Opaque, false), // not differentiable
            CacheKey::hash_data(b"t"),
        );
        registry.register_meta(
            "linear",
            make_meta(FilterKind::Trainable, true),
            CacheKey::hash_data(b"l"),
        );

        let result = compile(&graph, &registry, CompileMode::Inference, None).unwrap();

        assert_eq!(result.diagnostics.len(), 1);
        assert_eq!(result.diagnostics[0].node_id, "tree");
        assert_eq!(result.diagnostics[0].level, DiagnosticLevel::Warning);
        assert!(result.diagnostics[0].message.contains("gradient flow interrupted"));
    }

    #[test]
    fn no_diagnostic_when_all_differentiable() {
        let graph = linear_pipeline(vec![
            Node::new("a", "A", "F"),
            Node::new("b", "B", "F"),
        ]);

        let mut registry = SimpleFilterRegistry::new();
        register_nodes(
            &mut registry,
            &["a", "b"],
            make_meta(FilterKind::Trainable, true),
        );

        let result = compile(&graph, &registry, CompileMode::Inference, None).unwrap();
        assert!(result.diagnostics.is_empty());
    }

    #[test]
    fn compile_rejects_cycle() {
        let mut graph = Graph::new();
        graph.add_node(Node::new("a", "A", "F"));
        graph.add_node(Node::new("b", "B", "F"));
        graph.add_edge(Edge::data("e1", "a", "b"));
        graph.add_edge(Edge::data("e2", "b", "a"));

        let registry = SimpleFilterRegistry::new();
        let result = compile(&graph, &registry, CompileMode::Inference, None);
        assert!(matches!(result, Err(SomaError::CycleDetected)));
    }

    #[test]
    fn plan_summary_is_accurate() {
        let mut graph = Graph::new();
        graph.add_node(Node::new("root", "Root", "F"));
        graph.add_node(Node::new("b1", "B1", "F"));
        graph.add_node(Node::new("b2", "B2", "F"));
        graph.add_node(Node::new("end", "End", "F"));
        graph.add_edge(Edge::data("e1", "root", "b1"));
        graph.add_edge(Edge::data("e2", "root", "b2"));
        graph.add_edge(Edge::data("e3", "b1", "end"));
        graph.add_edge(Edge::data("e4", "b2", "end"));

        let mut registry = SimpleFilterRegistry::new();
        register_nodes(
            &mut registry,
            &["root", "b1", "b2", "end"],
            make_meta(FilterKind::Trainable, true),
        );

        let result = compile(&graph, &registry, CompileMode::Inference, None).unwrap();
        let summary = result.plan.summary();
        assert_eq!(summary.total_nodes, 4);
        assert_eq!(summary.parallel_branches, 2);
    }

    #[test]
    fn distribution_wraps_remote_nodes() {
        let graph = linear_pipeline(vec![
            Node::new("preprocess", "Preprocess", "F"),
            Node::new("gpu_train", "GpuTrain", "F"),
            Node::new("evaluate", "Evaluate", "F"),
        ]);

        let mut registry = SimpleFilterRegistry::new();
        // preprocess: local
        registry.register_meta(
            "preprocess",
            make_meta(FilterKind::Trainable, true),
            CacheKey::hash_data(b"pre"),
        );
        // gpu_train: remote on GPU tag
        let mut gpu_meta = make_meta(FilterKind::Trainable, true);
        gpu_meta.distribution = soma_core::filter::Distribution::Remote(
            soma_core::filter::RemoteTarget::Tag("gpu".into()),
        );
        registry.register_meta("gpu_train", gpu_meta, CacheKey::hash_data(b"gpu"));
        // evaluate: local
        registry.register_meta(
            "evaluate",
            make_meta(FilterKind::Trainable, true),
            CacheKey::hash_data(b"eval"),
        );

        let result = compile(&graph, &registry, CompileMode::Inference, None).unwrap();

        // Should be: Sequence(Execute(preprocess), Remote(gpu_train, ...), Execute(evaluate))
        if let ExecutionPlan::Sequence(steps) = &result.plan {
            assert_eq!(steps.len(), 3);
            assert!(matches!(&steps[0], ExecutionPlan::Execute { node_id } if node_id == "preprocess"));
            assert!(
                matches!(&steps[1], ExecutionPlan::Remote { node_id, target, .. }
                    if node_id == "gpu_train"
                    && *target == soma_core::filter::RemoteTarget::Tag("gpu".into())
                ),
                "expected Remote, got: {:?}", steps[1]
            );
            assert!(matches!(&steps[2], ExecutionPlan::Execute { node_id } if node_id == "evaluate"));
        } else {
            panic!("expected Sequence, got: {:?}", result.plan);
        }
    }

    #[test]
    fn local_distribution_not_wrapped() {
        let graph = linear_pipeline(vec![
            Node::new("a", "A", "F"),
            Node::new("b", "B", "F"),
        ]);

        let mut registry = SimpleFilterRegistry::new();
        register_nodes(
            &mut registry,
            &["a", "b"],
            make_meta(FilterKind::Trainable, true),
        );

        let result = compile(&graph, &registry, CompileMode::Inference, None).unwrap();

        // No Remote nodes
        let ids = result.plan.node_ids();
        assert_eq!(ids.len(), 2);
        // Should all be Execute, no Remote wrapper
        if let ExecutionPlan::Sequence(steps) = &result.plan {
            assert!(steps.iter().all(|s| matches!(s, ExecutionPlan::Execute { .. })));
        }
    }
}
