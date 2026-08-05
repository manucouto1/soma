//! Graph → ExecutionPlan compiler.
//!
//! Compilation phases: topological sort → parallelism detection →
//! cache resolution → schema validation → distribution wrapping → simplification.

use crate::plan::ExecutionPlan;
use somatize_core::cache::{CacheKey, CacheStore};
use somatize_core::control::LoopCondition;
use somatize_core::error::{Result, SomaError};
use somatize_core::filter::{Filter, FilterMeta};
use somatize_core::graph::{Graph, NodeId};
use somatize_core::node::NodeMeta;
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
    /// The node the diagnostic is about.
    pub node_id: NodeId,
    /// How seriously to take it.
    pub level: DiagnosticLevel,
    /// Human-readable description of what the compiler noticed.
    pub message: String,
}

/// Severity of a [`Diagnostic`]. Nothing here fails compilation — a
/// condition worth stopping for is returned as an error, not collected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticLevel {
    /// Probably not what the author intended (e.g. a gradient path broken
    /// by a non-differentiable node); the plan still compiles.
    Warning,
    /// Worth knowing, nothing to fix.
    Info,
}

/// Compiled result: the plan plus any diagnostics.
#[derive(Debug)]
pub struct CompileResult {
    /// The executable plan the runtime walks.
    pub plan: ExecutionPlan,
    /// What the compiler noticed along the way; never fatal (see
    /// [`DiagnosticLevel`]).
    pub diagnostics: Vec<Diagnostic>,
}

/// Registry that maps node IDs to their metadata.
///
/// The compiler needs metadata (cacheable, differentiable, schemas) but
/// not the implementations behind it. One required accessor, answering
/// for both kinds of node: an optional `step_meta` alongside a required
/// `meta` is how half the schema validation came to be skipped by
/// whichever registry forgot to override it.
pub trait NodeRegistry: Send + Sync {
    /// A node's contract — schemas, cacheability, effectfulness — whichever
    /// kind it is. `None` means the graph names a node nobody registered,
    /// which the compiler reports rather than guesses around.
    fn node_meta(&self, node_id: &str) -> Option<NodeMeta>;

    /// The node's configuration identity, folded into cache keys. Required
    /// rather than derived because only the registry knows how a node's
    /// configuration is canonicalized (Rust: canonical CBOR of fields;
    /// Python: qualname + config + source hash).
    fn config_hash(&self, node_id: &str) -> Option<CacheKey>;

    /// The computational view, for the phases that only make sense for a
    /// filter: gradient flow, differentiable collapsing.
    ///
    /// `None` for an effectful node — deliberately. Those phases ask
    /// "should a gradient pass through here", and the answer for a step
    /// is not "no, and warn about it" but "the question does not apply".
    fn meta(&self, node_id: &str) -> Option<FilterMeta> {
        self.node_meta(node_id)
            .filter(|m| !m.effectful)
            .map(|m| m.as_filter_meta())
    }
}

/// Simple in-memory node registry for compilation.
pub struct SimpleNodeRegistry {
    entries: HashMap<String, (NodeMeta, CacheKey)>,
}

impl SimpleNodeRegistry {
    /// An empty registry; populate it with [`register`](Self::register),
    /// [`register_meta`](Self::register_meta) or
    /// [`register_step_meta`](Self::register_step_meta).
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Register a step's metadata, so its schemas take part in validation.
    pub fn register_step_meta(
        &mut self,
        node_id: impl Into<String>,
        meta: somatize_core::step::StepMeta,
    ) {
        let id = node_id.into();
        // A step's config hash is not derivable from its metadata; the
        // compiler only needs one for cache resolution, which does not
        // apply to an effectful node.
        let hash = CacheKey::from_parts(&[b"step-meta", id.as_bytes()]);
        self.entries.insert(id, (meta.into(), hash));
    }

    /// Register a filter, taking metadata and config hash from the
    /// instance itself.
    pub fn register(&mut self, node_id: impl Into<String>, filter: &dyn Filter) {
        let id = node_id.into();
        self.entries
            .insert(id, (filter.meta().into(), filter.config_hash()));
    }

    /// Register filter metadata directly, for callers that have no filter
    /// instance to hand — a plan received over the wire, a test.
    pub fn register_meta(
        &mut self,
        node_id: impl Into<String>,
        meta: FilterMeta,
        config_hash: CacheKey,
    ) {
        self.entries
            .insert(node_id.into(), (meta.into(), config_hash));
    }
}

impl Default for SimpleNodeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl NodeRegistry for SimpleNodeRegistry {
    fn node_meta(&self, node_id: &str) -> Option<NodeMeta> {
        self.entries.get(node_id).map(|(m, _)| m.clone())
    }

    fn config_hash(&self, node_id: &str) -> Option<CacheKey> {
        self.entries.get(node_id).map(|(_, h)| h.clone())
    }
}

/// Graph-wide analysis shared by every level of plan construction.
///
/// Both maps are computed once over the whole graph. Sub-plans (loop bodies,
/// branch arms) project onto them rather than recomputing, so a node's
/// position relative to the rest of the graph is the same wherever it is
/// emitted.
struct PlanCtx<'b> {
    /// Topological level per node; nodes sharing a level are independent.
    levels: HashMap<&'b str, usize>,
    /// `dominators[n]` — every node that lies on all paths from a root to `n`.
    dominators: HashMap<&'b str, HashSet<&'b str>>,
}

impl<'b> PlanCtx<'b> {
    /// Does `d` lie on every path from a root to `n`? (Reflexive: `d` dominates itself.)
    fn dominates(&self, d: &str, n: &str) -> bool {
        self.dominators.get(n).is_some_and(|set| set.contains(d))
    }

    fn level_of(&self, n: &str) -> usize {
        self.levels.get(n).copied().unwrap_or(0)
    }

    /// Group nodes into ordered levels, dropping levels the subset doesn't occupy.
    fn group_by_level(&self, nodes: &[&'b str]) -> Vec<Vec<&'b str>> {
        let mut by_level: Vec<(usize, Vec<&'b str>)> = Vec::new();
        for &n in nodes {
            let lvl = self.level_of(n);
            match by_level.iter_mut().find(|(l, _)| *l == lvl) {
                Some((_, bucket)) => bucket.push(n),
                None => by_level.push((lvl, vec![n])),
            }
        }
        by_level.sort_by_key(|(l, _)| *l);
        by_level.into_iter().map(|(_, ns)| ns).collect()
    }

    /// Deterministic topological order: by level, then by id.
    fn in_topo_order(&self, set: HashSet<&'b str>) -> Vec<&'b str> {
        let mut out: Vec<&'b str> = set.into_iter().collect();
        out.sort_by(|a, b| self.level_of(a).cmp(&self.level_of(b)).then(a.cmp(b)));
        out
    }
}

/// Compiles a Graph into an ExecutionPlan.
pub struct Compiler<'a> {
    graph: &'a Graph,
    registry: &'a dyn NodeRegistry,
    mode: CompileMode,
    diagnostics: Vec<Diagnostic>,
}

impl<'a> Compiler<'a> {
    /// A compiler over `graph`, reading node contracts from `registry`.
    /// Nothing happens until [`compile`](Self::compile) is called.
    pub fn new(graph: &'a Graph, registry: &'a dyn NodeRegistry, mode: CompileMode) -> Self {
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

        // Check the shape of the graph itself
        self.check_connectivity();

        // Validate schema compatibility
        self.validate_schemas(&sorted)?;

        let ctx = PlanCtx {
            levels: self.compute_levels(&sorted),
            dominators: self.compute_dominators(&sorted),
        };

        // Reject ambiguous control flow before it can become a silent default
        self.validate_control_flow(&sorted, &ctx)?;

        // Build the structural plan (detect parallelism)
        let plan = self.plan_subset(&sorted, &ctx)?;

        // The plan carries no `Cached` nodes: cache lookups are resolved at
        // runtime, per node. A caller passing a cache gets a note saying so
        // — this used to be a whole "phase" that transformed nothing.
        if cache.is_some()
            && self.mode != CompileMode::NoCache
            && let Some(&first) = sorted.first()
        {
            self.diagnostics.push(Diagnostic {
                node_id: first.to_string(),
                level: DiagnosticLevel::Info,
                message: "cache lookups are resolved at runtime per node \
                          (key = hash(config + state + input)); the compiled plan \
                          contains no Cached nodes"
                    .to_string(),
            });
        }

        // Resolve distribution (wrap Remote nodes)
        let plan = self.resolve_distribution(plan);

        // Collapse consecutive differentiable nodes into Composite blocks
        let plan = self.collapse_differentiable(plan);

        let plan = plan.simplify();

        Ok(CompileResult {
            plan,
            diagnostics: self.diagnostics,
        })
    }

    /// Reject control flow whose meaning would otherwise be decided at
    /// runtime by whichever node happened to finish last.
    fn validate_control_flow<'b>(&self, sorted: &[&'b str], ctx: &PlanCtx<'b>) -> Result<()> {
        use somatize_core::graph::NodeKind;

        let all: HashSet<&str> = sorted.iter().copied().collect();

        for &node_id in sorted {
            let Some(node) = self.graph.node(node_id) else {
                continue;
            };
            match &node.kind {
                NodeKind::Loop { until, .. } => {
                    let body = self.claimed_subset(node_id, &all, ctx);
                    if body.is_empty() {
                        return Err(SomaError::Compilation(format!(
                            "loop `{node_id}` has an empty body: it needs at least one \
                             control edge to the node that starts each iteration"
                        )));
                    }
                    if matches!(until, LoopCondition::BodyTerminal) {
                        let terminals = self.body_terminals(&body);
                        if terminals.len() != 1 {
                            return Err(SomaError::Compilation(format!(
                                "loop `{node_id}` cannot infer its stop condition: its body has \
                                 {} terminal nodes ({}). Name the deciding node explicitly with \
                                 `LoopCondition::WhenSignaled`, or use `LoopCondition::Exhaust` \
                                 to always run `max_iterations` times",
                                terminals.len(),
                                terminals.join(", ")
                            )));
                        }
                    }
                    if let LoopCondition::WhenSignaled(target) = until
                        && !body.contains(&target.as_str())
                    {
                        return Err(SomaError::Compilation(format!(
                            "loop `{node_id}` waits on `{target}`, which is not in its body \
                             ({}) — it would never be re-evaluated",
                            body.join(", ")
                        )));
                    }
                }
                NodeKind::Branch { arms: declared } => {
                    let edges = self.control_targets(node_id, &all);
                    if edges.is_empty() {
                        return Err(SomaError::Compilation(format!(
                            "branch `{node_id}` has no arms: arms are the control edges \
                             leaving it, each labelled with the value that selects it"
                        )));
                    }
                    let mut seen: HashSet<String> = HashSet::new();
                    for (target, label) in &edges {
                        let label = label.clone().unwrap_or_else(|| target.to_string());
                        if !seen.insert(label.clone()) {
                            return Err(SomaError::Compilation(format!(
                                "branch `{node_id}` has two arms labelled `{label}` — \
                                 the second could never be selected"
                            )));
                        }
                    }

                    // When the node declares its label set, hold the edges to
                    // it in both directions. A mislabelled edge is otherwise
                    // an arm that simply never fires — visible only as a
                    // wrong answer, and only sometimes.
                    if !declared.is_empty() {
                        let declared_set: HashSet<&str> =
                            declared.iter().map(String::as_str).collect();

                        for label in &seen {
                            if !declared_set.contains(label.as_str())
                                && !somatize_core::control::is_default_arm(label)
                            {
                                return Err(SomaError::Compilation(format!(
                                    "branch `{node_id}` has an edge labelled `{label}`, which \
                                     is not among its declared arms ({}). Fix the label, or \
                                     declare it",
                                    declared.join(", ")
                                )));
                            }
                        }
                        for label in declared {
                            if !seen.contains(label) {
                                return Err(SomaError::Compilation(format!(
                                    "branch `{node_id}` declares arm `{label}` but no control \
                                     edge is labelled with it, so selecting it would fail at \
                                     runtime"
                                )));
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Nodes in `body` that no other body node depends on.
    fn body_terminals<'b>(&self, body: &[&'b str]) -> Vec<&'b str> {
        let member: HashSet<&str> = body.iter().copied().collect();
        body.iter()
            .copied()
            .filter(|n| {
                !self
                    .graph
                    .successors(n)
                    .iter()
                    .any(|s| member.contains(s as &str))
            })
            .collect()
    }

    /// Resolve `BodyTerminal` to the concrete node the executor will read.
    ///
    /// `validate_control_flow` has already rejected a body without exactly
    /// one terminal, so the error arm cannot fire. It is an error rather
    /// than a fallback because the fallback was `Exhaust`: a debug build
    /// asserted, and a release build quietly turned "stop when the body
    /// says so" into "run the full iteration count" — a loop that costs N
    /// times what it should, reported as success.
    fn resolve_loop_condition(
        &self,
        node_id: &str,
        until: &LoopCondition,
        body: &[&str],
    ) -> Result<LoopCondition> {
        match until {
            LoopCondition::BodyTerminal => match self.body_terminals(body).as_slice() {
                [only] => Ok(LoopCondition::WhenSignaled((*only).to_string())),
                terminals => Err(SomaError::Compilation(format!(
                    "loop `{node_id}` stops on its body terminal, but the body has {} \
                     of them{}. Name the one that decides with \
                     `LoopCondition::WhenSignaled`, or use `Exhaust` to always run \
                     the full count",
                    terminals.len(),
                    if terminals.is_empty() {
                        String::new()
                    } else {
                        format!(" ({})", terminals.join(", "))
                    }
                ))),
            },
            other => Ok(other.clone()),
        }
    }

    /// Plan a set of nodes, emitting each exactly once.
    ///
    /// A `Loop` or `Branch` in the set *owns* its body / arms: those nodes are
    /// compiled inside the construct and excluded from this level. Without
    /// that exclusion the body would run once more after the loop finished,
    /// and every arm would run unconditionally after the branch had already
    /// picked one.
    fn plan_subset<'b>(&self, nodes: &[&'b str], ctx: &PlanCtx<'b>) -> Result<ExecutionPlan> {
        let member: HashSet<&str> = nodes.iter().copied().collect();

        let mut owned: HashSet<&str> = HashSet::new();
        for &n in nodes {
            for m in self.owned_by(n, &member, ctx) {
                owned.insert(m);
            }
        }

        // Nodes not claimed by a construct, grouped by their graph-wide
        // topological level so relative ordering survives the projection.
        let top: Vec<&str> = nodes
            .iter()
            .copied()
            .filter(|n| !owned.contains(n))
            .collect();

        let mut plan_steps: Vec<ExecutionPlan> = Vec::new();
        for level in ctx.group_by_level(&top) {
            if level.len() == 1 {
                plan_steps.push(self.plan_for_node(level[0], ctx)?);
            } else {
                let branches: Vec<ExecutionPlan> = level
                    .iter()
                    .map(|id| self.plan_for_node(id, ctx))
                    .collect::<Result<_>>()?;
                plan_steps.push(ExecutionPlan::Parallel(branches));
            }
        }

        Ok(match plan_steps.len() {
            0 => ExecutionPlan::Empty,
            1 => plan_steps.into_iter().next().unwrap(),
            _ => ExecutionPlan::Sequence(plan_steps),
        })
    }

    /// The nodes a control-flow construct claims from `member`.
    ///
    /// Both `Loop` and `Branch` reach their sub-plans through **control**
    /// edges; a data edge leaving either is an ordinary downstream dependency.
    /// A target claims every node it dominates, so a node reachable from two
    /// arms is dominated by neither and stays outside — running once after the
    /// branch, which is what a convergence point should do.
    fn owned_by<'b>(
        &self,
        node_id: &'b str,
        member: &HashSet<&'b str>,
        ctx: &PlanCtx<'b>,
    ) -> Vec<&'b str> {
        use somatize_core::graph::NodeKind;

        let Some(node) = self.graph.node(node_id) else {
            return Vec::new();
        };
        if !matches!(
            node.kind,
            NodeKind::Loop { .. } | NodeKind::Branch { .. } | NodeKind::Step { .. }
        ) {
            return Vec::new();
        }

        let mut claimed = Vec::new();
        for (entry, _) in self.control_targets(node_id, member) {
            for &m in member {
                if m != node_id && ctx.dominates(entry, m) {
                    claimed.push(m);
                }
            }
        }
        claimed
    }

    /// Targets of control edges leaving `node_id`, restricted to `member`.
    fn control_targets<'b>(
        &self,
        node_id: &str,
        member: &HashSet<&'b str>,
    ) -> Vec<(&'b str, Option<String>)> {
        use somatize_core::graph::EdgeKind;

        self.graph
            .edges
            .iter()
            .filter(|e| e.source == node_id && e.kind == EdgeKind::Control)
            .filter_map(|e| member.get(e.target.as_str()).map(|t| (*t, e.label.clone())))
            .collect()
    }

    /// Generate the execution plan for a single node based on its kind.
    fn plan_for_node<'b>(&self, node_id: &'b str, ctx: &PlanCtx<'b>) -> Result<ExecutionPlan> {
        use somatize_core::graph::NodeKind;

        let node = match self.graph.node(node_id) {
            Some(n) => n,
            None => {
                return Ok(ExecutionPlan::Execute {
                    node_id: node_id.to_string(),
                });
            }
        };

        Ok(match &node.kind {
            NodeKind::Filter { .. } => ExecutionPlan::Execute {
                node_id: node_id.to_string(),
            },

            NodeKind::Step { .. } => {
                // Control edges leaving a step are the places it may hand
                // control to. Claimed the same way branch arms are, so each
                // target is compiled once, inside the step that reaches it.
                let all: HashSet<&str> = ctx.levels.keys().copied().collect();
                let handoffs: Vec<(NodeId, ExecutionPlan)> = self
                    .control_targets(node_id, &all)
                    .into_iter()
                    .map(|(target, _)| {
                        let nodes = self.dominated_subset(target, &all, ctx);
                        Ok((target.to_string(), self.plan_subset(&nodes, ctx)?))
                    })
                    .collect::<Result<_>>()?;
                ExecutionPlan::Step {
                    node_id: node_id.to_string(),
                    handoffs,
                }
            }

            NodeKind::SubGraph { graph } => {
                // Recursively compile the inner graph. An inner error is this
                // graph's error: the old fallback emitted a bare `Execute` for
                // the node, which deferred the failure to runtime under a
                // different name — inconsistent with the unknown-kind arm
                // below, which refuses rather than guesses.
                Compiler::new(graph, self.registry, self.mode)
                    .compile(None)?
                    .plan
            }

            NodeKind::Loop {
                max_iterations,
                until,
            } => {
                let all: HashSet<&str> = ctx.levels.keys().copied().collect();
                let body_nodes = self.claimed_subset(node_id, &all, ctx);
                let body = if body_nodes.is_empty() {
                    ExecutionPlan::Empty
                } else {
                    self.plan_subset(&body_nodes, ctx)?
                };
                ExecutionPlan::Loop {
                    node_id: node_id.to_string(),
                    body: Box::new(body),
                    max_iterations: *max_iterations,
                    until: self.resolve_loop_condition(node_id, until, &body_nodes)?,
                    // Whatever the stop condition is, a single-terminal body
                    // has one obvious thing to hand to the next pass.
                    carry_from: match self.body_terminals(&body_nodes).as_slice() {
                        [only] => Some((*only).to_string()),
                        _ => None,
                    },
                }
            }

            NodeKind::Branch { .. } => {
                let all: HashSet<&str> = ctx.levels.keys().copied().collect();
                let arms: Vec<(String, ExecutionPlan)> = self
                    .control_targets(node_id, &all)
                    .into_iter()
                    .map(|(target, label)| {
                        let label = label.unwrap_or_else(|| target.to_string());
                        let arm_nodes = self.dominated_subset(target, &all, ctx);
                        Ok((label, self.plan_subset(&arm_nodes, ctx)?))
                    })
                    .collect::<Result<_>>()?;
                ExecutionPlan::Branch {
                    node_id: node_id.to_string(),
                    arms,
                }
            }

            // `NodeKind` is `#[non_exhaustive]` and lives in another crate,
            // so this arm cannot be deleted — but it must not stay silent.
            // Falling through to `Execute` compiled an unknown kind as a
            // plain filter: a loop that never iterated, a step that was
            // never driven, and no diagnostic anywhere. Refusing to plan
            // what this compiler does not understand is the only safe
            // answer, and it turns a future omission into a clear error.
            other => {
                return Err(SomaError::Compilation(format!(
                    "node `{node_id}` has kind {other:?}, which this compiler \
                     does not know how to plan; the runtime would have run it \
                     as an ordinary filter"
                )));
            }
        })
    }

    /// Every node claimed by `node_id`'s control edges, in topological order.
    fn claimed_subset<'b>(
        &self,
        node_id: &'b str,
        universe: &HashSet<&'b str>,
        ctx: &PlanCtx<'b>,
    ) -> Vec<&'b str> {
        let mut claimed: HashSet<&str> = HashSet::new();
        for (entry, _) in self.control_targets(node_id, universe) {
            claimed.extend(self.dominated_subset(entry, universe, ctx));
        }
        ctx.in_topo_order(claimed)
    }

    /// `entry` plus everything it dominates, in topological order.
    fn dominated_subset<'b>(
        &self,
        entry: &'b str,
        universe: &HashSet<&'b str>,
        ctx: &PlanCtx<'b>,
    ) -> Vec<&'b str> {
        let set: HashSet<&str> = universe
            .iter()
            .copied()
            .filter(|&m| ctx.dominates(entry, m))
            .collect();
        ctx.in_topo_order(set)
    }

    /// Topological level of each node: `max(predecessor levels) + 1`.
    /// Nodes sharing a level have no dependency between them.
    fn compute_levels<'b>(&self, sorted: &[&'b str]) -> HashMap<&'b str, usize> {
        let mut node_level: HashMap<&str, usize> = HashMap::new();

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
        }

        node_level
    }

    /// Dominator sets over the DAG: `d` dominates `n` when every path from a
    /// root to `n` passes through `d`. Computed in topological order as
    /// `dom(n) = {n} ∪ ⋂ dom(pred)`.
    fn compute_dominators<'b>(&self, sorted: &[&'b str]) -> HashMap<&'b str, HashSet<&'b str>> {
        let mut dom: HashMap<&str, HashSet<&str>> = HashMap::new();

        for &node in sorted {
            let preds = self.graph.predecessors(node);
            let mut set: HashSet<&str> = HashSet::new();

            let mut pred_sets = preds.iter().filter_map(|p| dom.get(p));
            if let Some(first) = pred_sets.next() {
                set = first.clone();
                for other in pred_sets {
                    set.retain(|d| other.contains(d));
                }
            }
            set.insert(node);
            dom.insert(node, set);
        }

        dom
    }

    /// Cache resolution happens at RUNTIME, not here.
    ///
    /// The compiler never sees the dataset, so any key it could derive
    /// (formerly `H(config ‖ predecessor keys)`) is independent of the
    /// input data — the same graph on two different datasets would
    /// collide. The executor computes the real key
    /// `hash(config + state + input)` per node with the materialized
    /// input in hand, and skips execution on a hit.
    /// Wrap nodes with Remote distribution in ExecutionPlan::Remote.
    fn resolve_distribution(&self, plan: ExecutionPlan) -> ExecutionPlan {
        match plan {
            ExecutionPlan::Execute { ref node_id } | ExecutionPlan::Step { ref node_id, .. } => {
                if let Some(meta) = self.registry.node_meta(node_id) {
                    match &meta.distribution {
                        somatize_core::filter::Distribution::Remote(target) => {
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
                steps
                    .into_iter()
                    .map(|s| self.resolve_distribution(s))
                    .collect(),
            ),
            ExecutionPlan::Parallel(branches) => ExecutionPlan::Parallel(
                branches
                    .into_iter()
                    .map(|b| self.resolve_distribution(b))
                    .collect(),
            ),
            ExecutionPlan::Composite { ref node_ids } => {
                // If ALL nodes in the composite have a Remote target, wrap the
                // entire composite in a single Remote (using the first node's
                // target). Otherwise keep it local.
                let targets: Vec<_> = node_ids
                    .iter()
                    .filter_map(|nid| {
                        self.registry
                            .node_meta(nid)
                            .and_then(|m| match &m.distribution {
                                somatize_core::filter::Distribution::Remote(t) => Some(t.clone()),
                                _ => None,
                            })
                    })
                    .collect();

                if targets.len() == node_ids.len() && !targets.is_empty() {
                    let first_id = node_ids[0].clone();
                    ExecutionPlan::Remote {
                        node_id: first_id,
                        target: targets.into_iter().next().unwrap(),
                        plan: Box::new(plan),
                    }
                } else {
                    plan
                }
            }
            other => other,
        }
    }

    /// Collapse consecutive differentiable Execute nodes into Composite blocks.
    ///
    /// A `Composite` groups nodes that should share a PyTorch autograd session.
    /// Only groups 2+ consecutive `Execute` nodes where `meta.differentiable == true`.
    fn collapse_differentiable(&self, plan: ExecutionPlan) -> ExecutionPlan {
        match plan {
            ExecutionPlan::Sequence(steps) => {
                let mut result: Vec<ExecutionPlan> = Vec::new();
                let mut diff_group: Vec<String> = Vec::new();

                for step in steps {
                    if let ExecutionPlan::Execute { ref node_id } = step
                        && self
                            .registry
                            .meta(node_id)
                            .map(|m| m.differentiable)
                            .unwrap_or(false)
                    {
                        diff_group.push(node_id.clone());
                        continue;
                    }
                    // Flush accumulated differentiable group
                    Self::flush_diff_group(&mut diff_group, &mut result);
                    result.push(self.collapse_differentiable(step));
                }
                Self::flush_diff_group(&mut diff_group, &mut result);

                if result.len() == 1 {
                    result.pop().unwrap()
                } else {
                    ExecutionPlan::Sequence(result)
                }
            }
            ExecutionPlan::Parallel(branches) => ExecutionPlan::Parallel(
                branches
                    .into_iter()
                    .map(|b| self.collapse_differentiable(b))
                    .collect(),
            ),
            ExecutionPlan::Remote {
                node_id,
                target,
                plan,
            } => ExecutionPlan::Remote {
                node_id,
                target,
                plan: Box::new(self.collapse_differentiable(*plan)),
            },
            other => other,
        }
    }

    fn flush_diff_group(group: &mut Vec<String>, result: &mut Vec<ExecutionPlan>) {
        if group.len() > 1 {
            result.push(ExecutionPlan::Composite {
                node_ids: std::mem::take(group),
            });
        } else if let Some(id) = group.pop() {
            result.push(ExecutionPlan::Execute { node_id: id });
        }
    }

    /// Validate schema compatibility between connected filters.
    ///
    /// For each edge (A → B), checks that A's output_schema is compatible
    /// with B's input_schema. Emits warnings (not errors) for mismatches,
    /// since schemas are optional and None means "accepts anything".
    /// What a node accepts, whether it is a filter or a step.
    fn input_schema_of(&self, node_id: &str) -> Option<somatize_core::schema::Schema> {
        self.registry
            .node_meta(node_id)
            .and_then(|m| m.input_schema)
    }

    /// What a node produces, whether it is a filter or a step.
    fn output_schema_of(&self, node_id: &str) -> Option<somatize_core::schema::Schema> {
        self.registry
            .node_meta(node_id)
            .and_then(|m| m.output_schema)
    }

    /// Check that every edge could carry what flows along it.
    ///
    /// Two severities, because two very different things get called a
    /// "schema mismatch":
    ///
    /// - **Warning** — the dtypes differ but could plausibly line up (`f32`
    ///   into `f64`, a fixed shape into a dynamic one). Long-standing
    ///   behaviour; plenty of working pipelines rely on it.
    /// - **Error** — no reading of the producer could satisfy the consumer:
    ///   a tensor arriving where a conversation is expected. Across 1600+
    ///   annotated multi-agent traces this class — context lost or malformed
    ///   at a handoff — is the single largest bucket of failures after bad
    ///   specifications. It is cheap to catch here and expensive to catch
    ///   after a few thousand tokens.
    fn validate_schemas(&mut self, sorted: &[&str]) -> Result<()> {
        for &node_id in sorted {
            // Skip if this node accepts anything
            let Some(expected_input) = self.input_schema_of(node_id) else {
                continue;
            };

            for pred_id in self.graph.predecessors(node_id) {
                let Some(actual_output) = self.output_schema_of(pred_id) else {
                    continue; // predecessor output unknown, skip
                };

                // No possible reading — refuse to build the graph.
                if actual_output.is_incompatible_with(&expected_input) {
                    return Err(SomaError::Compilation(format!(
                        "`{pred_id}` outputs {actual_output} but `{node_id}` expects \
                         {expected_input}, and there is no conversion between them. \
                         Insert a node that adapts one to the other"
                    )));
                }

                let same_dtype = actual_output.dtype == expected_input.dtype;
                let both_numeric =
                    actual_output.dtype.is_numeric() && expected_input.dtype.is_numeric();

                // Warn on a shape that does not line up, or on an implicit
                // change of numeric width. Stay quiet about the promotions the
                // runtime performs by design (text → conversation, anything →
                // json): those are the intended way to connect such nodes, and
                // warning about them would train people to ignore warnings.
                if (same_dtype && !actual_output.is_compatible_with(&expected_input))
                    || (!same_dtype && both_numeric)
                {
                    self.diagnostics.push(Diagnostic {
                        node_id: node_id.to_string(),
                        level: DiagnosticLevel::Warning,
                        message: format!(
                            "schema mismatch: `{pred_id}` outputs {actual_output} \
                             but `{node_id}` expects {expected_input}",
                        ),
                    });
                }
            }
        }
        Ok(())
    }

    /// Check gradient flow and emit warnings for each interruption.
    ///
    /// Gradient flow can restart after an opaque node (differentiable nodes
    /// after an opaque one can still propagate gradients among themselves),
    /// but gradients from before the interruption are lost.
    /// Report parts of the architecture that are not wired into it.
    ///
    /// A node with no edges at all is silently a second root: roots receive
    /// the graph's input, so it runs, on data it was never meant to see,
    /// and its output goes nowhere. The DSL makes this easy to write by
    /// accident, because Python binds `>>` tighter than `|` — the fork in
    /// `A() | B() >> C()` is `A() | (B() >> C())`, and `A` is left
    /// dangling. Nothing else catches it: it is not a cycle, not a
    /// duplicate id, not a dangling edge endpoint, and the schemas of a
    /// node nobody feeds are trivially satisfied.
    ///
    /// A leaf — a node whose output nobody consumes — is Info rather than
    /// Warning, because fan-out to several leaves is a legitimate shape.
    /// It is worth saying only when there is more than one, since `forward`
    /// returns the leaf that actually ran and the others are computed and
    /// dropped.
    fn check_connectivity(&mut self) {
        if self.graph.nodes.len() < 2 {
            return;
        }

        let mut leaves = Vec::new();
        for node in &self.graph.nodes {
            let id = node.id.as_str();
            let has_input = !self.graph.predecessors(id).is_empty();
            let has_output = !self.graph.successors(id).is_empty();

            if !has_input && !has_output {
                self.diagnostics.push(Diagnostic {
                    node_id: id.to_string(),
                    level: DiagnosticLevel::Warning,
                    message: format!(
                        "`{id}` has no edges. It is therefore a root: it will run on the \
                         graph's input, and its output will be discarded. If it was meant \
                         to be part of the pipeline, connect it; if it is a spawn target, \
                         register it with `register_step` instead of adding a node."
                    ),
                });
            } else if !has_output {
                leaves.push(id.to_string());
            }
        }

        if leaves.len() > 1 {
            self.diagnostics.push(Diagnostic {
                node_id: leaves[0].clone(),
                level: DiagnosticLevel::Info,
                message: format!(
                    "{} nodes produce output nobody consumes ({}). `forward` returns the \
                     leaf that actually ran; the others are computed and dropped.",
                    leaves.len(),
                    leaves.join(", "),
                ),
            });
        }
    }

    fn check_gradient_flow(&mut self, sorted: &[&str]) {
        // Starts false, not true. A non-differentiable node only interrupts
        // a gradient if there is one to interrupt — and the first node in
        // topological order has nothing upstream of it. Starting true made
        // every graph of ordinary filters warn about its own first node,
        // which is most preprocessing pipelines, and a warning that fires
        // on correct code teaches people to stop reading warnings.
        let mut gradient_flows = false;

        for &node_id in sorted {
            if let Some(meta) = self.registry.meta(node_id) {
                if gradient_flows && !meta.differentiable {
                    self.diagnostics.push(Diagnostic {
                        node_id: node_id.to_string(),
                        level: DiagnosticLevel::Warning,
                        message: format!(
                            "gradient flow interrupted at `{}` ({:?}). \
                             Gradients from upstream will not reach downstream filters \
                             through this node.",
                            node_id, meta.kind,
                        ),
                    });
                    gradient_flows = false;
                } else if !gradient_flows && meta.differentiable {
                    // Gradient flow restarts: differentiable nodes after the
                    // interruption can propagate gradients among themselves
                    gradient_flows = true;
                }
            }
        }
    }
}

/// Convenience function: compile a graph with default settings.
pub fn compile(
    graph: &Graph,
    registry: &dyn NodeRegistry,
    mode: CompileMode,
    cache: Option<&dyn CacheStore>,
) -> Result<CompileResult> {
    Compiler::new(graph, registry, mode).compile(cache)
}

/// Compile a graph for streaming execution.
///
/// Produces an `ExecutionPlan::Stream` wrapping the topologically sorted
/// node chain, which the runtime executes chunk by chunk through the
/// same primitives `run_node` uses.
///
/// Streaming executes a single linear chain of filters — it used to
/// accept any DAG and silently run it as a chain, which for a diamond
/// is simply the wrong answer. So this validates what the executor can
/// honour:
///
/// - every node has at most one predecessor and one successor;
/// - no node is a step (the effect journal keys by `(run, node, turn)`,
///   so chunk 2 would replay chunk 1's effects — no defensible
///   semantics);
/// - `chunk_size > 0`.
pub fn compile_stream(
    graph: &Graph,
    registry: &dyn NodeRegistry,
    chunk_size: usize,
) -> Result<CompileResult> {
    graph.validate()?;
    let sorted = graph.topological_sort()?;

    if sorted.is_empty() {
        return Ok(CompileResult {
            plan: ExecutionPlan::Empty,
            diagnostics: Vec::new(),
        });
    }

    if chunk_size == 0 {
        return Err(SomaError::Compilation(
            "stream chunk_size must be at least 1".into(),
        ));
    }

    for id in &sorted {
        let (preds, succs) = (graph.predecessors(id), graph.successors(id));
        if preds.len() > 1 || succs.len() > 1 {
            return Err(SomaError::Compilation(format!(
                "streaming executes a single linear chain; node `{id}` has {} \
                 predecessors and {} successors — restructure the graph or use \
                 the standard forward",
                preds.len(),
                succs.len(),
            )));
        }
        match registry.node_meta(id) {
            Some(meta) if meta.effectful => {
                return Err(SomaError::Compilation(format!(
                    "step `{id}` cannot be streamed: effect journaling has no \
                     per-chunk semantics. Run the graph with the standard forward"
                )));
            }
            Some(_) => {}
            None => {
                return Err(SomaError::Compilation(format!(
                    "graph names node `{id}` but nothing with that id is registered"
                )));
            }
        }
    }

    let node_ids: Vec<NodeId> = sorted.into_iter().map(|s| s.to_string()).collect();
    let plan = ExecutionPlan::Stream {
        node_ids,
        chunk_size,
    };

    Ok(CompileResult {
        plan,
        diagnostics: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use somatize_core::cache::EntryMeta;
    use somatize_core::error::SomaError;
    use somatize_core::filter::{FilterKind, StreamMode};
    use somatize_core::graph::{Edge, Graph, Node, linear_pipeline};
    use somatize_core::value::Value;
    use std::collections::HashSet;
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
            deterministic: true,
            stream_mode: StreamMode::FixedState,
            distribution: somatize_core::filter::Distribution::Local,
            input_schema: None,
            output_schema: None,
        }
    }

    fn register_nodes(registry: &mut SimpleNodeRegistry, ids: &[&str], meta: FilterMeta) {
        for (i, id) in ids.iter().enumerate() {
            let hash = CacheKey::from_parts(&[id.as_bytes(), &[i as u8]]);
            registry.register_meta(*id, meta.clone(), hash);
        }
    }

    // ── Tests ──

    #[test]
    fn compile_empty_graph() {
        let graph = Graph::new();
        let registry = SimpleNodeRegistry::new();
        let result = compile(&graph, &registry, CompileMode::Inference, None).unwrap();
        assert!(matches!(result.plan, ExecutionPlan::Empty));
    }

    #[test]
    fn compile_single_node() {
        let mut graph = Graph::new();
        graph.add_node(Node::new("a", "A", "F"));
        let mut registry = SimpleNodeRegistry::new();
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
        let mut registry = SimpleNodeRegistry::new();
        register_nodes(
            &mut registry,
            &["a", "b", "c"],
            make_meta(FilterKind::Trainable, true),
        );

        let result = compile(&graph, &registry, CompileMode::Inference, None).unwrap();

        // All 3 nodes are differentiable → collapsed into Composite
        if let ExecutionPlan::Composite { node_ids } = &result.plan {
            assert_eq!(node_ids, &["a", "b", "c"]);
        } else {
            panic!("expected Composite, got: {:?}", result.plan);
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

        let mut registry = SimpleNodeRegistry::new();
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

        let mut registry = SimpleNodeRegistry::new();
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
    fn cache_resolution_is_deferred_to_runtime() {
        let graph = linear_pipeline(vec![
            Node::new("a", "Scaler", "F"),
            Node::new("b", "PCA", "F"),
            Node::new("c", "SVM", "F"),
        ]);

        let mut registry = SimpleNodeRegistry::new();
        register_nodes(
            &mut registry,
            &["a", "b", "c"],
            make_meta(FilterKind::Trainable, true),
        );

        // Even with a populated cache, the compiler must never emit
        // Cached nodes: its keys cannot include the input data, so a
        // compile-time hit could serve results from a different dataset.
        // The executor resolves cache hits per node at runtime.
        let a_config = registry.config_hash("a").unwrap();
        let a_cache_key = CacheKey::from_parts(&[&a_config.0]);
        let cache = MockCacheStore::new();
        cache.insert(a_cache_key);

        let result = compile(&graph, &registry, CompileMode::Inference, Some(&cache)).unwrap();

        assert!(
            !format!("{:?}", result.plan).contains("Cached"),
            "compiler must not emit Cached nodes, got: {:?}",
            result.plan
        );
        assert!(
            result
                .diagnostics
                .iter()
                .any(|d| d.level == DiagnosticLevel::Info
                    && d.message.contains("resolved at runtime")),
            "expected an informational diagnostic about runtime cache resolution"
        );
    }

    #[test]
    fn no_cache_mode_skips_all_caching() {
        let graph = linear_pipeline(vec![Node::new("a", "A", "F"), Node::new("b", "B", "F")]);

        let mut registry = SimpleNodeRegistry::new();
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
        assert!(!format!("{:?}", result.plan).contains("Cached"));
    }

    #[test]
    fn differentiable_mode_skips_output_caching() {
        let graph = linear_pipeline(vec![Node::new("a", "A", "F"), Node::new("b", "B", "F")]);

        let mut registry = SimpleNodeRegistry::new();
        register_nodes(
            &mut registry,
            &["a", "b"],
            make_meta(FilterKind::Trainable, true),
        );

        let a_config = registry.config_hash("a").unwrap();
        let a_key = CacheKey::from_parts(&[&a_config.0]);
        let cache = MockCacheStore::new();
        cache.insert(a_key);

        let result = compile(&graph, &registry, CompileMode::Differentiable, Some(&cache)).unwrap();

        // Differentiable mode should not cache forward outputs
        assert!(!format!("{:?}", result.plan).contains("Cached"));
    }

    #[test]
    fn gradient_flow_diagnostic_on_opaque() {
        let graph = linear_pipeline(vec![
            Node::new("scaler", "Scaler", "F"),
            Node::new("tree", "DecisionTree", "F"),
            Node::new("linear", "Linear", "F"),
        ]);

        let mut registry = SimpleNodeRegistry::new();
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
        assert!(
            result.diagnostics[0]
                .message
                .contains("gradient flow interrupted")
        );
    }

    #[test]
    fn no_diagnostic_when_all_differentiable() {
        let graph = linear_pipeline(vec![Node::new("a", "A", "F"), Node::new("b", "B", "F")]);

        let mut registry = SimpleNodeRegistry::new();
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

        let registry = SimpleNodeRegistry::new();
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

        let mut registry = SimpleNodeRegistry::new();
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

        let mut registry = SimpleNodeRegistry::new();
        // preprocess: local
        registry.register_meta(
            "preprocess",
            make_meta(FilterKind::Trainable, true),
            CacheKey::hash_data(b"pre"),
        );
        // gpu_train: remote on GPU tag
        let mut gpu_meta = make_meta(FilterKind::Trainable, true);
        gpu_meta.distribution = somatize_core::filter::Distribution::Remote(
            somatize_core::filter::RemoteTarget::Tag("gpu".into()),
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
            assert!(
                matches!(&steps[0], ExecutionPlan::Execute { node_id } if node_id == "preprocess")
            );
            assert!(
                matches!(&steps[1], ExecutionPlan::Remote { node_id, target, .. }
                    if node_id == "gpu_train"
                    && *target == somatize_core::filter::RemoteTarget::Tag("gpu".into())
                ),
                "expected Remote, got: {:?}",
                steps[1]
            );
            assert!(
                matches!(&steps[2], ExecutionPlan::Execute { node_id } if node_id == "evaluate")
            );
        } else {
            panic!("expected Sequence, got: {:?}", result.plan);
        }
    }

    #[test]
    fn local_distribution_not_wrapped() {
        let graph = linear_pipeline(vec![Node::new("a", "A", "F"), Node::new("b", "B", "F")]);

        let mut registry = SimpleNodeRegistry::new();
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
            assert!(
                steps
                    .iter()
                    .all(|s| matches!(s, ExecutionPlan::Execute { .. }))
            );
        }
    }

    // ── compile_stream ──

    #[test]
    fn stream_compiles_a_linear_chain() {
        let graph = linear_pipeline(vec![Node::new("a", "A", "F"), Node::new("b", "B", "F")]);
        let mut registry = SimpleNodeRegistry::new();
        register_nodes(
            &mut registry,
            &["a", "b"],
            make_meta(FilterKind::Stateless, false),
        );

        let result = compile_stream(&graph, &registry, 64).unwrap();
        let ExecutionPlan::Stream {
            node_ids,
            chunk_size,
        } = result.plan
        else {
            panic!("expected a Stream plan");
        };
        assert_eq!(node_ids, vec!["a", "b"]);
        assert_eq!(chunk_size, 64);
    }

    #[test]
    fn stream_of_an_empty_graph_is_empty() {
        let result = compile_stream(&Graph::new(), &SimpleNodeRegistry::new(), 64).unwrap();
        assert!(matches!(result.plan, ExecutionPlan::Empty));
    }

    #[test]
    fn stream_rejects_a_zero_chunk() {
        let graph = linear_pipeline(vec![Node::new("a", "A", "F")]);
        let mut registry = SimpleNodeRegistry::new();
        register_nodes(
            &mut registry,
            &["a"],
            make_meta(FilterKind::Stateless, false),
        );

        let err = compile_stream(&graph, &registry, 0).unwrap_err();
        assert!(err.to_string().contains("chunk_size"), "{err}");
    }

    /// A diamond used to stream as a chain in topological order — a
    /// silently wrong answer. Now it is a compile error naming the node.
    #[test]
    fn stream_rejects_a_non_linear_graph_by_name() {
        let mut graph = Graph::new();
        for id in ["a", "b", "c", "d"] {
            graph.add_node(Node::new(id, id, "F"));
        }
        graph.add_edge(Edge::data("e1", "a", "b"));
        graph.add_edge(Edge::data("e2", "a", "c"));
        graph.add_edge(Edge::data("e3", "b", "d"));
        graph.add_edge(Edge::data("e4", "c", "d"));
        let mut registry = SimpleNodeRegistry::new();
        register_nodes(
            &mut registry,
            &["a", "b", "c", "d"],
            make_meta(FilterKind::Stateless, false),
        );

        let err = compile_stream(&graph, &registry, 64).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("`a`"), "should name the forking node: {msg}");
        assert!(msg.contains("linear chain"), "{msg}");
    }

    /// The effect journal keys by (run, node, turn): chunk 2 would replay
    /// chunk 1's effects. There is no defensible semantics, so refuse.
    #[test]
    fn stream_rejects_a_step_by_name() {
        let graph = linear_pipeline(vec![Node::new("a", "A", "F"), Node::new("s", "S", "Step")]);
        let mut registry = SimpleNodeRegistry::new();
        register_nodes(
            &mut registry,
            &["a"],
            make_meta(FilterKind::Stateless, false),
        );
        registry.register_step_meta("s", somatize_core::step::StepMeta::new("S"));

        let err = compile_stream(&graph, &registry, 64).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("`s`"), "{msg}");
        assert!(msg.contains("cannot be streamed"), "{msg}");
    }

    #[test]
    fn stream_reports_an_unregistered_node() {
        let graph = linear_pipeline(vec![Node::new("ghost", "G", "F")]);
        let err = compile_stream(&graph, &SimpleNodeRegistry::new(), 64).unwrap_err();
        assert!(err.to_string().contains("`ghost`"), "{err}");
    }
}
