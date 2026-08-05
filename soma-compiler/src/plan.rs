//! Execution plan — the compiled representation of a pipeline.
//!
//! Variants: Sequence, Parallel, Execute, Loop, Branch, Remote, Stream, Empty.
//! Plans are data-free (no filter implementations) and serializable.

use serde::{Deserialize, Serialize};
use somatize_core::control::LoopCondition;
use somatize_core::filter::RemoteTarget;
use somatize_core::graph::NodeId;
use std::fmt;

/// A compiled execution plan produced by the compiler.
///
/// This is a recursive tree that the runtime walks to execute a pipeline.
/// The compiler resolves caching, parallelism, and distribution before
/// the runtime sees the plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ExecutionPlan {
    /// Execute steps sequentially, one after another.
    Sequence(Vec<ExecutionPlan>),

    /// Execute branches concurrently (fork-join).
    Parallel(Vec<ExecutionPlan>),

    /// Execute a single filter node.
    Execute {
        /// The graph node to execute.
        node_id: NodeId,
    },

    /// Run an effectful step to completion: poll, perform its effects,
    /// repeat. Distinct from `Execute` because the runtime has to drive a
    /// turn loop and journal what it performs, not call a function once.
    Step {
        /// The effectful node the runtime drives.
        node_id: NodeId,
        /// Where this step may hand control, by target node id.
        ///
        /// A handoff is a branch the *step* decides rather than a condition
        /// value, so it compiles the same way: each target is claimed by the
        /// step and appears exactly once, inside it. A `Goto` naming
        /// something not listed here is an error, not a jump into the dark.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        handoffs: Vec<(NodeId, ExecutionPlan)>,
    },

    /// Iterate: run `body` until `until` says stop, or `max_iterations` is hit.
    Loop {
        /// The loop controller node — the id events and assignments are
        /// reported under, distinct from any node inside `body`.
        node_id: NodeId,
        /// The sub-plan executed once per iteration.
        body: Box<ExecutionPlan>,
        /// Hard iteration cap; `None` leaves stopping entirely to `until`.
        max_iterations: Option<usize>,
        /// Already resolved by the compiler — never `BodyTerminal` here.
        /// The executor reads the signal from exactly this node.
        #[serde(default)]
        until: LoopCondition,
        /// The node whose output each pass hands to the next one.
        ///
        /// Separate from `until` on purpose: what a loop carries and what
        /// tells it to stop are different questions. A debate that runs a
        /// fixed number of rounds has no stop signal at all, but every round
        /// still has to start from what the last one said — otherwise the
        /// loop just repeats its first iteration.
        ///
        /// `None` when the body has no single terminal to carry from.
        #[serde(default)]
        carry_from: Option<NodeId>,
    },

    /// Conditional branching: evaluate condition, pick an arm.
    Branch {
        /// The node whose output selects an arm. The selector is control,
        /// not data: the chosen arm receives the branch's *input*.
        node_id: NodeId,
        /// `(label, sub-plan)` per arm; the condition value picks by label.
        arms: Vec<(String, ExecutionPlan)>,
    },

    /// Execute a sub-plan on a remote worker.
    Remote {
        /// The node the distribution directive was attached to. The wrapped
        /// `plan` names it again, which is why this wrapper contributes no
        /// ids of its own to `node_ids()`.
        node_id: NodeId,
        /// Where to run: a specific worker by id, or any worker with a tag.
        target: RemoteTarget,
        /// The sub-plan the remote worker executes.
        plan: Box<ExecutionPlan>,
    },

    /// Execute multiple differentiable nodes as a single block.
    /// The executor passes tensors directly between filters (no Value conversion),
    /// preserving PyTorch autograd for gradient flow.
    Composite {
        /// The differentiable nodes fused into the block, in execution order.
        node_ids: Vec<NodeId>,
    },

    /// Streaming execution: process input in chunks through a filter chain.
    /// Each filter's StreamMode (FixedState/Evolving/Barrier) defines its
    /// per-chunk contract. Results flow progressively — no full materialization.
    Stream {
        /// The filter chain each chunk flows through, in order.
        node_ids: Vec<NodeId>,
        /// How many input rows each chunk carries.
        chunk_size: usize,
    },

    /// No-op: nothing to execute (e.g. empty graph).
    Empty,
}

impl ExecutionPlan {
    /// The node ids this variant introduces itself, excluding its children.
    ///
    /// `Remote` introduces none: it wraps a plan that already names the
    /// node. Counting it here as well is what made `node_ids()` return the
    /// same id twice for every remote node — and `LocalRunner::fit`, which
    /// iterates that list, fit it twice.
    fn own_node_ids(&self) -> &[String] {
        match self {
            Self::Execute { node_id }
            | Self::Step { node_id, .. }
            | Self::Loop { node_id, .. }
            | Self::Branch { node_id, .. } => std::slice::from_ref(node_id),
            Self::Composite { node_ids } | Self::Stream { node_ids, .. } => node_ids,
            Self::Remote { .. } | Self::Sequence(_) | Self::Parallel(_) | Self::Empty => &[],
        }
    }

    /// The sub-plans nested inside this one, each with its edge label if it
    /// has one — a branch arm's label, a handoff's target.
    ///
    /// One structural walk, so the accessors below cannot disagree about
    /// the shape of the tree. They used to: `node_count` skipped a step's
    /// handoffs while `node_ids` collected them, so an agentic plan
    /// reported fewer nodes than it had.
    pub fn children(&self) -> impl Iterator<Item = (Option<&str>, &ExecutionPlan)> {
        // Spelled out rather than defaulted with `_ => &[]`. A wildcard here
        // is how a variant that owns sub-plans became invisible to
        // `node_count`/`node_ids` once already: the compiler cannot warn
        // about a case that is already handled. Listing every variant means
        // adding one breaks this walk at compile time, where the omission
        // is cheap to see.
        let labelled: &[(String, ExecutionPlan)] = match self {
            Self::Step { handoffs, .. } => handoffs,
            Self::Branch { arms, .. } => arms,
            Self::Sequence(_)
            | Self::Parallel(_)
            | Self::Execute { .. }
            | Self::Loop { .. }
            | Self::Remote { .. }
            | Self::Composite { .. }
            | Self::Stream { .. }
            | Self::Empty => &[],
        };
        let plain: &[ExecutionPlan] = match self {
            Self::Sequence(steps) | Self::Parallel(steps) => steps,
            Self::Execute { .. }
            | Self::Step { .. }
            | Self::Loop { .. }
            | Self::Branch { .. }
            | Self::Remote { .. }
            | Self::Composite { .. }
            | Self::Stream { .. }
            | Self::Empty => &[],
        };
        let single: Option<&ExecutionPlan> = match self {
            Self::Loop { body, .. } => Some(body),
            Self::Remote { plan, .. } => Some(plan),
            Self::Sequence(_)
            | Self::Parallel(_)
            | Self::Execute { .. }
            | Self::Step { .. }
            | Self::Branch { .. }
            | Self::Composite { .. }
            | Self::Stream { .. }
            | Self::Empty => None,
        };

        labelled
            .iter()
            .map(|(l, p)| (Some(l.as_str()), p))
            .chain(plain.iter().map(|p| (None, p)))
            .chain(single.map(|p| (None, p)))
    }

    /// Count total nodes in the plan.
    pub fn node_count(&self) -> usize {
        self.own_node_ids().len() + self.children().map(|(_, p)| p.node_count()).sum::<usize>()
    }

    /// Count parallel branches at the top level of the plan.
    ///
    /// Top level only, deliberately: this feeds a run's summary, and a
    /// fan-out inside a loop body happens once per iteration rather than
    /// once per run.
    pub fn parallel_branch_count(&self) -> usize {
        match self {
            Self::Parallel(branches) => branches.len(),
            Self::Sequence(steps) => steps.iter().map(|s| s.parallel_branch_count()).sum(),
            _ => 0,
        }
    }

    /// Collect all node IDs referenced in the plan.
    pub fn node_ids(&self) -> Vec<&str> {
        let mut ids: Vec<&str> = self.own_node_ids().iter().map(String::as_str).collect();
        for (_, child) in self.children() {
            ids.extend(child.node_ids());
        }
        ids
    }

    /// Create a PlanSummary for event payloads.
    pub fn summary(&self) -> somatize_core::event::PlanSummary {
        somatize_core::event::PlanSummary {
            total_nodes: self.node_count(),
            // Cache resolution moved to runtime; plans carry no cached nodes.
            cached_nodes: 0,
            parallel_branches: self.parallel_branch_count(),
        }
    }

    /// Flatten unnecessary nesting (e.g. Sequence of one element).
    pub fn simplify(self) -> Self {
        match self {
            Self::Sequence(mut steps) => {
                steps = steps.into_iter().map(|s| s.simplify()).collect();
                steps.retain(|s| !matches!(s, Self::Empty));
                match steps.len() {
                    0 => Self::Empty,
                    1 => steps.into_iter().next().unwrap(),
                    _ => Self::Sequence(steps),
                }
            }
            Self::Parallel(mut branches) => {
                branches = branches.into_iter().map(|b| b.simplify()).collect();
                branches.retain(|b| !matches!(b, Self::Empty));
                match branches.len() {
                    0 => Self::Empty,
                    1 => branches.into_iter().next().unwrap(),
                    _ => Self::Parallel(branches),
                }
            }
            other => other,
        }
    }
}

impl ExecutionPlan {
    /// Render the execution plan as a Mermaid flowchart.
    pub fn to_mermaid(&self) -> String {
        let mut out = String::from("graph TD\n");
        let mut counter = 0;
        self.mermaid_nodes(&mut out, &mut counter, None);
        out
    }

    /// Renders directly rather than over [`Self::children`], and so does
    /// [`Self::graph_nodes`], because the two do not draw the same picture:
    /// mermaid synthesises an `arm_N` node between a branch and each arm
    /// and draws handoffs as dotted edges to the target, while `to_graph`
    /// parents an arm straight to the branch and puts the label on the
    /// edge. Folding them together would have to change one of the two
    /// outputs. They share the shape of the recursion, not its result.
    fn mermaid_nodes(&self, out: &mut String, counter: &mut usize, parent: Option<&str>) {
        use std::fmt::Write;
        match self {
            Self::Execute { node_id } => {
                let _ = writeln!(out, "    {node_id}[{node_id}]");
                if let Some(p) = parent {
                    let _ = writeln!(out, "    {p} --> {node_id}");
                }
            }
            Self::Step { node_id, handoffs } => {
                // Parallelogram — an effectful node reaches outside the graph.
                let _ = writeln!(out, "    {node_id}[/{node_id}/]");
                if let Some(p) = parent {
                    let _ = writeln!(out, "    {p} --> {node_id}");
                }
                for (target, plan) in handoffs {
                    let _ = writeln!(out, "    {node_id} -.->|{target}| {target}");
                    plan.mermaid_nodes(out, counter, None);
                }
            }
            Self::Sequence(steps) => {
                let mut prev = parent.map(String::from);
                for step in steps {
                    step.mermaid_nodes(out, counter, prev.as_deref());
                    prev = step.first_node_id().map(String::from);
                }
            }
            Self::Parallel(branches) => {
                let fork_id = format!("fork_{counter}");
                *counter += 1;
                let _ = writeln!(out, "    {fork_id}{{{{fork}}}}");
                if let Some(p) = parent {
                    let _ = writeln!(out, "    {p} --> {fork_id}");
                }
                for branch in branches {
                    branch.mermaid_nodes(out, counter, Some(&fork_id));
                }
            }
            Self::Loop {
                node_id,
                body,
                max_iterations,
                ..
            } => {
                let label = match max_iterations {
                    Some(n) => format!("{node_id} loop max={n}"),
                    None => format!("{node_id} loop"),
                };
                let _ = writeln!(out, "    {node_id}(({label}))");
                if let Some(p) = parent {
                    let _ = writeln!(out, "    {p} --> {node_id}");
                }
                body.mermaid_nodes(out, counter, Some(node_id));
            }
            Self::Branch { node_id, arms } => {
                let _ = writeln!(out, "    {node_id}{{{{{node_id}}}}}");
                if let Some(p) = parent {
                    let _ = writeln!(out, "    {p} --> {node_id}");
                }
                for (label, plan) in arms {
                    let arm_id = format!("arm_{counter}");
                    *counter += 1;
                    let _ = writeln!(out, "    {node_id} -->|{label}| {arm_id}[{label}]");
                    plan.mermaid_nodes(out, counter, Some(&arm_id));
                }
            }
            Self::Remote {
                node_id,
                target,
                plan,
            } => {
                let _ = writeln!(out, "    {node_id}>{{{node_id} remote: {target:?}}}]");
                if let Some(p) = parent {
                    let _ = writeln!(out, "    {p} --> {node_id}");
                }
                plan.mermaid_nodes(out, counter, Some(node_id));
            }
            Self::Composite { node_ids } | Self::Stream { node_ids, .. } => {
                use std::fmt::Write;
                let stream_label = matches!(self, Self::Stream { .. });
                let mut prev: Option<&str> = None;
                for nid in node_ids {
                    if stream_label {
                        let _ = writeln!(out, "    {nid}([{nid} stream])");
                    } else {
                        let _ = writeln!(out, "    {nid}[{nid}]");
                    }
                    if let Some(p) = prev.or(parent) {
                        let _ = writeln!(out, "    {p} --> {nid}");
                    }
                    prev = Some(nid);
                }
            }
            Self::Empty => {}
        }
    }

    fn first_node_id(&self) -> Option<&str> {
        match self {
            Self::Execute { node_id } | Self::Step { node_id, .. } => Some(node_id),
            Self::Sequence(steps) => steps.first().and_then(|s| s.first_node_id()),
            Self::Parallel(_) => None,
            Self::Loop { node_id, .. }
            | Self::Branch { node_id, .. }
            | Self::Remote { node_id, .. } => Some(node_id),
            Self::Composite { node_ids } | Self::Stream { node_ids, .. } => {
                node_ids.first().map(|s| s.as_str())
            }
            Self::Empty => None,
        }
    }

    /// Synthesize a displayable [`Graph`](somatize_core::graph::Graph)
    /// from this plan — the same node synthesis as [`Self::to_mermaid`]
    /// (fork nodes for `Parallel`, arm nodes for `Branch`, pills for
    /// streams) — so every Graph renderer applies: `to_svg()`,
    /// `to_mermaid()`, `to_graphviz()`.
    pub fn to_graph(&self) -> somatize_core::graph::Graph {
        let mut g = somatize_core::graph::Graph::new();
        let mut counter = 0usize;
        self.graph_nodes(&mut g, &mut counter, None, None);
        g
    }

    fn add_edge(
        g: &mut somatize_core::graph::Graph,
        source: &str,
        target: &str,
        label: Option<&str>,
    ) {
        let mut edge =
            somatize_core::graph::Edge::data(format!("e{}", g.edges.len()), source, target);
        edge.label = label.map(str::to_string);
        g.add_edge(edge);
    }

    fn graph_nodes(
        &self,
        g: &mut somatize_core::graph::Graph,
        counter: &mut usize,
        parent: Option<&str>,
        edge_label: Option<&str>,
    ) {
        use somatize_core::graph::Node;
        match self {
            Self::Execute { node_id } => {
                g.add_node(Node::new(node_id, node_id, node_id));
                if let Some(p) = parent {
                    Self::add_edge(g, p, node_id, edge_label);
                }
            }
            Self::Step { node_id, handoffs } => {
                g.add_node(Node::step(node_id, node_id));
                if let Some(p) = parent {
                    Self::add_edge(g, p, node_id, edge_label);
                }
                for (target, plan) in handoffs {
                    plan.graph_nodes(g, counter, Some(node_id), Some(target));
                }
            }
            Self::Sequence(steps) => {
                let mut prev = parent.map(String::from);
                let mut label = edge_label;
                for step in steps {
                    step.graph_nodes(g, counter, prev.as_deref(), label);
                    label = None; // only the first hop carries the arm label
                    prev = step.first_node_id().map(String::from);
                }
            }
            Self::Parallel(branches) => {
                let fork_id = format!("fork_{counter}");
                *counter += 1;
                let mut fork = Node::branch(fork_id.clone());
                fork.label = "fork".to_string();
                g.add_node(fork);
                if let Some(p) = parent {
                    Self::add_edge(g, p, &fork_id, edge_label);
                }
                for branch in branches {
                    branch.graph_nodes(g, counter, Some(&fork_id), None);
                }
            }
            Self::Loop {
                node_id,
                body,
                max_iterations,
                ..
            } => {
                g.add_node(Node::loop_node(node_id.clone(), *max_iterations));
                if let Some(p) = parent {
                    Self::add_edge(g, p, node_id, edge_label);
                }
                body.graph_nodes(g, counter, Some(node_id), None);
            }
            Self::Branch { node_id, arms } => {
                g.add_node(Node::branch(node_id.clone()));
                if let Some(p) = parent {
                    Self::add_edge(g, p, node_id, edge_label);
                }
                for (label, plan) in arms {
                    plan.graph_nodes(g, counter, Some(node_id), Some(label));
                }
            }
            Self::Remote {
                node_id,
                target,
                plan,
            } => {
                let mut node = Node::subgraph(node_id.clone(), somatize_core::graph::Graph::new());
                node.label = format!("{node_id} (remote {target:?})");
                g.add_node(node);
                if let Some(p) = parent {
                    Self::add_edge(g, p, node_id, edge_label);
                }
                plan.graph_nodes(g, counter, Some(node_id), None);
            }
            Self::Composite { node_ids } | Self::Stream { node_ids, .. } => {
                let stream = matches!(self, Self::Stream { .. });
                let mut prev: Option<&str> = None;
                let mut label = edge_label;
                for nid in node_ids {
                    if stream {
                        let mut node = Node::loop_node(nid.clone(), None);
                        node.label = format!("{nid} stream");
                        g.add_node(node);
                    } else {
                        g.add_node(Node::new(nid, nid, nid));
                    }
                    if let Some(p) = prev.or(parent) {
                        Self::add_edge(g, p, nid, label);
                    }
                    label = None;
                    prev = Some(nid);
                }
            }
            Self::Empty => {}
        }
    }
}

impl fmt::Display for ExecutionPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_indent(f, 0)
    }
}

impl ExecutionPlan {
    fn fmt_indent(&self, f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
        let pad = "  ".repeat(indent);
        match self {
            Self::Sequence(steps) => {
                writeln!(f, "{pad}Sequence:")?;
                for step in steps {
                    step.fmt_indent(f, indent + 1)?;
                }
                Ok(())
            }
            Self::Parallel(branches) => {
                writeln!(f, "{pad}Parallel:")?;
                for branch in branches {
                    branch.fmt_indent(f, indent + 1)?;
                }
                Ok(())
            }
            Self::Execute { node_id } => writeln!(f, "{pad}Execute({node_id})"),
            Self::Step { node_id, handoffs } => {
                writeln!(f, "{pad}Step({node_id})")?;
                for (target, plan) in handoffs {
                    writeln!(f, "{pad}  ~>{target}:")?;
                    plan.fmt_indent(f, indent + 2)?;
                }
                Ok(())
            }
            Self::Loop {
                node_id,
                body,
                max_iterations,
                ..
            } => {
                writeln!(f, "{pad}Loop({node_id}, max={max_iterations:?}):")?;
                body.fmt_indent(f, indent + 1)
            }
            Self::Branch { node_id, arms } => {
                writeln!(f, "{pad}Branch({node_id}):")?;
                for (label, plan) in arms {
                    writeln!(f, "{pad}  [{label}]:")?;
                    plan.fmt_indent(f, indent + 2)?;
                }
                Ok(())
            }
            Self::Remote {
                node_id,
                target,
                plan,
            } => {
                writeln!(f, "{pad}Remote({node_id}, target={target:?}):")?;
                plan.fmt_indent(f, indent + 1)
            }
            Self::Composite { node_ids } => {
                let ids = node_ids
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(" \u{2192} ");
                writeln!(f, "{pad}Composite[{ids}]")
            }
            Self::Stream {
                node_ids,
                chunk_size,
            } => {
                let ids = node_ids
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(" \u{2192} ");
                writeln!(f, "{pad}Stream[{ids}](chunk_size={chunk_size})")
            }
            Self::Empty => writeln!(f, "{pad}Empty"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Remote` wraps a plan that already names the node, so counting the
    /// wrapper's id as well listed it twice. `LocalRunner::fit` iterates
    /// this list, so a remote trainable node was fitted twice.
    #[test]
    fn a_remote_node_is_listed_once() {
        let plan = ExecutionPlan::Remote {
            node_id: "n".into(),
            target: somatize_core::filter::RemoteTarget::Tag("gpu".into()),
            plan: Box::new(ExecutionPlan::Execute {
                node_id: "n".into(),
            }),
        };
        assert_eq!(plan.node_ids(), vec!["n"]);
        assert_eq!(plan.node_count(), 1);
    }

    /// `node_count` and `node_ids` walk the same tree and must agree.
    /// `node_count` used to skip a step's handoffs while `node_ids`
    /// collected them, so an agentic plan reported fewer nodes than it ran.
    #[test]
    fn the_two_walks_agree_on_a_plan_with_handoffs() {
        let plan = ExecutionPlan::Step {
            node_id: "router".into(),
            handoffs: vec![
                (
                    "billing".into(),
                    ExecutionPlan::Execute {
                        node_id: "billing".into(),
                    },
                ),
                (
                    "tech".into(),
                    ExecutionPlan::Sequence(vec![
                        ExecutionPlan::Execute {
                            node_id: "triage".into(),
                        },
                        ExecutionPlan::Execute {
                            node_id: "tech".into(),
                        },
                    ]),
                ),
            ],
        };

        assert_eq!(plan.node_ids(), vec!["router", "billing", "triage", "tech"]);
        assert_eq!(plan.node_count(), plan.node_ids().len());
    }

    /// Whatever the shape, the two accessors count the same tree.
    #[test]
    fn node_count_is_the_length_of_node_ids() {
        let plan = ExecutionPlan::Sequence(vec![
            ExecutionPlan::Execute {
                node_id: "prep".into(),
            },
            ExecutionPlan::Parallel(vec![
                ExecutionPlan::Execute {
                    node_id: "a".into(),
                },
                ExecutionPlan::Loop {
                    node_id: "refine".into(),
                    body: Box::new(ExecutionPlan::Execute {
                        node_id: "draft".into(),
                    }),
                    max_iterations: Some(3),
                    until: somatize_core::control::LoopCondition::Exhaust,
                    carry_from: None,
                },
            ]),
            ExecutionPlan::Branch {
                node_id: "route".into(),
                arms: vec![(
                    "left".into(),
                    ExecutionPlan::Execute {
                        node_id: "l".into(),
                    },
                )],
            },
        ]);
        assert_eq!(plan.node_count(), plan.node_ids().len());
    }

    #[test]
    fn node_count_linear() {
        let plan = ExecutionPlan::Sequence(vec![
            ExecutionPlan::Execute {
                node_id: "a".into(),
            },
            ExecutionPlan::Execute {
                node_id: "b".into(),
            },
            ExecutionPlan::Execute {
                node_id: "c".into(),
            },
        ]);
        assert_eq!(plan.node_count(), 3);
    }

    #[test]
    fn parallel_branch_count() {
        let plan = ExecutionPlan::Sequence(vec![
            ExecutionPlan::Execute {
                node_id: "a".into(),
            },
            ExecutionPlan::Parallel(vec![
                ExecutionPlan::Execute {
                    node_id: "b".into(),
                },
                ExecutionPlan::Execute {
                    node_id: "c".into(),
                },
                ExecutionPlan::Execute {
                    node_id: "d".into(),
                },
            ]),
            ExecutionPlan::Execute {
                node_id: "e".into(),
            },
        ]);
        assert_eq!(plan.parallel_branch_count(), 3);
        assert_eq!(plan.node_count(), 5);
    }

    #[test]
    fn node_ids_collected() {
        let plan = ExecutionPlan::Sequence(vec![
            ExecutionPlan::Execute {
                node_id: "a".into(),
            },
            ExecutionPlan::Execute {
                node_id: "b".into(),
            },
        ]);
        let ids = plan.node_ids();
        assert_eq!(ids, vec!["a", "b"]);
    }

    #[test]
    fn simplify_removes_empty() {
        let plan = ExecutionPlan::Sequence(vec![
            ExecutionPlan::Empty,
            ExecutionPlan::Execute {
                node_id: "a".into(),
            },
            ExecutionPlan::Empty,
        ]);
        let simplified = plan.simplify();
        assert!(matches!(simplified, ExecutionPlan::Execute { .. }));
    }

    #[test]
    fn simplify_unwraps_single_element() {
        let plan = ExecutionPlan::Sequence(vec![ExecutionPlan::Execute {
            node_id: "a".into(),
        }]);
        let simplified = plan.simplify();
        assert!(matches!(simplified, ExecutionPlan::Execute { .. }));
    }

    #[test]
    fn simplify_preserves_multi() {
        let plan = ExecutionPlan::Sequence(vec![
            ExecutionPlan::Execute {
                node_id: "a".into(),
            },
            ExecutionPlan::Execute {
                node_id: "b".into(),
            },
        ]);
        let simplified = plan.simplify();
        assert!(matches!(simplified, ExecutionPlan::Sequence(_)));
    }

    #[test]
    fn display_format() {
        let plan = ExecutionPlan::Sequence(vec![
            ExecutionPlan::Execute {
                node_id: "scaler".into(),
            },
            ExecutionPlan::Parallel(vec![
                ExecutionPlan::Execute {
                    node_id: "pca".into(),
                },
                ExecutionPlan::Execute {
                    node_id: "umap".into(),
                },
            ]),
            ExecutionPlan::Execute {
                node_id: "svm".into(),
            },
        ]);
        let output = format!("{plan}");
        assert!(output.contains("Sequence:"));
        assert!(output.contains("Parallel:"));
        assert!(output.contains("Execute(scaler)"));
        assert!(output.contains("Execute(pca)"));
    }

    #[test]
    fn summary_values() {
        let plan = ExecutionPlan::Sequence(vec![
            ExecutionPlan::Execute {
                node_id: "a".into(),
            },
            ExecutionPlan::Parallel(vec![
                ExecutionPlan::Execute {
                    node_id: "b".into(),
                },
                ExecutionPlan::Execute {
                    node_id: "c".into(),
                },
            ]),
            ExecutionPlan::Execute {
                node_id: "d".into(),
            },
        ]);
        let summary = plan.summary();
        assert_eq!(summary.total_nodes, 4);
        assert_eq!(summary.cached_nodes, 0);
        assert_eq!(summary.parallel_branches, 2);
    }

    #[test]
    fn serde_roundtrip() {
        let plan = ExecutionPlan::Sequence(vec![
            ExecutionPlan::Execute {
                node_id: "a".into(),
            },
            ExecutionPlan::Execute {
                node_id: "b".into(),
            },
        ]);
        let json = serde_json::to_string(&plan).unwrap();
        let deserialized: ExecutionPlan = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.node_count(), 2);
    }

    #[test]
    fn empty_plan() {
        let plan = ExecutionPlan::Empty;
        assert_eq!(plan.node_count(), 0);
        assert!(plan.node_ids().is_empty());
    }

    #[test]
    fn to_mermaid_sequence() {
        let plan = ExecutionPlan::Sequence(vec![
            ExecutionPlan::Execute {
                node_id: "scaler".into(),
            },
            ExecutionPlan::Execute {
                node_id: "model".into(),
            },
        ]);
        let m = plan.to_mermaid();
        assert!(m.starts_with("graph TD"));
        assert!(m.contains("scaler[scaler]"));
        assert!(m.contains("model[model]"));
        assert!(m.contains("scaler --> model"));
    }

    #[test]
    fn to_mermaid_parallel() {
        let plan = ExecutionPlan::Parallel(vec![
            ExecutionPlan::Execute {
                node_id: "a".into(),
            },
            ExecutionPlan::Execute {
                node_id: "b".into(),
            },
        ]);
        let m = plan.to_mermaid();
        assert!(m.contains("fork_0{"));
        assert!(m.contains("fork_0 --> a"));
        assert!(m.contains("fork_0 --> b"));
    }
}

#[cfg(test)]
mod to_graph_tests {
    use super::*;

    #[test]
    fn plan_to_graph_mirrors_mermaid_synthesis() {
        let plan = ExecutionPlan::Sequence(vec![
            ExecutionPlan::Execute {
                node_id: "load".into(),
            },
            ExecutionPlan::Parallel(vec![
                ExecutionPlan::Execute {
                    node_id: "a".into(),
                },
                ExecutionPlan::Execute {
                    node_id: "b".into(),
                },
            ]),
        ]);
        let g = plan.to_graph();
        let ids: Vec<&str> = g.nodes.iter().map(|n| n.id.as_str()).collect();
        assert_eq!(ids, vec!["load", "fork_0", "a", "b"]);
        assert_eq!(g.nodes[1].label, "fork");
        let edges: Vec<(&str, &str)> = g
            .edges
            .iter()
            .map(|e| (e.source.as_str(), e.target.as_str()))
            .collect();
        assert_eq!(
            edges,
            vec![("load", "fork_0"), ("fork_0", "a"), ("fork_0", "b")]
        );
        // Every Graph renderer now applies to the plan.
        let svg = g.to_svg();
        assert!(svg.starts_with("<svg"));
        assert!(svg.contains(">fork</text>"));
    }

    #[test]
    fn plan_to_graph_branch_arms_carry_edge_labels() {
        let plan = ExecutionPlan::Branch {
            node_id: "check".into(),
            arms: vec![
                (
                    "converged".into(),
                    ExecutionPlan::Execute {
                        node_id: "stop".into(),
                    },
                ),
                (
                    "continue".into(),
                    ExecutionPlan::Execute {
                        node_id: "train".into(),
                    },
                ),
            ],
        };
        let g = plan.to_graph();
        let labels: Vec<Option<&str>> = g.edges.iter().map(|e| e.label.as_deref()).collect();
        assert_eq!(labels, vec![Some("converged"), Some("continue")]);
    }
}
