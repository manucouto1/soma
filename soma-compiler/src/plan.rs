//! Execution plan — the compiled representation of a pipeline.
//!
//! Variants: Sequence, Parallel, Execute, Loop, Branch, Remote, Stream, Empty.
//! Plans are data-free (no filter implementations) and serializable.

use serde::{Deserialize, Serialize};
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
    Execute { node_id: NodeId },

    /// Iterate: execute body for each item in a collection.
    Loop {
        node_id: NodeId,
        body: Box<ExecutionPlan>,
        max_iterations: Option<usize>,
    },

    /// Conditional branching: evaluate condition, pick an arm.
    Branch {
        node_id: NodeId,
        arms: Vec<(String, ExecutionPlan)>,
    },

    /// Execute a sub-plan on a remote worker.
    Remote {
        node_id: NodeId,
        target: RemoteTarget,
        plan: Box<ExecutionPlan>,
    },

    /// Execute multiple differentiable nodes as a single block.
    /// The executor passes tensors directly between filters (no Value conversion),
    /// preserving PyTorch autograd for gradient flow.
    Composite { node_ids: Vec<NodeId> },

    /// Streaming execution: process input in chunks through a filter chain.
    /// Each filter's StreamMode (FixedState/Evolving/Barrier) defines its
    /// per-chunk contract. Results flow progressively — no full materialization.
    Stream {
        node_ids: Vec<NodeId>,
        chunk_size: usize,
    },

    /// No-op: nothing to execute (e.g. empty graph).
    Empty,
}

impl ExecutionPlan {
    /// Count total nodes in the plan.
    pub fn node_count(&self) -> usize {
        match self {
            Self::Execute { .. } => 1,
            Self::Composite { node_ids } | Self::Stream { node_ids, .. } => node_ids.len(),
            Self::Sequence(steps) | Self::Parallel(steps) => {
                steps.iter().map(|s| s.node_count()).sum()
            }
            Self::Loop { body, .. } => 1 + body.node_count(),
            Self::Branch { arms, .. } => {
                1 + arms.iter().map(|(_, p)| p.node_count()).sum::<usize>()
            }
            Self::Remote { plan, .. } => plan.node_count(),
            Self::Empty => 0,
        }
    }

    /// Count parallel branches at the top level of the plan.
    pub fn parallel_branch_count(&self) -> usize {
        match self {
            Self::Parallel(branches) => branches.len(),
            Self::Sequence(steps) => steps.iter().map(|s| s.parallel_branch_count()).sum(),
            Self::Execute { .. }
            | Self::Loop { .. }
            | Self::Branch { .. }
            | Self::Remote { .. }
            | Self::Composite { .. }
            | Self::Stream { .. }
            | Self::Empty => 0,
        }
    }

    /// Collect all node IDs referenced in the plan.
    pub fn node_ids(&self) -> Vec<&str> {
        match self {
            Self::Execute { node_id } => vec![node_id.as_str()],
            Self::Sequence(steps) | Self::Parallel(steps) => {
                steps.iter().flat_map(|s| s.node_ids()).collect()
            }
            Self::Loop { node_id, body, .. } => {
                let mut ids = vec![node_id.as_str()];
                ids.extend(body.node_ids());
                ids
            }
            Self::Branch { node_id, arms, .. } => {
                let mut ids = vec![node_id.as_str()];
                for (_, p) in arms {
                    ids.extend(p.node_ids());
                }
                ids
            }
            Self::Remote { node_id, plan, .. } => {
                let mut ids = vec![node_id.as_str()];
                ids.extend(plan.node_ids());
                ids
            }
            Self::Composite { node_ids } | Self::Stream { node_ids, .. } => {
                node_ids.iter().map(|s| s.as_str()).collect()
            }
            Self::Empty => vec![],
        }
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

    fn mermaid_nodes(&self, out: &mut String, counter: &mut usize, parent: Option<&str>) {
        use std::fmt::Write;
        match self {
            Self::Execute { node_id } => {
                let _ = writeln!(out, "    {node_id}[{node_id}]");
                if let Some(p) = parent {
                    let _ = writeln!(out, "    {p} --> {node_id}");
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
            Self::Execute { node_id } => Some(node_id),
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
            Self::Loop {
                node_id,
                body,
                max_iterations,
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
