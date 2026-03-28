use serde::{Deserialize, Serialize};
use soma_core::cache::CacheKey;
use soma_core::graph::NodeId;
use std::fmt;

/// A compiled execution plan produced by the compiler.
///
/// This is a recursive tree that the runtime walks to execute a pipeline.
/// The compiler resolves caching, parallelism, and distribution before
/// the runtime sees the plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionPlan {
    /// Execute steps sequentially, one after another.
    Sequence(Vec<ExecutionPlan>),

    /// Execute branches concurrently (fork-join).
    Parallel(Vec<ExecutionPlan>),

    /// Execute a single filter node.
    Execute { node_id: NodeId },

    /// Load result from cache (resolved at compile time).
    Cached { node_id: NodeId, key: CacheKey },

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

    /// No-op: nothing to execute (e.g. empty graph).
    Empty,
}

impl ExecutionPlan {
    /// Count total nodes in the plan (Execute + Cached).
    pub fn node_count(&self) -> usize {
        match self {
            Self::Execute { .. } | Self::Cached { .. } => 1,
            Self::Sequence(steps) | Self::Parallel(steps) => {
                steps.iter().map(|s| s.node_count()).sum()
            }
            Self::Loop { body, .. } => 1 + body.node_count(),
            Self::Branch { arms, .. } => {
                1 + arms.iter().map(|(_, p)| p.node_count()).sum::<usize>()
            }
            Self::Empty => 0,
        }
    }

    /// Count cached nodes in the plan.
    pub fn cached_count(&self) -> usize {
        match self {
            Self::Cached { .. } => 1,
            Self::Execute { .. } => 0,
            Self::Sequence(steps) | Self::Parallel(steps) => {
                steps.iter().map(|s| s.cached_count()).sum()
            }
            Self::Loop { body, .. } => body.cached_count(),
            Self::Branch { arms, .. } => arms.iter().map(|(_, p)| p.cached_count()).sum(),
            Self::Empty => 0,
        }
    }

    /// Count parallel branches at the top level of the plan.
    pub fn parallel_branch_count(&self) -> usize {
        match self {
            Self::Parallel(branches) => branches.len(),
            Self::Sequence(steps) => steps.iter().map(|s| s.parallel_branch_count()).sum(),
            _ => 0,
        }
    }

    /// Collect all node IDs referenced in the plan.
    pub fn node_ids(&self) -> Vec<&str> {
        match self {
            Self::Execute { node_id } | Self::Cached { node_id, .. } => vec![node_id.as_str()],
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
            Self::Empty => vec![],
        }
    }

    /// Create a PlanSummary for event payloads.
    pub fn summary(&self) -> soma_core::event::PlanSummary {
        soma_core::event::PlanSummary {
            total_nodes: self.node_count(),
            cached_nodes: self.cached_count(),
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
            Self::Cached { node_id, key } => writeln!(f, "{pad}Cached({node_id}, {key})"),
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
        assert_eq!(plan.cached_count(), 0);
    }

    #[test]
    fn cached_count() {
        let plan = ExecutionPlan::Sequence(vec![
            ExecutionPlan::Cached {
                node_id: "a".into(),
                key: CacheKey::hash_data(b"a"),
            },
            ExecutionPlan::Execute {
                node_id: "b".into(),
            },
            ExecutionPlan::Cached {
                node_id: "c".into(),
                key: CacheKey::hash_data(b"c"),
            },
        ]);
        assert_eq!(plan.node_count(), 3);
        assert_eq!(plan.cached_count(), 2);
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
            ExecutionPlan::Cached {
                node_id: "a".into(),
                key: CacheKey::hash_data(b"a"),
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
            ExecutionPlan::Cached {
                node_id: "a".into(),
                key: CacheKey::hash_data(b"a"),
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
        assert_eq!(summary.cached_nodes, 1);
        assert_eq!(summary.parallel_branches, 2);
    }

    #[test]
    fn serde_roundtrip() {
        let plan = ExecutionPlan::Sequence(vec![
            ExecutionPlan::Cached {
                node_id: "a".into(),
                key: CacheKey::hash_data(b"test"),
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
        assert_eq!(plan.cached_count(), 0);
        assert!(plan.node_ids().is_empty());
    }
}
