//! Derivation moves — the *edge* of the experiment pool.
//!
//! VisTrails' insight about workflow-evolution provenance is that the
//! interesting object is not the workflow but the **change applied to
//! its parent**. A tree of records tells you what you ran; a tree of
//! records plus the move on each edge tells you what you *tried*, and
//! whether it worked.
//!
//! A [`DerivationMove`] is stored as a field of the child record, never
//! as a separate journal line. One node, one edge, one append: an edge
//! can never end up orphaned by a crash between two writes, and the
//! journal's append-only crash safety is not duplicated.
//!
//! [`derive`](fn@derive) is a pure function over two records, so the same code
//! serves both automatic capture and the on-demand `kb_diff` tool.
//! When the parent carries no architecture — a legacy record, or one
//! whose run directory is gone — the result is a single
//! [`Change::Unspecified`]. Saying "something changed, and I cannot
//! say what" is worth more than inventing a plausible diff.

use crate::record::ExperimentRecord;
use serde::{Deserialize, Serialize};
use somatize_core::fingerprint::ArchitectureFingerprint;
use somatize_core::summary::round4;
use std::collections::{BTreeMap, BTreeSet};

/// One atomic difference between a parent experiment and its child.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "change")]
#[non_exhaustive]
pub enum Change {
    /// A node the parent's graph did not have.
    NodeAdded {
        /// The new node's id.
        node: String,
        /// The filter type behind it.
        filter: String,
    },
    /// A node the child's graph no longer has.
    NodeRemoved {
        /// The removed node's id.
        node: String,
        /// The filter type it carried.
        filter: String,
    },
    /// Same node id, different filter behind it.
    NodeReplaced {
        /// The node id both graphs share.
        node: String,
        /// The parent's filter type.
        from: String,
        /// The child's filter type.
        to: String,
    },
    /// Same filter, different configuration (the node's config hash
    /// moved). The *values* are not recoverable from a fingerprint —
    /// only the fact that they differ.
    NodeReconfigured {
        /// The node whose configuration moved.
        node: String,
        /// The parent's config hash.
        from_hash: String,
        /// The child's config hash.
        to_hash: String,
    },
    /// A data edge only the child's graph has.
    EdgeAdded {
        /// The edge's source node.
        source: String,
        /// The edge's target node.
        target: String,
    },
    /// A data edge only the parent's graph had.
    EdgeRemoved {
        /// The edge's source node.
        source: String,
        /// The edge's target node.
        target: String,
    },
    /// The same parameter key, set to a different value.
    ParamChanged {
        /// The parameter key.
        key: String,
        /// The parent's value.
        from: serde_json::Value,
        /// The child's value.
        to: serde_json::Value,
    },
    /// A parameter only the child sets.
    ParamAdded {
        /// The parameter key.
        key: String,
        /// The value the child sets it to.
        value: serde_json::Value,
    },
    /// A parameter only the parent set.
    ParamRemoved {
        /// The parameter key.
        key: String,
        /// The value the parent had set.
        value: serde_json::Value,
    },
    /// A study's search space moved (different dimensions searched).
    SearchSpaceChanged {
        /// Dimensions only the child searches.
        added: Vec<String>,
        /// Dimensions only the parent searched.
        removed: Vec<String>,
    },
    /// The code changed underneath: different git commit.
    CodeChanged {
        /// The parent's commit sha, when it recorded one.
        from_sha: Option<String>,
        /// The child's commit sha, when it recorded one.
        to_sha: Option<String>,
    },
    /// There was a move, but the evidence to describe it is gone.
    Unspecified {
        /// Why the diff could not be computed.
        reason: String,
    },
}

impl Change {
    /// Short human rendering, used to build a move's summary line.
    pub fn describe(&self) -> String {
        match self {
            Self::NodeAdded { node, filter } => format!("+node {node} ({filter})"),
            Self::NodeRemoved { node, filter } => format!("-node {node} ({filter})"),
            Self::NodeReplaced { node, from, to } => format!("{node}: {from} → {to}"),
            Self::NodeReconfigured { node, .. } => format!("{node} reconfigured"),
            Self::EdgeAdded { source, target } => format!("+edge {source}→{target}"),
            Self::EdgeRemoved { source, target } => format!("-edge {source}→{target}"),
            Self::ParamChanged { key, from, to } => {
                format!("{key}: {} → {}", compact(from), compact(to))
            }
            Self::ParamAdded { key, value } => format!("+{key}={}", compact(value)),
            Self::ParamRemoved { key, value } => format!("-{key}={}", compact(value)),
            Self::SearchSpaceChanged { added, removed } => {
                let mut parts = Vec::new();
                if !added.is_empty() {
                    parts.push(format!("+{}", added.join(",")));
                }
                if !removed.is_empty() {
                    parts.push(format!("-{}", removed.join(",")));
                }
                format!("search space {}", parts.join(" "))
            }
            Self::CodeChanged { from_sha, to_sha } => format!(
                "code {} → {}",
                short_sha(from_sha.as_deref()),
                short_sha(to_sha.as_deref())
            ),
            Self::Unspecified { reason } => format!("unspecified ({reason})"),
        }
    }
}

/// How one metric moved between parent and child.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MetricDelta {
    /// The parent's value.
    pub before: f64,
    /// The child's value.
    pub after: f64,
    /// `after - before`. Signed on purpose: whether that is good news
    /// depends on the objective's direction, which the move does not
    /// presume to know.
    pub delta: f64,
}

/// The edge from a parent experiment to this one: what was changed, and
/// what it did to the numbers.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct DerivationMove {
    /// Parent experiment id.
    pub from: String,
    /// Child experiment id (this record).
    pub to: String,
    /// The atomic differences, in the deterministic order
    /// [`derive`](fn@derive) emits them. Never empty from [`derive`](fn@derive):
    /// when nothing is visible it holds one [`Change::Unspecified`].
    pub changes: Vec<Change>,
    /// Per-metric movement, for metrics both runs reported.
    #[serde(default)]
    pub metric_delta: BTreeMap<String, MetricDelta>,
    /// Deterministic one-line rendering of the move.
    #[serde(default)]
    pub summary: String,
}

impl DerivationMove {
    /// Whether anything at all is known to have changed.
    pub fn is_empty(&self) -> bool {
        self.changes.is_empty() && self.metric_delta.is_empty()
    }
}

/// Changes named in a summary before it says "+N more".
const SUMMARY_CHANGES: usize = 4;
/// Metrics named in a summary before it stops listing them.
const SUMMARY_METRICS: usize = 3;

/// Diff two experiment records into the move that separates them.
///
/// Pure and deterministic: same pair, same move. Used both when a run
/// finishes (parent resolved from `.soma/HEAD`) and by `kb_diff` on any
/// two records the user names.
pub fn derive(parent: &ExperimentRecord, child: &ExperimentRecord) -> DerivationMove {
    let mut changes = Vec::new();

    match (&parent.architecture, &child.architecture) {
        (Some(before), Some(after)) => changes.extend(architecture_changes(before, after)),
        (None, Some(_)) => changes.push(Change::Unspecified {
            reason: format!("no architecture recorded for parent {}", parent.id),
        }),
        (Some(_), None) => changes.push(Change::Unspecified {
            reason: format!("no architecture recorded for child {}", child.id),
        }),
        (None, None) => {}
    }

    changes.extend(param_changes(&parent.params, &child.params));

    let (before_sha, after_sha) = (
        parent.git.as_ref().and_then(|g| g.sha.clone()),
        child.git.as_ref().and_then(|g| g.sha.clone()),
    );
    if before_sha != after_sha && (before_sha.is_some() || after_sha.is_some()) {
        changes.push(Change::CodeChanged {
            from_sha: before_sha,
            to_sha: after_sha,
        });
    }

    // Nothing observable moved, yet the user branched deliberately —
    // say so rather than emitting an empty, uninformative edge.
    if changes.is_empty() {
        changes.push(Change::Unspecified {
            reason: "no difference visible in architecture, params or code".into(),
        });
    }

    let metric_delta = metric_deltas(parent, child);
    let mut derivation = DerivationMove {
        from: parent.id.clone(),
        to: child.id.clone(),
        changes,
        metric_delta,
        summary: String::new(),
    };
    derivation.summary = summarize_move(&derivation);
    derivation
}

/// Node- and edge-level differences between two fingerprints.
fn architecture_changes(
    before: &ArchitectureFingerprint,
    after: &ArchitectureFingerprint,
) -> Vec<Change> {
    let mut changes = Vec::new();
    let ids: BTreeSet<&String> = before.nodes.keys().chain(after.nodes.keys()).collect();
    for id in ids {
        match (before.nodes.get(id), after.nodes.get(id)) {
            (None, Some(filter)) => changes.push(Change::NodeAdded {
                node: id.clone(),
                filter: filter.clone(),
            }),
            (Some(filter), None) => changes.push(Change::NodeRemoved {
                node: id.clone(),
                filter: filter.clone(),
            }),
            (Some(from), Some(to)) if from != to => changes.push(Change::NodeReplaced {
                node: id.clone(),
                from: from.clone(),
                to: to.clone(),
            }),
            (Some(_), Some(_)) => {
                // Same filter type: did its configuration move? Only
                // knowable when both runs captured a config hash.
                if let (Some(from), Some(to)) =
                    (before.node_config.get(id), after.node_config.get(id))
                    && from != to
                {
                    changes.push(Change::NodeReconfigured {
                        node: id.clone(),
                        from_hash: from.clone(),
                        to_hash: to.clone(),
                    });
                }
            }
            (None, None) => unreachable!("id came from one of the two maps"),
        }
    }

    let before_edges: BTreeSet<_> = before.edges.iter().collect();
    let after_edges: BTreeSet<_> = after.edges.iter().collect();
    for edge in after_edges.difference(&before_edges) {
        changes.push(Change::EdgeAdded {
            source: edge.source.clone(),
            target: edge.target.clone(),
        });
    }
    for edge in before_edges.difference(&after_edges) {
        changes.push(Change::EdgeRemoved {
            source: edge.source.clone(),
            target: edge.target.clone(),
        });
    }
    changes
}

fn param_changes(
    before: &std::collections::BTreeMap<String, serde_json::Value>,
    after: &std::collections::BTreeMap<String, serde_json::Value>,
) -> Vec<Change> {
    let keys: BTreeSet<&String> = before.keys().chain(after.keys()).collect();
    keys.into_iter()
        .filter_map(|key| match (before.get(key), after.get(key)) {
            (Some(from), Some(to)) if from != to => Some(Change::ParamChanged {
                key: key.clone(),
                from: from.clone(),
                to: to.clone(),
            }),
            (None, Some(value)) => Some(Change::ParamAdded {
                key: key.clone(),
                value: value.clone(),
            }),
            (Some(value), None) => Some(Change::ParamRemoved {
                key: key.clone(),
                value: value.clone(),
            }),
            _ => None,
        })
        .collect()
}

/// Movement of every metric both records report.
fn metric_deltas(
    parent: &ExperimentRecord,
    child: &ExperimentRecord,
) -> BTreeMap<String, MetricDelta> {
    parent
        .metrics
        .iter()
        .filter_map(|(name, before)| {
            let after = *child.metrics.get(name)?;
            Some((
                name.clone(),
                MetricDelta {
                    before: *before,
                    after,
                    delta: after - before,
                },
            ))
        })
        .collect()
}

/// One deterministic line: what changed, then what it did.
fn summarize_move(derivation: &DerivationMove) -> String {
    let mut described: Vec<String> = derivation
        .changes
        .iter()
        .take(SUMMARY_CHANGES)
        .map(Change::describe)
        .collect();
    if derivation.changes.len() > SUMMARY_CHANGES {
        described.push(format!(
            "+{} more",
            derivation.changes.len() - SUMMARY_CHANGES
        ));
    }
    let mut summary = described.join("; ");

    if !derivation.metric_delta.is_empty() {
        let moved: Vec<String> = derivation
            .metric_delta
            .iter()
            .take(SUMMARY_METRICS)
            .map(|(name, d)| format!("{name} {}{}", sign(d.delta), round4(d.delta.abs())))
            .collect();
        summary.push_str(&format!(" ⇒ {}", moved.join(", ")));
    }
    summary
}

fn sign(delta: f64) -> &'static str {
    if delta > 0.0 {
        "+"
    } else if delta < 0.0 {
        "−"
    } else {
        "±"
    }
}

fn short_sha(sha: Option<&str>) -> String {
    match sha {
        Some(sha) => sha.chars().take(8).collect(),
        None => "?".to_string(),
    }
}

/// Compact JSON rendering for a parameter value inside a one-liner.
fn compact(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Number(n) => n.to_string(),
        other => {
            let text = other.to_string();
            somatize_core::summary::one_line(&text, 40)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use somatize_core::fingerprint::EdgeRef;
    use somatize_core::graph::{Node, linear_pipeline};
    use somatize_core::tracking::GitInfo;

    fn fingerprint(nodes: &[(&str, &str)], edges: &[(&str, &str)]) -> ArchitectureFingerprint {
        ArchitectureFingerprint {
            digest: format!("{nodes:?}{edges:?}"),
            nodes: nodes
                .iter()
                .map(|(id, filter)| ((*id).to_string(), format!("filter:{filter}")))
                .collect(),
            edges: edges
                .iter()
                .map(|(s, t)| EdgeRef {
                    source: (*s).to_string(),
                    target: (*t).to_string(),
                    kind: "data".into(),
                })
                .collect(),
            n_nodes: nodes.len(),
            n_edges: edges.len(),
            node_config: BTreeMap::new(),
        }
    }

    fn record(id: &str) -> ExperimentRecord {
        ExperimentRecord::new(id, format!("{id}-run"))
    }

    #[test]
    fn a_swapped_filter_reads_as_a_replacement() {
        let mut parent = record("p");
        parent.architecture = Some(fingerprint(&[("a", "Scaler"), ("b", "SVM")], &[("a", "b")]));
        let mut child = record("c");
        child.architecture = Some(fingerprint(
            &[("a", "Scaler"), ("b", "RandomForest")],
            &[("a", "b")],
        ));

        let move_ = derive(&parent, &child);
        assert_eq!(move_.from, "p");
        assert_eq!(move_.to, "c");
        assert_eq!(
            move_.changes,
            vec![Change::NodeReplaced {
                node: "b".into(),
                from: "filter:SVM".into(),
                to: "filter:RandomForest".into(),
            }]
        );
        assert_eq!(move_.summary, "b: filter:SVM → filter:RandomForest");
    }

    #[test]
    fn added_and_removed_nodes_and_edges() {
        let mut parent = record("p");
        parent.architecture = Some(fingerprint(&[("a", "A"), ("b", "B")], &[("a", "b")]));
        let mut child = record("c");
        child.architecture = Some(fingerprint(
            &[("a", "A"), ("c", "C")],
            &[("a", "c"), ("c", "a")],
        ));

        let changes = derive(&parent, &child).changes;
        assert!(changes.contains(&Change::NodeAdded {
            node: "c".into(),
            filter: "filter:C".into()
        }));
        assert!(changes.contains(&Change::NodeRemoved {
            node: "b".into(),
            filter: "filter:B".into()
        }));
        assert!(changes.contains(&Change::EdgeAdded {
            source: "a".into(),
            target: "c".into()
        }));
        assert!(changes.contains(&Change::EdgeRemoved {
            source: "a".into(),
            target: "b".into()
        }));
    }

    #[test]
    fn a_reconfigured_node_needs_both_config_hashes() {
        let mut before = fingerprint(&[("model", "SVM")], &[]);
        let mut after = before.clone();

        // Without hashes there is nothing to compare — no change claimed.
        let mut parent = record("p");
        parent.architecture = Some(before.clone());
        let mut child = record("c");
        child.architecture = Some(after.clone());
        assert!(matches!(
            derive(&parent, &child).changes.as_slice(),
            [Change::Unspecified { .. }]
        ));

        before.node_config = BTreeMap::from([("model".into(), "aaa".into())]);
        after.node_config = BTreeMap::from([("model".into(), "bbb".into())]);
        parent.architecture = Some(before);
        child.architecture = Some(after);
        assert_eq!(
            derive(&parent, &child).changes,
            vec![Change::NodeReconfigured {
                node: "model".into(),
                from_hash: "aaa".into(),
                to_hash: "bbb".into(),
            }]
        );
    }

    #[test]
    fn params_produce_signed_metric_deltas() {
        let mut parent = record("p");
        parent.params.insert("lr".into(), serde_json::json!(0.01));
        parent
            .params
            .insert("dropout".into(), serde_json::json!(0.1));
        parent.metrics.insert("val_f1".into(), 0.80);
        parent.metrics.insert("loss".into(), 0.50);
        parent.metrics.insert("only_parent".into(), 1.0);

        let mut child = record("c");
        child.params.insert("lr".into(), serde_json::json!(0.05));
        child.params.insert("seed".into(), serde_json::json!(7));
        child.metrics.insert("val_f1".into(), 0.87);
        child.metrics.insert("loss".into(), 0.42);

        let move_ = derive(&parent, &child);
        assert_eq!(
            move_.changes,
            vec![
                Change::ParamRemoved {
                    key: "dropout".into(),
                    value: serde_json::json!(0.1)
                },
                Change::ParamChanged {
                    key: "lr".into(),
                    from: serde_json::json!(0.01),
                    to: serde_json::json!(0.05)
                },
                Change::ParamAdded {
                    key: "seed".into(),
                    value: serde_json::json!(7)
                },
            ]
        );
        // Only metrics both runs reported are comparable.
        assert_eq!(move_.metric_delta.len(), 2);
        let f1 = move_.metric_delta["val_f1"];
        assert_eq!((f1.before, f1.after), (0.80, 0.87));
        assert!((f1.delta - 0.07).abs() < 1e-9);
        assert!(move_.metric_delta["loss"].delta < 0.0);
        assert!(move_.summary.contains("val_f1 +0.07"), "{}", move_.summary);
        assert!(move_.summary.contains("loss −0.08"), "{}", move_.summary);
    }

    #[test]
    fn a_missing_parent_architecture_is_stated_not_invented() {
        let parent = record("p");
        let mut child = record("c");
        child.architecture = Some(fingerprint(&[("a", "A")], &[]));

        let move_ = derive(&parent, &child);
        assert_eq!(
            move_.changes,
            vec![Change::Unspecified {
                reason: "no architecture recorded for parent p".into()
            }]
        );
        assert!(
            move_.summary.starts_with("unspecified ("),
            "{}",
            move_.summary
        );
    }

    #[test]
    fn an_identical_pair_still_records_an_honest_edge() {
        let parent = record("p");
        let child = record("c");
        let move_ = derive(&parent, &child);
        assert!(
            !move_.is_empty(),
            "a branch is a fact even when nothing moved"
        );
        assert_eq!(
            move_.changes,
            vec![Change::Unspecified {
                reason: "no difference visible in architecture, params or code".into()
            }]
        );
    }

    #[test]
    fn a_code_change_is_a_change() {
        let mut parent = record("p");
        parent.git = Some(GitInfo {
            sha: Some("abcdef1234567890".into()),
            branch: Some("main".into()),
            dirty: Some(false),
        });
        let mut child = record("c");
        child.git = Some(GitInfo {
            sha: Some("0987654321fedcba".into()),
            ..GitInfo::default()
        });
        let move_ = derive(&parent, &child);
        assert_eq!(
            move_.changes,
            vec![Change::CodeChanged {
                from_sha: Some("abcdef1234567890".into()),
                to_sha: Some("0987654321fedcba".into()),
            }]
        );
        assert_eq!(move_.summary, "code abcdef12 → 09876543");
    }

    #[test]
    fn the_summary_caps_long_move_lists() {
        let mut parent = record("p");
        let mut child = record("c");
        for i in 0..10 {
            parent.params.insert(format!("p{i}"), serde_json::json!(i));
            child
                .params
                .insert(format!("p{i}"), serde_json::json!(i + 1));
        }
        let move_ = derive(&parent, &child);
        assert_eq!(move_.changes.len(), 10);
        assert!(move_.summary.contains("+6 more"), "{}", move_.summary);
    }

    #[test]
    fn derive_is_deterministic_and_roundtrips() {
        let mut parent = record("p");
        parent.architecture = Some(
            ArchitectureFingerprint::of(&linear_pipeline(vec![
                Node::new("a", "Scaler", "StandardScaler"),
                Node::new("b", "Model", "SVM"),
            ]))
            .unwrap(),
        );
        parent.metrics.insert("f1".into(), 0.5);
        let mut child = record("c");
        child.architecture = Some(
            ArchitectureFingerprint::of(&linear_pipeline(vec![
                Node::new("a", "Scaler", "StandardScaler"),
                Node::new("b", "Model", "RandomForest"),
            ]))
            .unwrap(),
        );
        child.metrics.insert("f1".into(), 0.6);

        let first = derive(&parent, &child);
        for _ in 0..5 {
            assert_eq!(derive(&parent, &child), first);
        }
        let json = serde_json::to_string(&first).unwrap();
        let back: DerivationMove = serde_json::from_str(&json).unwrap();
        assert_eq!(back, first);
        // Tagged representation keeps the variant name in the JSON.
        assert!(json.contains("\"change\":\"NodeReplaced\""), "{json}");
    }
}
