//! Architecture fingerprints — a stable identity for a graph's *shape*.
//!
//! Two questions the experiment pool has to answer are different enough
//! to need two answers:
//!
//! - *"Is this the exact same architecture I already ran?"* →
//!   [`ArchitectureFingerprint::digest`], an exact hash over node ids,
//!   node kinds and edges. Sensitive to renaming, which is what makes it
//!   usable as a dedup key.
//! - *"What have I run that looks like this?"* →
//!   [`ArchitectureFingerprint::node_tokens`] / [`edge_tokens`], bags of
//!   *type* tokens that carry no node ids at all, compared with
//!   [`structural_similarity`]. Renaming `scaler` to `norm` leaves them
//!   untouched.
//!
//! Both are derived from the same canonical form: nodes sorted by id,
//! edges sorted, with `edge.id`, `node.label` and `node.target`
//! excluded — they are cosmetic or deployment detail, not architecture.
//! `SubGraph` nodes recurse *by digest*, so nesting terminates and the
//! result stays independent of traversal order.
//!
//! [`edge_tokens`]: ArchitectureFingerprint::edge_tokens

use crate::canon::hash_canonical;
use crate::error::Result;
use crate::graph::{EdgeKind, Graph, NodeKind};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Structural identity of a graph, exact and fuzzy.
///
/// Written to `fingerprint.json` in every tracked run directory and
/// copied into the `ExperimentRecord` so the pool can rank past work by
/// architectural resemblance without re-reading any graph.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArchitectureFingerprint {
    /// Hex SHA-256 of the canonical form — exact, id-sensitive.
    pub digest: String,
    /// One type token per node, sorted (duplicates kept: two scalers
    /// are structurally different from one).
    pub node_tokens: Vec<String>,
    /// One `source>target` type token per edge, sorted, duplicates kept.
    pub edge_tokens: Vec<String>,
    pub n_nodes: usize,
    pub n_edges: usize,
    /// Per-node filter config hash, keyed by node id. Empty unless the
    /// caller had a filter registry to hand (soma-core has no access to
    /// filter instances; the Python binding fills this in at run start).
    #[serde(default)]
    pub node_config: BTreeMap<String, String>,
}

impl ArchitectureFingerprint {
    /// Fingerprint `graph`. Errors only if the canonical form is not
    /// serializable, which for a `Graph` means a bug, not bad input.
    pub fn of(graph: &Graph) -> Result<Self> {
        let canonical = canonical_form(graph)?;
        let digest = hash_canonical(&canonical)?.to_hex();
        let mut node_tokens: Vec<String> = canonical.nodes.iter().map(|n| n.1.clone()).collect();
        let mut edge_tokens: Vec<String> = canonical
            .edges
            .iter()
            .map(|(source, target, kind)| {
                let arrow = if kind == "control" { "~>" } else { ">" };
                let source = token_of(&canonical, source);
                let target = token_of(&canonical, target);
                format!("{source}{arrow}{target}")
            })
            .collect();
        node_tokens.sort();
        edge_tokens.sort();
        Ok(Self {
            digest,
            n_nodes: canonical.nodes.len(),
            n_edges: canonical.edges.len(),
            node_tokens,
            edge_tokens,
            node_config: BTreeMap::new(),
        })
    }

    /// Attach per-node config hashes (node id → hex hash).
    pub fn with_node_config(mut self, node_config: BTreeMap<String, String>) -> Self {
        self.node_config = node_config;
        self
    }

    /// Short prefix of the digest, for display.
    pub fn short(&self) -> &str {
        let end = self.digest.len().min(12);
        &self.digest[..end]
    }
}

/// Structural resemblance of two fingerprints in `[0, 1]`.
///
/// `0.6 · jaccard(node tokens) + 0.4 · jaccard(edge tokens)`, where
/// jaccard is the *multiset* variant (intersection sums per-token
/// minima, union sums maxima) so node counts matter. Deterministic and
/// linear — deliberately not graph isomorphism, which is both expensive
/// and too strict for "these two look alike".
///
/// Two empty fingerprints score `1.0` (identical), an empty against a
/// non-empty scores `0.0`.
pub fn structural_similarity(a: &ArchitectureFingerprint, b: &ArchitectureFingerprint) -> f64 {
    0.6 * multiset_jaccard(&a.node_tokens, &b.node_tokens)
        + 0.4 * multiset_jaccard(&a.edge_tokens, &b.edge_tokens)
}

/// One-line human description of a graph's topology, for the
/// `pipeline_summary` field of an experiment record.
///
/// Linear chains render as `a → b → c`; anything with fan-out renders
/// as the node list plus an edge count. Truncated so a summary never
/// dominates a search result.
pub fn pipeline_summary(graph: &Graph) -> String {
    const MAX_NODES: usize = 8;

    if graph.nodes.is_empty() {
        return "empty graph".to_string();
    }
    let sorted = graph.topological_sort().unwrap_or_default();
    let order: Vec<&str> = if sorted.len() == graph.nodes.len() {
        sorted
    } else {
        // Cyclic or malformed: fall back to declaration order.
        graph.nodes.iter().map(|n| n.id.as_str()).collect()
    };
    let described: Vec<String> = order
        .iter()
        .take(MAX_NODES)
        .map(|id| match graph.node(id).map(|n| &n.kind) {
            Some(NodeKind::Filter { filter_name }) if filter_name != id => {
                format!("{id}({filter_name})")
            }
            Some(NodeKind::SubGraph { graph }) => format!("{id}[{} nodes]", graph.nodes.len()),
            Some(NodeKind::Loop { .. }) => format!("{id}[loop]"),
            Some(NodeKind::Branch) => format!("{id}[branch]"),
            _ => (*id).to_string(),
        })
        .collect();
    let mut summary = described.join(" → ");
    if order.len() > MAX_NODES {
        summary.push_str(&format!(" → … (+{} more)", order.len() - MAX_NODES));
    }
    let is_chain = graph.edges.len() + 1 == graph.nodes.len()
        && graph
            .nodes
            .iter()
            .all(|n| graph.predecessors(&n.id).len() <= 1 && graph.successors(&n.id).len() <= 1);
    if is_chain {
        summary
    } else {
        format!(
            "{summary} ({} nodes, {} edges)",
            graph.nodes.len(),
            graph.edges.len()
        )
    }
}

/// The canonical, cosmetics-free view a digest is taken over.
#[derive(Debug, Serialize)]
struct CanonicalGraph {
    /// `(node id, type token)`, sorted by id.
    nodes: Vec<(String, String)>,
    /// `(source, target, kind)`, sorted.
    edges: Vec<(String, String, String)>,
}

fn canonical_form(graph: &Graph) -> Result<CanonicalGraph> {
    let mut nodes: Vec<(String, String)> = graph
        .nodes
        .iter()
        .map(|node| Ok((node.id.clone(), kind_token(&node.kind)?)))
        .collect::<Result<_>>()?;
    nodes.sort();
    let mut edges: Vec<(String, String, String)> = graph
        .edges
        .iter()
        .map(|edge| {
            let kind = match edge.kind {
                EdgeKind::Data => "data",
                EdgeKind::Control => "control",
            };
            (edge.source.clone(), edge.target.clone(), kind.to_string())
        })
        .collect();
    edges.sort();
    Ok(CanonicalGraph { nodes, edges })
}

/// Type token for a node kind — no ids, no labels, no targets.
///
/// A sub-graph collapses to its own digest, which keeps recursion
/// bounded by nesting depth and independent of the order the inner
/// nodes were declared in.
fn kind_token(kind: &NodeKind) -> Result<String> {
    Ok(match kind {
        NodeKind::Filter { filter_name } => format!("filter:{filter_name}"),
        NodeKind::SubGraph { graph } => {
            let inner = ArchitectureFingerprint::of(graph)?;
            format!("subgraph:{}", inner.short())
        }
        NodeKind::Loop { max_iterations } => match max_iterations {
            Some(n) => format!("loop:{n}"),
            None => "loop:*".to_string(),
        },
        NodeKind::Branch => "branch".to_string(),
    })
}

/// Type token of the node `id` refers to. A dangling edge endpoint
/// (possible in an unvalidated graph) tokenizes as `missing`.
fn token_of(canonical: &CanonicalGraph, id: &str) -> String {
    canonical
        .nodes
        .binary_search_by(|(node_id, _)| node_id.as_str().cmp(id))
        .map(|i| canonical.nodes[i].1.clone())
        .unwrap_or_else(|_| "missing".to_string())
}

/// Jaccard over multisets: `Σ min(count) / Σ max(count)`.
fn multiset_jaccard(a: &[String], b: &[String]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let mut counts: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    for token in a {
        counts.entry(token).or_default().0 += 1;
    }
    for token in b {
        counts.entry(token).or_default().1 += 1;
    }
    let (mut intersection, mut union) = (0usize, 0usize);
    for (left, right) in counts.values() {
        intersection += left.min(right);
        union += left.max(right);
    }
    if union == 0 {
        return 1.0;
    }
    intersection as f64 / union as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Edge, Node, linear_pipeline};

    fn chain() -> Graph {
        linear_pipeline(vec![
            Node::new("a", "Scaler", "StandardScaler"),
            Node::new("b", "Reducer", "PCA"),
            Node::new("c", "Model", "SVM"),
        ])
    }

    #[test]
    fn digest_is_deterministic_across_declaration_order() {
        let forward = chain();
        let mut shuffled = Graph::new();
        for node in forward.nodes.iter().rev() {
            shuffled.add_node(node.clone());
        }
        for edge in forward.edges.iter().rev() {
            shuffled.add_edge(edge.clone());
        }
        let a = ArchitectureFingerprint::of(&forward).unwrap();
        let b = ArchitectureFingerprint::of(&shuffled).unwrap();
        assert_eq!(a.digest, b.digest);
        assert_eq!(a.node_tokens, b.node_tokens);
        assert_eq!(a.edge_tokens, b.edge_tokens);
    }

    #[test]
    fn digest_ignores_cosmetics_but_not_structure() {
        let base = ArchitectureFingerprint::of(&chain()).unwrap();

        // Labels and targets are cosmetic / deployment detail.
        let mut cosmetic = chain();
        cosmetic.nodes[0].label = "renamed for the paper".into();
        cosmetic.nodes[1].target = Some("gpu".into());
        cosmetic.edges[0].id = "totally-different-edge-id".into();
        cosmetic.edges[0].label = Some("x".into());
        assert_eq!(
            base.digest,
            ArchitectureFingerprint::of(&cosmetic).unwrap().digest
        );

        // The filter behind a node is not.
        let mut swapped = chain();
        swapped.nodes[2].kind = NodeKind::Filter {
            filter_name: "RandomForest".into(),
        };
        assert_ne!(
            base.digest,
            ArchitectureFingerprint::of(&swapped).unwrap().digest
        );

        // Neither is an extra edge.
        let mut extra = chain();
        extra.add_edge(Edge::data("skip", "a", "c"));
        assert_ne!(
            base.digest,
            ArchitectureFingerprint::of(&extra).unwrap().digest
        );
    }

    #[test]
    fn digest_is_id_sensitive_but_tokens_are_not() {
        let base = ArchitectureFingerprint::of(&chain()).unwrap();
        let renamed = linear_pipeline(vec![
            Node::new("first", "Scaler", "StandardScaler"),
            Node::new("second", "Reducer", "PCA"),
            Node::new("third", "Model", "SVM"),
        ]);
        let renamed = ArchitectureFingerprint::of(&renamed).unwrap();
        assert_ne!(base.digest, renamed.digest, "digest seeds exact dedup");
        assert_eq!(base.node_tokens, renamed.node_tokens);
        assert_eq!(base.edge_tokens, renamed.edge_tokens);
        assert_eq!(structural_similarity(&base, &renamed), 1.0);
    }

    #[test]
    fn subgraph_recursion_is_by_digest_and_order_independent() {
        let inner_a = chain();
        let mut inner_b = Graph::new();
        for node in inner_a.nodes.iter().rev() {
            inner_b.add_node(node.clone());
        }
        for edge in inner_a.edges.iter().rev() {
            inner_b.add_edge(edge.clone());
        }
        let mut outer_a = Graph::new();
        outer_a.add_node(Node::subgraph("stage", inner_a));
        let mut outer_b = Graph::new();
        outer_b.add_node(Node::subgraph("stage", inner_b));
        assert_eq!(
            ArchitectureFingerprint::of(&outer_a).unwrap().digest,
            ArchitectureFingerprint::of(&outer_b).unwrap().digest
        );

        // A different inner graph changes the outer digest.
        let mut inner_c = chain();
        inner_c.add_node(Node::filter("Calibrator"));
        let mut outer_c = Graph::new();
        outer_c.add_node(Node::subgraph("stage", inner_c));
        assert_ne!(
            ArchitectureFingerprint::of(&outer_a).unwrap().digest,
            ArchitectureFingerprint::of(&outer_c).unwrap().digest
        );
    }

    #[test]
    fn similarity_is_bounded_symmetric_and_ordered() {
        let base = ArchitectureFingerprint::of(&chain()).unwrap();

        let mut one_swap = chain();
        one_swap.nodes[2].kind = NodeKind::Filter {
            filter_name: "RandomForest".into(),
        };
        let one_swap = ArchitectureFingerprint::of(&one_swap).unwrap();

        let unrelated = ArchitectureFingerprint::of(&linear_pipeline(vec![
            Node::filter("Tokenizer"),
            Node::filter("Transformer"),
        ]))
        .unwrap();

        for (a, b) in [
            (&base, &base),
            (&base, &one_swap),
            (&base, &unrelated),
            (&one_swap, &unrelated),
        ] {
            let s = structural_similarity(a, b);
            assert!((0.0..=1.0).contains(&s), "out of bounds: {s}");
            assert_eq!(s, structural_similarity(b, a), "not symmetric");
        }
        assert_eq!(structural_similarity(&base, &base), 1.0);
        assert!(structural_similarity(&base, &one_swap) > structural_similarity(&base, &unrelated));
        assert_eq!(structural_similarity(&base, &unrelated), 0.0);
    }

    #[test]
    fn similarity_counts_duplicates() {
        let one =
            ArchitectureFingerprint::of(&linear_pipeline(vec![Node::filter("Dense")])).unwrap();
        let three = ArchitectureFingerprint::of(&linear_pipeline(vec![
            Node::filter_with_id("d1", "Dense"),
            Node::filter_with_id("d2", "Dense"),
            Node::filter_with_id("d3", "Dense"),
        ]))
        .unwrap();
        let s = structural_similarity(&one, &three);
        assert!(
            s > 0.0 && s < 1.0,
            "stacking layers must move the score: {s}"
        );
    }

    #[test]
    fn empty_graphs_are_identical_to_each_other() {
        let empty = ArchitectureFingerprint::of(&Graph::new()).unwrap();
        assert_eq!(empty.n_nodes, 0);
        assert_eq!(structural_similarity(&empty, &empty), 1.0);
        let non_empty = ArchitectureFingerprint::of(&chain()).unwrap();
        assert_eq!(structural_similarity(&empty, &non_empty), 0.0);
    }

    #[test]
    fn control_edges_are_distinct_from_data_edges() {
        let mut data = Graph::new();
        data.add_node(Node::filter("A"));
        data.add_node(Node::filter("B"));
        data.add_edge(Edge::data("e", "A", "B"));
        let mut control = Graph::new();
        control.add_node(Node::filter("A"));
        control.add_node(Node::filter("B"));
        control.add_edge(Edge::control("e", "A", "B"));
        let data = ArchitectureFingerprint::of(&data).unwrap();
        let control = ArchitectureFingerprint::of(&control).unwrap();
        assert_ne!(data.digest, control.digest);
        assert_ne!(data.edge_tokens, control.edge_tokens);
    }

    #[test]
    fn fingerprint_roundtrips_and_tolerates_missing_node_config() {
        let fp = ArchitectureFingerprint::of(&chain())
            .unwrap()
            .with_node_config(BTreeMap::from([("a".to_string(), "deadbeef".to_string())]));
        let json = serde_json::to_string(&fp).unwrap();
        let back: ArchitectureFingerprint = serde_json::from_str(&json).unwrap();
        assert_eq!(back, fp);
        assert_eq!(back.node_config["a"], "deadbeef");

        let legacy = serde_json::json!({
            "digest": "abc",
            "node_tokens": ["filter:A"],
            "edge_tokens": [],
            "n_nodes": 1,
            "n_edges": 0,
        });
        let back: ArchitectureFingerprint = serde_json::from_value(legacy).unwrap();
        assert!(back.node_config.is_empty());
    }

    #[test]
    fn pipeline_summary_reads_as_a_chain_or_reports_shape() {
        assert_eq!(pipeline_summary(&Graph::new()), "empty graph");
        assert_eq!(
            pipeline_summary(&chain()),
            "a(StandardScaler) → b(PCA) → c(SVM)"
        );

        let mut forked = chain();
        forked.add_node(Node::filter("Aux"));
        forked.add_edge(Edge::data("fork", "a", "Aux"));
        let summary = pipeline_summary(&forked);
        assert!(summary.contains("4 nodes, 3 edges"), "{summary}");

        let mut wide = Graph::new();
        for i in 0..12 {
            wide.add_node(Node::filter_with_id(format!("n{i}"), "Dense"));
        }
        let summary = pipeline_summary(&wide);
        assert!(summary.contains("+4 more"), "{summary}");
    }

    #[test]
    fn pipeline_summary_survives_a_cycle() {
        let mut cyclic = Graph::new();
        cyclic.add_node(Node::filter("A"));
        cyclic.add_node(Node::filter("B"));
        cyclic.add_edge(Edge::data("e1", "A", "B"));
        cyclic.add_edge(Edge::data("e2", "B", "A"));
        assert!(cyclic.topological_sort().is_err());
        let summary = pipeline_summary(&cyclic);
        assert!(summary.contains('A') && summary.contains('B'), "{summary}");
    }
}
