//! Computational graph — DAG of filter nodes connected by edges.
//!
//! The graph is the user-facing representation of a pipeline topology.
//! It gets compiled into an [`ExecutionPlan`] by the compiler.

use crate::error::{Result, SomaError};
use crate::strategy::TrainingStrategy;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Unique identifier for a node in a graph.
///
/// Currently a type alias. Will be promoted to a newtype in a future version
/// for stronger type safety (tracked in architecture-review.md).
pub type NodeId = String;

/// Unique identifier for an edge in a graph.
pub type EdgeId = String;

/// What kind of computation a node represents.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum NodeKind {
    /// A single filter (the common case).
    Filter { filter_name: String },
    /// A nested sub-graph (compiled recursively).
    SubGraph { graph: Box<Graph> },
    /// A loop node — body is the sub-graph of successors.
    Loop { max_iterations: Option<usize> },
    /// A branch/conditional node — arms determined by control edges.
    Branch,
}

/// A node in the computational graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub label: String,
    pub kind: NodeKind,
}

impl Node {
    /// Create a filter node (backward-compatible with old 3-arg constructor).
    pub fn new(
        id: impl Into<String>,
        label: impl Into<String>,
        filter_name: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            kind: NodeKind::Filter {
                filter_name: filter_name.into(),
            },
        }
    }

    /// Create a filter node with explicit id and filter_name.
    pub fn filter_with_id(id: impl Into<String>, filter_name: impl Into<String>) -> Self {
        let id = id.into();
        Self {
            label: id.clone(),
            id,
            kind: NodeKind::Filter {
                filter_name: filter_name.into(),
            },
        }
    }

    /// Create a filter node where id defaults to filter_name.
    pub fn filter(filter_name: impl Into<String>) -> Self {
        let name = filter_name.into();
        Self {
            id: name.clone(),
            label: name.clone(),
            kind: NodeKind::Filter { filter_name: name },
        }
    }

    /// Create a sub-graph node.
    pub fn subgraph(id: impl Into<String>, graph: Graph) -> Self {
        let id = id.into();
        Self {
            id: id.clone(),
            label: id,
            kind: NodeKind::SubGraph {
                graph: Box::new(graph),
            },
        }
    }

    /// Create a loop node.
    pub fn loop_node(id: impl Into<String>, max_iterations: Option<usize>) -> Self {
        let id = id.into();
        Self {
            id: id.clone(),
            label: id,
            kind: NodeKind::Loop { max_iterations },
        }
    }

    /// Create a branch/conditional node.
    pub fn branch(id: impl Into<String>) -> Self {
        let id = id.into();
        Self {
            id: id.clone(),
            label: id,
            kind: NodeKind::Branch,
        }
    }

    /// Get the filter name if this is a Filter node.
    pub fn filter_name(&self) -> Option<&str> {
        match &self.kind {
            NodeKind::Filter { filter_name } => Some(filter_name),
            _ => None,
        }
    }
}

/// Type of connection between nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeKind {
    /// Normal data flow: output of source becomes input of target.
    Data,
    /// Control flow edge (for conditional/loop logic).
    Control,
}

/// A directed edge connecting two nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub id: EdgeId,
    pub source: NodeId,
    pub target: NodeId,
    pub kind: EdgeKind,
    pub label: Option<String>,
}

impl Edge {
    pub fn data(
        id: impl Into<String>,
        source: impl Into<String>,
        target: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            source: source.into(),
            target: target.into(),
            kind: EdgeKind::Data,
            label: None,
        }
    }

    pub fn control(
        id: impl Into<String>,
        source: impl Into<String>,
        target: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            source: source.into(),
            target: target.into(),
            kind: EdgeKind::Control,
            label: None,
        }
    }
}

/// A directed graph of computational nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    /// Training strategy for distributed execution.
    /// Inherited by subgraphs unless overridden.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training_strategy: Option<TrainingStrategy>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            training_strategy: None,
        }
    }

    /// Set the training strategy for this graph.
    pub fn with_strategy(mut self, strategy: TrainingStrategy) -> Self {
        self.training_strategy = Some(strategy);
        self
    }

    /// Set the training strategy (mutable).
    pub fn set_strategy(&mut self, strategy: TrainingStrategy) {
        self.training_strategy = Some(strategy);
    }

    /// Get the effective training strategy (defaults to Local).
    pub fn effective_strategy(&self) -> &TrainingStrategy {
        static LOCAL: TrainingStrategy = TrainingStrategy::Local;
        self.training_strategy.as_ref().unwrap_or(&LOCAL)
    }

    pub fn add_node(&mut self, node: Node) {
        self.nodes.push(node);
    }

    /// Add a filter node using the filter name as the node id.
    /// If a node with that name already exists, appends a suffix.
    pub fn add_filter(&mut self, filter_name: impl Into<String>) -> &str {
        let name = filter_name.into();
        let id = if self.nodes.iter().any(|n| n.id == name) {
            let mut i = 2;
            loop {
                let candidate = format!("{name}_{i}");
                if !self.nodes.iter().any(|n| n.id == candidate) {
                    break candidate;
                }
                i += 1;
            }
        } else {
            name.clone()
        };
        self.nodes.push(Node::filter_with_id(&id, &name));
        &self.nodes.last().unwrap().id
    }

    pub fn add_edge(&mut self, edge: Edge) {
        self.edges.push(edge);
    }

    /// Connect two nodes with a data edge (auto-generates edge id).
    pub fn connect(&mut self, source: impl Into<String>, target: impl Into<String>) {
        let id = format!("e_{}", self.edges.len());
        self.edges.push(Edge::data(id, source, target));
    }

    /// Get a node by its ID.
    pub fn node(&self, id: &str) -> Option<&Node> {
        self.nodes.iter().find(|n| n.id == id)
    }

    /// Get all node IDs.
    pub fn node_ids(&self) -> Vec<&str> {
        self.nodes.iter().map(|n| n.id.as_str()).collect()
    }

    /// Get predecessors of a node (nodes with edges pointing to it).
    pub fn predecessors(&self, node_id: &str) -> Vec<&str> {
        self.edges
            .iter()
            .filter(|e| e.target == node_id)
            .map(|e| e.source.as_str())
            .collect()
    }

    /// Get successors of a node (nodes it points to).
    pub fn successors(&self, node_id: &str) -> Vec<&str> {
        self.edges
            .iter()
            .filter(|e| e.source == node_id)
            .map(|e| e.target.as_str())
            .collect()
    }

    /// Find root nodes (no incoming edges).
    pub fn roots(&self) -> Vec<&str> {
        let has_incoming: HashSet<&str> = self.edges.iter().map(|e| e.target.as_str()).collect();
        self.nodes
            .iter()
            .filter(|n| !has_incoming.contains(n.id.as_str()))
            .map(|n| n.id.as_str())
            .collect()
    }

    /// Find leaf nodes (no outgoing edges).
    pub fn leaves(&self) -> Vec<&str> {
        let has_outgoing: HashSet<&str> = self.edges.iter().map(|e| e.source.as_str()).collect();
        self.nodes
            .iter()
            .filter(|n| !has_outgoing.contains(n.id.as_str()))
            .map(|n| n.id.as_str())
            .collect()
    }

    /// Compute in-degree for each node.
    fn in_degrees(&self) -> HashMap<&str, usize> {
        let mut degrees: HashMap<&str, usize> =
            self.nodes.iter().map(|n| (n.id.as_str(), 0)).collect();
        for edge in &self.edges {
            *degrees.entry(edge.target.as_str()).or_insert(0) += 1;
        }
        degrees
    }

    /// Topological sort using Kahn's algorithm.
    /// Returns Err if the graph contains a cycle.
    pub fn topological_sort(&self) -> Result<Vec<&str>> {
        let mut in_deg = self.in_degrees();
        let mut queue: Vec<&str> = in_deg
            .iter()
            .filter(|(_, deg)| **deg == 0)
            .map(|(&id, _)| id)
            .collect();
        queue.sort(); // deterministic order

        let mut sorted = Vec::with_capacity(self.nodes.len());

        while let Some(node) = queue.pop() {
            sorted.push(node);
            let mut next = Vec::new();
            for succ in self.successors(node) {
                if let Some(deg) = in_deg.get_mut(succ) {
                    *deg -= 1;
                    if *deg == 0 {
                        next.push(succ);
                    }
                }
            }
            next.sort();
            // Insert at beginning so we process in deterministic order
            for n in next.into_iter().rev() {
                queue.push(n);
            }
        }

        if sorted.len() != self.nodes.len() {
            return Err(SomaError::CycleDetected);
        }

        Ok(sorted)
    }

    /// Validate the graph structure (recursively validates sub-graphs).
    pub fn validate(&self) -> Result<()> {
        // Check for duplicate node IDs
        let mut seen = HashSet::new();
        for node in &self.nodes {
            if !seen.insert(&node.id) {
                return Err(SomaError::Compilation(format!(
                    "duplicate node id: `{}`",
                    node.id
                )));
            }
        }

        // Check that all edge endpoints reference existing nodes
        let node_ids: HashSet<&str> = self.nodes.iter().map(|n| n.id.as_str()).collect();
        for edge in &self.edges {
            if !node_ids.contains(edge.source.as_str()) {
                return Err(SomaError::NodeNotFound(edge.source.clone()));
            }
            if !node_ids.contains(edge.target.as_str()) {
                return Err(SomaError::NodeNotFound(edge.target.clone()));
            }
        }

        // Check for cycles
        self.topological_sort()?;

        // Recursively validate sub-graphs
        for node in &self.nodes {
            if let NodeKind::SubGraph { graph } = &node.kind {
                graph.validate()?;
            }
        }

        Ok(())
    }
}

// ── Visualization ──

impl Graph {
    /// Render as a Mermaid diagram.
    ///
    /// ```text
    /// graph LR
    ///     scaler[scaler]
    ///     model[model]
    ///     scaler --> model
    /// ```
    pub fn to_mermaid(&self) -> String {
        use std::fmt::Write;
        let mut out = String::from("graph LR\n");
        for node in &self.nodes {
            let shape = match &node.kind {
                NodeKind::Filter { .. } => format!("    {}[{}]", node.id, node.label),
                NodeKind::SubGraph { .. } => format!("    {}[[{}]]", node.id, node.label),
                NodeKind::Loop { max_iterations } => {
                    let label = match max_iterations {
                        Some(n) => format!("{} (max {})", node.label, n),
                        None => node.label.clone(),
                    };
                    format!("    {}(({}))", node.id, label)
                }
                NodeKind::Branch => format!("    {}{{{{{}}}}}", node.id, node.label),
            };
            let _ = writeln!(out, "{shape}");
        }
        for edge in &self.edges {
            let arrow = match edge.kind {
                EdgeKind::Data => "-->",
                EdgeKind::Control => "-.->",
            };
            if let Some(label) = &edge.label {
                let _ = writeln!(
                    out,
                    "    {} {}|{}| {}",
                    edge.source, arrow, label, edge.target
                );
            } else {
                let _ = writeln!(out, "    {} {} {}", edge.source, arrow, edge.target);
            }
        }
        out
    }

    /// Render as Graphviz DOT format.
    pub fn to_graphviz(&self) -> String {
        use std::fmt::Write;
        let mut out = String::from("digraph G {\n    rankdir=LR;\n");
        for node in &self.nodes {
            let shape = match &node.kind {
                NodeKind::Filter { .. } => "box",
                NodeKind::SubGraph { .. } => "doubleoctagon",
                NodeKind::Loop { .. } => "ellipse",
                NodeKind::Branch => "diamond",
            };
            let _ = writeln!(
                out,
                "    \"{}\" [label=\"{}\" shape={}];",
                node.id, node.label, shape
            );
        }
        for edge in &self.edges {
            let style = match edge.kind {
                EdgeKind::Data => "",
                EdgeKind::Control => " [style=dashed]",
            };
            let label = edge
                .label
                .as_ref()
                .map(|l| format!(" [label=\"{l}\"]"))
                .unwrap_or_default();
            let attrs = if style.is_empty() && label.is_empty() {
                String::new()
            } else if label.is_empty() {
                style.to_string()
            } else {
                label
            };
            let _ = writeln!(
                out,
                "    \"{}\" -> \"{}\"{};",
                edge.source, edge.target, attrs
            );
        }
        out.push_str("}\n");
        out
    }

    /// Render as an ASCII text tree for terminal display.
    pub fn to_text(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        let sorted = self.topological_sort().unwrap_or_default();
        let total_nodes = self.nodes.len();
        let total_edges = self.edges.len();
        let _ = writeln!(out, "Graph ({total_nodes} nodes, {total_edges} edges)");

        for (i, node_id) in sorted.iter().enumerate() {
            let node = match self.node(node_id) {
                Some(n) => n,
                None => continue,
            };
            let is_last = i == sorted.len() - 1;
            let prefix = if is_last { "└── " } else { "├── " };
            let kind_tag = match &node.kind {
                NodeKind::Filter { filter_name } => {
                    if filter_name == &node.id {
                        String::new()
                    } else {
                        format!(" ({})", filter_name)
                    }
                }
                NodeKind::SubGraph { graph } => {
                    format!(" [subgraph: {} nodes]", graph.nodes.len())
                }
                NodeKind::Loop { max_iterations } => match max_iterations {
                    Some(n) => format!(" [loop max={n}]"),
                    None => " [loop]".into(),
                },
                NodeKind::Branch => " [branch]".into(),
            };
            let preds = self.predecessors(node_id);
            let pred_info = if preds.is_empty() {
                String::new()
            } else {
                format!(" ← {}", preds.join(", "))
            };
            let _ = writeln!(out, "{prefix}{}{kind_tag}{pred_info}", node.id);
        }
        out
    }
}

impl std::fmt::Display for Graph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_text())
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for constructing linear pipelines easily.
pub fn linear_pipeline(nodes: Vec<Node>) -> Graph {
    let mut graph = Graph::new();
    for (i, node) in nodes.iter().enumerate() {
        graph.add_node(node.clone());
        if i > 0 {
            graph.add_edge(Edge::data(format!("e_{}", i), &nodes[i - 1].id, &node.id));
        }
    }
    graph
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_linear_graph() -> Graph {
        linear_pipeline(vec![
            Node::new("a", "Scaler", "StandardScaler"),
            Node::new("b", "PCA", "PCA"),
            Node::new("c", "SVM", "SVM"),
        ])
    }

    #[test]
    fn linear_pipeline_structure() {
        let g = sample_linear_graph();
        assert_eq!(g.nodes.len(), 3);
        assert_eq!(g.edges.len(), 2);
    }

    #[test]
    fn roots_and_leaves() {
        let g = sample_linear_graph();
        assert_eq!(g.roots(), vec!["a"]);
        assert_eq!(g.leaves(), vec!["c"]);
    }

    #[test]
    fn predecessors_and_successors() {
        let g = sample_linear_graph();
        assert!(g.predecessors("a").is_empty());
        assert_eq!(g.predecessors("b"), vec!["a"]);
        assert_eq!(g.successors("a"), vec!["b"]);
        assert_eq!(g.successors("b"), vec!["c"]);
        assert!(g.successors("c").is_empty());
    }

    #[test]
    fn topological_sort_linear() {
        let g = sample_linear_graph();
        let sorted = g.topological_sort().unwrap();
        assert_eq!(sorted, vec!["a", "b", "c"]);
    }

    #[test]
    fn topological_sort_parallel() {
        let mut g = Graph::new();
        g.add_node(Node::new("root", "Root", "Input"));
        g.add_node(Node::new("b1", "Branch1", "F1"));
        g.add_node(Node::new("b2", "Branch2", "F2"));
        g.add_node(Node::new("merge", "Merge", "Merge"));
        g.add_edge(Edge::data("e1", "root", "b1"));
        g.add_edge(Edge::data("e2", "root", "b2"));
        g.add_edge(Edge::data("e3", "b1", "merge"));
        g.add_edge(Edge::data("e4", "b2", "merge"));

        let sorted = g.topological_sort().unwrap();
        // root must be first, merge must be last
        assert_eq!(sorted[0], "root");
        assert_eq!(sorted[3], "merge");
        // b1 and b2 can be in any order between root and merge
        let middle: HashSet<&str> = sorted[1..3].iter().copied().collect();
        assert!(middle.contains("b1"));
        assert!(middle.contains("b2"));
    }

    #[test]
    fn topological_sort_detects_cycle() {
        let mut g = Graph::new();
        g.add_node(Node::new("a", "A", "F"));
        g.add_node(Node::new("b", "B", "F"));
        g.add_edge(Edge::data("e1", "a", "b"));
        g.add_edge(Edge::data("e2", "b", "a")); // cycle!

        let result = g.topological_sort();
        assert!(matches!(result, Err(SomaError::CycleDetected)));
    }

    #[test]
    fn validate_accepts_valid_graph() {
        let g = sample_linear_graph();
        assert!(g.validate().is_ok());
    }

    #[test]
    fn validate_rejects_duplicate_ids() {
        let mut g = Graph::new();
        g.add_node(Node::new("a", "A", "F"));
        g.add_node(Node::new("a", "A2", "F"));
        assert!(matches!(g.validate(), Err(SomaError::Compilation(_))));
    }

    #[test]
    fn validate_rejects_missing_edge_target() {
        let mut g = Graph::new();
        g.add_node(Node::new("a", "A", "F"));
        g.add_edge(Edge::data("e1", "a", "nonexistent"));
        assert!(matches!(g.validate(), Err(SomaError::NodeNotFound(_))));
    }

    #[test]
    fn graph_serde_roundtrip() {
        let g = sample_linear_graph();
        let json = serde_json::to_string(&g).unwrap();
        let deserialized: Graph = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.nodes.len(), 3);
        assert_eq!(deserialized.edges.len(), 2);
    }

    #[test]
    fn empty_graph_is_valid() {
        let g = Graph::new();
        assert!(g.validate().is_ok());
        assert!(g.topological_sort().unwrap().is_empty());
    }

    #[test]
    fn single_node_graph() {
        let mut g = Graph::new();
        g.add_node(Node::new("solo", "Solo", "F"));
        assert_eq!(g.roots(), vec!["solo"]);
        assert_eq!(g.leaves(), vec!["solo"]);
        assert_eq!(g.topological_sort().unwrap(), vec!["solo"]);
    }

    // ── NodeKind tests ──

    #[test]
    fn node_filter_shorthand() {
        let n = Node::filter("StandardScaler");
        assert_eq!(n.id, "StandardScaler");
        assert_eq!(n.filter_name(), Some("StandardScaler"));
    }

    #[test]
    fn node_filter_with_id() {
        let n = Node::filter_with_id("my_scaler", "StandardScaler");
        assert_eq!(n.id, "my_scaler");
        assert_eq!(n.filter_name(), Some("StandardScaler"));
    }

    #[test]
    fn graph_add_filter_auto_names() {
        let mut g = Graph::new();
        g.add_filter("Scaler");
        g.add_filter("PCA");
        g.connect("Scaler", "PCA");

        assert!(g.validate().is_ok());
        assert_eq!(g.nodes.len(), 2);
        assert_eq!(g.nodes[0].id, "Scaler");
        assert_eq!(g.nodes[1].id, "PCA");
    }

    #[test]
    fn graph_add_filter_deduplicates() {
        let mut g = Graph::new();
        g.add_filter("Scaler");
        g.add_filter("Scaler"); // duplicate name → gets suffix

        assert_eq!(g.nodes.len(), 2);
        assert_eq!(g.nodes[0].id, "Scaler");
        assert_eq!(g.nodes[1].id, "Scaler_2");
    }

    #[test]
    fn subgraph_node() {
        let inner = linear_pipeline(vec![Node::new("a", "A", "F"), Node::new("b", "B", "F")]);

        let mut outer = Graph::new();
        outer.add_node(Node::new("input", "Input", "Input"));
        outer.add_node(Node::subgraph("pipeline", inner));
        outer.add_node(Node::new("output", "Output", "Output"));
        outer.add_edge(Edge::data("e1", "input", "pipeline"));
        outer.add_edge(Edge::data("e2", "pipeline", "output"));

        assert!(outer.validate().is_ok());
        assert_eq!(outer.nodes.len(), 3);

        // SubGraph node has no filter_name
        assert!(outer.node("pipeline").unwrap().filter_name().is_none());
    }

    #[test]
    fn loop_and_branch_nodes() {
        let mut g = Graph::new();
        g.add_node(Node::loop_node("train_loop", Some(100)));
        g.add_node(Node::branch("check_convergence"));
        g.add_edge(Edge::data("e1", "train_loop", "check_convergence"));

        assert!(g.validate().is_ok());
        assert!(matches!(
            g.node("train_loop").unwrap().kind,
            NodeKind::Loop {
                max_iterations: Some(100)
            }
        ));
        assert!(matches!(
            g.node("check_convergence").unwrap().kind,
            NodeKind::Branch
        ));
    }

    // ── Visualization tests ──

    #[test]
    fn to_mermaid_linear() {
        let g = sample_linear_graph();
        let m = g.to_mermaid();
        assert!(m.starts_with("graph LR"));
        assert!(m.contains("a[Scaler]"));
        assert!(m.contains("b[PCA]"));
        assert!(m.contains("c[SVM]"));
        assert!(m.contains("a --> b"));
        assert!(m.contains("b --> c"));
    }

    #[test]
    fn to_mermaid_branch_and_loop() {
        let mut g = Graph::new();
        g.add_node(Node::loop_node("train", Some(100)));
        g.add_node(Node::branch("check"));
        g.add_edge(Edge::data("e1", "train", "check"));

        let m = g.to_mermaid();
        assert!(m.contains("train((train (max 100)))"));
        assert!(m.contains("check{"));
        assert!(m.contains("train --> check"));
    }

    #[test]
    fn to_graphviz_output() {
        let g = sample_linear_graph();
        let dot = g.to_graphviz();
        assert!(dot.starts_with("digraph G {"));
        assert!(dot.contains("rankdir=LR"));
        assert!(dot.contains("\"a\" [label=\"Scaler\" shape=box]"));
        assert!(dot.contains("\"a\" -> \"b\""));
        assert!(dot.ends_with("}\n"));
    }

    #[test]
    fn to_text_output() {
        let g = sample_linear_graph();
        let text = g.to_text();
        assert!(text.contains("Graph (3 nodes, 2 edges)"));
        assert!(text.contains("a"));
        assert!(text.contains("b"));
        assert!(text.contains("c"));
        assert!(text.contains("← a"));
    }

    #[test]
    fn display_trait() {
        let g = sample_linear_graph();
        let s = format!("{g}");
        assert!(s.contains("Graph (3 nodes"));
    }

    #[test]
    fn node_kind_serde_roundtrip() {
        let inner = linear_pipeline(vec![Node::new("x", "X", "F")]);
        let nodes = vec![
            Node::filter("Scaler"),
            Node::subgraph("sub", inner),
            Node::loop_node("loop", Some(50)),
            Node::branch("cond"),
        ];

        for node in &nodes {
            let json = serde_json::to_string(node).unwrap();
            let parsed: Node = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed.id, node.id);
        }
    }
}
