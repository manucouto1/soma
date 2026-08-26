//! The CU1 questionnaire, answered.

use somatize_core::{Graph, GraphError, NodeId};

/// `a → b → c`, the linear pipeline almost everything else grows from.
fn linear() -> Graph {
    let mut g = Graph::new();
    for id in ["a", "b", "c"] {
        g.add_node(id).unwrap();
    }
    g.add_edge("a", "b").unwrap();
    g.add_edge("b", "c").unwrap();
    g
}

fn ids(nodes: Vec<&NodeId>) -> Vec<&str> {
    nodes.into_iter().map(NodeId::as_str).collect()
}

#[test]
fn an_empty_graph_is_valid() {
    let g = Graph::new();
    assert_eq!(g.len(), 0);
    assert!(g.is_empty());
    assert!(g.topological_sort().is_empty());
}

#[test]
fn a_single_node_graph_is_valid() {
    let mut g = Graph::new();
    g.add_node("alone").unwrap();
    assert_eq!(ids(g.roots()), ["alone"]);
    assert_eq!(ids(g.leaves()), ["alone"]);
}

#[test]
fn the_nodes_keep_their_insertion_order() {
    assert_eq!(
        linear()
            .nodes()
            .iter()
            .map(NodeId::as_str)
            .collect::<Vec<_>>(),
        ["a", "b", "c"]
    );
}

#[test]
fn a_linear_pipeline_has_the_structure_it_says() {
    let g = linear();
    assert_eq!(g.len(), 3);
    assert_eq!(g.edges().len(), 2);
    assert_eq!(ids(g.roots()), ["a"]);
    assert_eq!(ids(g.leaves()), ["c"]);
}

#[test]
fn free_id_suffixes_until_it_finds_a_gap() {
    let mut g = Graph::new();
    assert_eq!(g.free_id("clean").as_str(), "clean");
    g.add_node("clean").unwrap();
    assert_eq!(g.free_id("clean").as_str(), "clean_2");
    g.add_node("clean_2").unwrap();
    assert_eq!(g.free_id("clean").as_str(), "clean_3");
}

#[test]
fn two_nodes_cannot_share_a_name() {
    let mut g = Graph::new();
    g.add_node("a").unwrap();
    assert_eq!(
        g.add_node("a").unwrap_err(),
        GraphError::DuplicateNode("a".into())
    );
    assert_eq!(g.len(), 1);
}

#[test]
fn an_edge_needs_both_its_ends_to_exist() {
    let mut g = Graph::new();
    g.add_node("a").unwrap();
    assert_eq!(
        g.add_edge("a", "ghost").unwrap_err(),
        GraphError::UnknownNode("ghost".into())
    );
    assert_eq!(
        g.add_edge("ghost", "a").unwrap_err(),
        GraphError::UnknownNode("ghost".into())
    );
    assert!(g.edges().is_empty());
}

#[test]
fn the_same_edge_is_not_added_twice() {
    let mut g = linear();
    assert_eq!(
        g.add_edge("a", "b").unwrap_err(),
        GraphError::DuplicateEdge {
            from: "a".into(),
            to: "b".into()
        }
    );
    assert_eq!(g.edges().len(), 2);
}

#[test]
fn a_cycle_is_rejected_when_added_not_when_walked() {
    let mut g = linear();
    assert_eq!(
        g.add_edge("c", "a").unwrap_err(),
        GraphError::WouldCycle {
            from: "c".into(),
            to: "a".into()
        }
    );
    assert_eq!(g.edges().len(), 2);
}

#[test]
fn a_node_does_not_connect_to_itself() {
    let mut g = linear();
    assert!(matches!(
        g.add_edge("a", "a").unwrap_err(),
        GraphError::WouldCycle { .. }
    ));
}

#[test]
fn predecessors_and_successors() {
    let g = linear();
    assert_eq!(ids(g.predecessors(&"b".into())), ["a"]);
    assert_eq!(ids(g.successors(&"b".into())), ["c"]);
    assert!(g.predecessors(&"a".into()).is_empty());
    assert!(g.successors(&"c".into()).is_empty());
}

#[test]
fn roots_and_leaves_with_branches() {
    let mut g = Graph::new();
    for id in ["source_1", "source_2", "join", "output"] {
        g.add_node(id).unwrap();
    }
    g.add_edge("source_1", "join").unwrap();
    g.add_edge("source_2", "join").unwrap();
    g.add_edge("join", "output").unwrap();

    assert_eq!(ids(g.roots()), ["source_1", "source_2"]);
    assert_eq!(ids(g.leaves()), ["output"]);
    assert_eq!(
        ids(g.predecessors(&"join".into())),
        ["source_1", "source_2"]
    );
}

#[test]
fn topological_order_of_a_chain() {
    assert_eq!(ids(linear().topological_sort()), ["a", "b", "c"]);
}

#[test]
fn topological_order_with_parallel_branches() {
    let mut g = Graph::new();
    for id in ["input", "left", "right", "join"] {
        g.add_node(id).unwrap();
    }
    g.add_edge("input", "left").unwrap();
    g.add_edge("input", "right").unwrap();
    g.add_edge("left", "join").unwrap();
    g.add_edge("right", "join").unwrap();

    let order = ids(g.topological_sort());
    assert_eq!(order.len(), 4);
    assert_eq!(order[0], "input");
    assert_eq!(order[3], "join");
}

#[test]
fn the_topological_order_is_deterministic() {
    let g = linear();
    assert_eq!(ids(g.topological_sort()), ids(g.topological_sort()));
}

#[test]
fn loose_nodes_also_show_up_in_the_order() {
    let mut g = linear();
    g.add_node("loose").unwrap();
    assert_eq!(ids(g.topological_sort()), ["a", "b", "c", "loose"]);
}
