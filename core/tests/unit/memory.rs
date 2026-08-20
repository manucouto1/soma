//! What is remembered about each node, and what may honestly be kept.

use crate::doubles::Add;
use soma_next_core::{Graph, Memory, MemoryError, cacheable, node};

/// `a >> b >> c`, which is the shape the prefix rule is about.
fn chain() -> Graph {
    let (graph, _, _, _) = (node("a", Add(1.0)) >> node("b", Add(1.0)) >> node("c", Add(1.0)))
        .somatize()
        .unwrap();
    graph
}

/// Everything in this graph named, and everything frozen but the ones said to
/// be still training. With none, it is the state a legitimate cache starts from.
fn settled_but(graph: &Graph, moving: &[&str]) -> Memory {
    let mut memory = Memory::new();
    for id in graph.nodes() {
        memory.identify(id.clone(), "Add");
        if !moving.contains(&id.as_str()) {
            memory.freeze(id.clone(), None);
        }
    }
    memory
}

fn settled(graph: &Graph) -> Memory {
    settled_but(graph, &[])
}

#[test]
fn nothing_is_remembered_until_it_is_said() {
    let memory = Memory::new();
    let a = "a".into();

    assert!(memory.is_empty());
    assert_eq!(memory.len(), 0);
    assert!(!memory.is_frozen(&a));
    assert!(!memory.is_cached(&a));
    assert_eq!(memory.identity_of(&a), None);
    assert_eq!(memory.state_of(&a), None);
    assert_eq!(memory.salt_of(&a), None);
    assert_eq!(memory.fingerprint_of(&a), None);
}

#[test]
fn what_implements_a_node_is_what_names_it() {
    let mut memory = Memory::new();
    assert_eq!(memory.identify("a", "Encoder"), None);
    assert_eq!(memory.identity_of(&"a".into()), Some("Encoder"));
    // Saying it again says what it was called before, so a rename is visible.
    assert_eq!(
        memory.identify("a", "Embedder"),
        Some("Encoder".to_string())
    );
}

#[test]
fn frozen_without_a_state_is_still_frozen() {
    // A tokenizer has nothing to settle and is no less settled for it.
    let mut memory = Memory::new();
    memory.freeze("a", None);

    assert!(memory.is_frozen(&"a".into()));
    assert_eq!(memory.state_of(&"a".into()), None);
}

#[test]
fn freezing_twice_is_how_the_digest_arrives() {
    // `.frozen()` declares it with no digest; whoever knows how to hash the
    // weights says it again with one. Both are the same statement.
    let mut memory = Memory::new();
    memory.freeze("a", None);
    memory.freeze("a", Some("sha256:w".to_string()));

    assert!(memory.is_frozen(&"a".into()));
    assert_eq!(memory.state_of(&"a".into()), Some("sha256:w"));
}

#[test]
fn caching_carries_the_salt_it_was_asked_with() {
    let mut memory = Memory::new();
    memory.cache("a", None);
    memory.cache("b", Some("a100-fp16".to_string()));

    assert!(memory.is_cached(&"a".into()));
    assert_eq!(memory.salt_of(&"a".into()), None);
    assert_eq!(memory.salt_of(&"b".into()), Some("a100-fp16"));
}

#[test]
fn the_fingerprint_is_remembered_and_nothing_else_depends_on_it() {
    // It is metadata: it is compared on a hit and it is not in the key, so a
    // graph without one is perfectly cacheable.
    let graph = chain();
    let mut memory = settled(&graph);
    memory.cache("c", None);
    assert_eq!(cacheable(&graph, &memory), Ok(()));

    memory.written_as("c", "a1b2c3d4");
    assert_eq!(memory.fingerprint_of(&"c".into()), Some("a1b2c3d4"));
    assert_eq!(cacheable(&graph, &memory), Ok(()));
}

#[test]
fn it_counts_the_nodes_anything_was_said_about() {
    let mut memory = Memory::new();
    memory.freeze("a", None);
    memory.cache("a", None);
    memory.identify("a", "Add");
    assert_eq!(memory.len(), 1, "four things about one node is one node");

    memory.written_as("b", "a1b2c3d4");
    assert_eq!(memory.len(), 2);
}

#[test]
fn a_graph_that_keeps_nothing_is_always_fine() {
    let graph = chain();
    assert_eq!(cacheable(&graph, &Memory::new()), Ok(()));
}

#[test]
fn a_settled_prefix_may_be_kept() {
    let graph = chain();
    let mut memory = settled(&graph);
    memory.cache("c", None);

    assert_eq!(cacheable(&graph, &memory), Ok(()));
}

#[test]
fn a_node_that_still_changes_may_not_keep_its_own_output() {
    let graph = chain();
    let mut memory = Memory::new();
    memory.identify("c", "Add");
    memory.cache("c", None);

    assert_eq!(
        cacheable(&graph, &memory),
        Err(MemoryError::Unsettled {
            cached: "c".into(),
            moving: "c".into(),
        })
    );
}

#[test]
fn a_node_above_that_still_changes_stops_it_too() {
    // The whole point of the rule being about **prefixes**: freezing the node
    // and leaving what feeds it training would restore a value that is a leaf,
    // and everything above it would quietly stop learning.
    let graph = chain();
    let mut memory = settled(&graph);
    memory.cache("c", None);
    assert_eq!(cacheable(&graph, &memory), Ok(()), "all of it settled");

    // The same, with the far end of the chain left training.
    let mut thawed = settled_but(&graph, &["a"]);
    thawed.cache("c", None);

    assert_eq!(
        cacheable(&graph, &thawed),
        Err(MemoryError::Unsettled {
            cached: "c".into(),
            moving: "a".into(),
        })
    );
}

#[test]
fn the_nearest_one_is_the_one_named() {
    // Two things wrong above it: what the message points at is the one closest
    // to the problem, not the furthest.
    let graph = chain();
    let mut memory = settled_but(&graph, &["a", "b"]);
    memory.cache("c", None);

    assert_eq!(
        cacheable(&graph, &memory),
        Err(MemoryError::Unsettled {
            cached: "c".into(),
            moving: "b".into(),
        })
    );
}

#[test]
fn something_unnamed_above_it_leaves_nothing_to_key_with() {
    let graph = chain();
    let mut nameless = Memory::new();
    for id in graph.nodes() {
        nameless.freeze(id.clone(), None);
        if id.as_str() != "a" {
            nameless.identify(id.clone(), "Add");
        }
    }
    nameless.cache("c", None);

    assert_eq!(
        cacheable(&graph, &nameless),
        Err(MemoryError::Nameless {
            cached: "c".into(),
            nameless: "a".into(),
        })
    );
}

#[test]
fn what_is_below_a_cached_node_is_none_of_its_business() {
    // `a >> b >> c` with `b` cached: that `c` still trains is exactly the case
    // this exists for — a frozen encoder with a head being trained on top.
    let graph = chain();
    let mut memory = settled_but(&graph, &["c"]);
    memory.cache("b", None);

    assert_eq!(cacheable(&graph, &memory), Ok(()));
}

#[test]
fn the_errors_say_which_two_nodes_and_why() {
    let far = MemoryError::Unsettled {
        cached: "c".into(),
        moving: "a".into(),
    };
    let itself = MemoryError::Unsettled {
        cached: "c".into(),
        moving: "c".into(),
    };

    assert!(far.to_string().contains('a') && far.to_string().contains('c'));
    assert!(
        !itself.to_string().contains("which it reads"),
        "when it is itself, it should not talk about reading: {itself}"
    );
}
