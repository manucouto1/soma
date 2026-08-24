//! The engine, against Rust nodes: no Python in the way.

use crate::doubles::{
    Add, Anything, Cable, EachOne, Fail, Immediate, Journal, Ledger, Mean, MeetingPoint, Mirror,
    Miscounts, Notebook, Opaquely, Panics, Rendezvous, Told, Ubiquitous, Witness,
};
use soma_next_core::{
    Catalog, Ctx, Device, Executor, Graph, Host, Key, Keys, Memory, Node, NodeError, NodeId,
    Outcome, Placement, Plan, RunError, Value, compile, distribute, node,
};
use std::sync::Arc;

fn number(v: &Value) -> f64 {
    let Value::Number(x) = v else {
        panic!("expected a number, found {}", v.type_name());
    };
    *x
}

// ── Filters ──

#[test]
fn an_empty_plan_returns_its_input() {
    let out = Executor::new(&Catalog::new())
        .run(&Plan::Empty, Value::text("intact"))
        .unwrap();
    assert_eq!(out, Value::text("intact"));
}

#[test]
fn a_chain_chains_the_outputs() {
    let mut g = Graph::new();
    let mut c = Catalog::new();
    for (id, how_much) in [("a", 1.0), ("b", 10.0), ("c", 100.0)] {
        g.add_node(id).unwrap();
        c.insert(id, Arc::new(Add(how_much)));
    }
    g.add_edge("a", "b").unwrap();
    g.add_edge("b", "c").unwrap();

    let plan = compile(&g, &c).unwrap();
    let out = Executor::new(&c).run(&plan, Value::number(0.0)).unwrap();
    assert_eq!(number(&out), 111.0);
}

#[test]
fn several_leaves_come_out_as_a_map_keyed_by_name() {
    let mut g = Graph::new();
    let mut c = Catalog::new();
    for (id, how_much) in [("source", 1.0), ("left", 10.0), ("right", 100.0)] {
        g.add_node(id).unwrap();
        c.insert(id, Arc::new(Add(how_much)));
    }
    g.add_edge("source", "left").unwrap();
    g.add_edge("source", "right").unwrap();

    let plan = compile(&g, &c).unwrap();
    let out = Executor::new(&c).run(&plan, Value::number(0.0)).unwrap();

    // source leaves 1.0, and each branch starts from there.
    assert_eq!(
        out,
        Value::map(vec![
            ("left".to_string(), Value::number(11.0)),
            ("right".to_string(), Value::number(101.0)),
        ])
    );
    assert_eq!(out.get("right"), Some(&Value::number(101.0)));
}

#[test]
fn a_node_with_two_inputs_receives_a_map() {
    let mut g = Graph::new();
    let mut c = Catalog::new();
    for (id, how_much) in [("left", 10.0), ("right", 100.0)] {
        g.add_node(id).unwrap();
        c.insert(id, Arc::new(Add(how_much)));
    }
    g.add_node("join").unwrap();
    c.insert("join", Arc::new(Mean));
    g.add_edge("left", "join").unwrap();
    g.add_edge("right", "join").unwrap();

    let plan = compile(&g, &c).unwrap();
    let out = Executor::new(&c).run(&plan, Value::number(0.0)).unwrap();
    assert_eq!(number(&out), 55.0);
}

#[test]
fn a_diamond_comes_back_round() {
    let mut g = Graph::new();
    let mut c = Catalog::new();
    for (id, how_much) in [("source", 1.0), ("left", 10.0), ("right", 100.0)] {
        g.add_node(id).unwrap();
        c.insert(id, Arc::new(Add(how_much)));
    }
    g.add_node("join").unwrap();
    c.insert("join", Arc::new(Mean));
    for (a, b) in [
        ("source", "left"),
        ("source", "right"),
        ("left", "join"),
        ("right", "join"),
    ] {
        g.add_edge(a, b).unwrap();
    }

    let plan = compile(&g, &c).unwrap();
    let out = Executor::new(&c).run(&plan, Value::number(0.0)).unwrap();
    // source leaves 1.0; branches 11.0 and 101.0; mean 56.0.
    assert_eq!(number(&out), 56.0);
}

#[test]
fn a_filters_failure_says_which_node_it_was() {
    let mut g = Graph::new();
    let mut c = Catalog::new();
    g.add_node("bomb").unwrap();
    c.insert("bomb", Arc::new(Fail));

    let plan = compile(&g, &c).unwrap();
    let err = Executor::new(&c).run(&plan, Value::Null).unwrap_err();
    assert!(matches!(err, RunError::Node { ref node, .. } if node.as_str() == "bomb"));
    assert!(err.to_string().contains("I broke"));
}

// ── Steps ──

// ── A node keeps whatever it keeps, and the engine keeps nothing ──

#[test]
fn what_a_node_kept_is_still_there_the_next_time_the_graph_runs() {
    // The catalog holds **the node**, not a copy per run, so its state outlives
    // a `forward`. That is not incidental — a worker keeping its catalog is what
    // lets an activation stay alive on the far side of a cut, and CU14 rests on
    // it.
    //
    // The other face of it is a trap, and this is where it is written down: a
    // node that counts answers a second run differently from the first, and
    // nothing warns. The engine promises the same **plan**, not that a node
    // without memory is the only kind there is — and now that a node runs to the
    // end on its own, whatever it keeps is entirely its own business.
    struct Counts(std::sync::Mutex<f64>);
    impl Node for Counts {
        fn forward(&self, _input: &Value, _ctx: &Ctx<'_>) -> Result<Value, NodeError> {
            let mut times = self.0.lock().expect("nobody poisons this mutex");
            *times += 1.0;
            Ok(Value::number(*times))
        }
    }

    let mut g = Graph::new();
    let mut c = Catalog::new();
    g.add_node("counts").unwrap();
    c.insert("counts", Arc::new(Counts(std::sync::Mutex::new(0.0))));
    let plan = compile(&g, &c).unwrap();

    let once = Executor::new(&c).run(&plan, Value::Null).unwrap();
    let twice = Executor::new(&c).run(&plan, Value::Null).unwrap();

    assert_eq!(once, Value::number(1.0));
    assert_eq!(
        twice,
        Value::number(2.0),
        "the second run started from scratch"
    );
}

// ── Waves: what happens when two branches are launched at once ──

/// Builds the graph, compiles it and runs it with nodes that note where they
/// went. Returns the journal and what came out.
fn run_noting(
    nodes: &[&'static str],
    edges: &[(&str, &str)],
) -> (Arc<Journal>, Result<Value, RunError>) {
    let journal = Journal::new();
    let mut g = Graph::new();
    let mut c = Catalog::new();
    for id in nodes {
        g.add_node(*id).unwrap();
        c.insert(*id, Arc::new(Witness(id, Arc::clone(&journal))));
    }
    for (from, to) in edges {
        g.add_edge(*from, *to).unwrap();
    }
    let plan = compile(&g, &c).unwrap();
    let output = Executor::new(&c).run(&plan, Value::Null);
    (journal, output)
}

#[test]
fn the_real_execution_order_respects_the_edges() {
    // The plan states an order; this checks the one that **actually** happens,
    // with threads in the way. A node cannot have executed before any of its
    // predecessors.
    let edges = [
        ("a", "b"),
        ("b", "b2"),
        ("b2", "b3"),
        ("a", "c"),
        ("c", "c2"),
        ("b3", "d"),
        ("c2", "d"),
    ];
    let (journal, output) = run_noting(&["a", "b", "b2", "b3", "c", "c2", "d"], &edges);
    output.unwrap();

    let order = journal.order();
    assert_eq!(
        order.len(),
        7,
        "they all executed, and exactly once: {order:?}"
    );
    let when = |who: &str| order.iter().position(|n| n == who).unwrap();
    for (from, to) in edges {
        assert!(
            when(from) < when(to),
            "{to} executed before {from}: {order:?}"
        );
    }
}

#[test]
fn a_whole_branch_runs_on_the_same_thread() {
    // It is what decomposing by branch rather than by topological level buys:
    // the branch is pinned to a thread, and the day a node has a device — which
    // in torch is *thread-local* — it is set once and not at every step.
    let (journal, output) = run_noting(
        &["a", "b", "b2", "b3", "c", "c2", "d"],
        &[
            ("a", "b"),
            ("b", "b2"),
            ("b2", "b3"),
            ("a", "c"),
            ("c", "c2"),
            ("b3", "d"),
            ("c2", "d"),
        ],
    );
    output.unwrap();

    assert_eq!(journal.thread_of("b"), journal.thread_of("b2"));
    assert_eq!(journal.thread_of("b2"), journal.thread_of("b3"));
    assert_eq!(journal.thread_of("c"), journal.thread_of("c2"));
    assert_ne!(
        journal.thread_of("b"),
        journal.thread_of("c"),
        "the two branches cannot share a thread, or they are not running at once"
    );
    assert_eq!(
        journal.thread_of("a"),
        journal.thread_of("d"),
        "what is outside the wave runs on the executing thread"
    );
}

#[test]
fn the_branches_of_a_wave_really_do_run_at_the_same_time() {
    // Without sleeping for a millisecond: the two nodes agree to meet, and were
    // they executed one after the other the first would wait for the second
    // until the deadline ran out.
    let point = MeetingPoint::new();
    let mut g = Graph::new();
    let mut c = Catalog::new();
    for id in ["left", "right"] {
        g.add_node(id).unwrap();
        c.insert(
            id,
            Arc::new(Rendezvous {
                point: Arc::clone(&point),
                how_many: 2,
                fails: None,
            }),
        );
    }
    let plan = compile(&g, &c).unwrap();
    assert!(
        matches!(plan, Plan::Wave(_)),
        "two unrelated nodes are a wave"
    );

    Executor::new(&c).run(&plan, Value::Null).unwrap();
}

#[test]
fn three_branches_also_go_at_once() {
    let point = MeetingPoint::new();
    let mut g = Graph::new();
    let mut c = Catalog::new();
    g.add_node("source").unwrap();
    c.insert("source", Arc::new(Immediate));
    for id in ["x", "y", "z"] {
        g.add_node(id).unwrap();
        g.add_edge("source", id).unwrap();
        c.insert(
            id,
            Arc::new(Rendezvous {
                point: Arc::clone(&point),
                how_many: 3,
                fails: None,
            }),
        );
    }
    let plan = compile(&g, &c).unwrap();
    Executor::new(&c).run(&plan, Value::Null).unwrap();
}

#[test]
fn the_diamond_gives_the_same_result_with_a_wave_as_without_one() {
    // The result cannot depend on whether the branches get spread out or not.
    let mut g = Graph::new();
    let mut c = Catalog::new();
    for (id, how_much) in [("source", 1.0), ("left", 10.0), ("right", 100.0)] {
        g.add_node(id).unwrap();
        c.insert(id, Arc::new(Add(how_much)));
    }
    g.add_node("join").unwrap();
    c.insert("join", Arc::new(Mean));
    g.add_edge("source", "left").unwrap();
    g.add_edge("source", "right").unwrap();
    g.add_edge("left", "join").unwrap();
    g.add_edge("right", "join").unwrap();

    let plan = compile(&g, &c).unwrap();
    let out = Executor::new(&c).run(&plan, Value::number(0.0)).unwrap();
    // 0 → 1 → {11, 101} → mean 56
    assert_eq!(number(&out), 56.0);
}

#[test]
fn what_each_branch_produces_reaches_whoever_reads_it() {
    // Branches work on a copy of what was produced and are merged on rejoining;
    // the join node has to see what both produced, including what happened
    // inside each branch.
    let mut g = Graph::new();
    let mut c = Catalog::new();
    for (id, how_much) in [
        ("a", 1.0),
        ("b", 10.0),
        ("b2", 20.0),
        ("c", 100.0),
        ("c2", 200.0),
    ] {
        g.add_node(id).unwrap();
        c.insert(id, Arc::new(Add(how_much)));
    }
    g.add_node("d").unwrap();
    c.insert("d", Arc::new(Mean));
    for (from, to) in [
        ("a", "b"),
        ("b", "b2"),
        ("a", "c"),
        ("c", "c2"),
        ("b2", "d"),
        ("c2", "d"),
    ] {
        g.add_edge(from, to).unwrap();
    }

    let plan = compile(&g, &c).unwrap();
    let out = Executor::new(&c).run(&plan, Value::number(0.0)).unwrap();
    // 0 → 1 → branch b: 11, 31 · branch c: 101, 301 → mean 166
    assert_eq!(number(&out), 166.0);
}

#[test]
fn two_failing_branches_always_give_the_first_ones_error() {
    // Both genuinely fail at the same time — they agree to meet before breaking
    // — so which one fails first on the clock is a race. The error that counts
    // cannot depend on it: it is the first declared branch's.
    let point = MeetingPoint::new();
    let mut g = Graph::new();
    let mut c = Catalog::new();
    for (id, message) in [
        ("left", "the left one broke"),
        ("right", "the right one broke"),
    ] {
        g.add_node(id).unwrap();
        c.insert(
            id,
            Arc::new(Rendezvous {
                point: Arc::clone(&point),
                how_many: 2,
                fails: Some(message),
            }),
        );
    }
    let plan = compile(&g, &c).unwrap();

    let RunError::Node { node, source } = Executor::new(&c).run(&plan, Value::Null).unwrap_err()
    else {
        panic!("expected a node failure");
    };
    assert_eq!(node.as_str(), "left");
    assert_eq!(source.message(), "the left one broke");
}

#[test]
fn one_branchs_failure_does_not_hide_the_others_if_that_one_comes_first() {
    // The same graph with the branches the other way round gives the other
    // error: it is not that "left" always wins, it is that declaration order
    // wins.
    let mut g = Graph::new();
    let mut c = Catalog::new();
    g.add_node("right").unwrap();
    c.insert("right", Arc::new(Fail));
    g.add_node("left").unwrap();
    c.insert("left", Arc::new(Fail));

    let plan = compile(&g, &c).unwrap();
    let RunError::Node { node, .. } = Executor::new(&c).run(&plan, Value::Null).unwrap_err() else {
        panic!("expected a node failure");
    };
    assert_eq!(node.as_str(), "right", "`right` was declared first");
}

#[test]
#[should_panic(expected = "I blew up")]
fn a_panic_inside_a_branch_is_not_swallowed() {
    let mut g = Graph::new();
    let mut c = Catalog::new();
    g.add_node("healthy").unwrap();
    c.insert("healthy", Arc::new(Immediate));
    g.add_node("bad").unwrap();
    c.insert("bad", Arc::new(Panics));

    let plan = compile(&g, &c).unwrap();
    let _ = Executor::new(&c).run(&plan, Value::Null);
}

#[test]
fn a_wave_that_is_the_whole_plan_returns_the_map_of_its_leaves() {
    let mut g = Graph::new();
    let mut c = Catalog::new();
    for (id, how_much) in [("one", 1.0), ("other", 2.0)] {
        g.add_node(id).unwrap();
        c.insert(id, Arc::new(Add(how_much)));
    }
    let plan = compile(&g, &c).unwrap();
    let out = Executor::new(&c).run(&plan, Value::number(10.0)).unwrap();

    assert_eq!(
        out,
        Value::map(vec![
            ("one".to_string(), Value::number(11.0)),
            ("other".to_string(), Value::number(12.0)),
        ])
    );
}

#[test]
fn a_graph_that_is_not_series_parallel_still_executes_correctly() {
    // The N: not a single wave, but the result is right and `d` sees both
    // inputs.
    let mut g = Graph::new();
    let mut c = Catalog::new();
    for (id, how_much) in [("a", 1.0), ("b", 2.0), ("c", 100.0)] {
        g.add_node(id).unwrap();
        c.insert(id, Arc::new(Add(how_much)));
    }
    g.add_node("d").unwrap();
    c.insert("d", Arc::new(Mean));
    for (from, to) in [("a", "c"), ("a", "d"), ("b", "d")] {
        g.add_edge(from, to).unwrap();
    }

    let plan = compile(&g, &c).unwrap();
    assert!(
        !format!("{plan:?}").contains("Wave"),
        "the N has no series-parallel tree"
    );
    let out = Executor::new(&c).run(&plan, Value::number(0.0)).unwrap();
    // leaves: c = 1+100 = 101, d = mean(a=1, b=2) = 1.5
    assert_eq!(
        out,
        Value::map(vec![
            ("c".to_string(), Value::number(101.0)),
            ("d".to_string(), Value::number(1.5)),
        ])
    );
}

// ── The placement: it reaches the node, and does nothing else ──

/// A graph of witnesses that note where they were told to run.
fn witnesses(ids: &[&'static str], ledger: &Arc<Ledger>) -> (Graph, Catalog) {
    let mut g = Graph::new();
    let mut c = Catalog::new();
    for id in ids {
        g.add_node(*id).unwrap();
        c.insert(*id, Arc::new(Ubiquitous(id, ledger.clone())));
    }
    (g, c)
}

#[test]
fn without_a_placement_the_node_sees_no_device() {
    let ledger = Ledger::new();
    let (g, c) = witnesses(&["a"], &ledger);

    let plan = compile(&g, &c).unwrap();
    Executor::new(&c).run(&plan, Value::Null).unwrap();

    assert_eq!(ledger.of("a"), None);
}

#[test]
fn the_node_sees_where_it_was_told_to_run() {
    let ledger = Ledger::new();
    let (g, c) = witnesses(&["a"], &ledger);
    let mut placement = Placement::new();
    placement.place("a", Device::Cuda(1));

    let plan = compile(&g, &c).unwrap();
    Executor::new(&c)
        .placed(&placement)
        .run(&plan, Value::Null)
        .unwrap();

    assert_eq!(ledger.of("a"), Some(Device::Cuda(1)));
}

#[test]
fn each_node_sees_its_own_and_only_its_own() {
    let ledger = Ledger::new();
    let (mut g, c) = witnesses(&["a", "b", "c"], &ledger);
    g.add_edge("a", "b").unwrap();
    g.add_edge("b", "c").unwrap();

    let mut placement = Placement::new();
    placement.place("a", Device::Cpu);
    placement.place("c", Device::Meta);

    let plan = compile(&g, &c).unwrap();
    Executor::new(&c)
        .placed(&placement)
        .run(&plan, Value::Null)
        .unwrap();

    assert_eq!(ledger.of("a"), Some(Device::Cpu));
    assert_eq!(ledger.of("b"), None, "nobody catches the neighbour's");
    assert_eq!(ledger.of("c"), Some(Device::Meta));
}

#[test]
fn the_branches_of_a_wave_see_different_devices_each_on_its_own_thread() {
    // It is the reason CU9 came before this: a whole branch runs on one thread,
    // so it can set its device once and have it hold.
    let ledger = Ledger::new();
    let (mut g, c) = witnesses(&["source", "left", "right"], &ledger);
    g.add_edge("source", "left").unwrap();
    g.add_edge("source", "right").unwrap();

    let mut placement = Placement::new();
    placement.place("left", Device::Cuda(0));
    placement.place("right", Device::Cuda(1));

    let plan = compile(&g, &c).unwrap();
    assert!(format!("{plan:?}").contains("Wave"));
    Executor::new(&c)
        .placed(&placement)
        .run(&plan, Value::Null)
        .unwrap();

    assert_eq!(ledger.of("left"), Some(Device::Cuda(0)));
    assert_eq!(ledger.of("right"), Some(Device::Cuda(1)));
}

#[test]
fn a_placement_that_names_someone_else_reaches_nobody() {
    // A bare `Placement` cannot check ids: that is done where there is a graph
    // in front of you. Here we only see that no other node swallows it.
    let ledger = Ledger::new();
    let (g, c) = witnesses(&["a"], &ledger);
    let mut placement = Placement::new();
    placement.place("other", Device::Cuda(0));

    let plan = compile(&g, &c).unwrap();
    Executor::new(&c)
        .placed(&placement)
        .run(&plan, Value::Null)
        .unwrap();

    assert_eq!(ledger.of("a"), None);
}

#[test]
fn placing_does_not_change_what_the_graph_produces() {
    let mut g = Graph::new();
    let mut c = Catalog::new();
    for (id, how_much) in [("a", 1.0), ("b", 10.0)] {
        g.add_node(id).unwrap();
        c.insert(id, Arc::new(Add(how_much)));
    }
    g.add_edge("a", "b").unwrap();
    let mut placement = Placement::new();
    placement.place("a", Device::Meta);
    placement.place("b", Device::Cuda(3));

    let plan = compile(&g, &c).unwrap();
    let without = Executor::new(&c).run(&plan, Value::number(0.0)).unwrap();
    let with = Executor::new(&c)
        .placed(&placement)
        .run(&plan, Value::number(0.0))
        .unwrap();

    assert_eq!(number(&without), 11.0);
    assert_eq!(without, with);
}

// ── What is distributed, while there is nobody to carry it ──
//
// The plan already knows how to say "this runs over there" and the engine does
// not yet know how to get there. What is pinned down here is that this shows: a
// slice placed away is **not** executed at home for lack of a transport.

#[test]
fn a_slice_on_another_host_without_a_transport_stops_saying_which() {
    let (g, c, placement, _) = (node("a", Add(1.0)) >> node("b", Add(1.0)).at("worker1"))
        .somatize()
        .unwrap();
    let plan = distribute(&compile(&g, &c).unwrap(), &placement);

    let err = Executor::new(&c)
        .placed(&placement)
        .run(&plan, Value::number(0.0))
        .unwrap_err();

    assert_eq!(err, RunError::NoTransport(Host::new("worker1")));
    assert!(
        err.to_string().contains("worker1"),
        "the name has to appear in the message: {err}"
    );
}

#[test]
fn without_distributing_the_same_graph_runs_as_always() {
    // The other half of the test above: what stops the execution is the
    // distribution, not the placement. The same graph, the same catalog, the
    // same `Placement`, and a plan that has not been through `distribute` runs
    // in full.
    let (g, c, placement, _) = (node("a", Add(1.0)) >> node("b", Add(1.0)).at("worker1"))
        .somatize()
        .unwrap();
    let plan = compile(&g, &c).unwrap();

    assert_eq!(
        number(
            &Executor::new(&c)
                .placed(&placement)
                .run(&plan, Value::number(0.0))
                .unwrap()
        ),
        2.0
    );
}

#[test]
fn what_runs_away_still_counts_as_a_leaf() {
    // `terminals` goes through a `Remote`: distributing does not change what the
    // graph produces, only where. If it did not go through, the output of a
    // distributed plan would stop being a map and nobody would find out until
    // there was a transport.
    let (g, c, placement, _) = (node("a", Add(1.0)).at("w1") | node("b", Add(2.0)))
        .somatize()
        .unwrap();
    let plan = distribute(&compile(&g, &c).unwrap(), &placement);

    // Both branches are leaves; the first is away and fails first.
    assert_eq!(
        Executor::new(&c)
            .placed(&placement)
            .run(&plan, Value::number(0.0))
            .unwrap_err(),
        RunError::NoTransport(Host::new("w1"))
    );
}

// ── What crosses over to a transport, and what comes back ──
//
// No processes: that belongs to `soma-next-transport`. What is pinned down here
// is the core's seam — what gets sent, what comes back and where it is merged.

/// The same graph built twice: one for here, one for "there".
fn both_sides(nodes: &[(&str, f64)], edges: &[(&str, &str)]) -> (Graph, Catalog, Catalog) {
    let mut g = Graph::new();
    let (mut here, mut there) = (Catalog::new(), Catalog::new());
    for (id, how_much) in nodes {
        g.add_node(*id).unwrap();
        here.insert(*id, Arc::new(Add(*how_much)));
        there.insert(*id, Arc::new(Add(*how_much)));
    }
    for (from, to) in edges {
        g.add_edge(*from, *to).unwrap();
    }
    (g, here, there)
}

fn away(ids: &[&str]) -> Placement {
    let mut placement = Placement::new();
    for id in ids {
        placement.place_at(*id, Host::new("there"));
    }
    placement
}

#[test]
fn the_transport_is_sent_only_what_the_slice_needs() {
    // `c` reads from `b`, and `b` ran here. What travels is `b` and nothing
    // else: `a` was also produced here and nobody over there looks at it.
    let (g, here, there) = both_sides(
        &[("a", 1.0), ("b", 10.0), ("c", 100.0)],
        &[("a", "b"), ("b", "c")],
    );
    let placement = away(&["c"]);
    let mirror = Mirror::new(there);
    let plan = distribute(&compile(&g, &here).unwrap(), &placement);

    let output = Executor::new(&here)
        .placed(&placement)
        .reaching("there", &mirror)
        .run(&plan, Value::number(0.0))
        .unwrap();

    assert_eq!(number(&output), 111.0);
    let trips = mirror.trips();
    assert_eq!(trips.len(), 1, "a single trip");
    assert_eq!(
        trips[0].known,
        vec![(NodeId::from("b"), Value::number(11.0))],
        "something is missing from or spare on the wire"
    );
}

#[test]
fn a_slice_that_reads_nothing_from_here_travels_without_cargo() {
    let (g, here, there) = both_sides(&[("a", 1.0)], &[]);
    let placement = away(&["a"]);
    let mirror = Mirror::new(there);
    let plan = distribute(&compile(&g, &here).unwrap(), &placement);

    Executor::new(&here)
        .placed(&placement)
        .reaching("there", &mirror)
        .run(&plan, Value::number(5.0))
        .unwrap();

    let trips = mirror.trips();
    assert!(trips[0].known.is_empty(), "it reads from nobody here");
    assert_eq!(
        trips[0].input,
        Value::number(5.0),
        "but it does need the graph's input"
    );
}

#[test]
fn what_comes_back_is_read_by_whoever_comes_next() {
    let (g, here, there) = both_sides(
        &[("a", 1.0), ("b", 10.0), ("c", 100.0)],
        &[("a", "b"), ("b", "c")],
    );
    let placement = away(&["b"]);
    let mirror = Mirror::new(there);
    let plan = distribute(&compile(&g, &here).unwrap(), &placement);

    assert_eq!(
        number(
            &Executor::new(&here)
                .placed(&placement)
                .reaching("there", &mirror)
                .run(&plan, Value::number(0.0))
                .unwrap()
        ),
        111.0
    );
}

#[test]
fn the_placement_travels_with_the_slice() {
    let (g, here, there) = both_sides(&[("a", 1.0)], &[]);
    let mut placement = away(&["a"]);
    placement.place("a", Device::Meta);
    let mirror = Mirror::new(there);
    let plan = distribute(&compile(&g, &here).unwrap(), &placement);

    Executor::new(&here)
        .placed(&placement)
        .reaching("there", &mirror)
        .run(&plan, Value::number(0.0))
        .unwrap();

    assert_eq!(
        mirror.trips()[0].placement.of(&"a".into()),
        Some(&Device::Meta)
    );
}

#[test]
fn whatever_the_transport_says_is_attributed_to_the_host() {
    let (g, here, _) = both_sides(&[("a", 1.0)], &[]);
    let placement = away(&["a"]);
    let cable = Cable("the network went down");
    let plan = distribute(&compile(&g, &here).unwrap(), &placement);

    let err = Executor::new(&here)
        .placed(&placement)
        .reaching("there", &cable)
        .run(&plan, Value::number(0.0))
        .unwrap_err();

    assert_eq!(
        err,
        RunError::Transport {
            host: Host::new("there"),
            source: soma_next_core::TransportError::new("the network went down"),
        }
    );
    let said = err.to_string();
    assert!(
        said.contains("there") && said.contains("the network went down"),
        "{said}"
    );
}

#[test]
fn distributed_or_not_the_result_is_the_same() {
    // The invariant that makes distributing a decision and not a change of
    // semantics. Over a diamond, which is where most things can go wrong.
    let mut g = Graph::new();
    let (mut here, mut there) = (Catalog::new(), Catalog::new());
    for id in ["s", "l", "r"] {
        g.add_node(id).unwrap();
        here.insert(id, Arc::new(Add(1.0)));
        there.insert(id, Arc::new(Add(1.0)));
    }
    g.add_node("j").unwrap();
    here.insert("j", Arc::new(Mean));
    there.insert("j", Arc::new(Mean));
    for (from, to) in [("s", "l"), ("s", "r"), ("l", "j"), ("r", "j")] {
        g.add_edge(from, to).unwrap();
    }

    let whole = Executor::new(&here)
        .run(&compile(&g, &here).unwrap(), Value::number(0.0))
        .unwrap();

    for sent_away in [
        vec!["l"],
        vec!["l", "r"],
        vec!["j"],
        vec!["s", "l", "r", "j"],
    ] {
        let placement = away(&sent_away);
        let mirror = Mirror::new(there.clone());
        let plan = distribute(&compile(&g, &here).unwrap(), &placement);
        let distributed = Executor::new(&here)
            .placed(&placement)
            .reaching("there", &mirror)
            .run(&plan, Value::number(0.0))
            .unwrap();

        assert_eq!(
            distributed, whole,
            "sending {sent_away:?} away gives something else"
        );
    }
}

// ── `resume`: what a worker does on receiving a slice ──

#[test]
fn resume_feeds_in_what_it_was_given_as_if_it_had_produced_it() {
    let mut c = Catalog::new();
    c.insert("b", Arc::new(Add(10.0)));
    let plan = Plan::Execute {
        node: "b".into(),
        from: vec!["a".into()],
    };

    let outcome = Executor::new(&c)
        .resume(
            &plan,
            Value::Null,
            vec![(NodeId::from("a"), Value::number(1.0))],
            Vec::new(),
        )
        .unwrap();

    assert_eq!(number(&outcome.last), 11.0);
}

#[test]
fn resume_does_not_return_what_arrived() {
    // Whoever sent it already has it; returning it would pay for the wire twice.
    let mut c = Catalog::new();
    c.insert("b", Arc::new(Add(10.0)));
    let plan = Plan::Execute {
        node: "b".into(),
        from: vec!["a".into()],
    };

    let outcome = Executor::new(&c)
        .resume(
            &plan,
            Value::Null,
            vec![(NodeId::from("a"), Value::number(1.0))],
            Vec::new(),
        )
        .unwrap();

    assert_eq!(
        outcome.produced,
        vec![(NodeId::from("b"), Value::number(11.0))]
    );
}

#[test]
fn resume_returns_what_was_produced_ordered_by_id() {
    // A `HashMap` iterates differently in each process, and this crosses exactly
    // that boundary: without an order, two equal runs would give different bytes.
    let mut g = Graph::new();
    let mut c = Catalog::new();
    for id in ["z", "a", "m"] {
        g.add_node(id).unwrap();
        c.insert(id, Arc::new(Add(1.0)));
    }
    let plan = compile(&g, &c).unwrap();

    let outcome = Executor::new(&c)
        .resume(&plan, Value::number(0.0), Vec::new(), Vec::new())
        .unwrap();

    assert_eq!(
        outcome
            .produced
            .iter()
            .map(|(id, _)| id.as_str())
            .collect::<Vec<_>>(),
        ["a", "m", "z"]
    );
}

#[test]
fn resume_of_an_empty_plan_returns_its_input_and_nothing_produced() {
    let c = Catalog::new();
    assert_eq!(
        Executor::new(&c)
            .resume(&Plan::Empty, Value::number(7.0), Vec::new(), Vec::new())
            .unwrap(),
        Outcome {
            last: Value::number(7.0),
            produced: Vec::new(),
            keys: Vec::new(),
        }
    );
}

#[test]
fn what_stayed_over_there_names_both_ends_when_somebody_here_reads_it() {
    // The slice answers fine: what it produced in the middle was read where it
    // ran, and its own value travels. What cannot happen is somebody **here**
    // reading that middle — and the walk says which value and which reader,
    // instead of finding a hole where a predecessor should be.
    let mut here = Catalog::new();
    here.insert("opaque", Arc::new(Immediate));
    here.insert("reads", Arc::new(Add(1.0)));
    here.insert("c", Arc::new(Add(1.0)));
    let mut there = Catalog::new();
    there.insert("opaque", Arc::new(Opaquely));
    there.insert("reads", Arc::new(Anything));
    let mirror = Mirror::new(there);

    let step = |id: &str, from: &[&str]| Plan::Execute {
        node: id.into(),
        from: from.iter().map(|each| NodeId::from(*each)).collect(),
    };
    let plan = Plan::Sequence(vec![
        Plan::Remote {
            host: Host::new("there"),
            inner: Box::new(Plan::Sequence(vec![
                step("opaque", &[]),
                step("reads", &["opaque"]),
            ])),
        },
        step("c", &["opaque"]),
    ]);

    let said = Executor::new(&here)
        .reaching("there", &mirror)
        .run(&plan, Value::Null)
        .unwrap_err()
        .to_string();

    assert!(said.contains("`c`") && said.contains("`opaque`"), "{said}");
    assert!(said.contains("stayed where it ran"), "{said}");
}

#[test]
fn an_outcome_leaves_behind_what_cannot_travel_and_keeps_what_it_answers_with() {
    // `last` is the value of the slice itself and has a reader on the other side
    // by definition, so it is not filtered: refusing it is the honest answer.
    // What is filtered is the middle of the slice, which was read where it ran.
    let outcome = Outcome {
        last: Value::number(1.0),
        produced: vec![
            (NodeId::from("live"), Value::opaque(7u32)),
            (NodeId::from("plain"), Value::number(2.0)),
        ],
        keys: Vec::new(),
    };

    assert_eq!(
        outcome.travelling().produced,
        vec![(NodeId::from("plain"), Value::number(2.0))]
    );
}

// ── What is remembered, and what does not get run twice ──
//
// The engine's half of the cache: a key travelling beside what was produced,
// a lookup before the node and a write after it. What a key is *made of* is a
// keeper's business and lives with the double.

/// A graph of one node that writes itself down when it runs.
fn watched(who: &'static str) -> (Graph, Catalog, Memory, Arc<Journal>) {
    let journal = Journal::new();
    let (graph, catalog, _, memory) = node(who, Witness(who, journal.clone()))
        .frozen()
        .cached()
        .somatize()
        .unwrap();
    (graph, catalog, memory, journal)
}

#[test]
fn what_is_kept_is_not_computed_again() {
    let (g, c, memory, journal) = watched("encoder");
    let plan = compile(&g, &c).unwrap();
    let notebook = Notebook::new();

    let first = Executor::new(&c)
        .remembering(&memory)
        .keeping(&notebook)
        .run(&plan, Value::number(7.0))
        .unwrap();
    let second = Executor::new(&c)
        .remembering(&memory)
        .keeping(&notebook)
        .run(&plan, Value::number(7.0))
        .unwrap();

    assert_eq!(number(&first), 7.0);
    assert_eq!(number(&second), 7.0);
    assert_eq!(
        notebook.under(&notebook.names()[0]),
        Some(Value::number(7.0)),
        "and what is kept under that name is what it produced"
    );
    assert_eq!(
        journal.order(),
        ["encoder"],
        "the second run had the answer already: the node must not have been asked"
    );
}

#[test]
fn a_different_input_is_a_different_name_and_the_node_runs() {
    // The other half of the one above, and the reason the root is the one place
    // content is hashed: a cache that answers the same for two inputs is not a
    // cache, it is a bug.
    let (g, c, memory, journal) = watched("encoder");
    let plan = compile(&g, &c).unwrap();
    let notebook = Notebook::new();

    for input in [1.0, 2.0] {
        Executor::new(&c)
            .remembering(&memory)
            .keeping(&notebook)
            .run(&plan, Value::number(input))
            .unwrap();
    }

    assert_eq!(journal.order(), ["encoder", "encoder"]);
    assert_eq!(notebook.names().len(), 2);
}

#[test]
fn what_is_above_names_what_is_below() {
    // The Merkle rule, seen from outside: the same node, the same code, and a
    // predecessor settled at another state — another name, and no hit.
    let name_of_head = |state: &str| {
        let (g, c, _, mut memory) = (node("encoder", Add(1.0)).frozen().cached()
            >> node("head", Add(1.0)).frozen().cached())
        .somatize()
        .unwrap();
        memory.freeze("encoder", Some(state.to_string()));
        let plan = compile(&g, &c).unwrap();
        let notebook = Notebook::new();
        Executor::new(&c)
            .remembering(&memory)
            .keeping(&notebook)
            .run(&plan, Value::number(0.0))
            .unwrap();
        notebook.names().last().cloned().expect("head was kept")
    };

    assert_ne!(
        name_of_head("weights-of-monday"),
        name_of_head("weights-of-tuesday")
    );
}

#[test]
fn the_fingerprint_of_the_code_is_not_part_of_the_name() {
    // Deliberate, and the whole reason the fingerprint is written *beside* the
    // value: a cosmetic refactor must not invalidate half the store in silence.
    // What it does is get compared on a hit and said out loud.
    let (g, c, memory, journal) = watched("encoder");
    let plan = compile(&g, &c).unwrap();
    let notebook = Notebook::new();

    let mut written_yesterday = memory.clone();
    written_yesterday.written_as("encoder", "v1");
    let mut written_today = memory.clone();
    written_today.written_as("encoder", "v2");

    Executor::new(&c)
        .remembering(&written_yesterday)
        .keeping(&notebook)
        .run(&plan, Value::number(7.0))
        .unwrap();
    Executor::new(&c)
        .remembering(&written_today)
        .keeping(&notebook)
        .run(&plan, Value::number(7.0))
        .unwrap();

    assert_eq!(notebook.names().len(), 1, "one name, not two");
    assert_eq!(
        journal.order(),
        ["encoder"],
        "and the second run used what was kept by the first"
    );
    assert_eq!(
        notebook.said_of(&notebook.names()[0]),
        [
            ("node".to_string(), "encoder".to_string()),
            ("fingerprint".to_string(), "v1".to_string())
        ],
        "what is written beside it is what produced it, not what asked for it"
    );
}

#[test]
fn a_node_that_keeps_nothing_still_passes_its_name_on() {
    // `.cached()` is opt-in because keeping costs; not declaring it must not
    // break the chain, or declaring it node by node would be declaring it for
    // the whole graph.
    let journal = Journal::new();
    let (g, c, _, memory) = (node("encoder", Witness("encoder", journal.clone())).frozen()
        >> node("head", Witness("head", journal.clone()))
            .frozen()
            .cached())
    .somatize()
    .unwrap();
    let plan = compile(&g, &c).unwrap();
    let notebook = Notebook::new();

    for _ in 0..2 {
        Executor::new(&c)
            .remembering(&memory)
            .keeping(&notebook)
            .run(&plan, Value::number(7.0))
            .unwrap();
    }

    assert_eq!(
        notebook.names().len(),
        1,
        "only what asked to be kept is kept"
    );
    assert_eq!(
        notebook.said_of(&notebook.names()[0])[0].1,
        "head",
        "and it is the one that asked"
    );
    assert_eq!(
        journal.order(),
        ["encoder", "head"],
        "the head was named out of a name nobody kept — and the second time round \
         nothing needed the encoder, because the head's answer was already there"
    );
}

#[test]
fn what_cannot_be_named_is_not_kept_and_is_not_an_error() {
    // An `Opaque` root: there is nothing to hash, so nothing below it has a
    // name. The run goes on exactly as it did before any of this existed.
    let journal = Journal::new();
    let (g, c, _, memory) = node("encoder", Witness("encoder", journal.clone()))
        .frozen()
        .cached()
        .somatize()
        .unwrap();
    let plan = compile(&g, &c).unwrap();
    let notebook = Notebook::new();

    for _ in 0..2 {
        Executor::new(&c)
            .remembering(&memory)
            .keeping(&notebook)
            .run(&plan, Value::opaque(7u32))
            .unwrap();
    }

    assert!(notebook.names().is_empty());
    assert_eq!(journal.order(), ["encoder", "encoder"]);
}

#[test]
fn nothing_is_named_without_somebody_to_hash_it() {
    // The whole thing is behind `keeping`: a graph that declares a cache and an
    // executor that was given no keeper runs as it always did.
    let (g, c, _, journal) = watched("encoder");
    let plan = compile(&g, &c).unwrap();

    for _ in 0..2 {
        Executor::new(&c).run(&plan, Value::number(7.0)).unwrap();
    }
    assert_eq!(journal.order(), ["encoder", "encoder"]);
}

#[test]
fn the_names_a_slice_brings_are_not_the_names_it_gives() {
    // What a worker answers with. The same retention as `produced`: what came in
    // does not come back, or every hop would grow the answer.
    let (_, c, _, memory) = (node("encoder", Add(1.0)).frozen().cached()
        >> node("head", Add(1.0)).frozen().cached())
    .somatize()
    .unwrap();
    let notebook = Notebook::new();

    // As if `encoder` had run at home and only `head` were sent away.
    let brought = Key::new("what-the-encoder-was-called");
    let outcome = Executor::new(&c)
        .remembering(&memory)
        .keeping(&notebook)
        .resume(
            &Plan::Execute {
                node: "head".into(),
                from: vec!["encoder".into()],
            },
            Value::Null,
            vec![(NodeId::from("encoder"), Value::number(1.0))],
            vec![(NodeId::from("encoder"), Keys::One(brought.clone()))],
        )
        .unwrap();

    assert_eq!(
        outcome
            .keys
            .iter()
            .map(|(id, _)| id.as_str())
            .collect::<Vec<_>>(),
        ["head"],
        "what it was given back is what it named itself"
    );
    assert_eq!(outcome.keys[0].1, Keys::One(notebook.names()[0].clone()));
    assert_ne!(outcome.keys[0].1, Keys::One(brought.clone()));
}

#[test]
fn the_names_and_what_is_remembered_cross_to_the_other_side() {
    // A slice that leaves has to be able to name what it produces, and for that
    // it needs the names of what it reads and the table that says what any of it
    // is. Both travel in the `Cargo`, like the placement and for the same
    // reason: they are data.
    let (g, c, placement, memory) = (node("encoder", Add(1.0)).frozen().cached()
        >> node("head", Add(1.0)).frozen().cached().at("worker1"))
    .somatize()
    .unwrap();
    let plan = distribute(&compile(&g, &c).unwrap(), &placement);
    let notebook = Notebook::new();
    let mirror = Mirror::new(c.clone());

    Executor::new(&c)
        .remembering(&memory)
        .keeping(&notebook)
        .placed(&placement)
        .reaching("worker1", &mirror)
        .run(&plan, Value::number(0.0))
        .unwrap();

    let trips = mirror.trips();
    assert_eq!(
        trips[0]
            .keys
            .iter()
            .map(|(id, _)| id.as_str())
            .collect::<Vec<_>>(),
        ["encoder"],
        "the names of what it reads, and only those"
    );
    assert_eq!(trips[0].keys[0].1, Keys::One(notebook.names()[0].clone()));
    assert!(trips[0].memory.is_cached(&"head".into()));
}

#[test]
fn what_is_remembered_travels_with_nobody_here_to_keep_anything() {
    // The two are not one call for exactly this: a coordinator that keeps
    // nothing still has to say what the nodes are, or whoever does keep things
    // over there can name none of them.
    let (g, c, placement, memory) = node("head", Add(1.0))
        .frozen()
        .cached()
        .at("worker1")
        .somatize()
        .unwrap();
    let plan = distribute(&compile(&g, &c).unwrap(), &placement);
    let mirror = Mirror::new(c.clone());

    Executor::new(&c)
        .remembering(&memory)
        .placed(&placement)
        .reaching("worker1", &mirror)
        .run(&plan, Value::number(0.0))
        .unwrap();

    let trips = mirror.trips();
    assert!(trips[0].memory.is_cached(&"head".into()));
    assert!(trips[0].memory.is_frozen(&"head".into()));
    assert_eq!(
        trips[0].memory.identity_of(&"head".into()),
        Some("unit::doubles::Add")
    );
}

// ── A node that maps, and a cache with the grain of an item ──

/// A one-node graph whose node maps, cached, with a journal of what it was made
/// to look at.
fn mapping() -> (Graph, Catalog, Memory, Arc<Journal>) {
    let journal = Journal::new();
    let (graph, catalog, _, memory) = node("embed", EachOne(journal.clone()))
        .frozen()
        .cached()
        .mapped()
        .somatize()
        .unwrap();
    (graph, catalog, memory, journal)
}

fn documents(many: &[f64]) -> Value {
    Value::list(many.iter().copied().map(Value::number).collect::<Vec<_>>())
}

fn tens(many: &[f64]) -> Vec<f64> {
    many.iter().map(|x| x * 10.0).collect()
}

fn each(out: &Value) -> Vec<f64> {
    let Value::List(items) = out else {
        panic!("expected a list, found {}", out.type_name())
    };
    items.iter().map(number).collect()
}

#[test]
fn a_node_that_maps_answers_one_for_each_item() {
    let (g, c, memory, _) = mapping();
    let plan = compile(&g, &c).unwrap();

    let out = Executor::new(&c)
        .remembering(&memory)
        .keeping(&Notebook::new())
        .run(&plan, documents(&[1.0, 2.0, 3.0]))
        .unwrap();

    assert_eq!(each(&out), tens(&[1.0, 2.0, 3.0]));
}

#[test]
fn and_it_maps_with_nobody_keeping_anything_at_all() {
    // `.mapped()` is a contract before it is an optimization: hand it a list,
    // get a list as long. That stays true with no keeper in sight.
    let (g, c, _, _) = mapping();
    let plan = compile(&g, &c).unwrap();

    let out = Executor::new(&c).run(&plan, documents(&[4.0])).unwrap();

    assert_eq!(each(&out), tens(&[4.0]));
}

#[test]
fn the_second_run_looks_at_nothing() {
    let (g, c, memory, journal) = mapping();
    let plan = compile(&g, &c).unwrap();
    let notebook = Notebook::new();
    let run = || {
        Executor::new(&c)
            .remembering(&memory)
            .keeping(&notebook)
            .run(&plan, documents(&[1.0, 2.0, 3.0]))
            .unwrap()
    };

    run();
    let second = run();

    assert_eq!(each(&second), tens(&[1.0, 2.0, 3.0]));
    assert_eq!(
        journal.order().len(),
        3,
        "it looked at something the second time round"
    );
}

#[test]
fn a_new_item_among_old_ones_is_the_only_one_looked_at() {
    // **The whole reason this exists.** With one name per node, adding a
    // document changes the name of the list and all of them miss; with one per
    // item, the old ones are read back and the new one runs — and the order of
    // the answer is the order of the input, not the order things were computed.
    let (g, c, memory, journal) = mapping();
    let plan = compile(&g, &c).unwrap();
    let notebook = Notebook::new();
    let run = |input: Value| {
        Executor::new(&c)
            .remembering(&memory)
            .keeping(&notebook)
            .run(&plan, input)
            .unwrap()
    };

    run(documents(&[1.0, 2.0, 3.0]));
    let again = run(documents(&[9.0, 1.0, 2.0, 3.0]));

    assert_eq!(each(&again), tens(&[9.0, 1.0, 2.0, 3.0]));
    assert_eq!(
        journal.order(),
        ["Number(1)", "Number(2)", "Number(3)", "Number(9)"],
        "the three it already had were looked at again"
    );
}

#[test]
fn an_item_is_named_after_itself_and_not_after_where_it_sits() {
    // The same document in another list is the same item. If a name were built
    // out of a position, this would miss on all four.
    let (g, c, memory, journal) = mapping();
    let plan = compile(&g, &c).unwrap();
    let notebook = Notebook::new();
    let run = |input: Value| {
        Executor::new(&c)
            .remembering(&memory)
            .keeping(&notebook)
            .run(&plan, input)
            .unwrap()
    };

    run(documents(&[1.0, 2.0, 3.0, 4.0]));
    let shuffled = run(documents(&[4.0, 3.0, 2.0, 1.0]));

    assert_eq!(each(&shuffled), tens(&[4.0, 3.0, 2.0, 1.0]));
    assert_eq!(journal.order().len(), 4, "the same four, in another order");
}

#[test]
fn what_is_not_a_list_is_refused_with_the_node_and_what_arrived() {
    let (g, c, memory, _) = mapping();
    let plan = compile(&g, &c).unwrap();

    let said = Executor::new(&c)
        .remembering(&memory)
        .keeping(&Notebook::new())
        .run(&plan, Value::number(1.0))
        .unwrap_err()
        .to_string();

    assert!(said.contains("embed"), "{said}");
    assert!(said.contains("number"), "{said}");
}

#[test]
fn and_so_is_an_answer_with_the_wrong_number_of_items() {
    let (g, c, _, memory) = node("miscounts", Miscounts).mapped().somatize().unwrap();
    let plan = compile(&g, &c).unwrap();

    let said = Executor::new(&c)
        .remembering(&memory)
        .run(&plan, documents(&[1.0, 2.0, 3.0]))
        .unwrap_err()
        .to_string();

    assert!(said.contains("3 items"), "{said}");
    assert!(said.contains("1"), "{said}");
}

#[test]
fn what_reads_a_mapped_node_is_named_after_the_whole_list() {
    // A node downstream is not mapped: it reads the list, all of it, so its name
    // has to depend on all of it. Change one item and it has to miss.
    let journal = Journal::new();
    let (g, c, _, memory) = (node("embed", EachOne(journal.clone()))
        .frozen()
        .cached()
        .mapped()
        >> node("head", Witness("head", journal.clone()))
            .frozen()
            .cached())
    .somatize()
    .unwrap();
    let plan = compile(&g, &c).unwrap();
    let notebook = Notebook::new();
    let run = |input: Value| {
        Executor::new(&c)
            .remembering(&memory)
            .keeping(&notebook)
            .run(&plan, input)
            .unwrap()
    };

    run(documents(&[1.0, 2.0]));
    let heads = journal.order().iter().filter(|who| *who == "head").count();
    run(documents(&[1.0, 2.0]));
    let same = journal.order().iter().filter(|who| *who == "head").count();
    run(documents(&[1.0, 3.0]));
    let changed = journal.order().iter().filter(|who| *who == "head").count();

    assert_eq!(heads, 1);
    assert_eq!(same, 1, "the same list: the head had its answer already");
    assert_eq!(changed, 2, "one item changed and the head has to run again");
}

// ── What only fed an answer that was already kept ──
//
// A name is knowable before anything runs — that is what `key_for` is for — so
// the engine can ask what it already has and then not compute what only fed one
// of those answers. The whole section is about the difference between *not
// keeping* something and *not needing* it.

/// An encoder under a head, both settled and only the head kept.
fn under_a_kept_head() -> (Arc<Journal>, Graph, Catalog, Memory) {
    let journal = Journal::new();
    let (g, c, _, memory) = (node("encoder", Witness("encoder", journal.clone())).frozen()
        >> node("head", Witness("head", journal.clone()))
            .frozen()
            .cached())
    .somatize()
    .unwrap();
    (journal, g, c, memory)
}

#[test]
fn what_only_fed_an_answer_that_was_kept_is_not_run() {
    // The expensive half of a graph is usually the half at the top: a settled
    // encoder, a dataset. Running it to throw its result away a microsecond
    // later is the cost this removes.
    let (journal, g, c, memory) = under_a_kept_head();
    let plan = compile(&g, &c).unwrap();
    let notebook = Notebook::new();

    for _ in 0..2 {
        Executor::new(&c)
            .remembering(&memory)
            .keeping(&notebook)
            .run(&plan, Value::number(7.0))
            .unwrap();
    }

    assert_eq!(journal.order(), ["encoder", "head"]);
}

#[test]
fn and_the_answer_is_still_the_answer() {
    let (_, g, c, memory) = under_a_kept_head();
    let plan = compile(&g, &c).unwrap();
    let notebook = Notebook::new();
    let asked = || {
        Executor::new(&c)
            .remembering(&memory)
            .keeping(&notebook)
            .run(&plan, Value::number(7.0))
            .unwrap()
    };

    assert_eq!(asked(), asked(), "the second run said something else");
}

#[test]
fn and_it_says_so_rather_than_leaving_a_hole_in_the_record() {
    // A node that is simply absent cannot be told from one that was never in
    // the graph, and *why is there no time for `encoder`* is a question whoever
    // reads a run will have.
    let (_, g, c, memory) = under_a_kept_head();
    let plan = compile(&g, &c).unwrap();
    let notebook = Notebook::new();
    let told = Told::new();
    for _ in 0..2 {
        Executor::new(&c)
            .remembering(&memory)
            .keeping(&notebook)
            .watching(&told)
            .run(&plan, Value::number(7.0))
            .unwrap();
    }

    assert!(told.kinds().contains(&"spared".to_string()));
    assert!(
        told.all().iter().any(|fact| matches!(
            fact,
            soma_next_core::Fact::Spared { node } if node.as_str() == "encoder"
        )),
        "it has to say which one",
    );
}

#[test]
fn but_what_somebody_else_still_reads_is_run() {
    // The rule is about **every** reader. One of them having its answer already
    // says nothing about the others, and skipping here would be wrong output
    // rather than a slow run.
    let journal = Journal::new();
    let mut g = Graph::new();
    let mut c = Catalog::new();
    let mut memory = Memory::new();
    for id in ["encoder", "head", "other"] {
        g.add_node(id).unwrap();
        c.insert(id, Arc::new(Witness(id, journal.clone())));
        memory.identify(id, "Witness");
        memory.freeze(id, Some("settled".into()));
    }
    g.add_edge("encoder", "head").unwrap();
    g.add_edge("encoder", "other").unwrap();
    memory.cache("head", None);

    let plan = compile(&g, &c).unwrap();
    let notebook = Notebook::new();
    for _ in 0..2 {
        Executor::new(&c)
            .remembering(&memory)
            .keeping(&notebook)
            .run(&plan, Value::number(7.0))
            .unwrap();
    }

    assert_eq!(
        journal
            .order()
            .iter()
            .filter(|who| *who == "encoder")
            .count(),
        2,
        "`other` reads it and `other` was never kept",
    );
}

#[test]
fn a_node_that_maps_keeps_everything_above_it() {
    // The one place this has to give up, and it gives up in the safe
    // direction: the names of a mapped node's answers are made out of the
    // **items**, so they are not knowable until it has them. It counts as a
    // miss and what feeds it stays.
    let journal = Journal::new();
    let mut g = Graph::new();
    let mut c = Catalog::new();
    let mut memory = Memory::new();
    g.add_node("encoder").unwrap();
    c.insert("encoder", Arc::new(Witness("encoder", journal.clone())));
    g.add_node("each").unwrap();
    c.insert("each", Arc::new(EachOne(journal.clone())));
    g.add_edge("encoder", "each").unwrap();
    for id in ["encoder", "each"] {
        memory.identify(id, "Witness");
        memory.freeze(id, Some("settled".into()));
    }
    memory.map("each");
    memory.cache("each", None);

    let plan = compile(&g, &c).unwrap();
    let notebook = Notebook::new();
    let items = Value::list(vec![Value::number(1.0), Value::number(2.0)]);
    for _ in 0..2 {
        Executor::new(&c)
            .remembering(&memory)
            .keeping(&notebook)
            .run(&plan, items.clone())
            .unwrap();
    }

    assert_eq!(
        journal
            .order()
            .iter()
            .filter(|who| *who == "encoder")
            .count(),
        2,
        "a mapped node cannot be foreseen, so what feeds it has to run",
    );
}

#[test]
fn and_a_slice_nobody_needs_is_not_sent_at_all() {
    // The saving is not the work over there, it is the **round trip**: the
    // client works out that nothing reads what comes back, so no message is
    // written at all.
    let (g, here, there) = both_sides(&[("a", 1.0), ("b", 10.0)], &[("a", "b")]);
    let placement = away(&["a"]);
    let mirror = Mirror::new(there);
    let plan = distribute(&compile(&g, &here).unwrap(), &placement);
    let mut memory = Memory::new();
    for id in ["a", "b"] {
        memory.identify(id, "Add");
        memory.freeze(id, Some("settled".into()));
    }
    memory.cache("b", None);
    let notebook = Notebook::new();

    for _ in 0..2 {
        Executor::new(&here)
            .placed(&placement)
            .reaching("there", &mirror)
            .remembering(&memory)
            .keeping(&notebook)
            .run(&plan, Value::number(0.0))
            .unwrap();
    }

    assert_eq!(mirror.trips().len(), 1, "the second run went nowhere");
}

// ── The names, asked for without a run ──
//
// `foreseen` is public because the answer is worth having on its own. Two
// versions of one graph name a node differently exactly when its recipe
// changed, so comparing two sets of names says what an edit did — before
// anybody pays to find out. What a run does with the answer is unchanged; this
// section is about asking for it alone.

/// A chain of three, all named by `somatize`, the middle one kept.
fn a_chain_kept_in_the_middle(salt: Option<&str>) -> (Graph, Catalog, Memory) {
    let (g, c, _, mut memory) =
        (node("a", Add(1.0)) >> node("b", Add(10.0)) >> node("c", Add(100.0)))
            .somatize()
            .unwrap();
    memory.cache("b", salt.map(str::to_string));
    (g, c, memory)
}

/// The one name of a node that maps nothing.
fn only_name(named: &std::collections::HashMap<NodeId, Keys>, id: &str) -> Key {
    match named.get(&NodeId::from(id)) {
        Some(Keys::One(key)) => key.clone(),
        other => panic!("`{id}` should have had one name, and had {other:?}"),
    }
}

#[test]
fn the_names_foreseen_are_the_names_things_are_kept_under() {
    // The whole contract in one assertion: what the cold pass says a node's
    // output will be called is what the run then calls it. A `foreseen` that
    // drifted from `key_for` would be a diff that quietly lies.
    let (g, c, memory) = a_chain_kept_in_the_middle(None);
    let plan = compile(&g, &c).unwrap();
    let notebook = Notebook::new();
    let executor = Executor::new(&c).remembering(&memory).keeping(&notebook);

    let (named, _) = executor.foreseen(&plan, &Value::number(0.0));
    executor.run(&plan, Value::number(0.0)).unwrap();

    assert_eq!(notebook.names(), vec![only_name(&named, "b")]);
}

#[test]
fn nothing_ran_to_find_out() {
    // A node that cannot run at all still has a name: the recipe is enough, and
    // that is what makes the question askable about a graph nobody can execute
    // here — no GPU, no dataset, no weights.
    let (g, c, _, mut memory) = (node("a", Add(1.0)) >> node("b", Panics))
        .somatize()
        .unwrap();
    memory.cache("b", None);
    let plan = compile(&g, &c).unwrap();
    let notebook = Notebook::new();

    let (named, _) = Executor::new(&c)
        .remembering(&memory)
        .keeping(&notebook)
        .foreseen(&plan, &Value::number(0.0));

    assert_eq!(named.len(), 2, "both were named and neither was asked");
}

#[test]
fn a_changed_recipe_renames_what_is_under_it_and_nothing_above() {
    // The property the whole thing rests on. A salt is the smallest change to a
    // recipe there is, and it has to reach every name below it and no name
    // above it: that asymmetry is what tells an edit that invalidated an
    // encoder from one that only touched the head.
    let (g, c, plain) = a_chain_kept_in_the_middle(None);
    let (_, _, salted) = a_chain_kept_in_the_middle(Some("a100-fp16"));
    let plan = compile(&g, &c).unwrap();
    let notebook = Notebook::new();
    let names = |memory: &Memory| {
        Executor::new(&c)
            .remembering(memory)
            .keeping(&notebook)
            .foreseen(&plan, &Value::number(0.0))
            .0
    };

    let (before, after) = (names(&plain), names(&salted));

    assert_eq!(
        only_name(&before, "a"),
        only_name(&after, "a"),
        "what is above the change keeps its name, and so its kept answer",
    );
    assert_ne!(only_name(&before, "b"), only_name(&after, "b"));
    assert_ne!(
        only_name(&before, "c"),
        only_name(&after, "c"),
        "a name is made of names, so the change reaches everything under it",
    );
}

#[test]
fn what_cannot_be_foreseen_is_missing_rather_than_wrong() {
    // A mapped node is named by the content of its items, which nobody has
    // yet. It is left out, and so is everything under it: whoever compares two
    // of these has to read the absence as "cannot tell". Saying "unchanged"
    // here would be the one answer that costs somebody a week.
    let (g, c, _, memory) = (node("a", Add(1.0)).mapped() >> node("b", Mean))
        .somatize()
        .unwrap();
    let plan = compile(&g, &c).unwrap();
    let notebook = Notebook::new();

    let (named, _) = Executor::new(&c)
        .remembering(&memory)
        .keeping(&notebook)
        .foreseen(&plan, &Value::list(vec![Value::number(1.0)]));

    assert!(!named.contains_key(&NodeId::from("a")));
    assert!(!named.contains_key(&NodeId::from("b")));
}

#[test]
fn without_a_keeper_nothing_is_named() {
    // The same silence a run gets, and for the same reason: the core has no
    // algorithm to hash with. A caller that forgot the store gets an empty
    // answer, not a wrong one.
    let (g, c, memory) = a_chain_kept_in_the_middle(None);
    let plan = compile(&g, &c).unwrap();

    let (named, unneeded) = Executor::new(&c)
        .remembering(&memory)
        .foreseen(&plan, &Value::number(0.0));

    assert!(named.is_empty() && unneeded.is_empty());
}
