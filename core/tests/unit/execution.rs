//! The engine, against Rust filters and steps: no Python in the way.

use crate::doubles::{
    Add, AlwaysNull, Ask, Cable, Fail, Immediate, Insatiable, Journal, Ledger, Mean, MeetingPoint,
    Mirror, Panics, Rendezvous, RendezvousDriver, Shout, Ubiquitous, Witness,
};
use soma_next_core::{
    Catalog, Ctx, Device, Executor, Graph, Host, Node, NodeError, NodeId, Outcome, Placement, Plan,
    RunError, Transition, Value, compile, distribute, node,
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

#[test]
fn a_step_that_finishes_on_the_first_turn_needs_no_driver() {
    let mut g = Graph::new();
    let mut c = Catalog::new();
    g.add_node("already").unwrap();
    c.insert("already", Arc::new(Immediate));

    let plan = compile(&g, &c).unwrap();
    let out = Executor::new(&c).run(&plan, Value::text("echo")).unwrap();
    assert_eq!(out, Value::text("echo"));
}

#[test]
fn a_step_asks_for_something_and_the_driver_gives_it() {
    let mut g = Graph::new();
    let mut c = Catalog::new();
    g.add_node("question").unwrap();
    c.insert("question", Arc::new(Ask(vec![Value::text("hello")])));

    let plan = compile(&g, &c).unwrap();
    let shout = Shout;
    let out = Executor::new(&c)
        .with_driver(&shout)
        .run(&plan, Value::Null)
        .unwrap();
    assert_eq!(out, Value::text("HELLO"));
}

#[test]
fn without_a_driver_a_step_that_asks_fails_saying_so() {
    let mut g = Graph::new();
    let mut c = Catalog::new();
    g.add_node("question").unwrap();
    c.insert("question", Arc::new(Ask(vec![Value::text("hello")])));

    let plan = compile(&g, &c).unwrap();
    let err = Executor::new(&c).run(&plan, Value::Null).unwrap_err();
    assert_eq!(err, RunError::NoDriver("question".into()));
    assert!(err.to_string().contains("no driver"));
}

#[test]
fn the_drivers_failure_is_attributed_to_the_step_that_asked() {
    let mut g = Graph::new();
    let mut c = Catalog::new();
    g.add_node("question").unwrap();
    // Shout only knows text; it is asked with a number.
    c.insert("question", Arc::new(Ask(vec![Value::number(1.0)])));

    let plan = compile(&g, &c).unwrap();
    let shout = Shout;
    let err = Executor::new(&c)
        .with_driver(&shout)
        .run(&plan, Value::Null)
        .unwrap_err();
    assert!(matches!(err, RunError::Driver { ref node, .. } if node.as_str() == "question"));
}

#[test]
fn a_step_that_cannot_stop_spends_its_turns_and_says_so() {
    let mut g = Graph::new();
    let mut c = Catalog::new();
    g.add_node("never").unwrap();
    c.insert("never", Arc::new(Insatiable));

    let plan = compile(&g, &c).unwrap();
    let driver = AlwaysNull;
    let err = Executor::new(&c)
        .with_driver(&driver)
        .run(&plan, Value::Null)
        .unwrap_err();
    assert!(matches!(err, RunError::TurnLimit { ref node, .. } if node.as_str() == "never"));
    assert!(err.to_string().contains("cannot stop"));
}

// ── Filters and steps in the same chain ──

#[test]
fn a_filter_and_a_step_chain_without_knowing_about_each_other() {
    let mut g = Graph::new();
    let mut c = Catalog::new();
    g.add_node("add").unwrap();
    g.add_node("echo").unwrap();
    g.add_edge("add", "echo").unwrap();
    c.insert("add", Arc::new(Add(1.0)));
    c.insert("echo", Arc::new(Immediate));

    let plan = compile(&g, &c).unwrap();
    let out = Executor::new(&c).run(&plan, Value::number(41.0)).unwrap();
    assert_eq!(number(&out), 42.0);
}

#[test]
fn a_node_can_fail_halfway_through_its_turns() {
    struct GivesUp;
    impl Node for GivesUp {
        fn forward(&self, _input: &Value, _ctx: &Ctx<'_>) -> Result<Transition, NodeError> {
            Err(NodeError::new("I cannot"))
        }
    }

    let mut g = Graph::new();
    let mut c = Catalog::new();
    g.add_node("gives_up").unwrap();
    c.insert("gives_up", Arc::new(GivesUp));

    let plan = compile(&g, &c).unwrap();
    let err = Executor::new(&c).run(&plan, Value::Null).unwrap_err();
    assert!(matches!(err, RunError::Node { ref node, .. } if node.as_str() == "gives_up"));
}

// ── What merging the two contracts makes possible ──

#[test]
fn a_node_can_evolve_from_always_finishing_to_asking_for_a_turn() {
    // With two traits this meant rewriting the type (error[E0119] if you tried
    // to have both). Here it is one more branch in the same body.
    struct Evolves;
    impl Node for Evolves {
        fn forward(&self, input: &Value, ctx: &Ctx<'_>) -> Result<Transition, NodeError> {
            if ctx.turn > 0 {
                // We already asked: the answer is what the driver brought.
                return Ok(Transition::Done(ctx.results[0].clone()));
            }
            match input {
                Value::Number(x) if *x < 0.0 => {
                    Ok(Transition::Await(vec![Value::text("negative")]))
                }
                other => Ok(Transition::Done(other.clone())),
            }
        }
    }

    let mut g = Graph::new();
    let mut c = Catalog::new();
    g.add_node("evolves").unwrap();
    c.insert("evolves", Arc::new(Evolves));
    let plan = compile(&g, &c).unwrap();

    // With positive input it asks for nothing, so it does not even need a driver.
    let out = Executor::new(&c).run(&plan, Value::number(1.0)).unwrap();
    assert_eq!(out, Value::number(1.0));

    // With negative input it asks for a turn, in the same node.
    let shout = Shout;
    let out = Executor::new(&c)
        .with_driver(&shout)
        .run(&plan, Value::number(-1.0))
        .unwrap();
    assert_eq!(out, Value::text("NEGATIVE"));
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
fn two_branches_can_keep_the_driver_busy_at_the_same_time() {
    // Where a wave wins beyond argument: two nodes waiting on something
    // outside. The driver does not serve the second until the first has
    // arrived, so if they were not concurrent the deadline would run out.
    let point = MeetingPoint::new();
    let mut g = Graph::new();
    let mut c = Catalog::new();
    for id in ["one", "other"] {
        g.add_node(id).unwrap();
        c.insert(id, Arc::new(Ask(vec![Value::text("?")])));
    }
    let plan = compile(&g, &c).unwrap();

    let driver = RendezvousDriver {
        point: Arc::clone(&point),
        how_many: 2,
    };
    let out = Executor::new(&c)
        .with_driver(&driver)
        .run(&plan, Value::Null)
        .unwrap();

    assert_eq!(
        out,
        Value::map(vec![
            ("one".to_string(), Value::text("served")),
            ("other".to_string(), Value::text("served")),
        ])
    );
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
    let (g, c, placement) = (node("a", Add(1.0)) >> node("b", Add(1.0)).at("worker1"))
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
    let (g, c, placement) = (node("a", Add(1.0)) >> node("b", Add(1.0)).at("worker1"))
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
    let (g, c, placement) = (node("a", Add(1.0)).at("w1") | node("b", Add(2.0)))
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
        .resume(&plan, Value::number(0.0), Vec::new())
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
            .resume(&Plan::Empty, Value::number(7.0), Vec::new())
            .unwrap(),
        Outcome {
            last: Value::number(7.0),
            produced: Vec::new(),
        }
    );
}
