//! Compiling: from the structure to the decided shape.
//!
//! Half concrete shapes — which tree comes out of which graph — and half
//! invariants that hold for any graph. The invariants are what would have caught
//! the bug that killed `Plan::Parallel`, hence a battery and not one case.

use crate::doubles::Add;
use somatize_core::{
    Catalog, CompileError, Destination, Device, Graph, Host, NodeId, Placement, Plan, compile,
    distribute, node,
};
use std::collections::HashSet;
use std::sync::Arc;

/// A graph with these nodes and these edges, all with an implementation.
fn graph_with(nodes: &[&str], edges: &[(&str, &str)]) -> (Graph, Catalog) {
    let mut g = Graph::new();
    let mut c = Catalog::new();
    for id in nodes {
        g.add_node(*id).unwrap();
        c.insert(*id, Arc::new(Add(1.0)));
    }
    for (from, to) in edges {
        g.add_edge(*from, *to).unwrap();
    }
    (g, c)
}

fn with_filters(ids: &[&str]) -> (Graph, Catalog) {
    graph_with(ids, &[])
}

fn execute(node: &str, from: &[&str]) -> Plan {
    Plan::Execute {
        node: node.into(),
        from: from.iter().map(|id| NodeId::from(*id)).collect(),
    }
}

fn plan_of(nodes: &[&str], edges: &[(&str, &str)]) -> Plan {
    let (g, c) = graph_with(nodes, edges);
    compile(&g, &c).unwrap()
}

#[test]
fn an_empty_graph_compiles_to_nothing() {
    assert_eq!(
        compile(&Graph::new(), &Catalog::new()).unwrap(),
        Plan::Empty
    );
}

#[test]
fn a_single_filter_is_not_wrapped_in_a_sequence() {
    let (g, c) = with_filters(&["a"]);
    assert_eq!(compile(&g, &c).unwrap(), execute("a", &[]));
}

#[test]
fn every_step_carries_where_its_input_comes_from() {
    assert_eq!(
        plan_of(&["a", "b", "c"], &[("a", "b"), ("b", "c")]),
        Plan::Sequence(vec![
            execute("a", &[]),
            execute("b", &["a"]),
            execute("c", &["b"]),
        ])
    );
}

#[test]
fn a_linear_chain_compiles_to_what_it_did_before_waves() {
    // The regression that matters most: everything closed from CU2 to CU8 is a
    // chain, and its plan has to come out identical, without a single wave.
    let plan = plan_of(&["a", "b", "c", "d"], &[("a", "b"), ("b", "c"), ("c", "d")]);
    assert_eq!(
        plan,
        Plan::Sequence(vec![
            execute("a", &[]),
            execute("b", &["a"]),
            execute("c", &["b"]),
            execute("d", &["c"]),
        ])
    );
}

#[test]
fn a_node_without_an_implementation_never_gets_to_execute() {
    let mut g = Graph::new();
    g.add_node("orphan").unwrap();
    assert_eq!(
        compile(&g, &Catalog::new()).unwrap_err(),
        CompileError::NoImplementation("orphan".into())
    );
}

#[test]
fn two_loose_nodes_are_a_wave_of_two_branches() {
    // With no edges at all, the graph is two components: `a | b`.
    assert_eq!(
        plan_of(&["a", "b"], &[]),
        Plan::Wave(vec![execute("a", &[]), execute("b", &[])])
    );
}

#[test]
fn opening_into_two_branches_puts_them_in_a_wave() {
    assert_eq!(
        plan_of(
            &["source", "left", "right"],
            &[("source", "left"), ("source", "right")]
        ),
        Plan::Sequence(vec![
            execute("source", &[]),
            Plan::Wave(vec![
                execute("left", &["source"]),
                execute("right", &["source"]),
            ]),
        ])
    );
}

#[test]
fn closing_two_branches_is_one_node_reading_from_two() {
    assert_eq!(
        plan_of(
            &["left", "right", "join"],
            &[("left", "join"), ("right", "join")]
        ),
        Plan::Sequence(vec![
            Plan::Wave(vec![execute("left", &[]), execute("right", &[])]),
            execute("join", &["left", "right"]),
        ])
    );
}

#[test]
fn a_diamond_executes_the_join_node_exactly_once() {
    // The case that broke `Plan::Parallel`: its branches overlapped and `join`
    // ended up in both. Connected components cannot, so it is emitted outside.
    assert_eq!(
        plan_of(
            &["source", "left", "right", "join"],
            &[
                ("source", "left"),
                ("source", "right"),
                ("left", "join"),
                ("right", "join"),
            ]
        ),
        Plan::Sequence(vec![
            execute("source", &[]),
            Plan::Wave(vec![
                execute("left", &["source"]),
                execute("right", &["source"]),
            ]),
            execute("join", &["left", "right"]),
        ])
    );
}

#[test]
fn a_branch_of_several_nodes_is_a_single_branch_of_the_wave() {
    // `a >> (b >> b2 >> b3 | c >> c2) >> d`, which rules out grouping by
    // topological level: `b2` does not wait on `c` and the branch fits a thread.
    assert_eq!(
        plan_of(
            &["a", "b", "b2", "b3", "c", "c2", "d"],
            &[
                ("a", "b"),
                ("b", "b2"),
                ("b2", "b3"),
                ("a", "c"),
                ("c", "c2"),
                ("b3", "d"),
                ("c2", "d"),
            ]
        ),
        Plan::Sequence(vec![
            execute("a", &[]),
            Plan::Wave(vec![
                Plan::Sequence(vec![
                    execute("b", &["a"]),
                    execute("b2", &["b"]),
                    execute("b3", &["b2"]),
                ]),
                Plan::Sequence(vec![execute("c", &["a"]), execute("c2", &["c"])]),
            ]),
            execute("d", &["b3", "c2"]),
        ])
    );
}

#[test]
fn the_series_cut_does_not_split_a_branch_down_the_middle() {
    // `(a >> a2 | b) >> (c | d)`. No node everything passes through, so a
    // "barrier node" would put `a` in a wave with `b` and leave `a2` loose. The
    // series cut looks at both whole ends and recovers `a >> a2` in one piece.
    assert_eq!(
        plan_of(
            &["a", "a2", "b", "c", "d"],
            &[
                ("a", "a2"),
                ("a2", "c"),
                ("a2", "d"),
                ("b", "c"),
                ("b", "d"),
            ]
        ),
        Plan::Sequence(vec![
            Plan::Wave(vec![
                Plan::Sequence(vec![execute("a", &[]), execute("a2", &["a"])]),
                execute("b", &[]),
            ]),
            Plan::Wave(vec![execute("c", &["a2", "b"]), execute("d", &["a2", "b"]),]),
        ])
    );
}

#[test]
fn two_consecutive_waves_do_not_merge_into_one() {
    // `(a | b) >> (c | d)`: four nodes independent pairwise, but `c` and `d`
    // depend on the first two. That is two waves, not one of four.
    assert_eq!(
        plan_of(
            &["a", "b", "c", "d"],
            &[("a", "c"), ("a", "d"), ("b", "c"), ("b", "d")]
        ),
        Plan::Sequence(vec![
            Plan::Wave(vec![execute("a", &[]), execute("b", &[])]),
            Plan::Wave(vec![execute("c", &["a", "b"]), execute("d", &["a", "b"])]),
        ])
    );
}

#[test]
fn a_wave_can_carry_another_inside() {
    // `a >> ((b >> (c | d) >> e) | f) >> g`: the long branch has its own fan.
    // The tree nests as deep as the expression.
    assert_eq!(
        plan_of(
            &["a", "b", "c", "d", "e", "f", "g"],
            &[
                ("a", "b"),
                ("b", "c"),
                ("b", "d"),
                ("c", "e"),
                ("d", "e"),
                ("a", "f"),
                ("e", "g"),
                ("f", "g"),
            ]
        ),
        Plan::Sequence(vec![
            execute("a", &[]),
            Plan::Wave(vec![
                Plan::Sequence(vec![
                    execute("b", &["a"]),
                    Plan::Wave(vec![execute("c", &["b"]), execute("d", &["b"])]),
                    execute("e", &["c", "d"]),
                ]),
                execute("f", &["a"]),
            ]),
            execute("g", &["e", "f"]),
        ])
    );
}

#[test]
fn three_branches_give_a_wave_of_three() {
    let Plan::Sequence(steps) = plan_of(
        &["source", "x", "y", "z"],
        &[("source", "x"), ("source", "y"), ("source", "z")],
    ) else {
        panic!("a fan compiles to a sequence");
    };
    let Plan::Wave(branches) = &steps[1] else {
        panic!("the second step is the wave");
    };
    assert_eq!(branches.len(), 3);
}

#[test]
fn two_unrelated_graphs_are_two_branches_even_if_each_is_long() {
    assert_eq!(
        plan_of(&["a", "a2", "b", "b2"], &[("a", "a2"), ("b", "b2")]),
        Plan::Wave(vec![
            Plan::Sequence(vec![execute("a", &[]), execute("a2", &["a"])]),
            Plan::Sequence(vec![execute("b", &[]), execute("b2", &["b"])]),
        ])
    );
}

#[test]
fn the_n_has_no_tree_and_is_walked_in_sequence() {
    // `a→c, a→d, b→d` is the minimal pattern that is not series-parallel
    // (Valdes, Tarjan and Lawler, 1982): no cut and no components, so it is
    // walked in a row. It cannot be written with the DSL.
    assert_eq!(
        plan_of(&["a", "b", "c", "d"], &[("a", "c"), ("a", "d"), ("b", "d")]),
        Plan::Sequence(vec![
            execute("a", &[]),
            execute("b", &[]),
            execute("c", &["a"]),
            execute("d", &["a", "b"]),
        ])
    );
}

#[test]
fn an_n_does_not_spoil_the_parallelism_of_what_sits_beside_it() {
    // The N hangs off its own roots and does not touch `other`, which is a
    // separate component: the healthy part is still a branch of the wave.
    let Plan::Wave(branches) = plan_of(
        &["a", "b", "c", "d", "other"],
        &[("a", "c"), ("a", "d"), ("b", "d")],
    ) else {
        panic!("the N and the loose node are two components");
    };
    assert_eq!(
        branches.len(),
        2,
        "the whole N is one branch, `other` is the other"
    );
    assert_eq!(branches[1], execute("other", &[]));
}

/// One topology from the battery. It is named so the failure says which it was.
struct Topology {
    name: &'static str,
    nodes: Vec<&'static str>,
    edges: Vec<(&'static str, &'static str)>,
}

fn topology(
    name: &'static str,
    nodes: Vec<&'static str>,
    edges: Vec<(&'static str, &'static str)>,
) -> Topology {
    Topology { name, nodes, edges }
}

/// Every interesting topology, so the invariants can be run against them all.
fn battery() -> Vec<Topology> {
    vec![
        topology("one node", vec!["a"], vec![]),
        topology("chain", vec!["a", "b", "c"], vec![("a", "b"), ("b", "c")]),
        topology("loose", vec!["a", "b", "c"], vec![]),
        topology(
            "diamond",
            vec!["s", "l", "r", "j"],
            vec![("s", "l"), ("s", "r"), ("l", "j"), ("r", "j")],
        ),
        topology(
            "long branches",
            vec!["a", "b", "b2", "b3", "c", "c2", "d"],
            vec![
                ("a", "b"),
                ("b", "b2"),
                ("b2", "b3"),
                ("a", "c"),
                ("c", "c2"),
                ("b3", "d"),
                ("c2", "d"),
            ],
        ),
        topology(
            "uneven branches with no single join",
            vec!["a", "a2", "b", "c", "d"],
            vec![
                ("a", "a2"),
                ("a2", "c"),
                ("a2", "d"),
                ("b", "c"),
                ("b", "d"),
            ],
        ),
        topology(
            "nested wave",
            vec!["a", "b", "c", "d", "e", "f", "g"],
            vec![
                ("a", "b"),
                ("b", "c"),
                ("b", "d"),
                ("c", "e"),
                ("d", "e"),
                ("a", "f"),
                ("e", "g"),
                ("f", "g"),
            ],
        ),
        topology(
            "the N",
            vec!["a", "b", "c", "d"],
            vec![("a", "c"), ("a", "d"), ("b", "d")],
        ),
        topology(
            "N with a healthy neighbour",
            vec!["a", "b", "c", "d", "other", "other2"],
            vec![("a", "c"), ("a", "d"), ("b", "d"), ("other", "other2")],
        ),
        topology(
            "fan of three that rejoins",
            vec!["s", "x", "y", "z", "j"],
            vec![
                ("s", "x"),
                ("s", "y"),
                ("s", "z"),
                ("x", "j"),
                ("y", "j"),
                ("z", "j"),
            ],
        ),
    ]
}

/// The plan's steps, in the order they would execute. A wave's branches are
/// flattened one after another: they are independent, so any interleaving does.
fn steps(plan: &Plan) -> Vec<(NodeId, Vec<NodeId>)> {
    // The core walks its own plan now; this is here so the invariants below read
    // as they did, and so a change to that walk is felt from outside the crate.
    plan.steps()
        .map(|step| (step.node.clone(), step.from.to_vec()))
        .collect()
}

#[test]
fn no_node_executes_twice_or_is_left_out() {
    // The invariant `Plan::Parallel` broke: on a diamond it executed the join
    // node twice because its branches overlapped.
    for Topology { name, nodes, edges } in battery() {
        let (g, c) = graph_with(&nodes, &edges);
        let executed: Vec<NodeId> = steps(&compile(&g, &c).unwrap())
            .into_iter()
            .map(|(node, _)| node)
            .collect();

        let unique: HashSet<&NodeId> = executed.iter().collect();
        assert_eq!(
            unique.len(),
            executed.len(),
            "`{name}` executes some node twice: {executed:?}"
        );
        assert_eq!(
            unique.len(),
            g.len(),
            "`{name}` leaves some node of the graph unexecuted"
        );
    }
}

#[test]
fn the_order_the_plan_dictates_respects_the_edges() {
    // That a node does not execute before its predecessors is what makes its
    // input exist by the time it goes looking for it.
    for Topology { name, nodes, edges } in battery() {
        let (g, c) = graph_with(&nodes, &edges);
        let order: Vec<NodeId> = steps(&compile(&g, &c).unwrap())
            .into_iter()
            .map(|(node, _)| node)
            .collect();

        for (i, node) in order.iter().enumerate() {
            for pred in g.predecessors(node) {
                let before = order
                    .iter()
                    .position(|n| n == pred)
                    .expect("it is in the plan");
                assert!(
                    before < i,
                    "`{name}`: {node} executes before its predecessor {pred}"
                );
            }
        }
    }
}

#[test]
fn every_step_declares_exactly_its_predecessors_in_the_graph() {
    for Topology { name, nodes, edges } in battery() {
        let (g, c) = graph_with(&nodes, &edges);
        for (node, from) in steps(&compile(&g, &c).unwrap()) {
            let expected: Vec<NodeId> = g.predecessors(&node).into_iter().cloned().collect();
            assert_eq!(from, expected, "`{name}`: wrong `from` for {node}");
        }
    }
}

#[test]
fn the_branches_of_a_wave_share_no_node() {
    // What makes merging what each branch produced unable to clobber anything,
    // and it is free from the branches being connected components.
    fn check(plan: &Plan, name: &str) {
        match plan {
            Plan::Empty | Plan::Execute { .. } => {}
            Plan::Remote { inner, .. } => check(inner, name),
            Plan::Sequence(plans) => plans.iter().for_each(|p| check(p, name)),
            Plan::Wave(branches) => {
                let mut seen: HashSet<NodeId> = HashSet::new();
                for branch in branches {
                    for (node, _) in steps(branch) {
                        assert!(
                            seen.insert(node.clone()),
                            "`{name}`: {node} is in two branches of the same wave"
                        );
                    }
                    check(branch, name);
                }
            }
        }
    }
    for Topology { name, nodes, edges } in battery() {
        let (g, c) = graph_with(&nodes, &edges);
        check(&compile(&g, &c).unwrap(), name);
    }
}

#[test]
fn the_same_graph_always_compiles_the_same() {
    // Without this, `plan()` would be useless and so would whatever cache comes.
    for Topology { name, nodes, edges } in battery() {
        let (g, c) = graph_with(&nodes, &edges);
        let first = compile(&g, &c).unwrap();
        for _ in 0..5 {
            assert_eq!(compile(&g, &c).unwrap(), first, "`{name}` is not stable");
        }
    }
}

fn placed(hosts: &[(&str, &str)]) -> Placement {
    let mut placement = Placement::new();
    for (id, host) in hosts {
        placement.place_at(*id, Host::new(*host));
    }
    placement
}

fn distributed(nodes: &[&str], edges: &[(&str, &str)], hosts: &[(&str, &str)]) -> Plan {
    distribute(&plan_of(nodes, edges), &placed(hosts))
}

/// A trip to `host` with this inside.
fn at(host: &str, inner: Plan) -> Plan {
    Plan::Remote {
        host: Host::new(host),
        inner: Box::new(inner),
    }
}

#[test]
fn without_any_host_the_plan_comes_out_identical() {
    // What separating `compile` from `distribute` buys: with nobody placing a
    // host, the second step does not exist. Over the battery, not one case.
    for Topology { name, nodes, edges } in battery() {
        let (g, c) = graph_with(&nodes, &edges);
        let plan = compile(&g, &c).unwrap();
        assert_eq!(
            distribute(&plan, &Placement::new()),
            plan,
            "`{name}` changes shape without anyone having sent it anywhere"
        );
    }
}

#[test]
fn placing_devices_only_distributes_nothing_either() {
    // The CU10 invariant is still alive: a device is inert as far as the
    // traversal goes. Only a host moves anything.
    let (g, c, placement, _) = (node("a", Add(1.0)).on(Device::Cuda(0))
        >> node("b", Add(1.0)).on(Device::Cpu))
    .somatize()
    .unwrap();
    let plan = compile(&g, &c).unwrap();

    assert_eq!(distribute(&plan, &placement), plan);
}

#[test]
fn a_whole_chain_on_one_host_is_a_single_trip() {
    // The most that can be grouped, which is where the benefit comes from.
    assert_eq!(
        distributed(
            &["a", "b", "c"],
            &[("a", "b"), ("b", "c")],
            &[("a", "w1"), ("b", "w1"), ("c", "w1")],
        ),
        at(
            "w1",
            Plan::Sequence(vec![
                execute("a", &[]),
                execute("b", &["a"]),
                execute("c", &["b"]),
            ])
        )
    );
}

#[test]
fn a_consecutive_run_is_not_split_into_one_trip_per_node() {
    // The one that catches the bug: `decompose` leaves sequences flat, so
    // wrapping child by child would give three `Remote`s for one trip.
    assert_eq!(
        distributed(
            &["a", "b", "c", "d"],
            &[("a", "b"), ("b", "c"), ("c", "d")],
            &[("b", "w1"), ("c", "w1"), ("d", "w1")],
        ),
        Plan::Sequence(vec![
            execute("a", &[]),
            at(
                "w1",
                Plan::Sequence(vec![
                    execute("b", &["a"]),
                    execute("c", &["b"]),
                    execute("d", &["c"]),
                ])
            ),
        ])
    );
}

#[test]
fn you_come_back_from_a_host_and_carry_on_here() {
    assert_eq!(
        distributed(&["a", "b", "c"], &[("a", "b"), ("b", "c")], &[("b", "w1")],),
        Plan::Sequence(vec![
            execute("a", &[]),
            at("w1", execute("b", &["a"])),
            execute("c", &["b"]),
        ])
    );
}

#[test]
fn two_different_hosts_are_not_merged() {
    assert_eq!(
        distributed(&["a", "b"], &[("a", "b")], &[("a", "w1"), ("b", "w2")],),
        Plan::Sequence(vec![
            at("w1", execute("a", &[])),
            at("w2", execute("b", &["a"])),
        ])
    );
}

#[test]
fn a_run_of_one_is_not_wrapped_in_a_sequence_of_one() {
    // The shape cannot depend on how you arrived at it.
    let one = distributed(&["a", "b"], &[("a", "b")], &[("b", "w1")]);
    let Plan::Sequence(steps) = &one else {
        panic!("a sequence was expected: {one:?}")
    };
    assert_eq!(steps[1], at("w1", execute("b", &["a"])));
}

#[test]
fn each_branch_of_a_wave_travels_to_its_own_host() {
    assert_eq!(
        distributed(
            &["s", "l", "r"],
            &[("s", "l"), ("s", "r")],
            &[("l", "w1"), ("r", "w2")],
        ),
        Plan::Sequence(vec![
            execute("s", &[]),
            Plan::Wave(vec![
                at("w1", execute("l", &["s"])),
                at("w2", execute("r", &["s"])),
            ]),
        ])
    );
}

#[test]
fn a_whole_wave_on_one_host_travels_once_with_the_wave_inside() {
    // The `Remote` goes on the outside and the `Wave` on the inside, not the
    // other way round: the concurrency happens **there**, where the nodes are.
    assert_eq!(
        distributed(
            &["s", "l", "r"],
            &[("s", "l"), ("s", "r")],
            &[("l", "w1"), ("r", "w1")],
        ),
        Plan::Sequence(vec![
            execute("s", &[]),
            at(
                "w1",
                Plan::Wave(vec![execute("l", &["s"]), execute("r", &["s"])])
            ),
        ])
    );
}

#[test]
fn a_whole_long_branch_on_one_host_is_one_trip() {
    // `a >> (b >> b2 | c) >> d`, with the long branch away.
    let plan = distributed(
        &["a", "b", "b2", "c", "d"],
        &[("a", "b"), ("b", "b2"), ("a", "c"), ("b2", "d"), ("c", "d")],
        &[("b", "w1"), ("b2", "w1")],
    );

    assert_eq!(
        plan,
        Plan::Sequence(vec![
            execute("a", &[]),
            Plan::Wave(vec![
                at(
                    "w1",
                    Plan::Sequence(vec![execute("b", &["a"]), execute("b2", &["b"])])
                ),
                execute("c", &["a"]),
            ]),
            execute("d", &["b2", "c"]),
        ])
    );
}

#[test]
fn distributing_twice_gives_the_same_thing() {
    // Idempotent, and over the whole battery with half a placement applied:
    // going through twice cannot wrap twice.
    for Topology { name, nodes, edges } in battery() {
        let hosts: Vec<(&str, &str)> = nodes
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 2 == 0)
            .map(|(_, id)| (*id, "w1"))
            .collect();
        let placement = placed(&hosts);
        let plan = distribute(&plan_of(&nodes, &edges), &placement);

        assert_eq!(
            distribute(&plan, &placement),
            plan,
            "`{name}` gets wrapped twice"
        );
    }
}

#[test]
fn placing_a_node_that_is_not_there_distributes_nothing() {
    // A `Placement` is a bare map: it does not check that the ids exist, and
    // naming a stranger cannot change the plan of those that are there.
    let plan = plan_of(&["a", "b"], &[("a", "b")]);
    assert_eq!(distribute(&plan, &placed(&[("ghost", "w1")])), plan);
}

#[test]
fn an_empty_plan_travels_nowhere() {
    assert_eq!(
        distribute(&Plan::Empty, &placed(&[("a", "w1")])),
        Plan::Empty
    );
}

#[test]
fn the_same_plan_and_the_same_placement_always_distribute_the_same() {
    let placement = placed(&[("b", "w1"), ("c", "w1")]);
    let plan = plan_of(&["a", "b", "c"], &[("a", "b"), ("b", "c")]);
    let first = distribute(&plan, &placement);
    for _ in 0..5 {
        assert_eq!(distribute(&plan, &placement), first);
    }
}

#[test]
fn the_steps_come_out_in_the_order_they_were_declared() {
    // Observable: it is the order a sequence runs in and the order a wave's
    // branches were written. A walk that reversed it would pass every count.
    let plan = plan_of(&["a", "b", "c"], &[("a", "b"), ("b", "c")]);

    assert_eq!(
        plan.steps()
            .map(|step| step.node.as_str())
            .collect::<Vec<_>>(),
        vec!["a", "b", "c"]
    );
}

#[test]
fn a_step_carries_what_it_reads_and_not_only_who_it_is() {
    let plan = plan_of(&["a", "b", "mean"], &[("a", "mean"), ("b", "mean")]);
    let join = plan
        .steps()
        .find(|step| step.node.as_str() == "mean")
        .expect("the join is in there");

    assert_eq!(
        join.from.iter().map(NodeId::as_str).collect::<Vec<_>>(),
        vec!["a", "b"]
    );
}

#[test]
fn every_node_of_the_graph_is_a_step_of_its_plan_exactly_once() {
    for Topology { name, nodes, edges } in battery() {
        let plan = plan_of(&nodes, &edges);
        let walked: Vec<&NodeId> = plan.steps().map(|step| step.node).collect();

        assert_eq!(walked.len(), nodes.len(), "{name}");
        assert_eq!(
            walked.iter().collect::<HashSet<_>>().len(),
            nodes.len(),
            "{name}"
        );
    }
}

#[test]
fn the_steps_of_a_distributed_plan_are_the_same_steps() {
    // What a plan **does** does not depend on where it does it, which is why
    // this walk enters a `Remote` and the other one does not.
    let plan = plan_of(&["a", "b"], &[("a", "b")]);
    let mut placement = Placement::new();
    placement.place_at(NodeId::from("b"), Host::from("gpu"));
    let spread = distribute(&plan, &placement);

    assert_ne!(spread, plan, "it did travel");
    assert_eq!(
        spread.steps().collect::<Vec<_>>(),
        plan.steps().collect::<Vec<_>>()
    );
}

#[test]
fn nothing_to_do_walks_to_nothing() {
    assert_eq!(Plan::Empty.steps().count(), 0);
    assert_eq!(Plan::Empty.destinations().count(), 0);
}

#[test]
fn destinations_stop_where_a_slice_already_says_where_it_goes() {
    // The one line the two walks differ in, and what makes `distribute`
    // idempotent: a plan that has already travelled is not opened up again.
    let plan = plan_of(&["a", "b"], &[("a", "b")]);
    let mut placement = Placement::new();
    placement.place_at(NodeId::from("b"), Host::from("gpu"));
    let spread = distribute(&plan, &placement);

    let seen: Vec<Destination<'_>> = spread.destinations().collect();
    assert_eq!(seen.len(), 2, "{spread:?}");
    assert!(matches!(seen[0], Destination::Node(node) if node.as_str() == "a"));
    assert!(matches!(seen[1], Destination::Away(host) if host.to_string() == "gpu"));
}

#[test]
fn a_plan_that_has_not_travelled_has_a_destination_per_node() {
    let plan = plan_of(&["a", "b", "c"], &[("a", "b"), ("b", "c")]);

    assert!(
        plan.destinations()
            .all(|destination| matches!(destination, Destination::Node(_)))
    );
    assert_eq!(plan.destinations().count(), 3);
}

#[test]
fn a_device_is_not_a_destination_because_it_is_not_a_place_to_send_to() {
    // `.on("cuda:0")` says where inside a machine, `.at("gpu")` says which
    // machine. Only the second decides what travels.
    let plan = plan_of(&["a"], &[]);
    let mut placement = Placement::new();
    placement.place(NodeId::from("a"), Device::Cuda(0));

    assert_eq!(distribute(&plan, &placement), plan);
    assert!(matches!(
        plan.destinations().next(),
        Some(Destination::Node(_))
    ));
}
