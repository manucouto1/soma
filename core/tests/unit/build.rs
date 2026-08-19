//! The DSL: declaring the graph as an expression.

use crate::doubles::{Add, Immediate, Mean};
use soma_next_core::{Executor, GraphError, NodeId, Plan, Value, compile, node};

fn number(v: &Value) -> f64 {
    let Value::Number(x) = v else {
        panic!("expected a number, found {}", v.type_name());
    };
    *x
}

#[test]
fn a_chain() {
    let (g, c, _) = (node("a", Add(1.0)) >> node("b", Add(10.0)))
        .somatize()
        .unwrap();

    assert_eq!(g.len(), 2);
    assert_eq!(g.edges().len(), 1);
    let plan = compile(&g, &c).unwrap();
    assert_eq!(
        number(&Executor::new(&c).run(&plan, Value::number(0.0)).unwrap()),
        11.0
    );
}

#[test]
fn a_diamond_reads_at_a_glance() {
    let (g, c, _) = (node("source", Add(1.0))
        >> (node("left", Add(10.0)) | node("right", Add(100.0)))
        >> node("join", Mean))
    .somatize()
    .unwrap();

    assert_eq!(g.len(), 4);
    assert_eq!(g.edges().len(), 4);
    let plan = compile(&g, &c).unwrap();
    assert_eq!(
        number(&Executor::new(&c).run(&plan, Value::number(0.0)).unwrap()),
        56.0
    );
}

#[test]
fn branches_can_have_their_own_length() {
    let (g, _, _) = (node("source", Add(1.0))
        >> ((node("left", Add(1.0)) >> node("left2", Add(1.0))) | node("right", Add(1.0))))
    .somatize()
    .unwrap();

    assert_eq!(g.len(), 4);
    // source→left, left→left2, source→right
    assert_eq!(g.edges().len(), 3);
}

#[test]
fn a_node_that_asks_for_turns_and_one_that_does_not_mix_just_the_same() {
    let (g, c, _) = (node("add", Add(1.0)) >> node("echo", Immediate))
        .somatize()
        .unwrap();

    let plan = compile(&g, &c).unwrap();
    assert_eq!(
        number(&Executor::new(&c).run(&plan, Value::number(41.0)).unwrap()),
        42.0
    );
}

#[test]
fn a_repeated_id_is_caught_at_materialization() {
    let err = (node("a", Add(1.0)) >> node("a", Add(1.0)))
        .somatize()
        .unwrap_err();
    assert_eq!(err, GraphError::DuplicateNode("a".into()));
}

#[test]
fn the_failure_survives_whatever_you_glue_on_afterwards() {
    let broken = node("a", Add(1.0)) >> node("a", Add(1.0));
    let err = (broken >> node("b", Add(1.0))).somatize().unwrap_err();
    assert_eq!(err, GraphError::DuplicateNode("a".into()));
}

// ── The oracle: the plan reproduces the expression tree ──
//
// `>>` and `|` are serial and parallel composition, so the expression you write
// **is** a tree. `compile` does not receive it — the graph has only nodes and
// edges, and has to give the same thing when built in a loop with
// `node()`/`edge()`, which was decision 6 of CU5 — so it **recovers** it. These
// tests check that it recovers it whole: they are the oracle the whole
// decomposition comes from.

fn execute(id: &str, from: &[&str]) -> Plan {
    Plan::Execute {
        node: id.into(),
        from: from.iter().map(|f| NodeId::from(*f)).collect(),
    }
}

/// The plan of an expression, to compare against its tree.
macro_rules! plan_of {
    ($wire:expr) => {{
        let (g, c, _) = ($wire).somatize().unwrap();
        compile(&g, &c).unwrap()
    }};
}

#[test]
fn a_chain_is_a_sequence_and_nothing_more() {
    assert_eq!(
        plan_of!(node("a", Add(1.0)) >> node("b", Add(1.0)) >> node("c", Add(1.0))),
        Plan::Sequence(vec![
            execute("a", &[]),
            execute("b", &["a"]),
            execute("c", &["b"]),
        ])
    );
}

#[test]
fn a_lone_or_is_a_wave() {
    assert_eq!(
        plan_of!(node("a", Add(1.0)) | node("b", Add(1.0))),
        Plan::Wave(vec![execute("a", &[]), execute("b", &[])])
    );
}

#[test]
fn the_dsl_diamond_comes_out_exactly_as_written() {
    assert_eq!(
        plan_of!(
            node("s", Add(1.0))
                >> (node("l", Add(10.0)) | node("r", Add(100.0)))
                >> node("j", Mean)
        ),
        Plan::Sequence(vec![
            execute("s", &[]),
            Plan::Wave(vec![execute("l", &["s"]), execute("r", &["s"])]),
            execute("j", &["l", "r"]),
        ])
    );
}

#[test]
fn the_parentheses_of_a_long_branch_survive_the_graph() {
    // `a >> (b >> b2 >> b3 | c >> c2) >> d`, the case that forces a wave branch
    // to be a whole plan and not a lone step.
    assert_eq!(
        plan_of!(
            node("a", Add(1.0))
                >> ((node("b", Add(1.0)) >> node("b2", Add(1.0)) >> node("b3", Add(1.0)))
                    | (node("c", Add(1.0)) >> node("c2", Add(1.0))))
                >> node("d", Mean)
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
fn two_open_branches_followed_by_two_more() {
    // `(a >> a2 | b) >> (c | d)`: the counterexample that rules out cutting at a
    // "barrier node". The `a >> a2` branch has to come out in one piece.
    assert_eq!(
        plan_of!(
            ((node("a", Add(1.0)) >> node("a2", Add(1.0))) | node("b", Add(1.0)))
                >> (node("c", Mean) | node("d", Mean))
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
fn a_wave_inside_a_branch_of_another_wave() {
    // `a >> ((b >> (c | d) >> e) | f) >> g`: the tree nests as deep as the
    // parentheses do.
    assert_eq!(
        plan_of!(
            node("a", Add(1.0))
                >> ((node("b", Add(1.0))
                    >> (node("c", Add(1.0)) | node("d", Add(1.0)))
                    >> node("e", Mean))
                    | node("f", Add(1.0)))
                >> node("g", Mean)
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
fn three_branches_of_different_lengths() {
    assert_eq!(
        plan_of!(
            node("s", Add(1.0))
                >> (node("x", Add(1.0))
                    | (node("y", Add(1.0)) >> node("y2", Add(1.0)))
                    | (node("z", Add(1.0)) >> node("z2", Add(1.0)) >> node("z3", Add(1.0))))
                >> node("j", Mean)
        ),
        Plan::Sequence(vec![
            execute("s", &[]),
            Plan::Wave(vec![
                execute("x", &["s"]),
                Plan::Sequence(vec![execute("y", &["s"]), execute("y2", &["y"])]),
                Plan::Sequence(vec![
                    execute("z", &["s"]),
                    execute("z2", &["z"]),
                    execute("z3", &["z2"]),
                ]),
            ]),
            execute("j", &["x", "y2", "z3"]),
        ])
    );
}

#[test]
fn a_branch_that_opens_and_closes_inside_itself() {
    // The left branch is a whole diamond; the right one, a single node.
    assert_eq!(
        plan_of!(
            ((node("p", Add(1.0))
                >> (node("q", Add(1.0)) | node("r", Add(1.0)))
                >> node("s", Mean))
                | node("t", Add(1.0)))
        ),
        Plan::Wave(vec![
            Plan::Sequence(vec![
                execute("p", &[]),
                Plan::Wave(vec![execute("q", &["p"]), execute("r", &["p"])]),
                execute("s", &["q", "r"]),
            ]),
            execute("t", &[]),
        ])
    );
}
