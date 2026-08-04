//! Control-flow compilation: `Loop` bodies and `Branch` arms.
//!
//! These paths had zero coverage. The invariant they enforce is that a node
//! consumed as a loop body or a branch arm is compiled *once* — as part of the
//! enclosing construct — and never again as a top-level step. Without it the
//! loop body runs an extra time after the loop finishes, and every branch arm
//! runs unconditionally after the branch has already selected one.

use somatize_compiler::plan::ExecutionPlan;
use somatize_compiler::{CompileMode, SimpleNodeRegistry, compile};
use somatize_core::cache::CacheKey;
use somatize_core::control::LoopCondition;
use somatize_core::filter::{Distribution, FilterKind, FilterMeta, StreamMode};
use somatize_core::graph::{Edge, Graph, Node};

fn meta(name: &str) -> FilterMeta {
    FilterMeta {
        name: name.into(),
        kind: FilterKind::Stateless,
        cacheable: true,
        differentiable: false,
        deterministic: true,
        stream_mode: StreamMode::FixedState,
        distribution: Distribution::Local,
        input_schema: None,
        output_schema: None,
    }
}

fn registry(node_ids: &[&str]) -> SimpleNodeRegistry {
    let mut reg = SimpleNodeRegistry::new();
    for id in node_ids {
        reg.register_meta(*id, meta(id), CacheKey::from_parts(&[id.as_bytes()]));
    }
    reg
}

/// How many times does `node_id` appear as an `Execute` anywhere in the plan?
fn execute_count(plan: &ExecutionPlan, node_id: &str) -> usize {
    match plan {
        ExecutionPlan::Execute { node_id: n } => usize::from(n == node_id),
        ExecutionPlan::Sequence(steps) | ExecutionPlan::Parallel(steps) => {
            steps.iter().map(|s| execute_count(s, node_id)).sum()
        }
        ExecutionPlan::Loop { body, .. } => execute_count(body, node_id),
        ExecutionPlan::Branch { arms, .. } => {
            arms.iter().map(|(_, p)| execute_count(p, node_id)).sum()
        }
        ExecutionPlan::Remote { plan, .. } => execute_count(plan, node_id),
        ExecutionPlan::Composite { node_ids } | ExecutionPlan::Stream { node_ids, .. } => {
            node_ids.iter().filter(|n| *n == node_id).count()
        }
        ExecutionPlan::Empty => 0,
        _ => 0,
    }
}

/// Is there a `Loop` anywhere in the plan?
fn find_loop(plan: &ExecutionPlan) -> Option<&ExecutionPlan> {
    match plan {
        ExecutionPlan::Loop { .. } => Some(plan),
        ExecutionPlan::Sequence(steps) | ExecutionPlan::Parallel(steps) => {
            steps.iter().find_map(find_loop)
        }
        ExecutionPlan::Branch { arms, .. } => arms.iter().find_map(|(_, p)| find_loop(p)),
        ExecutionPlan::Remote { plan, .. } => find_loop(plan),
        _ => None,
    }
}

fn find_branch(plan: &ExecutionPlan) -> Option<&ExecutionPlan> {
    match plan {
        ExecutionPlan::Branch { .. } => Some(plan),
        ExecutionPlan::Sequence(steps) | ExecutionPlan::Parallel(steps) => {
            steps.iter().find_map(find_branch)
        }
        ExecutionPlan::Loop { body, .. } => find_branch(body),
        ExecutionPlan::Remote { plan, .. } => find_branch(plan),
        _ => None,
    }
}

// ── Loop ──

/// `loop_node -> body` must compile to a single `Loop` whose body is `body`,
/// with `body` appearing exactly once in the whole plan.
#[test]
fn loop_body_is_compiled_exactly_once() {
    let mut g = Graph::new();
    g.add_node(Node::loop_node("refine", Some(5)));
    g.add_node(Node::filter_with_id("body", "body"));
    g.add_edge(Edge::control("e1", "refine", "body"));

    let reg = registry(&["body"]);
    let result = compile(&g, &reg, CompileMode::Inference, None).expect("compiles");

    assert_eq!(
        execute_count(&result.plan, "body"),
        1,
        "loop body compiled more than once — it would run again after the loop:\n{}",
        result.plan
    );

    let lp = find_loop(&result.plan).expect("plan contains a Loop");
    let ExecutionPlan::Loop {
        body,
        max_iterations,
        ..
    } = lp
    else {
        unreachable!()
    };
    assert_eq!(*max_iterations, Some(5));
    assert_eq!(execute_count(body, "body"), 1);
}

/// A loop whose body is a chain (`refine -> a -> b`) must contain both nodes
/// inside the loop, each exactly once.
#[test]
fn loop_body_chain_is_fully_contained() {
    let mut g = Graph::new();
    g.add_node(Node::loop_node("refine", Some(3)));
    g.add_node(Node::filter_with_id("a", "a"));
    g.add_node(Node::filter_with_id("b", "b"));
    g.add_edge(Edge::control("e1", "refine", "a"));
    g.add_edge(Edge::data("e2", "a", "b"));

    let reg = registry(&["a", "b"]);
    let result = compile(&g, &reg, CompileMode::Inference, None).expect("compiles");

    assert_eq!(execute_count(&result.plan, "a"), 1, "\n{}", result.plan);
    assert_eq!(execute_count(&result.plan, "b"), 1, "\n{}", result.plan);

    let lp = find_loop(&result.plan).expect("plan contains a Loop");
    let ExecutionPlan::Loop { body, .. } = lp else {
        unreachable!()
    };
    assert_eq!(execute_count(body, "a"), 1, "`a` must be inside the loop");
    assert_eq!(execute_count(body, "b"), 1, "`b` must be inside the loop");
}

/// Nodes downstream of the loop are not part of the body: `refine -> body`,
/// `refine -> after` where `after` is reached by a *data* edge from the loop.
#[test]
fn nodes_after_the_loop_stay_outside_it() {
    let mut g = Graph::new();
    g.add_node(Node::filter_with_id("prep", "prep"));
    g.add_node(Node::loop_node("refine", Some(2)));
    g.add_node(Node::filter_with_id("body", "body"));
    g.add_edge(Edge::data("e0", "prep", "refine"));
    g.add_edge(Edge::control("e1", "refine", "body"));

    let reg = registry(&["prep", "body"]);
    let result = compile(&g, &reg, CompileMode::Inference, None).expect("compiles");

    assert_eq!(execute_count(&result.plan, "prep"), 1);
    assert_eq!(execute_count(&result.plan, "body"), 1, "\n{}", result.plan);

    let lp = find_loop(&result.plan).expect("plan contains a Loop");
    let ExecutionPlan::Loop { body, .. } = lp else {
        unreachable!()
    };
    assert_eq!(
        execute_count(body, "prep"),
        0,
        "`prep` is upstream of the loop, not its body"
    );
}

// ── Loop stop conditions ──

/// A single-terminal body resolves `BodyTerminal` to that node, so the
/// executor never has to guess which output decides.
#[test]
fn body_terminal_resolves_to_the_single_terminal_node() {
    let mut g = Graph::new();
    g.add_node(Node::loop_node("refine", Some(4)));
    g.add_node(Node::filter_with_id("draft", "draft"));
    g.add_node(Node::filter_with_id("judge", "judge"));
    g.add_edge(Edge::control("e1", "refine", "draft"));
    g.add_edge(Edge::data("e2", "draft", "judge"));

    let reg = registry(&["draft", "judge"]);
    let result = compile(&g, &reg, CompileMode::Inference, None).expect("compiles");

    let ExecutionPlan::Loop { until, .. } = find_loop(&result.plan).expect("a Loop") else {
        unreachable!()
    };
    assert_eq!(
        *until,
        LoopCondition::WhenSignaled("judge".into()),
        "the terminal node decides, not whoever ran last"
    );
}

/// A fan-out body has two terminals, so there is no single node whose output
/// means "stop". Guessing here is what made the old executor's
/// `execution_order.last()` non-deterministic under a parallel body.
#[test]
fn ambiguous_body_terminal_is_a_compile_error() {
    let mut g = Graph::new();
    g.add_node(Node::loop_node("refine", Some(4)));
    g.add_node(Node::filter_with_id("head", "head"));
    g.add_node(Node::filter_with_id("left", "left"));
    g.add_node(Node::filter_with_id("right", "right"));
    g.add_edge(Edge::control("e1", "refine", "head"));
    g.add_edge(Edge::data("e2", "head", "left"));
    g.add_edge(Edge::data("e3", "head", "right"));

    let reg = registry(&["head", "left", "right"]);
    let err = compile(&g, &reg, CompileMode::Inference, None)
        .expect_err("two terminals must not silently pick one");

    let msg = err.to_string();
    assert!(msg.contains("refine"), "should name the loop: {msg}");
    assert!(
        msg.contains("WhenSignaled") || msg.contains("Exhaust"),
        "should say how to fix it: {msg}"
    );
}

/// The ambiguity is resolvable by naming the deciding node.
#[test]
fn explicit_condition_resolves_ambiguity() {
    let mut g = Graph::new();
    g.add_node(Node::loop_until(
        "refine",
        Some(4),
        LoopCondition::WhenSignaled("left".into()),
    ));
    g.add_node(Node::filter_with_id("head", "head"));
    g.add_node(Node::filter_with_id("left", "left"));
    g.add_node(Node::filter_with_id("right", "right"));
    g.add_edge(Edge::control("e1", "refine", "head"));
    g.add_edge(Edge::data("e2", "head", "left"));
    g.add_edge(Edge::data("e3", "head", "right"));

    let reg = registry(&["head", "left", "right"]);
    let result = compile(&g, &reg, CompileMode::Inference, None).expect("compiles");

    let ExecutionPlan::Loop { until, .. } = find_loop(&result.plan).expect("a Loop") else {
        unreachable!()
    };
    assert_eq!(*until, LoopCondition::WhenSignaled("left".into()));
}

/// `carry_from` is populated from the body's single terminal — what one
/// iteration produced is what the next one reads. The executor applies it,
/// but only the compiler writes it: dropping these three lines would
/// compile a refine loop that redrafts from the original input every pass,
/// and until now nothing asserted they exist.
#[test]
fn the_compiler_carries_the_body_terminal_back() {
    let mut g = Graph::new();
    g.add_node(Node::loop_node("refine", Some(4)));
    g.add_node(Node::filter_with_id("draft", "draft"));
    g.add_node(Node::filter_with_id("judge", "judge"));
    g.add_edge(Edge::control("e1", "refine", "draft"));
    g.add_edge(Edge::data("e2", "draft", "judge"));

    let reg = registry(&["draft", "judge"]);
    let result = compile(&g, &reg, CompileMode::Inference, None).expect("compiles");

    let ExecutionPlan::Loop { carry_from, .. } = find_loop(&result.plan).expect("a Loop") else {
        unreachable!()
    };
    assert_eq!(
        carry_from.as_deref(),
        Some("judge"),
        "the single terminal is what the next iteration must read"
    );
}

/// Two terminals mean there is no single thing to hand to the next pass;
/// the compiler must leave `carry_from` empty rather than pick one, even
/// when an explicit stop condition makes the loop itself compilable.
#[test]
fn an_ambiguous_body_carries_nothing() {
    let mut g = Graph::new();
    g.add_node(Node::loop_until(
        "refine",
        Some(4),
        LoopCondition::WhenSignaled("left".into()),
    ));
    g.add_node(Node::filter_with_id("head", "head"));
    g.add_node(Node::filter_with_id("left", "left"));
    g.add_node(Node::filter_with_id("right", "right"));
    g.add_edge(Edge::control("e1", "refine", "head"));
    g.add_edge(Edge::data("e2", "head", "left"));
    g.add_edge(Edge::data("e3", "head", "right"));

    let reg = registry(&["head", "left", "right"]);
    let result = compile(&g, &reg, CompileMode::Inference, None).expect("compiles");

    let ExecutionPlan::Loop { carry_from, .. } = find_loop(&result.plan).expect("a Loop") else {
        unreachable!()
    };
    assert_eq!(
        carry_from, &None,
        "with two terminals, guessing a carry would silently feed the \
         loop one branch's output"
    );
}

/// Waiting on a node outside the body would hang: it is never re-evaluated.
#[test]
fn condition_outside_the_body_is_a_compile_error() {
    let mut g = Graph::new();
    g.add_node(Node::filter_with_id("prep", "prep"));
    g.add_node(Node::loop_until(
        "refine",
        Some(4),
        LoopCondition::WhenSignaled("prep".into()),
    ));
    g.add_node(Node::filter_with_id("body", "body"));
    g.add_edge(Edge::data("e0", "prep", "refine"));
    g.add_edge(Edge::control("e1", "refine", "body"));

    let reg = registry(&["prep", "body"]);
    let err = compile(&g, &reg, CompileMode::Inference, None)
        .expect_err("a condition outside the body never changes");
    assert!(err.to_string().contains("prep"), "{err}");
}

/// A loop with no control edge has no body to run.
#[test]
fn loop_without_a_body_is_a_compile_error() {
    let mut g = Graph::new();
    g.add_node(Node::filter_with_id("prep", "prep"));
    g.add_node(Node::loop_node("refine", Some(4)));
    g.add_edge(Edge::data("e0", "prep", "refine"));

    let reg = registry(&["prep"]);
    let err =
        compile(&g, &reg, CompileMode::Inference, None).expect_err("empty loop body is useless");
    assert!(err.to_string().contains("empty body"), "{err}");
}

// ── Branch ──

/// Each arm is compiled once, inside the `Branch` — never again as a
/// top-level step (which would run every arm unconditionally).
#[test]
fn branch_arms_are_compiled_exactly_once() {
    let mut g = Graph::new();
    g.add_node(Node::branch("router"));
    g.add_node(Node::filter_with_id("billing", "billing"));
    g.add_node(Node::filter_with_id("tech", "tech"));
    g.add_edge(Edge::control("e1", "router", "billing").with_label("billing"));
    g.add_edge(Edge::control("e2", "router", "tech").with_label("tech"));

    let reg = registry(&["billing", "tech"]);
    let result = compile(&g, &reg, CompileMode::Inference, None).expect("compiles");

    assert_eq!(
        execute_count(&result.plan, "billing"),
        1,
        "arm compiled more than once — it would run even when not selected:\n{}",
        result.plan
    );
    assert_eq!(execute_count(&result.plan, "tech"), 1, "\n{}", result.plan);

    let br = find_branch(&result.plan).expect("plan contains a Branch");
    let ExecutionPlan::Branch { arms, .. } = br else {
        unreachable!()
    };
    assert_eq!(arms.len(), 2);
    let labels: Vec<&str> = arms.iter().map(|(l, _)| l.as_str()).collect();
    assert!(labels.contains(&"billing"), "labels were {labels:?}");
    assert!(labels.contains(&"tech"), "labels were {labels:?}");
}

/// A `Branch` selects arms via control edges. A data edge leaving the branch
/// node is a normal downstream dependency, not an arm.
#[test]
fn branch_arms_come_only_from_control_edges() {
    let mut g = Graph::new();
    g.add_node(Node::branch("router"));
    g.add_node(Node::filter_with_id("yes", "yes"));
    g.add_node(Node::filter_with_id("no", "no"));
    g.add_node(Node::filter_with_id("sink", "sink"));
    g.add_edge(Edge::control("e1", "router", "yes").with_label("yes"));
    g.add_edge(Edge::control("e2", "router", "no").with_label("no"));
    // Not an arm — plain data dependency, runs regardless of the branch taken.
    g.add_edge(Edge::data("e3", "yes", "sink"));

    let reg = registry(&["yes", "no", "sink"]);
    let result = compile(&g, &reg, CompileMode::Inference, None).expect("compiles");

    let br = find_branch(&result.plan).expect("plan contains a Branch");
    let ExecutionPlan::Branch { arms, .. } = br else {
        unreachable!()
    };
    assert_eq!(
        arms.len(),
        2,
        "only control edges are arms, got {:?}",
        arms.iter().map(|(l, _)| l).collect::<Vec<_>>()
    );

    assert_eq!(execute_count(&result.plan, "yes"), 1, "\n{}", result.plan);
    assert_eq!(execute_count(&result.plan, "no"), 1, "\n{}", result.plan);
    assert_eq!(execute_count(&result.plan, "sink"), 1, "\n{}", result.plan);
}

/// Two arms with the same label: the second could never be selected.
#[test]
fn duplicate_arm_labels_are_a_compile_error() {
    let mut g = Graph::new();
    g.add_node(Node::branch("router"));
    g.add_node(Node::filter_with_id("a", "a"));
    g.add_node(Node::filter_with_id("b", "b"));
    g.add_edge(Edge::control("e1", "router", "a").with_label("retry"));
    g.add_edge(Edge::control("e2", "router", "b").with_label("retry"));

    let reg = registry(&["a", "b"]);
    let err = compile(&g, &reg, CompileMode::Inference, None).expect_err("duplicate arm labels");
    assert!(err.to_string().contains("retry"), "{err}");
}

/// A branch with no control edges has nothing to select between.
#[test]
fn branch_without_arms_is_a_compile_error() {
    let mut g = Graph::new();
    g.add_node(Node::filter_with_id("prep", "prep"));
    g.add_node(Node::branch("router"));
    g.add_edge(Edge::data("e0", "prep", "router"));

    let reg = registry(&["prep"]);
    let err = compile(&g, &reg, CompileMode::Inference, None).expect_err("branch with no arms");
    assert!(err.to_string().contains("no arms"), "{err}");
}

/// Branch arms that are themselves chains stay wholly inside their arm.
#[test]
fn branch_arm_chain_is_fully_contained() {
    let mut g = Graph::new();
    g.add_node(Node::branch("router"));
    g.add_node(Node::filter_with_id("a1", "a1"));
    g.add_node(Node::filter_with_id("a2", "a2"));
    g.add_node(Node::filter_with_id("b1", "b1"));
    g.add_edge(Edge::control("e1", "router", "a1").with_label("a"));
    g.add_edge(Edge::control("e2", "router", "b1").with_label("b"));
    g.add_edge(Edge::data("e3", "a1", "a2"));

    let reg = registry(&["a1", "a2", "b1"]);
    let result = compile(&g, &reg, CompileMode::Inference, None).expect("compiles");

    for n in ["a1", "a2", "b1"] {
        assert_eq!(
            execute_count(&result.plan, n),
            1,
            "`{n}` appears more than once:\n{}",
            result.plan
        );
    }

    let br = find_branch(&result.plan).expect("plan contains a Branch");
    let ExecutionPlan::Branch { arms, .. } = br else {
        unreachable!()
    };
    let arm_a = arms
        .iter()
        .find(|(l, _)| l == "a")
        .map(|(_, p)| p)
        .expect("arm `a`");
    assert_eq!(execute_count(arm_a, "a1"), 1);
    assert_eq!(
        execute_count(arm_a, "a2"),
        1,
        "the whole `a` chain belongs to the arm"
    );
    assert_eq!(execute_count(arm_a, "b1"), 0);
}
