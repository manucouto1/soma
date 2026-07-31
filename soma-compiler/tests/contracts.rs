//! Typed contracts on edges and branch arms.
//!
//! The motivation is empirical. Of 1600+ annotated multi-agent traces
//! (MAST, NeurIPS 2025), ~42% of failures are specification/design and ~37%
//! are inter-agent misalignment — context lost or malformed at a handoff.
//! Both are contract problems, not model problems, and Soma is unusual in
//! having a compiler that can refuse them.
//!
//! Two severities, deliberately:
//! - *plausible but unequal* (f32 → f64) stays a warning, as it always was;
//! - *no possible reading* (tensor → conversation) is a compile error.

use somatize_compiler::{CompileMode, DiagnosticLevel, SimpleFilterRegistry, compile};
use somatize_core::cache::CacheKey;
use somatize_core::filter::{Distribution, FilterKind, FilterMeta, StreamMode};
use somatize_core::graph::{Edge, Graph, Node};
use somatize_core::schema::{DataType, Schema};
use somatize_core::step::StepMeta;

fn filter_meta(name: &str, input: Option<Schema>, output: Option<Schema>) -> FilterMeta {
    FilterMeta {
        name: name.into(),
        kind: FilterKind::Stateless,
        cacheable: true,
        differentiable: false,
        deterministic: true,
        stream_mode: StreamMode::FixedState,
        distribution: Distribution::Local,
        input_schema: input,
        output_schema: output,
    }
}

fn key(id: &str) -> CacheKey {
    CacheKey::from_parts(&[id.as_bytes()])
}

// ── Coercion rules ──

#[test]
fn text_flows_into_a_conversation() {
    // A bare prompt promotes to a user turn — `Messages::from_value` does it.
    assert!(DataType::Utf8.can_coerce_to(&DataType::Messages));
    assert!(DataType::Messages.can_coerce_to(&DataType::Utf8));
}

#[test]
fn json_is_the_dynamic_type() {
    for dt in [
        DataType::Float64,
        DataType::Utf8,
        DataType::Messages,
        DataType::Bytes,
    ] {
        assert!(dt.can_coerce_to(&DataType::Json), "{dt} → json");
        assert!(DataType::Json.can_coerce_to(&dt), "json → {dt}");
    }
}

#[test]
fn numeric_widths_are_coercible_to_each_other() {
    assert!(DataType::Float32.can_coerce_to(&DataType::Float64));
    assert!(DataType::Int64.can_coerce_to(&DataType::Float64));
}

/// The pairs that make a handoff fail.
#[test]
fn tensors_and_conversations_do_not_mix() {
    assert!(!DataType::Float64.can_coerce_to(&DataType::Messages));
    assert!(!DataType::Messages.can_coerce_to(&DataType::Float64));
    assert!(!DataType::Float64.can_coerce_to(&DataType::Utf8));
    assert!(!DataType::Bytes.can_coerce_to(&DataType::Messages));
}

// ── Edge validation ──

/// A tensor feeding a node that wants a conversation cannot work under any
/// reading, so it must not compile.
#[test]
fn an_impossible_edge_is_a_compile_error() {
    let mut g = Graph::new();
    g.add_node(Node::filter_with_id("encoder", "encoder"));
    g.add_node(Node::step("agent", "Agent"));
    g.add_edge(Edge::data("e", "encoder", "agent"));

    let mut reg = SimpleFilterRegistry::new();
    reg.register_meta(
        "encoder",
        filter_meta(
            "encoder",
            None,
            Some(Schema::vector(DataType::Float64, 128)),
        ),
        key("encoder"),
    );
    reg.register_step_meta(
        "agent",
        StepMeta::new("Agent").with_input_schema(Schema::messages()),
    );

    let err = compile(&g, &reg, CompileMode::Inference, None)
        .expect_err("a tensor cannot be a conversation");
    let msg = err.to_string();
    assert!(msg.contains("encoder"), "{msg}");
    assert!(msg.contains("agent"), "{msg}");
    assert!(msg.contains("no conversion"), "{msg}");
}

/// The step's *output* is checked too — a handoff runs in both directions.
#[test]
fn a_step_output_meeting_a_tensor_input_is_rejected() {
    let mut g = Graph::new();
    g.add_node(Node::step("agent", "Agent"));
    g.add_node(Node::filter_with_id("classifier", "classifier"));
    g.add_edge(Edge::data("e", "agent", "classifier"));

    let mut reg = SimpleFilterRegistry::new();
    reg.register_step_meta(
        "agent",
        StepMeta::new("Agent").with_output_schema(Schema::messages()),
    );
    reg.register_meta(
        "classifier",
        filter_meta(
            "classifier",
            Some(Schema::vector(DataType::Float64, 10)),
            None,
        ),
        key("classifier"),
    );

    let err =
        compile(&g, &reg, CompileMode::Inference, None).expect_err("messages are not a tensor");
    assert!(err.to_string().contains("no conversion"), "{err}");
}

/// Text into a conversation is a real coercion and must compile cleanly —
/// this is the ordinary first hop of an agentic graph.
#[test]
fn text_into_a_conversation_compiles_without_complaint() {
    let mut g = Graph::new();
    g.add_node(Node::filter_with_id("prompt", "prompt"));
    g.add_node(Node::step("agent", "Agent"));
    g.add_edge(Edge::data("e", "prompt", "agent"));

    let mut reg = SimpleFilterRegistry::new();
    reg.register_meta(
        "prompt",
        filter_meta("prompt", None, Some(Schema::text())),
        key("prompt"),
    );
    reg.register_step_meta(
        "agent",
        StepMeta::new("Agent").with_input_schema(Schema::messages()),
    );

    let result = compile(&g, &reg, CompileMode::Inference, None).expect("compiles");
    assert!(
        result
            .diagnostics
            .iter()
            .all(|d| d.level != DiagnosticLevel::Warning || !d.message.contains("schema")),
        "a legitimate coercion produced a schema warning: {:?}",
        result.diagnostics
    );
}

/// A merely unequal pair still only warns — existing pipelines depend on it.
#[test]
fn a_plausible_mismatch_is_still_only_a_warning() {
    let mut g = Graph::new();
    g.add_node(Node::filter_with_id("a", "a"));
    g.add_node(Node::filter_with_id("b", "b"));
    g.add_edge(Edge::data("e", "a", "b"));

    let mut reg = SimpleFilterRegistry::new();
    reg.register_meta(
        "a",
        filter_meta("a", None, Some(Schema::vector(DataType::Float32, 8))),
        key("a"),
    );
    reg.register_meta(
        "b",
        filter_meta("b", Some(Schema::vector(DataType::Float64, 8)), None),
        key("b"),
    );

    let result = compile(&g, &reg, CompileMode::Inference, None).expect("still compiles");
    assert!(
        result
            .diagnostics
            .iter()
            .any(|d| d.message.contains("schema mismatch")),
        "expected a warning, got {:?}",
        result.diagnostics
    );
}

// ── Branch arms ──

/// Declaring the arms lets a mislabelled edge be caught before the run.
#[test]
fn an_edge_labelling_an_undeclared_arm_is_rejected() {
    let mut g = Graph::new();
    g.add_node(Node::branch_over("router", ["billing", "tech"]));
    g.add_node(Node::filter_with_id("billing", "billing"));
    g.add_node(Node::filter_with_id("tech", "tech"));
    g.add_edge(Edge::control("e1", "router", "billing").with_label("billing"));
    // Typo: the router will never produce "tec".
    g.add_edge(Edge::control("e2", "router", "tech").with_label("tec"));

    let mut reg = SimpleFilterRegistry::new();
    for id in ["billing", "tech"] {
        reg.register_meta(id, filter_meta(id, None, None), key(id));
    }

    let err = compile(&g, &reg, CompileMode::Inference, None).expect_err("typo in an arm label");
    let msg = err.to_string();
    assert!(msg.contains("tec"), "should name the bad label: {msg}");
    assert!(
        msg.contains("billing"),
        "should list the declared arms: {msg}"
    );
}

/// A declared arm with no edge would fail the moment it was selected.
#[test]
fn a_declared_arm_without_an_edge_is_rejected() {
    let mut g = Graph::new();
    g.add_node(Node::branch_over("router", ["billing", "tech"]));
    g.add_node(Node::filter_with_id("billing", "billing"));
    g.add_edge(Edge::control("e1", "router", "billing").with_label("billing"));

    let mut reg = SimpleFilterRegistry::new();
    reg.register_meta(
        "billing",
        filter_meta("billing", None, None),
        key("billing"),
    );

    let err = compile(&g, &reg, CompileMode::Inference, None).expect_err("unreachable arm");
    assert!(err.to_string().contains("tech"), "{err}");
}

#[test]
fn declared_arms_matching_their_edges_compile() {
    let mut g = Graph::new();
    g.add_node(Node::branch_over("router", ["billing", "tech"]));
    g.add_node(Node::filter_with_id("billing", "billing"));
    g.add_node(Node::filter_with_id("tech", "tech"));
    g.add_edge(Edge::control("e1", "router", "billing").with_label("billing"));
    g.add_edge(Edge::control("e2", "router", "tech").with_label("tech"));

    let mut reg = SimpleFilterRegistry::new();
    for id in ["billing", "tech"] {
        reg.register_meta(id, filter_meta(id, None, None), key(id));
    }

    compile(&g, &reg, CompileMode::Inference, None).expect("compiles");
}

/// A `default` arm is always allowed — it is the sanctioned catch-all, and
/// declaring it explicitly should not be required.
#[test]
fn a_default_arm_needs_no_declaration() {
    let mut g = Graph::new();
    g.add_node(Node::branch_over("router", ["billing"]));
    g.add_node(Node::filter_with_id("billing", "billing"));
    g.add_node(Node::filter_with_id("other", "other"));
    g.add_edge(Edge::control("e1", "router", "billing").with_label("billing"));
    g.add_edge(Edge::control("e2", "router", "other").with_label("default"));

    let mut reg = SimpleFilterRegistry::new();
    for id in ["billing", "other"] {
        reg.register_meta(id, filter_meta(id, None, None), key(id));
    }

    compile(&g, &reg, CompileMode::Inference, None).expect("compiles");
}

/// Not declaring arms keeps the old behaviour: infer them from the edges.
#[test]
fn undeclared_arms_are_still_inferred() {
    let mut g = Graph::new();
    g.add_node(Node::branch("router"));
    g.add_node(Node::filter_with_id("a", "a"));
    g.add_node(Node::filter_with_id("b", "b"));
    g.add_edge(Edge::control("e1", "router", "a").with_label("a"));
    g.add_edge(Edge::control("e2", "router", "b").with_label("b"));

    let mut reg = SimpleFilterRegistry::new();
    for id in ["a", "b"] {
        reg.register_meta(id, filter_meta(id, None, None), key(id));
    }

    compile(&g, &reg, CompileMode::Inference, None).expect("compiles");
}
