//! What the engine says while it works.
//!
//! The other answer a run gives. `run` returns a value; this is everything that
//! happened on the way to it, and the two come out of different holes on
//! purpose: a value arrives when it is over, and a fact arrives when it happens.

use crate::doubles::{Add, Cable, EachOne, Fail, Journal, Mirror, Notebook, Told, Witness};
use soma_next_core::{
    Catalog, Executor, Fact, Graph, Host, NodeId, Placement, Plan, Value, compile, distribute, node,
};
use std::sync::Arc;

/// A chain of `Add`s, both here and "there", and where each of them runs.
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

// ── Nobody is watching, which is most runs ──

#[test]
fn a_run_nobody_watches_behaves_exactly_as_it_did() {
    let (g, c, _) = both_sides(&[("a", 1.0), ("b", 10.0)], &[("a", "b")]);
    let plan = compile(&g, &c).unwrap();

    let out = Executor::new(&c).run(&plan, Value::number(0.0)).unwrap();

    assert_eq!(out, Value::number(11.0));
}

// ── What one forward says ──

#[test]
fn every_node_that_ran_is_said_so_in_the_order_it_ran() {
    let (g, c, _) = both_sides(&[("a", 1.0), ("b", 10.0)], &[("a", "b")]);
    let plan = compile(&g, &c).unwrap();
    let told = Told::new();

    Executor::new(&c)
        .watching(&told)
        .run(&plan, Value::number(0.0))
        .unwrap();

    assert_eq!(told.ran(), ["a", "b"]);
}

#[test]
fn a_run_ends_with_one_fact_that_says_it_is_over() {
    let (g, c, _) = both_sides(&[("a", 1.0)], &[]);
    let plan = compile(&g, &c).unwrap();
    let told = Told::new();

    Executor::new(&c)
        .watching(&told)
        .run(&plan, Value::number(0.0))
        .unwrap();

    assert_eq!(told.kinds(), ["ran", "finished"]);
}

#[test]
fn a_node_that_failed_says_which_one_before_the_run_stops() {
    // The whole reason this is a fact and not a line in `RunError`: by the time
    // the caller reads the error the run is over, and whoever was watching
    // wanted to know which node while it was happening.
    let mut g = Graph::new();
    let mut c = Catalog::new();
    g.add_node("a").unwrap();
    g.add_node("boom").unwrap();
    g.add_edge("a", "boom").unwrap();
    c.insert("a", Arc::new(Add(1.0)));
    c.insert("boom", Arc::new(Fail));
    let plan = compile(&g, &c).unwrap();
    let told = Told::new();

    Executor::new(&c)
        .watching(&told)
        .run(&plan, Value::number(0.0))
        .unwrap_err();

    assert_eq!(told.kinds(), ["ran", "failed", "broke"]);
    let Fact::Failed { node, why } = &told.all()[1] else {
        panic!("expected the node's failure, found {:?}", told.all()[1]);
    };
    assert_eq!(node, &NodeId::from("boom"));
    assert!(
        why.contains("I broke"),
        "it carries what the node said: {why}"
    );
}

#[test]
fn a_run_that_could_not_finish_is_still_closed() {
    // A record has to end whichever way the run did, or whoever is writing one
    // never learns that this forward is over.
    let (g, c, _) = both_sides(&[("a", 1.0)], &[]);
    let placement = away(&["a"]);
    let plan = distribute(&compile(&g, &c).unwrap(), &placement);
    let told = Told::new();

    Executor::new(&c)
        .placed(&placement)
        .reaching("there", &Cable("no route to `there`"))
        .watching(&told)
        .run(&plan, Value::number(0.0))
        .unwrap_err();

    assert_eq!(told.kinds(), ["broke"]);
    assert!(told.all()[0].ends_a_run());
}

#[test]
fn an_empty_plan_still_says_it_finished() {
    let told = Told::new();

    Executor::new(&Catalog::new())
        .watching(&told)
        .run(&Plan::Empty, Value::text("intact"))
        .unwrap();

    assert_eq!(told.kinds(), ["finished"]);
}

// ── The cache, which is the interesting half of what a run does ──

#[test]
fn a_hit_and_a_miss_are_two_different_facts() {
    let journal = Journal::new();
    let (g, c, _, memory) = node("encoder", Witness("encoder", journal.clone()))
        .frozen()
        .cached()
        .somatize()
        .unwrap();
    let plan = compile(&g, &c).unwrap();
    let notebook = Notebook::new();
    let (first, again) = (Told::new(), Told::new());

    for told in [&first, &again] {
        Executor::new(&c)
            .remembering(&memory)
            .keeping(&notebook)
            .watching(told)
            .run(&plan, Value::number(7.0))
            .unwrap();
    }

    assert_eq!(first.kinds(), ["ran", "kept", "finished"]);
    assert_eq!(
        again.kinds(),
        ["recalled", "finished"],
        "a hit does not run the node, so there is nothing to say it did"
    );
}

#[test]
fn a_node_that_maps_says_how_many_items_it_did_not_have_to_compute() {
    // The grain CU16 separated: one number could not say this, because half of
    // a mapped node's items being new is the normal case.
    let journal = Journal::new();
    let (g, c, _, memory) = node("embed", EachOne(journal.clone()))
        .frozen()
        .cached()
        .mapped()
        .somatize()
        .unwrap();
    let plan = compile(&g, &c).unwrap();
    let notebook = Notebook::new();
    let documents =
        |many: &[f64]| Value::list(many.iter().copied().map(Value::number).collect::<Vec<_>>());

    let told = Told::new();
    Executor::new(&c)
        .remembering(&memory)
        .keeping(&notebook)
        .run(&plan, documents(&[1.0, 2.0]))
        .unwrap();
    Executor::new(&c)
        .remembering(&memory)
        .keeping(&notebook)
        .watching(&told)
        .run(&plan, documents(&[1.0, 2.0, 3.0]))
        .unwrap();

    let Some(Fact::Items { node, of, recalled }) = told.all().into_iter().next() else {
        panic!("expected the items first, found {:?}", told.kinds());
    };
    assert_eq!(node, NodeId::from("embed"));
    assert_eq!(
        (of, recalled),
        (3, 2),
        "two were already there and one is new"
    );
}

// ── And what happened on the other machine ──

#[test]
fn what_ran_over_there_arrives_here_saying_where_it_ran() {
    let (g, here, there) = both_sides(&[("a", 1.0), ("b", 10.0)], &[("a", "b")]);
    let placement = away(&["b"]);
    let plan = distribute(&compile(&g, &here).unwrap(), &placement);
    let told = Told::new();

    Executor::new(&here)
        .placed(&placement)
        .reaching("there", &Mirror::new(there))
        .watching(&told)
        .run(&plan, Value::number(0.0))
        .unwrap();

    // `a` here, `b` over there, the round trip, and the end.
    assert_eq!(told.kinds(), ["ran", "ran", "left", "finished"]);
    let facts = told.all();
    let (kind, fields) = facts[1].flattened();
    let said: std::collections::HashMap<_, _> = fields.into_iter().collect();
    assert_eq!(kind, "ran");
    assert_eq!(said["node"], "b");
    assert_eq!(
        said["host"], "there",
        "the engine over there does not know its own name: this one does"
    );
}

#[test]
fn the_round_trip_is_its_own_fact_and_not_the_sum_of_what_happened_there() {
    // It is the number that answers whether sending it was worth it, and the
    // wire is exactly the part that is not in any of the others.
    let (g, here, there) = both_sides(&[("a", 1.0)], &[]);
    let placement = away(&["a"]);
    let plan = distribute(&compile(&g, &here).unwrap(), &placement);
    let told = Told::new();

    Executor::new(&here)
        .placed(&placement)
        .reaching("there", &Mirror::new(there))
        .watching(&told)
        .run(&plan, Value::number(0.0))
        .unwrap();

    let Some(Fact::Left { host, .. }) = told
        .all()
        .into_iter()
        .find(|fact| matches!(fact, Fact::Left { .. }))
    else {
        panic!("nothing said the slice left: {:?}", told.kinds());
    };
    assert_eq!(host, Host::new("there"));
}

#[test]
fn a_slice_that_went_away_says_nothing_about_finishing() {
    // A `forward` is a run; a slice executed for somebody else is not one. If
    // `resume` said `finished` too, whoever writes records would close this
    // forward in the middle of it.
    let (g, here, there) = both_sides(&[("a", 1.0), ("b", 10.0)], &[("a", "b")]);
    let placement = away(&["b"]);
    let plan = distribute(&compile(&g, &here).unwrap(), &placement);
    let told = Told::new();

    Executor::new(&here)
        .placed(&placement)
        .reaching("there", &Mirror::new(there))
        .watching(&told)
        .run(&plan, Value::number(0.0))
        .unwrap();

    assert_eq!(
        told.kinds().iter().filter(|k| **k == "finished").count(),
        1,
        "exactly one of these ends the record"
    );
}

// ── When, and not only how long ──

#[test]
fn every_node_says_where_it_sat_on_the_run_s_own_timeline() {
    // What makes a picture of *what ran when* possible at all. A duration from
    // the run's start and not a wall clock, so it still means something when it
    // comes back from another machine.
    let (g, c, _) = both_sides(&[("a", 1.0), ("b", 10.0)], &[("a", "b")]);
    let plan = compile(&g, &c).unwrap();
    let told = Told::new();

    Executor::new(&c)
        .watching(&told)
        .run(&plan, Value::number(0.0))
        .unwrap();

    let began: Vec<_> = told
        .all()
        .into_iter()
        .filter_map(|fact| match fact {
            Fact::Ran { began, .. } => Some(began),
            _ => None,
        })
        .collect();
    assert_eq!(began.len(), 2);
    assert!(began[0] <= began[1], "`a` ran before `b`: {began:?}");
}

#[test]
fn a_slice_counts_from_its_own_start_and_not_from_the_run_s() {
    // `resume` is not a run. What a slice says about *when* is a fact about the
    // slice, and whoever draws a timeline adds the offset of the `Left` it
    // arrived under — two wall clocks would not have composed at all.
    let (g, here, there) = both_sides(&[("a", 1.0), ("b", 10.0)], &[("a", "b")]);
    let placement = away(&["b"]);
    let plan = distribute(&compile(&g, &here).unwrap(), &placement);
    let told = Told::new();

    Executor::new(&here)
        .placed(&placement)
        .reaching("there", &Mirror::new(there))
        .watching(&told)
        .run(&plan, Value::number(0.0))
        .unwrap();

    let elsewhere = told
        .all()
        .into_iter()
        .find_map(|fact| match fact {
            Fact::Elsewhere { saw, .. } => Some(*saw),
            _ => None,
        })
        .expect("something ran over there");
    let Fact::Ran { began, .. } = elsewhere else {
        panic!("expected a node running over there");
    };
    let Some(Fact::Left { began: left, .. }) = told
        .all()
        .into_iter()
        .find(|fact| matches!(fact, Fact::Left { .. }))
    else {
        panic!("nothing said the slice left");
    };
    assert!(
        began < left || began.as_micros() < 1_000,
        "the slice's own offset is small; it is not the run's clock: {began:?} vs {left:?}"
    );
}
