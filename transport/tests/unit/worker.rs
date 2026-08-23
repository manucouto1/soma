//! A slice of plan really running in another process.
//!
//! These tests start a separate binary and talk to it over pipes. No double
//! will do here: what is being checked is precisely that the process exists,
//! that the plan arrives, and that what it produces comes back.

use crate::doubles::{Add, Dir, catalog};
use soma_next_core::{
    Catalog, Device, Executor, Fact, Graph, Host, Memory, Placement, Plan, RunError, Value,
    Watcher, compile, distribute, node,
};
use soma_next_store::{Cache, Local};
use soma_next_transport::{Codec, CodecError, Worker};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Stands up the test worker.
fn worker() -> Worker {
    Worker::spawn(Command::new(env!("CARGO_BIN_EXE_test-worker")))
        .expect("cargo just built the binary")
}

fn number(v: &Value) -> f64 {
    let Value::Number(x) = v else {
        panic!("expected a number, found {}", v.type_name())
    };
    *x
}

/// A graph built by hand, with the catalog both sides know.
fn graph_with(nodes: &[&str], edges: &[(&str, &str)]) -> Graph {
    let mut g = Graph::new();
    for id in nodes {
        g.add_node(*id).unwrap();
    }
    for (from, to) in edges {
        g.add_edge(*from, *to).unwrap();
    }
    g
}

fn on_hosts(hosts: &[(&str, &str)]) -> Placement {
    let mut placement = Placement::new();
    for (id, host) in hosts {
        placement.place_at(*id, Host::new(*host));
    }
    placement
}

fn run(
    graph: &Graph,
    catalog: &Catalog,
    placement: &Placement,
    worker: &Worker,
    input: Value,
) -> Result<Value, RunError> {
    let plan = distribute(&compile(graph, catalog).unwrap(), placement);
    Executor::new(catalog)
        .placed(placement)
        .reaching("worker1", worker)
        .run(&plan, input)
}

// ── That the other process exists ──

#[test]
fn a_node_placed_away_really_runs_in_another_process() {
    // `where` returns the pid of whoever executes it. If the distribution were
    // a fiction and it executed here, it would return ours.
    let g = graph_with(&["where"], &[]);
    let c = catalog();
    let p = on_hosts(&[("where", "worker1")]);
    let w = worker();

    let output = run(&g, &c, &p, &w, Value::Null).unwrap();

    assert_ne!(
        number(&output) as u32,
        std::process::id(),
        "it ran in this process: the distribution is getting nowhere"
    );
    assert!(number(&output) > 0.0);
}

#[test]
fn the_same_graph_undistributed_runs_here() {
    // The other half: what changes is the placement, not the graph.
    let g = graph_with(&["where"], &[]);
    let c = catalog();
    let plan = compile(&g, &c).unwrap();

    let output = Executor::new(&c).run(&plan, Value::Null).unwrap();

    assert_eq!(number(&output) as u32, std::process::id());
}

// ── That the work arrives and comes back ──

#[test]
fn a_whole_chain_goes_away_and_comes_back_with_the_result() {
    let g = graph_with(&["a", "b", "c"], &[("a", "b"), ("b", "c")]);
    let c = catalog();
    let p = on_hosts(&[("a", "worker1"), ("b", "worker1"), ("c", "worker1")]);
    let w = worker();

    assert_eq!(
        number(&run(&g, &c, &p, &w, Value::number(0.0)).unwrap()),
        3.0
    );
}

#[test]
fn what_the_worker_produces_is_read_by_whoever_comes_next_here() {
    // The real seam: `b` runs there, `c` runs here and reads from `b`. If what
    // was produced did not come back, `gather` would not find its input.
    let g = graph_with(&["a", "b", "c"], &[("a", "b"), ("b", "c")]);
    let c = catalog();
    let p = on_hosts(&[("b", "worker1")]);
    let w = worker();

    assert_eq!(
        number(&run(&g, &c, &p, &w, Value::number(0.0)).unwrap()),
        3.0
    );
}

#[test]
fn what_is_produced_here_reaches_the_worker() {
    // And the opposite direction: `a` runs here, `b` runs there and reads from `a`.
    let g = graph_with(&["a", "b"], &[("a", "b")]);
    let c = catalog();
    let p = on_hosts(&[("b", "worker1")]);
    let w = worker();

    assert_eq!(
        number(&run(&g, &c, &p, &w, Value::number(10.0)).unwrap()),
        12.0
    );
}

#[test]
fn a_fan_in_with_one_branch_away_gives_the_same_as_all_of_it_here() {
    let g = graph_with(
        &["a", "left", "right", "join"],
        &[
            ("a", "left"),
            ("a", "right"),
            ("left", "join"),
            ("right", "join"),
        ],
    );
    let c = catalog();
    let w = worker();

    let here = run(&g, &c, &Placement::new(), &w, Value::number(0.0)).unwrap();
    let away = run(
        &g,
        &c,
        &on_hosts(&[("left", "worker1")]),
        &w,
        Value::number(0.0),
    )
    .unwrap();

    assert_eq!(number(&here), 2.0);
    assert_eq!(number(&away), number(&here));
}

#[test]
fn a_whole_wave_on_one_worker_goes_in_a_single_trip() {
    // And not one per branch, which is what I took for granted: `distribute`
    // wraps the **whole** wave, so out comes a `Remote` with the `Wave` inside.
    // Distributing does not break the concurrency, it moves it elsewhere.
    let g = graph_with(&["left", "right"], &[]);
    let c = catalog();
    let p = on_hosts(&[("left", "worker1"), ("right", "worker1")]);
    let w = worker();

    let plan = distribute(&compile(&g, &c).unwrap(), &p);
    assert!(
        matches!(&plan, Plan::Remote { inner, .. } if matches!(**inner, Plan::Wave(_))),
        "expected one trip with the wave inside: {plan:?}"
    );

    let output = run(&g, &c, &p, &w, Value::number(0.0)).unwrap();
    assert_eq!(
        output,
        Value::map(vec![
            ("left".to_string(), Value::number(1.0)),
            ("right".to_string(), Value::number(1.0)),
        ])
    );
}

#[test]
fn two_workers_are_two_hosts_and_two_processes() {
    let g = graph_with(&["left", "right"], &[]);
    let c = catalog();
    let mut p = Placement::new();
    p.place_at("left", Host::new("one"));
    p.place_at("right", Host::new("two"));

    let (a, b) = (worker(), worker());
    let plan = distribute(&compile(&g, &c).unwrap(), &p);
    let output = Executor::new(&c)
        .placed(&p)
        .reaching("one", &a)
        .reaching("two", &b)
        .run(&plan, Value::number(0.0))
        .unwrap();

    assert_eq!(
        output,
        Value::map(vec![
            ("left".to_string(), Value::number(1.0)),
            ("right".to_string(), Value::number(1.0)),
        ])
    );
}

#[test]
fn the_same_worker_serves_several_runs() {
    let g = graph_with(&["a"], &[]);
    let c = catalog();
    let p = on_hosts(&[("a", "worker1")]);
    let w = worker();

    for i in 0..5 {
        assert_eq!(
            number(&run(&g, &c, &p, &w, Value::number(i as f64)).unwrap()),
            i as f64 + 1.0
        );
    }
}

// ── The placement crosses, because it is data ──

#[test]
fn the_node_over_there_sees_the_device_it_was_given_here() {
    // What `placement.rs` has been promising since CU10: "the day a subgraph
    // travels to another machine, the placement travels with it".
    let g = graph_with(&["device"], &[]);
    let c = catalog();
    let mut p = Placement::new();
    p.place_at("device", Host::new("worker1"));
    p.place("device", Device::Meta);
    let w = worker();

    assert_eq!(
        run(&g, &c, &p, &w, Value::Null).unwrap(),
        Value::text("meta")
    );
}

#[test]
fn without_a_device_the_node_over_there_sees_none_either() {
    // "Wherever it lands" is still not "cpu" on the other side of the wire.
    let g = graph_with(&["device"], &[]);
    let c = catalog();
    let p = on_hosts(&[("device", "worker1")]);
    let w = worker();

    assert_eq!(run(&g, &c, &p, &w, Value::Null).unwrap(), Value::Null);
}

// ── What goes wrong ──

#[test]
fn a_name_nobody_resolves_is_not_executed_here_just_in_case() {
    let g = graph_with(&["a"], &[]);
    let c = catalog();
    let p = on_hosts(&[("a", "the_one_that_is_not_there")]);
    let plan = distribute(&compile(&g, &c).unwrap(), &p);
    let w = worker();

    let err = Executor::new(&c)
        .placed(&p)
        .reaching("worker1", &w)
        .run(&plan, Value::number(0.0))
        .unwrap_err();

    assert_eq!(
        err,
        RunError::NoTransport(Host::new("the_one_that_is_not_there"))
    );
}

#[test]
fn a_failure_over_there_comes_back_with_the_host_and_the_reason() {
    let g = graph_with(&["broken"], &[]);
    let c = catalog();
    let p = on_hosts(&[("broken", "worker1")]);
    let w = worker();

    let err = run(&g, &c, &p, &w, Value::Null).unwrap_err();
    let said = err.to_string();

    assert!(said.contains("worker1"), "the host is missing: {said}");
    assert!(said.contains("broken"), "the node is missing: {said}");
    assert!(said.contains("I broke"), "the reason is missing: {said}");
}

#[test]
fn an_opaque_bound_for_over_there_does_not_leave_this_process() {
    // The value is produced by a node here and read by one over there.
    let g = graph_with(&["opaque", "a"], &[("opaque", "a")]);
    let c = catalog();
    let p = on_hosts(&[("a", "worker1")]);
    let w = worker();

    let said = run(&g, &c, &p, &w, Value::Null).unwrap_err().to_string();

    assert!(said.contains("worker1"), "the host is missing: {said}");
    assert!(
        said.contains("does not cross"),
        "the reason is missing: {said}"
    );
}

#[test]
fn an_opaque_produced_over_there_does_not_come_back_either() {
    // The other direction, and the one that would be forgotten: the worker
    // cannot stay silent, because this side would be waiting for it forever.
    let g = graph_with(&["opaque"], &[]);
    let c = catalog();
    let p = on_hosts(&[("opaque", "worker1")]);
    let w = worker();

    let said = run(&g, &c, &p, &w, Value::Null).unwrap_err().to_string();

    assert!(said.contains("does not cross"), "{said}");
}

// ── With a codec, which is where the frontier really is ──

/// The same one the worker runs with `--codec`: two ends that do not agree on
/// how something is written down do not understand each other.
struct U32s;

const WRITTEN: &str = "__a_u32__";

impl Codec for U32s {
    fn packed(&self, value: &Value) -> Result<Value, CodecError> {
        match value.downcast::<u32>() {
            Some(n) => Ok(Value::map(vec![(
                WRITTEN.to_string(),
                Value::number(*n as f64),
            )])),
            None => match value {
                Value::Opaque(_) => Err(CodecError::new("only a `u32` is written down here")),
                other => Ok(other.clone()),
            },
        }
    }

    fn unpacked(&self, value: &Value) -> Result<Value, CodecError> {
        let Value::Map(pairs) = value else {
            return Ok(value.clone());
        };
        match &pairs[..] {
            [(key, Value::Number(n))] if key == WRITTEN => Ok(Value::opaque(*n as u32)),
            _ => Ok(value.clone()),
        }
    }
}

fn worker_with_codec() -> Worker {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_test-worker"));
    cmd.arg("--codec");
    Worker::spawn(cmd)
        .expect("cargo just built the binary")
        .packing(Arc::new(U32s))
}

#[test]
fn with_a_codec_an_opaque_produced_over_there_does_come_back() {
    // The frontier did not go away, it moved: from "an opaque" to "an opaque
    // nobody registered a codec for".
    let g = graph_with(&["opaque"], &[]);
    let c = catalog();
    let p = on_hosts(&[("opaque", "worker1")]);
    let w = worker_with_codec();

    let out = run(&g, &c, &p, &w, Value::Null).expect("it can be written down now");

    assert_eq!(
        out.downcast::<u32>().copied(),
        Some(7),
        "what came back is not the value that was produced over there"
    );
}

#[test]
fn and_one_bound_for_over_there_arrives_as_what_it_was() {
    // The other direction, and the one `reads` can answer: it says `1.0` only if
    // what it was handed is still an opaque on arrival.
    let g = graph_with(&["opaque", "reads"], &[("opaque", "reads")]);
    let c = catalog();
    let p = on_hosts(&[("reads", "worker1")]);
    let w = worker_with_codec();

    let out = run(&g, &c, &p, &w, Value::Null).expect("it crosses now");

    assert_eq!(number(&out), 1.0, "it did not arrive as an opaque");
}

#[test]
fn a_codec_that_cannot_write_the_slices_own_value_refuses_in_its_own_words() {
    // `last` is the value of the slice itself and has a reader here by
    // definition, so there is no leaving it behind: the codec's words are the
    // answer, and they are better than the wire's. **The words are the far
    // end's**, which is the half of this that could not be tested in one
    // process: the codec that failed is the one standing next to the node.
    let g = graph_with(&["unwritable"], &[]);
    let c = catalog();
    let p = on_hosts(&[("unwritable", "worker1")]);
    let w = worker_with_codec();

    let said = run(&g, &c, &p, &w, Value::Null).unwrap_err().to_string();

    assert!(
        said.contains("is all this codec knows how to write down"),
        "the codec over there did not speak: {said}"
    );
    assert!(said.contains("worker1"), "the host is missing: {said}");
}

#[test]
fn but_one_it_cannot_write_and_nobody_asked_for_stays_where_it_ran() {
    // CU14's rule, unchanged by any of this: an intermediate value is read by
    // the steps that ran with it, and one that cannot be written down is left
    // behind rather than refusing the whole answer.
    let g = graph_with(&["unwritable", "reads"], &[("unwritable", "reads")]);
    let c = catalog();
    let p = on_hosts(&[("unwritable", "worker1"), ("reads", "worker1")]);
    let w = worker_with_codec();

    let out = run(&g, &c, &p, &w, Value::Null).expect("nobody here reads the opaque");

    assert_eq!(number(&out), 1.0, "what it read was not the opaque");
}

#[test]
fn a_worker_that_does_not_pack_hands_the_node_what_it_was_sent() {
    // Both ends or neither. The client writes it down and the worker, with no
    // codec, never reads it back — so `reads` is handed a map and says so. It is
    // the failure this is allowed to have, and it is quiet: which is why nothing
    // installs one end without the other.
    let g = graph_with(&["opaque", "reads"], &[("opaque", "reads")]);
    let c = catalog();
    let p = on_hosts(&[("reads", "worker1")]);
    let w = worker().packing(Arc::new(U32s));

    let out = run(&g, &c, &p, &w, Value::Null).expect("what crosses is a map, and a map crosses");

    assert_eq!(number(&out), 0.0, "the worker cannot have read it back");
}

#[test]
fn an_opaque_read_only_over_there_does_not_stop_the_slice() {
    // CU12's debt, paid. An intermediate value of a slice is read by the steps
    // of that slice, which ran where it did: refusing the whole answer over one
    // was refusing the case this exists for — two steps on one host with
    // something live in between them.
    let g = graph_with(&["opaque", "reads"], &[("opaque", "reads")]);
    let c = catalog();
    let p = on_hosts(&[("opaque", "worker1"), ("reads", "worker1")]);
    let w = worker();

    let out = run(&g, &c, &p, &w, Value::Null).unwrap();

    assert_eq!(number(&out), 1.0, "what it read was not the opaque");
}

#[test]
fn an_id_the_worker_does_not_know_is_reported_instead_of_hanging() {
    // The price of the catalog not travelling, said with the name in front.
    let mut c = catalog();
    c.insert("only_here", Arc::new(Add(1.0)));
    let g = graph_with(&["only_here"], &[]);
    let p = on_hosts(&[("only_here", "worker1")]);
    let w = worker();

    let said = run(&g, &c, &p, &w, Value::number(0.0))
        .unwrap_err()
        .to_string();

    assert!(said.contains("only_here"), "{said}");
    assert!(said.contains("implementation"), "{said}");
}

#[test]
fn the_dsl_sends_a_slice_away_without_assembling_anything_by_hand() {
    // How it is really written, end to end.
    let (g, c, p, _) = (node("a", Add(1.0)) >> node("b", Add(1.0)).at("worker1"))
        .somatize()
        .unwrap();
    let w = worker();

    assert_eq!(
        number(&run(&g, &c, &p, &w, Value::number(0.0)).unwrap()),
        2.0
    );
}

// ── The whole thesis: two processes overlap ──

#[test]
fn two_workers_really_do_run_at_the_same_time() {
    // What having processes and not threads buys. The two nodes agree to meet
    // in a file, so if the distribution were sequential — or if it were a
    // fiction and both ran here — the first would wait for a second that has
    // not started, and the deadline would fire.
    let meeting = std::env::temp_dir().join(format!("soma-meeting-{}", std::process::id()));
    let _ = std::fs::remove_file(&meeting);

    let start = || {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_test-worker"));
        cmd.env("SOMA_RENDEZVOUS", &meeting)
            .env("SOMA_RENDEZVOUS_COUNT", "2");
        Worker::spawn(cmd).expect("cargo just built the binary")
    };
    let (one, two) = (start(), start());

    let g = graph_with(&["meet_one", "meet_two"], &[]);
    let c = catalog();
    let mut p = Placement::new();
    p.place_at("meet_one", Host::new("one"));
    p.place_at("meet_two", Host::new("two"));

    let plan = distribute(&compile(&g, &c).unwrap(), &p);
    let output = Executor::new(&c)
        .placed(&p)
        .reaching("one", &one)
        .reaching("two", &two)
        .run(&plan, Value::Null)
        .unwrap();

    let Value::Map(pairs) = &output else {
        panic!("two leaves were expected: {output:?}")
    };
    let pids: Vec<f64> = pairs.iter().map(|(_, v)| number(v)).collect();
    assert_ne!(pids[0], pids[1], "both slices went to the same process");
    assert!(
        pids.iter().all(|pid| *pid as u32 != std::process::id()),
        "one of them ran here"
    );

    let _ = std::fs::remove_file(&meeting);
}

#[test]
fn the_wave_that_travels_whole_runs_at_once_over_there() {
    // The consequence of the above, and the hardest to believe without testing
    // it: the worker receives a `Wave` and executes it **itself** on two
    // threads. A single process on the other side, two nodes that agree to meet
    // — and they arrive.
    let meeting = std::env::temp_dir().join(format!("soma-meeting-inside-{}", std::process::id()));
    let _ = std::fs::remove_file(&meeting);

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_test-worker"));
    cmd.env("SOMA_RENDEZVOUS", &meeting)
        .env("SOMA_RENDEZVOUS_COUNT", "2");
    let w = Worker::spawn(cmd).unwrap();

    let g = graph_with(&["meet_one", "meet_two"], &[]);
    let c = catalog();
    let p = on_hosts(&[("meet_one", "worker1"), ("meet_two", "worker1")]);

    let output = run(&g, &c, &p, &w, Value::Null).unwrap();

    let Value::Map(pairs) = &output else {
        panic!("two leaves were expected: {output:?}")
    };
    let pids: Vec<f64> = pairs.iter().map(|(_, v)| number(v)).collect();
    assert_eq!(
        pids[0], pids[1],
        "it is a single worker: the same pid twice"
    );
    assert_ne!(pids[0] as u32, std::process::id(), "it ran here");

    let _ = std::fs::remove_file(&meeting);
}

#[test]
fn two_trips_to_the_same_worker_queue_up_without_getting_lost() {
    // When the wave **cannot** be wrapped whole — one branch starts here — out
    // come two `Remote`s to the same host, and two threads call `dispatch` at
    // once. A pipe does not fit two conversations: the `Mutex` orders them.
    // What is checked is that ordering them does not mix them up.
    let g = graph_with(&["a", "left", "right"], &[("a", "right")]);
    let c = catalog();
    let p = on_hosts(&[("left", "worker1"), ("right", "worker1")]);
    let w = worker();

    let plan = distribute(&compile(&g, &c).unwrap(), &p);
    let Plan::Wave(branches) = &plan else {
        panic!("a wave was expected: {plan:?}")
    };
    assert_eq!(branches.len(), 2, "two branches, each with its own trip");

    assert_eq!(
        run(&g, &c, &p, &w, Value::number(0.0)).unwrap(),
        Value::map(vec![
            ("right".to_string(), Value::number(2.0)),
            ("left".to_string(), Value::number(1.0)),
        ])
    );
}

// ── What is written on `stdout` is not a message ──

#[test]
fn something_printed_on_the_workers_stdout_is_reported_and_does_not_hang() {
    // The cap in `frame` exists for exactly this: four ASCII characters read as
    // a length give well over a gigabyte, and without the cap the read would
    // wait forever for bytes that are never coming.
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_test-worker"));
    cmd.arg("--noisy");
    let w = Worker::spawn(cmd).unwrap();

    let g = graph_with(&["a"], &[]);
    let c = catalog();
    let p = on_hosts(&[("a", "worker1")]);

    let said = run(&g, &c, &p, &w, Value::number(0.0))
        .unwrap_err()
        .to_string();

    assert!(said.contains("MB"), "it does not say how big: {said}");
    assert!(
        said.contains("stderr"),
        "it does not say where to look: {said}"
    );
}

// ── The empty worker: the catalog arrives over the wire ──
//
// The other kind of worker. It starts without knowing what `x` is, and the
// client tells it. The artifact in these tests is plain text — `x=5` — and not a
// pickle, on purpose: what is being tested is the mechanism, and the mechanism
// knows nothing about Python.

use soma_next_transport::Artifact;

/// A worker that starts without a catalog.
fn empty() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_test-worker"));
    cmd.arg("--empty");
    cmd
}

fn adds(spec: &str) -> Artifact {
    // The id is set by whoever produces the artifact. Here, the text itself.
    Artifact::new("adds", format!("text:{spec}"), spec.as_bytes().to_vec())
}

fn run_provisioned(
    g: &Graph,
    c: &Catalog,
    p: &Placement,
    w: &Worker,
    input: Value,
) -> Result<Value, RunError> {
    let plan = distribute(&compile(g, c).unwrap(), p);
    Executor::new(c)
        .placed(p)
        .reaching("worker1", w)
        .run(&plan, input)
}

#[test]
fn an_empty_worker_receives_its_catalog_and_executes() {
    let w = Worker::spawn(empty())
        .unwrap()
        .carrying(adds("x=5"), "rust");

    // `x` has to exist here too, because `compile` requires it. It adds
    // something else on purpose: were it executed in this process, the result
    // would be 1.
    let mut c = Catalog::new();
    c.insert("x", Arc::new(Add(1.0)));
    let g = graph_with(&["x"], &[]);
    let p = on_hosts(&[("x", "worker1")]);

    assert_eq!(
        number(&run_provisioned(&g, &c, &p, &w, Value::number(0.0)).unwrap()),
        5.0,
        "it ran with the catalog from here, not with the one that was sent"
    );
}

#[test]
fn an_artifact_that_is_already_in_the_store_is_not_asked_for() {
    // The `have`/`want` finally having a `have`, and the way to see it from
    // outside is to make the client unable to answer: the second worker is
    // offered the **same id** with bytes that are not a spec. If it asked for
    // them, opening them would fail. It works, so it never asked.
    let shared = Dir::new();

    let first = Worker::spawn(empty_keeping(shared.path()))
        .unwrap()
        .carrying(adds("x=5"), "rust");
    assert_eq!(number(&run_it(&first)), 5.0);
    drop(first);

    let second = Worker::spawn(empty_keeping(shared.path()))
        .unwrap()
        .carrying(
            Artifact::new("adds", "text:x=5", b"this is not a spec".to_vec()),
            "rust",
        );

    assert_eq!(
        number(&run_it(&second)),
        5.0,
        "it asked the client instead of looking in the store"
    );
}

#[test]
fn without_a_store_the_next_worker_starts_from_nothing() {
    // The other half: the same two workers with nowhere to keep it. The second
    // one does ask, gets the bytes that are not a spec, and says so.
    let first = Worker::spawn(empty())
        .unwrap()
        .carrying(adds("x=5"), "rust");
    assert_eq!(number(&run_it(&first)), 5.0);
    drop(first);

    let second = Worker::spawn(empty()).unwrap().carrying(
        Artifact::new("adds", "text:x=5", b"this is not a spec".to_vec()),
        "rust",
    );

    let said = run_at(&second).unwrap_err().to_string();
    assert!(said.contains("is not `id=number`"), "{said}");
}

/// A worker that starts empty and keeps what it is sent in this directory.
fn empty_keeping(store: &std::path::Path) -> Command {
    let mut cmd = empty();
    cmd.args(["--store", &store.to_string_lossy()]);
    cmd
}

/// One node, `x`, executed over there.
fn run_it(w: &Worker) -> Value {
    run_at(w).unwrap()
}

fn run_at(w: &Worker) -> Result<Value, RunError> {
    let mut c = Catalog::new();
    c.insert("x", Arc::new(Add(1.0)));
    let g = graph_with(&["x"], &[]);
    let p = on_hosts(&[("x", "worker1")]);
    run_provisioned(&g, &c, &p, w, Value::number(0.0))
}

#[test]
fn the_artifact_is_sent_only_once() {
    // The `have`/`want`, observable from outside: the `times` node returns how
    // many catalogs have been built in the worker. If the second run resent the
    // artifact, it would go up to two.
    let w = Worker::spawn(empty())
        .unwrap()
        .carrying(adds("x=5"), "rust");
    let mut c = Catalog::new();
    c.insert("times", Arc::new(Add(0.0)));
    let g = graph_with(&["times"], &[]);
    let p = on_hosts(&[("times", "worker1")]);

    for _ in 0..3 {
        assert_eq!(
            number(&run_provisioned(&g, &c, &p, &w, Value::Null).unwrap()),
            1.0,
            "the catalog was built more than once"
        );
    }
}

#[test]
fn the_empty_worker_really_executes_in_another_process() {
    let w = Worker::spawn(empty())
        .unwrap()
        .carrying(adds("x=5"), "rust");
    let mut c = Catalog::new();
    c.insert("where", Arc::new(Add(0.0)));
    let g = graph_with(&["where"], &[]);
    let p = on_hosts(&[("where", "worker1")]);

    let output = run_provisioned(&g, &c, &p, &w, Value::Null).unwrap();

    assert_ne!(number(&output) as u32, std::process::id());
}

#[test]
fn a_runtime_the_worker_does_not_accept_is_rejected_at_the_greeting() {
    // The original's lesson, written as a test: it is rejected **before** trying
    // to rebuild anything, with both sides in the message.
    let w = Worker::spawn(empty())
        .unwrap()
        .carrying(adds("x=5"), "cpython-3.13");
    let mut c = Catalog::new();
    c.insert("x", Arc::new(Add(1.0)));
    let g = graph_with(&["x"], &[]);
    let p = on_hosts(&[("x", "worker1")]);

    let said = run_provisioned(&g, &c, &p, &w, Value::number(0.0))
        .unwrap_err()
        .to_string();

    assert!(
        said.contains("cpython-3.13"),
        "the client is missing: {said}"
    );
    assert!(said.contains("rust"), "the worker is missing: {said}");
}

#[test]
fn a_kind_of_artifact_it_does_not_know_is_rejected_by_name() {
    let artifact = Artifact::new("pickle", "sha256:abc", vec![1, 2, 3]);
    let w = Worker::spawn(empty()).unwrap().carrying(artifact, "rust");
    let mut c = Catalog::new();
    c.insert("x", Arc::new(Add(1.0)));
    let g = graph_with(&["x"], &[]);
    let p = on_hosts(&[("x", "worker1")]);

    let said = run_provisioned(&g, &c, &p, &w, Value::number(0.0))
        .unwrap_err()
        .to_string();

    assert!(said.contains("pickle"), "{said}");
}

#[test]
fn a_broken_artifact_is_rejected_saying_where() {
    let w = Worker::spawn(empty())
        .unwrap()
        .carrying(adds("x=not_a_number"), "rust");
    let mut c = Catalog::new();
    c.insert("x", Arc::new(Add(1.0)));
    let g = graph_with(&["x"], &[]);
    let p = on_hosts(&[("x", "worker1")]);

    let said = run_provisioned(&g, &c, &p, &w, Value::number(0.0))
        .unwrap_err()
        .to_string();

    assert!(said.contains("not_a_number"), "{said}");
}

// ── The artifact is set once ──

#[test]
fn offering_the_same_artifact_twice_does_nothing() {
    // The graph calls `offering` on every run, so repeating it has to be free.
    let w = Worker::spawn(empty())
        .unwrap()
        .carrying(adds("x=5"), "rust");
    let mut c = Catalog::new();
    c.insert("x", Arc::new(Add(1.0)));
    let g = graph_with(&["x"], &[]);
    let p = on_hosts(&[("x", "worker1")]);

    assert!(w.offering(adds("x=5"), "rust").is_ok());
    assert_eq!(
        number(&run_provisioned(&g, &c, &p, &w, Value::number(0.0)).unwrap()),
        5.0
    );
    assert!(
        w.offering(adds("x=5"), "rust").is_ok(),
        "even after greeting"
    );
}

#[test]
fn the_artifact_cannot_be_swapped_once_the_session_is_open() {
    // Staying quiet would leave the client believing it sent nodes that are not
    // there: the session opened with the first artifact and that is what the
    // worker has.
    let w = Worker::spawn(empty())
        .unwrap()
        .carrying(adds("x=5"), "rust");
    let mut c = Catalog::new();
    c.insert("x", Arc::new(Add(1.0)));
    let g = graph_with(&["x"], &[]);
    let p = on_hosts(&[("x", "worker1")]);
    run_provisioned(&g, &c, &p, &w, Value::number(0.0)).unwrap();

    let said = w.offering(adds("x=9"), "rust").unwrap_err().to_string();

    assert!(said.contains("without reconnecting"), "{said}");
}

#[test]
fn before_greeting_the_artifact_can_still_be_swapped() {
    // The greeting is lazy, so until the first job there is nothing to redo.
    let w = Worker::spawn(empty())
        .unwrap()
        .carrying(adds("x=5"), "rust");
    w.offering(adds("x=9"), "rust").unwrap();

    let mut c = Catalog::new();
    c.insert("x", Arc::new(Add(1.0)));
    let g = graph_with(&["x"], &[]);
    let p = on_hosts(&[("x", "worker1")]);

    assert_eq!(
        number(&run_provisioned(&g, &c, &p, &w, Value::number(0.0)).unwrap()),
        9.0
    );
}

// ── The two kinds of worker cannot be confused ──

#[test]
fn offering_an_artifact_to_a_worker_that_already_has_a_catalog_is_rejected() {
    // It is not ignored: throwing away its catalog or silently keeping it would
    // both be silently wrong answers.
    let w = Worker::spawn(Command::new(env!("CARGO_BIN_EXE_test-worker")))
        .unwrap()
        .carrying(adds("x=5"), "rust");
    let c = catalog();
    let g = graph_with(&["a"], &[]);
    let p = on_hosts(&[("a", "worker1")]);

    let said = run_provisioned(&g, &c, &p, &w, Value::number(0.0))
        .unwrap_err()
        .to_string();

    assert!(said.contains("already brings its catalog"), "{said}");
}

#[test]
fn an_empty_worker_nobody_provisions_is_rejected() {
    let w = Worker::spawn(empty()).unwrap();
    let c = catalog();
    let g = graph_with(&["a"], &[]);
    let p = on_hosts(&[("a", "worker1")]);

    let said = run_provisioned(&g, &c, &p, &w, Value::number(0.0))
        .unwrap_err()
        .to_string();

    assert!(said.contains("starts empty"), "{said}");
}

// ── Standing: the worker stops being a child of the client ──
//
// This is the real use case. Everything above starts the worker from the test
// itself, so it dies with it; here the worker stands up on its own, serves
// whoever connects, and is still alive when the client leaves.

use std::io::BufRead;

/// Stands up a standing worker and returns it and its address.
///
/// It reports the address on `stdout` — where there is no wire — because it is
/// asked for port `0`: picking a fixed number in a test is asking for two
/// concurrent runs to collide.
/// A worker standing on a port, and the guarantee that it stops.
///
/// A `Drop` and not a `kill()` on the last line of each test: a test that fails
/// never reaches its last line, and the orphan keeps the test binary's inherited
/// `stderr` open — so `cargo test` waits on that pipe instead of reporting the
/// failure, until whatever timeout is watching gives up.
struct Standing(std::process::Child);

impl Drop for Standing {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn standing(empty: bool) -> (Standing, String) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_test-worker"));
    cmd.args(["--listen", "127.0.0.1:0"]);
    if empty {
        cmd.arg("--empty");
    }
    let mut child = cmd
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("cargo just built the binary");

    let output = child.stdout.take().expect("asked for it with `piped`");
    let mut line = String::new();
    std::io::BufReader::new(output)
        .read_line(&mut line)
        .expect("the worker says where it ended up open before serving anyone");
    let addr = line
        .strip_prefix("LISTEN ")
        .unwrap_or_else(|| panic!("it did not say where it listens: {line:?}"))
        .trim()
        .to_string();
    (Standing(child), addr)
}

#[test]
fn a_standing_worker_serves_whoever_connects() {
    let (_standing, addr) = standing(false);
    let w = Worker::connect(&addr).unwrap();

    let g = graph_with(&["where"], &[]);
    let c = catalog();
    let p = on_hosts(&[("where", "worker1")]);
    let output = run(&g, &c, &p, &w, Value::Null).unwrap();

    assert_ne!(number(&output) as u32, std::process::id());
}

#[test]
fn the_worker_outlives_the_client_leaving() {
    // What separates a worker from a subprocess: two different clients, one
    // after the other, against the same process. If dying with the first were
    // the behaviour, the second could not even connect.
    let (_standing, addr) = standing(false);
    let g = graph_with(&["where"], &[]);
    let c = catalog();
    let p = on_hosts(&[("where", "worker1")]);

    let first = {
        let w = Worker::connect(&addr).unwrap();
        number(&run(&g, &c, &p, &w, Value::Null).unwrap())
    };
    let second = {
        let w = Worker::connect(&addr).unwrap();
        number(&run(&g, &c, &p, &w, Value::Null).unwrap())
    };

    assert_eq!(
        first, second,
        "it is not the same process from one client to the next"
    );
    assert_ne!(first as u32, std::process::id());
}

#[test]
fn a_standing_empty_worker_is_provisioned_and_executes() {
    let (_standing, addr) = standing(true);
    let w = Worker::connect(&addr)
        .unwrap()
        .carrying(adds("x=5"), "rust");

    let mut c = Catalog::new();
    c.insert("x", Arc::new(Add(1.0)));
    let g = graph_with(&["x"], &[]);
    let p = on_hosts(&[("x", "worker1")]);

    assert_eq!(
        number(&run(&g, &c, &p, &w, Value::number(0.0)).unwrap()),
        5.0
    );
}

#[test]
fn the_provisioned_catalog_survives_from_one_client_to_the_next() {
    // The `have`/`want` cache, now between clients and not only between runs:
    // the second client announces the same artifact and the worker answers that
    // it already has it. `times` counts how many catalogs were built there.
    let (_standing, addr) = standing(true);
    let mut c = Catalog::new();
    c.insert("times", Arc::new(Add(0.0)));
    let g = graph_with(&["times"], &[]);
    let p = on_hosts(&[("times", "worker1")]);

    let count = || {
        let w = Worker::connect(&addr)
            .unwrap()
            .carrying(adds("x=5"), "rust");
        number(&run(&g, &c, &p, &w, Value::Null).unwrap())
    };

    assert_eq!(count(), 1.0);
    assert_eq!(count(), 1.0, "the second client resent the artifact");
}

#[test]
fn a_client_is_not_left_working_against_another_ones_catalog() {
    // A worker holds **one** catalog, and two clients at once can bring
    // different artifacts. The second replaces it, and the first has to be told
    // — executing whatever is now there would run somebody else's
    // implementations, and with these ids it would do it in silence.
    let (_standing, addr) = standing(true);
    let first = Worker::connect(&addr)
        .unwrap()
        .carrying(adds("x=5"), "rust");
    let second = Worker::connect(&addr)
        .unwrap()
        .carrying(adds("x=100"), "rust");

    let mut c = Catalog::new();
    c.insert("x", Arc::new(Add(1.0)));
    let g = graph_with(&["x"], &[]);
    let p = on_hosts(&[("x", "worker1")]);

    assert_eq!(
        number(&run(&g, &c, &p, &first, Value::number(0.0)).unwrap()),
        5.0
    );
    assert_eq!(
        number(&run(&g, &c, &p, &second, Value::number(0.0)).unwrap()),
        100.0,
        "the second client's artifact is the one now loaded"
    );

    let complained = run(&g, &c, &p, &first, Value::number(0.0))
        .expect_err("the first client's catalog is no longer there")
        .to_string();

    assert!(complained.contains("text:x=100"), "{complained}");
    assert!(complained.contains("text:x=5"), "{complained}");
}

#[test]
fn connecting_where_there_is_nobody_is_reported_straight_away() {
    // And not when the first job is sent: a closed port is known on opening.
    assert!(Worker::connect("127.0.0.1:1").is_err());
}

#[test]
fn two_simultaneous_connections_against_the_same_standing_worker() {
    // The deadlock the integration test caught: two branches of a wave, two
    // connections, a single worker. If it served one at a time, the second
    // would sit in the `accept` queue waiting for the first to release its own
    // — and the first does not release it until the `forward` finishes.
    let (_standing, addr) = standing(false);
    let (one, other) = (
        Worker::connect(&addr).unwrap(),
        Worker::connect(&addr).unwrap(),
    );

    let g = graph_with(&["left", "right"], &[]);
    let c = catalog();
    let mut p = Placement::new();
    p.place_at("left", Host::new("w1"));
    p.place_at("right", Host::new("w2"));

    let plan = distribute(&compile(&g, &c).unwrap(), &p);
    let output = Executor::new(&c)
        .placed(&p)
        .reaching("w1", &one)
        .reaching("w2", &other)
        .run(&plan, Value::number(0.0))
        .unwrap();

    assert_eq!(
        output,
        Value::map(vec![
            ("left".to_string(), Value::number(1.0)),
            ("right".to_string(), Value::number(1.0)),
        ])
    );
}

// ── That a worker can already know the answer ──

#[test]
fn what_a_worker_already_kept_is_not_run_again() {
    // The `Keeper` reaching the far side. `counts` answers how many times it has
    // been asked **in that process**: a second run that answers `1` did not run
    // it, and the only way it could answer without running is out of the store.
    let shared = Dir::new();
    let w = Worker::spawn(keeping(shared.path())).unwrap();
    let (c, memory) = counting();

    assert_eq!(number(&run_counting(&c, &memory, &w)), 1.0);
    assert_eq!(
        number(&run_counting(&c, &memory, &w)),
        1.0,
        "it ran again over there: what was kept never got looked up"
    );
}

#[test]
fn without_a_keeper_the_same_worker_runs_it_every_time() {
    // The other half, and it is the one that proves the test above is not a
    // fiction: the same worker, the same graph, no keeper.
    let shared = Dir::new();
    let mut without = Command::new(env!("CARGO_BIN_EXE_test-worker"));
    without.args(["--store", &shared.path().to_string_lossy()]);
    let w = Worker::spawn(without).unwrap();
    let (c, memory) = counting();

    assert_eq!(number(&run_counting(&c, &memory, &w)), 1.0);
    assert_eq!(number(&run_counting(&c, &memory, &w)), 2.0);
}

#[test]
fn the_name_of_what_ran_over_there_comes_back() {
    // What makes the chain carry on **below** a slice that went away: whatever
    // reads `counts` next is named out of the name it was given over there.
    let shared = Dir::new();
    let w = Worker::spawn(keeping(shared.path())).unwrap();
    let (c, memory) = counting();
    let here = Dir::new();
    let store = Local::at(here.path()).unwrap();
    let cache = Cache::over(&store);

    let plan = distribute(
        &compile(&graph_with(&["counts"], &[]), &c).unwrap(),
        &on_hosts(&[("counts", "worker1")]),
    );
    let outcome = Executor::new(&c)
        .remembering(&memory)
        .keeping(&cache)
        .placed(&on_hosts(&[("counts", "worker1")]))
        .reaching("worker1", &w)
        .resume(&plan, Value::Null, Vec::new(), Vec::new())
        .unwrap();

    assert_eq!(
        outcome
            .keys
            .iter()
            .map(|(id, _)| id.as_str())
            .collect::<Vec<_>>(),
        ["counts"]
    );
}

/// A worker that keeps both what it is sent and what its nodes produce.
fn keeping(store: &std::path::Path) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_test-worker"));
    cmd.args(["--store", &store.to_string_lossy(), "--keeper"]);
    cmd
}

/// The catalog both sides know, and what is remembered of `counts`: settled, so
/// keeping its output is honest, and named, or there is nothing to build a key
/// out of.
fn counting() -> (Catalog, Memory) {
    let mut catalog = Catalog::new();
    catalog.insert("counts", Arc::new(Add(1.0)));
    let mut memory = Memory::new();
    memory.identify("counts", "Counts");
    memory.freeze("counts", None);
    memory.cache("counts", None);
    (catalog, memory)
}

/// `counts`, executed over there, with a keeper here so the table travels.
fn run_counting(c: &Catalog, memory: &Memory, w: &Worker) -> Value {
    let here = Dir::new();
    let store = Local::at(here.path()).unwrap();
    let cache = Cache::over(&store);
    let p = on_hosts(&[("counts", "worker1")]);
    let plan = distribute(&compile(&graph_with(&["counts"], &[]), c).unwrap(), &p);

    Executor::new(c)
        .remembering(memory)
        .keeping(&cache)
        .placed(&p)
        .reaching("worker1", w)
        .run(&plan, Value::Null)
        .unwrap()
}

#[test]
fn a_client_that_keeps_nothing_still_lets_the_worker_keep() {
    // What is remembered belongs to the **graph**, so it travels even when
    // whoever coordinates has nowhere to keep anything of its own. Before this,
    // the table only went out if there was a keeper here, and a worker with a
    // store was told nothing about the nodes and could name none of them.
    let shared = Dir::new();
    let w = Worker::spawn(keeping(shared.path())).unwrap();
    let (c, memory) = counting();
    let p = on_hosts(&[("counts", "worker1")]);
    let plan = distribute(&compile(&graph_with(&["counts"], &[]), &c).unwrap(), &p);
    let run = || {
        Executor::new(&c)
            .remembering(&memory)
            .placed(&p)
            .reaching("worker1", &w)
            .run(&plan, Value::Null)
            .unwrap()
    };

    assert_eq!(number(&run()), 1.0);
    assert_eq!(
        number(&run()),
        1.0,
        "the worker kept it, and this side keeps nothing at all"
    );
}

// ── What the far side says while it is still working ──

/// Keeps every fact and **when** it arrived, measured from a start the test
/// gives it.
struct Told {
    began: Instant,
    seen: Mutex<Vec<(Fact, Duration)>>,
}

impl Told {
    fn new() -> Self {
        Self {
            began: Instant::now(),
            seen: Mutex::new(Vec::new()),
        }
    }

    fn kinds(&self) -> Vec<String> {
        // Owned: a `Fact::Said` carries a kind that came off a wire, so
        // `flattened` borrows from the fact and there is nothing static to
        // hand back.
        self.seen
            .lock()
            .unwrap()
            .iter()
            .map(|(fact, _)| fact.flattened().0.to_string())
            .collect()
    }

    /// How long after the start the first fact about that node arrived.
    fn when(&self, node: &str) -> Option<Duration> {
        self.seen.lock().unwrap().iter().find_map(|(fact, when)| {
            let (_, fields) = fact.flattened();
            fields
                .iter()
                .any(|(name, what)| name == "node" && what == node)
                .then_some(*when)
        })
    }
}

impl Watcher for Told {
    fn saw(&self, fact: &Fact) {
        self.seen
            .lock()
            .unwrap()
            .push((fact.clone(), self.began.elapsed()));
    }
}

#[test]
fn what_a_real_worker_saw_comes_back_saying_it_was_that_worker() {
    let g = graph_with(&["a", "b"], &[("a", "b")]);
    let placement = on_hosts(&[("a", "worker1"), ("b", "worker1")]);
    let worker = worker();
    let catalog = catalog();
    let plan = distribute(&compile(&g, &catalog).unwrap(), &placement);
    let told = Told::new();

    Executor::new(&catalog)
        .placed(&placement)
        .reaching("worker1", &worker)
        .watching(&told)
        .run(&plan, Value::number(0.0))
        .unwrap();

    // `machine` first: the worker says what it looks like **before** the work
    // rather than after, because a reading taken once the slice is over is a
    // reading of a machine that has just stopped.
    assert_eq!(told.kinds(), ["machine", "ran", "ran", "left", "finished"]);
    let seen = told.seen.lock().unwrap();
    let (_, fields) = seen[1].0.flattened();
    let said: std::collections::HashMap<_, _> = fields.into_iter().collect();
    assert_eq!(said["node"], "a");
    assert_eq!(
        said["host"], "worker1",
        "a worker does not know its own name; the client does"
    );
    let (_, fields) = seen[0].0.flattened();
    let machine: std::collections::HashMap<_, _> = fields.into_iter().collect();
    assert_eq!(
        machine["host"], "worker1",
        "and the reading it sent about itself is attributed the same way, by \
         riding the fact that was already being wrapped"
    );
}

#[test]
fn a_fact_arrives_while_the_work_is_still_going_and_not_with_the_answer() {
    // The whole point of the slice, and the only assertion that can tell the
    // two apart: `slow` takes 300 ms, so the fact about `a` — which ran before
    // it — has to be here long before the answer is. Batched at the end, both
    // would land together.
    let g = graph_with(&["a", "slow"], &[("a", "slow")]);
    let placement = on_hosts(&[("a", "worker1"), ("slow", "worker1")]);
    let worker = worker();
    let catalog = catalog();
    let plan = distribute(&compile(&g, &catalog).unwrap(), &placement);
    let told = Told::new();

    let began = Instant::now();
    Executor::new(&catalog)
        .placed(&placement)
        .reaching("worker1", &worker)
        .watching(&told)
        .run(&plan, Value::number(0.0))
        .unwrap();
    let answered = began.elapsed();

    let first = told.when("a").expect("`a` ran and said so");
    assert!(
        answered >= Duration::from_millis(250),
        "the slow node did not take its time, so this proves nothing: {answered:?}"
    );
    assert!(
        first + Duration::from_millis(200) < answered,
        "the fact about `a` arrived at {first:?} and the answer at {answered:?}: \
         that is close enough together to be a batch, not a stream"
    );
}
