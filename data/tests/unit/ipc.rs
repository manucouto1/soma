//! Writing a frame down, so it can be kept and so it can be sent.

use crate::counting::Counting;
use crate::frame::batch;
use crate::parquet::{holding, numbers, reading};
use crate::tempdir;
use soma_next_core::{Codec, Packing};
use soma_next_core::{Executor, Value, compile};
use soma_next_data::{Frame, Ipc, Parquet, Span};
use soma_next_store::{Cache, Local};
use std::sync::Arc;
use std::sync::atomic::Ordering;

#[test]
fn a_frame_written_down_and_read_back_is_the_same_frame() {
    let there = Ipc.packed(&Frame::new(batch()).value()).unwrap();

    let back = Ipc.unpacked(&there).unwrap();

    assert_eq!(Frame::of(&back).unwrap().batch(), &batch());
}

#[test]
fn and_what_it_became_can_leave_the_process() {
    // The whole point of a codec: before, the value could not cross; after, it
    // is maps and bytes, and `travels` says so without being asked to relax.
    let written = Ipc.packed(&Frame::new(batch()).value()).unwrap();

    assert!(written.travels());
}

#[test]
fn a_frame_is_found_however_deep_it_is() {
    let deep = Value::map(vec![(
        "batch".to_string(),
        Value::list(vec![Frame::new(batch()).value()]),
    )]);

    let there = Ipc.packed(&deep).unwrap();
    assert!(there.travels());

    let back = Ipc.unpacked(&there).unwrap();
    let Some(Value::List(items)) = back.get("batch") else {
        panic!("the shape has to survive the round trip");
    };
    assert_eq!(Frame::of(&items[0]).unwrap().rows(), 3);
}

#[test]
fn a_value_with_nothing_opaque_in_it_comes_back_as_it_was() {
    let plain = Span::new(0, 64).value();

    assert_eq!(Ipc.packed(&plain).unwrap(), plain);
    assert_eq!(Ipc.unpacked(&plain).unwrap(), plain);
}

#[test]
fn an_opaque_that_is_not_a_frame_says_so() {
    let why = Ipc
        .packed(&Value::opaque("not a frame".to_string()))
        .unwrap_err();

    assert!(why.to_string().contains("not a frame"), "{why}");
}

#[test]
fn and_somebody_elses_kind_is_left_exactly_as_it_arrived() {
    // A tensor written down by the Python side, passing through a process that
    // only knows frames. Not ours to read, and not ours to lose.
    let theirs = soma_next_core::written_down("torch.Tensor", vec![1, 2, 3]);

    assert_eq!(Ipc.unpacked(&theirs).unwrap(), theirs);
}

#[test]
fn a_kept_frame_means_the_dataset_is_not_opened_again() {
    // The other way of not opening it, and the one that works when something
    // downstream **did** have to run: the frame itself is keepable, so the
    // source is the node that hits. The second pass never asks for the parquet.
    let where_ = tempdir::Dir::new();
    let store = Arc::new(Counting::over(Local::at(where_.path()).unwrap()));
    let dataset = holding(store.as_ref(), "data/numbers", &numbers(1000, 100));

    let asked = || {
        let source = Parquet::at(store.clone(), "data/numbers").unwrap();
        let (graph, catalog, memory, rows) = reading(source, true);
        let plan = compile(&graph, &catalog).unwrap();
        let cache = Cache::over(store.as_ref());
        let packing = Packing::over(&cache, &Ipc);
        let out = Executor::new(&catalog)
            .remembering(&memory)
            .keeping(&packing)
            .run(&plan, Span::new(0, 64).value())
            .unwrap();
        (out, rows.0.load(Ordering::SeqCst))
    };

    let (first, ran) = asked();
    assert_eq!((first, ran), (Value::number(64.0), 1));

    store.forget();
    let (again, ran) = asked();

    assert_eq!(again, Value::number(64.0), "the same answer");
    assert_eq!(ran, 0, "and nobody counted rows again");
    assert!(
        !store.fetched(&dataset),
        "the dataset was opened again, which is the thing this buys"
    );
}
