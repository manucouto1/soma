//! A dataset in a store, read by spans — and what that buys the cache.

use crate::counting::Counting;
use crate::frame::batch;
use crate::tempdir;
use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use somatize_core::{Catalog, Executor, Graph, Memory, Node, Value, compile};
use somatize_data::{Frame, Parquet, Span};
use somatize_store::{Cache, Digest, Local, Meta, Store};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

// ── Fixtures ──

/// `rows` numbered rows, cut into row groups of `per_group`.
pub fn numbers(rows: i64, per_group: usize) -> Vec<u8> {
    let schema = Schema::new(vec![Field::new("n", DataType::Int64, false)]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(Int64Array::from((0..rows).collect::<Vec<_>>()))],
    )
    .unwrap();
    written(&batch, per_group)
}

fn written(batch: &RecordBatch, per_group: usize) -> Vec<u8> {
    let mut out = Vec::new();
    let how = WriterProperties::builder()
        .set_max_row_group_row_count(Some(per_group))
        .build();
    let mut writer = ArrowWriter::try_new(&mut out, batch.schema(), Some(how)).unwrap();
    writer.write(batch).unwrap();
    writer.close().unwrap();
    out
}

/// A store with that file bound under that name.
pub fn holding(store: &dyn Store, name: &str, bytes: &[u8]) -> Digest {
    let digest = store.put(bytes).unwrap();
    store.bind(name, &digest, Meta::new()).unwrap();
    digest
}

/// The column of numbers a frame came back with.
fn column(frame: &Frame) -> Vec<i64> {
    frame
        .batch()
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .values()
        .to_vec()
}

// ── Reading ──

#[test]
fn a_span_is_the_rows_it_names() {
    let where_ = tempdir::Dir::new();
    let store = Arc::new(Local::at(where_.path()).unwrap());
    holding(store.as_ref(), "data/numbers", &numbers(100, 100));

    let source = Parquet::at(store, "data/numbers").unwrap();

    assert_eq!(
        column(&source.read(Span::new(10, 4)).unwrap()),
        [10, 11, 12, 13]
    );
}

#[test]
fn the_last_span_of_a_dataset_is_short_and_that_is_not_an_error() {
    let where_ = tempdir::Dir::new();
    let store = Arc::new(Local::at(where_.path()).unwrap());
    holding(store.as_ref(), "data/numbers", &numbers(10, 10));

    let source = Parquet::at(store, "data/numbers").unwrap();

    assert_eq!(source.read(Span::new(8, 64)).unwrap().rows(), 2);
}

#[test]
fn and_one_past_the_end_is_a_frame_with_no_rows_in_it() {
    // Which is how whoever is walking a dataset finds out they have arrived,
    // without asking anybody how long it was.
    let where_ = tempdir::Dir::new();
    let store = Arc::new(Local::at(where_.path()).unwrap());
    holding(store.as_ref(), "data/numbers", &numbers(10, 10));

    let source = Parquet::at(store, "data/numbers").unwrap();

    assert_eq!(source.read(Span::new(10, 8)).unwrap().rows(), 0);
}

#[test]
fn a_span_that_crosses_a_row_group_still_comes_back_as_one_frame() {
    // Parquet reads a row group at a time, so this arrives as two batches. What
    // was asked for was rows 6..14, and that is one frame.
    let where_ = tempdir::Dir::new();
    let store = Arc::new(Local::at(where_.path()).unwrap());
    holding(store.as_ref(), "data/numbers", &numbers(20, 5));

    let source = Parquet::at(store, "data/numbers").unwrap();

    assert_eq!(
        column(&source.read(Span::new(6, 8)).unwrap()),
        [6, 7, 8, 9, 10, 11, 12, 13]
    );
}

#[test]
fn the_columns_come_back_with_their_names_and_their_types() {
    let where_ = tempdir::Dir::new();
    let store = Arc::new(Local::at(where_.path()).unwrap());
    holding(store.as_ref(), "data/sms", &written(&batch(), 1024));

    let frame = Parquet::at(store, "data/sms")
        .unwrap()
        .read(Span::new(0, 2))
        .unwrap();

    let names: Vec<&str> = frame
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().as_str())
        .collect();
    assert_eq!(names, ["sms", "label"]);
    assert_eq!(frame.rows(), 2);
}

// ── Saying what it is, without reading itself ──

#[test]
fn the_version_is_what_the_store_already_knew() {
    let where_ = tempdir::Dir::new();
    let store = Arc::new(Local::at(where_.path()).unwrap());
    let digest = holding(store.as_ref(), "data/numbers", &numbers(10, 10));

    let source = Parquet::at(store, "data/numbers").unwrap();

    assert_eq!(source.version(), digest.as_str());
}

#[test]
fn the_same_data_under_two_names_is_the_same_version() {
    // Which is what makes a key describe the rows and not the paperwork: two
    // people who bound the same dataset differently share a cache.
    let where_ = tempdir::Dir::new();
    let store = Arc::new(Local::at(where_.path()).unwrap());
    let bytes = numbers(10, 10);
    holding(store.as_ref(), "mine/numbers", &bytes);
    holding(store.as_ref(), "yours/numbers", &bytes);

    let mine = Parquet::at(store.clone(), "mine/numbers").unwrap();
    let yours = Parquet::at(store, "yours/numbers").unwrap();

    assert_eq!(mine.version(), yours.version());
}

#[test]
fn and_different_data_under_one_name_is_a_different_version() {
    let where_ = tempdir::Dir::new();
    let store = Arc::new(Local::at(where_.path()).unwrap());
    holding(store.as_ref(), "data/numbers", &numbers(10, 10));
    let before = Parquet::at(store.clone(), "data/numbers")
        .unwrap()
        .version()
        .to_string();

    holding(store.as_ref(), "data/numbers", &numbers(20, 10));
    let after = Parquet::at(store, "data/numbers").unwrap();

    assert_ne!(after.version(), before);
}

#[test]
fn a_name_nobody_bound_says_so_before_anything_runs() {
    let where_ = tempdir::Dir::new();
    let store = Arc::new(Local::at(where_.path()).unwrap());

    let why = Parquet::at(store, "data/nothing").unwrap_err();

    assert!(why.message().contains("data/nothing"), "{why}");
}

// ── What it costs ──

#[test]
fn declaring_a_dataset_does_not_open_it() {
    // The half of a virtual table worth having: a graph that names a dataset
    // has not read it. One `resolve`, and the version is already known.
    let where_ = tempdir::Dir::new();
    let store = Arc::new(Counting::over(Local::at(where_.path()).unwrap()));
    holding(store.as_ref(), "data/numbers", &numbers(1000, 100));
    store.forget();

    let source = Parquet::at(store.clone(), "data/numbers").unwrap();

    assert_eq!(store.seen(), (1, 0), "one lookup, and no bytes");
    assert!(!source.version().is_empty());
}

#[test]
fn and_the_bytes_arrive_once_however_many_spans_are_asked_for() {
    let where_ = tempdir::Dir::new();
    let store = Arc::new(Counting::over(Local::at(where_.path()).unwrap()));
    holding(store.as_ref(), "data/numbers", &numbers(1000, 100));
    let source = Parquet::at(store.clone(), "data/numbers").unwrap();
    store.forget();

    for at in [0, 64, 128] {
        source.read(Span::new(at, 64)).unwrap();
    }

    assert_eq!(store.seen(), (0, 1), "three spans, one fetch");
}

// ── In a graph ──

/// Counts the rows it was handed, and how many times it was asked at all.
pub struct Rows(pub AtomicUsize);

impl Node for Rows {
    fn forward(
        &self,
        input: &Value,
        _ctx: &somatize_core::Ctx<'_>,
    ) -> Result<Value, somatize_core::NodeError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        let frame = Frame::of(input).expect("a frame arrived");
        Ok(Value::number(frame.rows() as f64))
    }
}

/// A graph of two nodes: the dataset, and something that reads a frame.
///
/// `keeping` is whether the **source**'s own output is worth keeping, which is
/// only honest once there is a codec that can write a frame down.
pub fn reading(source: Parquet, keeping: bool) -> (Graph, Catalog, Memory, Arc<Rows>) {
    let mut graph = Graph::new();
    let mut catalog = Catalog::new();
    let mut memory = Memory::new();

    memory.identify("sms", "Parquet");
    // The version, said the way the digest of settled weights is said: the
    // declaration froze it, and whoever knows how to hash says it again.
    memory.freeze("sms", Some(source.version().to_string()));
    if keeping {
        memory.cache("sms", None);
    }
    graph.add_node("sms").unwrap();
    catalog.insert("sms", Arc::new(source));

    let rows = Arc::new(Rows(AtomicUsize::new(0)));
    memory.identify("rows", "Rows");
    memory.freeze("rows", None);
    memory.cache("rows", None);
    graph.add_node("rows").unwrap();
    catalog.insert("rows", rows.clone());
    graph.add_edge("sms", "rows").unwrap();

    (graph, catalog, memory, rows)
}

#[test]
fn a_source_is_a_node_and_the_graph_is_handed_a_span() {
    let where_ = tempdir::Dir::new();
    let store = Arc::new(Local::at(where_.path()).unwrap());
    holding(store.as_ref(), "data/numbers", &numbers(1000, 100));
    let (graph, catalog, ..) = reading(Parquet::at(store, "data/numbers").unwrap(), false);

    let plan = compile(&graph, &catalog).unwrap();
    let out = Executor::new(&catalog)
        .run(&plan, Span::new(0, 64).value())
        .unwrap();

    assert_eq!(out, Value::number(64.0));
}

#[test]
fn the_second_run_finds_the_answer_under_a_name_it_could_work_out() {
    // The whole point, end to end. What names the rows is the span and the
    // version, and both are known **before anything is read** — so the second
    // run recognizes the question, does not answer it again, and **does not
    // open the dataset**: nothing reads what the source makes that is not
    // already kept, so the source has nothing left to be for.
    let where_ = tempdir::Dir::new();
    let disk = Local::at(where_.path()).unwrap();
    let store = Arc::new(Counting::over(disk));
    let dataset = holding(store.as_ref(), "data/numbers", &numbers(1000, 100));

    let asked = |store: Arc<Counting>| {
        let source = Parquet::at(store.clone(), "data/numbers").unwrap();
        let (graph, catalog, memory, rows) = reading(source, false);
        let plan = compile(&graph, &catalog).unwrap();
        let cache = Cache::over(store.as_ref());
        let out = Executor::new(&catalog)
            .remembering(&memory)
            .keeping(&cache)
            .run(&plan, Span::new(0, 64).value())
            .unwrap();
        (out, rows.0.load(Ordering::SeqCst))
    };

    let (first, ran) = asked(store.clone());
    assert_eq!((first, ran), (Value::number(64.0), 1));

    store.forget();
    let (again, ran) = asked(store.clone());

    assert_eq!(again, Value::number(64.0), "the same answer");
    assert_eq!(ran, 0, "and nobody counted rows again");
    assert!(
        !store.fetched(&dataset),
        "the parquet was opened again to feed a node that never ran",
    );
}
