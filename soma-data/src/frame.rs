//! A batch of columns, which is what a source answers with.

use arrow_array::RecordBatch;
use arrow_ipc::reader::StreamReader;
use arrow_ipc::writer::StreamWriter;
use arrow_schema::SchemaRef;
use arrow_select::concat::concat_batches;
use somatize_core::Value;
use std::fmt;

/// Rows and columns, together with what each column is called and holds.
///
/// A type of ours rather than a `RecordBatch` in an `Opaque` for two reasons
/// that are not style: a codec is an `impl` of somebody else's trait and the
/// orphan rule wants one of the two types to be ours, and this is the word the
/// design is written in — *the difference between training and deploying is how
/// many rows the frame brings*.
///
/// It is not a tensor and does not want to be. Numbers and fixed-size lists of
/// them are contiguous Arrow buffers, so converting is a reshape and not a copy;
/// text is not a tensor until something tokenizes it. **The conversion is a
/// node**, and the same node whether the frame came from a file or a topic.
#[derive(Debug, Clone)]
pub struct Frame(RecordBatch);

impl Frame {
    /// What gets written beside the bytes. Named after the **format** and not
    /// the language: what is on disk is an Arrow IPC stream, and whoever reads
    /// it back may be a `polars` on the other side of the wall.
    pub const KIND: &'static str = "arrow.RecordBatch";

    /// This batch, as a frame.
    pub fn new(batch: RecordBatch) -> Self {
        Self(batch)
    }

    /// How many rows it brings. The only number a caller usually wants: it is
    /// what says whether a span was short because the dataset ended.
    pub fn rows(&self) -> usize {
        self.0.num_rows()
    }

    /// What the columns are called and what they hold — **without reading a
    /// value**, which is the half of a virtual table worth having.
    pub fn schema(&self) -> &SchemaRef {
        self.0.schema_ref()
    }

    /// The batch itself, for whoever brought their own engine.
    pub fn batch(&self) -> &RecordBatch {
        &self.0
    }

    /// As a value, which is how it crosses an edge. `Opaque` and not a variant
    /// of its own: the core has no dependencies and is not going to learn what a
    /// column is.
    pub fn value(self) -> Value {
        Value::opaque(self)
    }

    /// The frame this value carries, if it carries one.
    pub fn of(value: &Value) -> Option<&Self> {
        value.downcast::<Self>()
    }

    /// What it weighs in bytes: Arrow IPC, buffers and all. No encoding pass and
    /// no per-value work, which is the reason Arrow is the type that crosses an
    /// edge rather than something converted to at the edges.
    pub fn written(&self) -> Result<Vec<u8>, FrameError> {
        let mut out = Vec::new();
        let mut writer = StreamWriter::try_new(&mut out, self.schema())
            .map_err(|e| FrameError(format!("a frame could not be written down: {e}")))?;
        writer
            .write(&self.0)
            .map_err(|e| FrameError(format!("a frame could not be written down: {e}")))?;
        writer
            .finish()
            .map_err(|e| FrameError(format!("a frame could not be written down: {e}")))?;
        drop(writer);
        Ok(out)
    }

    /// And back.
    ///
    /// One frame, however many batches the stream holds: what was written down
    /// was a frame, and a frame is what comes back.
    pub fn read(bytes: &[u8]) -> Result<Self, FrameError> {
        let reader = StreamReader::try_new(bytes, None)
            .map_err(|e| FrameError(format!("those bytes are not a frame: {e}")))?;
        let schema = reader.schema();
        let batches: Vec<RecordBatch> = reader
            .collect::<Result<_, _>>()
            .map_err(|e| FrameError(format!("those bytes are not a frame: {e}")))?;
        let whole = concat_batches(&schema, batches.iter())
            .map_err(|e| FrameError(format!("those rows would not join up: {e}")))?;
        Ok(Self(whole))
    }
}

/// Why that could not be written down, or read back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameError(String);

impl FrameError {
    /// The message.
    pub fn message(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for FrameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for FrameError {}
