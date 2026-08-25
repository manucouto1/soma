//! A parquet file in a store, read by spans.

use crate::{Frame, Span};
use arrow_schema::ArrowError;
use arrow_select::concat::concat_batches;
use bytes::Bytes;
use somatize_core::{Ctx, Node, NodeError, Value};
use somatize_store::{Digest, Store};
use std::fmt;
use std::sync::{Arc, OnceLock};
// The crate and this module have the same name, which is right — one file per
// type, and the type is `Parquet` — so every path to the crate says so.
use ::parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

/// A parquet file kept in a store, answering spans of rows.
///
/// ```ignore
/// let sms = Parquet::at(store, "data/sms")?;
/// memory.identify("sms", "Parquet");
/// memory.freeze("sms", Some(sms.version().to_string()));   // and it cost no bytes
/// ```
///
/// # It reads nothing until it is asked
///
/// Declaring it resolves the name and stops there: a graph that names a dataset
/// has not opened it, which is the half of a virtual table that is actually
/// worth having. The bytes arrive on the first span and are held from then on.
///
/// # It reads the version it stated, not the name it was given
///
/// A name in a store points at a digest, and this keeps the **digest**. So a
/// dataset rebound under the same name while a run is in flight does not change
/// what that run is reading, and the key the cache computed still describes the
/// rows that went in. Resolving once is not an optimization, it is what makes
/// the version true.
pub struct Parquet {
    store: Arc<dyn Store>,
    name: String,
    version: Digest,
    file: OnceLock<Bytes>,
}

impl Parquet {
    /// The parquet file bound under this name, or why there is none.
    ///
    /// One `resolve` and no bytes: what this costs is what makes stating a
    /// version affordable at all.
    pub fn at(store: Arc<dyn Store>, name: impl Into<String>) -> Result<Self, ParquetError> {
        let name = name.into();
        let bound = store
            .resolve(&name)
            .map_err(|e| ParquetError(format!("`{name}` could not be looked up: {e}")))?
            .ok_or_else(|| ParquetError(format!("nothing is bound to `{name}` in this store")))?;
        Ok(Self {
            store,
            name,
            version: bound.digest,
            file: OnceLock::new(),
        })
    }

    /// What this dataset is, for the key of everything computed from it.
    ///
    /// The digest of the content, which the store had already worked out when
    /// the bytes were put there. This is what goes to `Memory::freeze`, and it
    /// is the reason a source can be settled without reading itself.
    pub fn version(&self) -> &str {
        self.version.as_str()
    }

    /// The name it was declared under, which is the graph's word and not the
    /// data's.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The rows that span names.
    ///
    /// Short is not an error: the last span of a dataset is whatever is left,
    /// and one past the end is a frame with no rows — which is how a reader
    /// finds out it has arrived.
    pub fn read(&self, span: Span) -> Result<Frame, ParquetError> {
        let file = self.file()?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file.clone())
            .map_err(|e| self.unreadable(e))?
            .with_offset(span.at as usize)
            .with_limit(span.take as usize)
            .with_batch_size(span.take.max(1) as usize);
        let schema = builder.schema().clone();
        let read: Vec<_> = builder
            .build()
            .map_err(|e| self.unreadable(e))?
            .collect::<Result<_, ArrowError>>()
            .map_err(|e| self.unreadable(e))?;
        // Concatenated rather than handed over one at a time: a span that
        // crosses a row group comes back as two batches, and whoever asked for
        // rows 4096..8192 asked for one frame.
        let whole = concat_batches(&schema, read.iter()).map_err(|e| self.unreadable(e))?;
        Ok(Frame::new(whole))
    }

    /// The bytes, fetched once.
    fn file(&self) -> Result<&Bytes, ParquetError> {
        if let Some(held) = self.file.get() {
            return Ok(held);
        }
        let raw = self
            .store
            .get(&self.version)
            .map_err(|e| ParquetError(format!("`{}` could not be read: {e}", self.name)))?
            .ok_or_else(|| {
                ParquetError(format!(
                    "`{}` names {} and there are no such bytes here: a store can be \
                     swept, and what it says it has is a record, not a promise",
                    self.name,
                    self.version.as_str()
                ))
            })?;
        let _ = self.file.set(Bytes::from(raw));
        Ok(self.file.get().expect("just set"))
    }

    /// The same complaint however it arrives, with the name in front of it.
    fn unreadable(&self, why: impl fmt::Display) -> ParquetError {
        ParquetError(format!("`{}` is not readable as parquet: {why}", self.name))
    }
}

impl Node for Parquet {
    /// A span in, a [`Frame`] out.
    fn forward(&self, input: &Value, _ctx: &Ctx<'_>) -> Result<Value, NodeError> {
        let span = Span::of(input).map_err(|e| NodeError::new(e.message()))?;
        Ok(self
            .read(span)
            .map_err(|e| NodeError::new(e.message()))?
            .value())
    }
}

impl fmt::Debug for Parquet {
    /// Without the bytes, and without the store: what identifies it is the name
    /// it was declared under and the content it settled on.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Parquet")
            .field("name", &self.name)
            .field("version", &self.version.as_str())
            .finish()
    }
}

/// Why those rows could not be had.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParquetError(String);

impl ParquetError {
    /// The message.
    pub fn message(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ParquetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ParquetError {}
