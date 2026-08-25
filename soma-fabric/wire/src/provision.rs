//! Who turns an artifact into a catalog. The far side.
//!
//! It is the hole this crate leaves so as not to learn what a `cloudpickle` is.
//! A generic worker starts with a `Provision` and no catalog; when a client
//! arrives, it decides whether to accept it and what to build.
//!
//! Two methods, each answering a question that cannot be answered here:
//!
//! | question | why it is not ours |
//! |---|---|
//! | can this client provision me? | only whoever deserializes knows what couples to what |
//! | what comes out of these bytes? | we do not know what the bytes are |
//!
//! `accepts` is the original's lesson written as a method. The old soma's worker
//! chose an interpreter with `$SOMA_PYTHON` or `python3` from the `PATH`, and a
//! pickled filter can only be rebuilt by an interpreter "close enough" to the
//! one that pickled it — with a different one cloudpickle returns the class's
//! `__dict__` instead of an instance, which surfaces as `'dict' object is not
//! callable` from inside a subprocess, with nothing pointing at the version gap.
//! Hence the client **identifies itself** in the greeting: refusing on connect,
//! with both versions in front of you, is cheaper than anything afterwards.

use somatize_core::Catalog;
use std::fmt;

/// Knows how to turn an artifact into a catalog.
pub trait Provision: Send + Sync {
    /// Whether a client that identifies itself this way — `cpython-3.13/…`,
    /// opaque here — can provision this worker with an artifact of this kind.
    ///
    /// The `kind` matters: one carrying serialized objects demands that both
    /// sides look very much alike, one carrying names and state almost nothing.
    fn accepts(&self, runtime: &str, kind: &str) -> Result<(), ProvisionError>;

    /// What this artifact yields.
    fn provide(&self, kind: &str, bytes: &[u8]) -> Result<Provisioned, ProvisionError>;
}

/// What comes out of an artifact: the implementations.
pub struct Provisioned {
    /// Who executes each node.
    pub catalog: Catalog,
}

impl Provisioned {
    /// An artifact's implementations, unpacked.
    pub fn new(catalog: Catalog) -> Self {
        Self { catalog }
    }
}

/// Why a worker will not let itself be provisioned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProvisionError {
    /// The client and the worker cannot understand each other.
    Incompatible {
        /// How the client identified itself.
        client: String,
        /// And what there is on this side.
        worker: String,
    },
    /// A kind of artifact this worker cannot interpret.
    UnknownKind(String),
    /// It knew how to interpret it and could not.
    Broken(String),
}

impl fmt::Display for ProvisionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Incompatible { client, worker } => write!(
                f,
                "this worker runs `{worker}` and the client `{client}`: what was \
                 serialized there cannot be rebuilt here"
            ),
            Self::UnknownKind(kind) => {
                write!(f, "this worker cannot interpret a `{kind}` artifact")
            }
            Self::Broken(why) => write!(f, "the artifact could not be opened: {why}"),
        }
    }
}

impl std::error::Error for ProvisionError {}
