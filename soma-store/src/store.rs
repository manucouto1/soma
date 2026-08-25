//! Who keeps what is worth keeping. The hole.

use crate::Digest;
use std::fmt;

/// What you want to remember about something you stored, in the order you say
/// it: the code's fingerprint, which run produced it, what it is.
///
/// Text and not a closed type because the vocabulary is the caller's — the same
/// division of labour as an [`Artifact`]'s `kind`. What this crate does with it
/// is write it down and hand it back.
///
/// [`Artifact`]: https://docs.rs/somatize-fabric-wire
pub type Meta = Vec<(String, String)>;

/// A name, what it points at, and what was said about it.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Bound {
    /// What it is called: a cache key, an artifact's id.
    pub name: String,
    /// The bytes it points at.
    pub digest: Digest,
    /// What the caller wanted to remember.
    pub meta: Meta,
    /// When it was bound, in seconds since the epoch. Stamped here because a
    /// store you cannot sort by time is one you cannot explore.
    pub when: u64,
}

/// Keeps bytes by their content, and names that point at them.
pub trait Store: Send + Sync {
    /// Saves these bytes. Saving the same ones twice is the same as saving them
    /// once — that is what content addressing is for.
    fn put(&self, bytes: &[u8]) -> Result<Digest, StoreError>;

    /// The bytes, if they are here.
    fn get(&self, digest: &Digest) -> Result<Option<Vec<u8>>, StoreError>;

    /// Points a name at some bytes, with what you want to remember about it.
    ///
    /// Binding the same name again replaces it: a name is the question, and the
    /// answer can be refreshed — which is what `.overwrite()` will do.
    fn bind(&self, name: &str, digest: &Digest, meta: Meta) -> Result<(), StoreError>;

    /// Points a name at some bytes **only if nobody has**, and says whether it
    /// did. This is how work gets handed out.
    ///
    /// Not `resolve` and then `bind`: between the two, somebody else does the
    /// same, and two machines train the same round while nobody trains the next
    /// one. It has to be **one** operation that either takes the name or finds
    /// it taken, which is why it is on the trait and has no default — a default
    /// written out of the other two would be a race with a doc comment on it.
    ///
    /// Whoever claims it does the work. Whoever is told `false` goes and asks
    /// for the next thing.
    fn claim(&self, name: &str, digest: &Digest, meta: Meta) -> Result<bool, StoreError>;

    /// What that name points at, if anything.
    fn resolve(&self, name: &str) -> Result<Option<Bound>, StoreError>;

    /// The same for many at once, answering in the order they were asked.
    ///
    /// In the trait from the first day, and not for symmetry: a cache that works
    /// item by item asks thousands at a time, and against a store on the far end
    /// of a network that is thousands of round trips unless it is one call. The
    /// default is the loop; whoever can do better overrides it.
    fn resolve_many(&self, names: &[&str]) -> Result<Vec<Option<Bound>>, StoreError> {
        names.iter().map(|name| self.resolve(name)).collect()
    }

    /// The same for the bytes.
    fn get_many(&self, digests: &[&Digest]) -> Result<Vec<Option<Vec<u8>>>, StoreError> {
        digests.iter().map(|digest| self.get(digest)).collect()
    }

    /// Everything bound here, for whoever wants to look at what there is.
    ///
    /// A scan, and that is the point: the records are the truth, and an index
    /// that answers this faster is something you build from them and can throw
    /// away.
    fn bound(&self) -> Result<Vec<Bound>, StoreError>;
}

/// One record, in the JSON it is kept as.
///
/// Readable with `cat`, which was a requirement before it was a format: a store
/// whose truth you cannot look at is one you cannot debug at three in the
/// morning.
///
/// **Here and not in an implementor**, because it is what makes a directory and
/// a bucket the same store: two copies of this would drift, and the day they did
/// nothing would fail — the records would simply stop being each other's.
pub(crate) fn record(name: &str, digest: &Digest, meta: Meta) -> Result<Vec<u8>, StoreError> {
    let bound = Bound {
        name: name.to_string(),
        digest: digest.clone(),
        meta,
        when: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|since| since.as_secs())
            .unwrap_or(0),
    };
    serde_json::to_vec_pretty(&bound)
        .map_err(|e| StoreError::Corrupt(format!("that record cannot be written: {e}")))
}

/// The other direction.
pub(crate) fn read_record(bytes: &[u8]) -> Result<Bound, StoreError> {
    serde_json::from_slice(bytes)
        .map_err(|e| StoreError::Corrupt(format!("that record cannot be read: {e}")))
}

/// Why something could not be kept, or found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StoreError {
    /// The system said no: no permission, no space, no such directory.
    Io(String),
    /// Something is there and is not what it should be.
    Corrupt(String),
}

impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(why) => write!(f, "the store could not be reached: {why}"),
            Self::Corrupt(why) => write!(f, "what the store has is not what it should be: {why}"),
        }
    }
}

impl std::error::Error for StoreError {}
