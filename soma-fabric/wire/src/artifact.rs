//! What an empty worker is provisioned with.
//!
//! A generic worker starts **without a catalog**: it knows how to execute plans
//! and does not know what `tokenize` is. The artifact is what tells it, and this
//! crate **does not look at what it carries** — that is the
//! [`Provision`](crate::Provision)'s business. A `cloudpickle` of Python
//! objects, a zip of a package, or a factory name are the same `bytes` field
//! with a different `kind`.
//!
//! The `id` is set by whoever produces it rather than hashed here: **without
//! interpreting the content there is no criterion for saying when two artifacts
//! are the same one**, and two pickles of one catalog can differ byte for byte.
//! Whoever produces it knows what identifies it and says so; here we compare
//! strings.

/// What turns an empty worker into one that can execute your graph.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Artifact {
    /// How it must be interpreted: `pickle`, `package`, `factory`… Text and not
    /// an enum because the vocabulary belongs to the [`Provision`](crate::Provision).
    pub kind: String,
    /// Which one it is, so it is not sent twice. Set by whoever produces it.
    pub id: String,
    /// What it is. Nobody here looks at it.
    pub bytes: Vec<u8>,
}

impl Artifact {
    /// An artifact of this kind, with this identity and these bytes.
    pub fn new(kind: impl Into<String>, id: impl Into<String>, bytes: Vec<u8>) -> Self {
        Self {
            kind: kind.into(),
            id: id.into(),
            bytes,
        }
    }

    /// How it announces itself before being sent: kind and identity, without
    /// the weight.
    pub fn label(&self) -> Label {
        Label {
            kind: self.kind.clone(),
            id: self.id.clone(),
        }
    }
}

/// What an artifact is called, without the artifact.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Label {
    /// How it must be interpreted.
    pub kind: String,
    /// Which one it is.
    pub id: String,
}
