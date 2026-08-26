//! Where what is worth keeping is kept: bytes by content, and names that point
//! at them. The hole the core left, plugged with directories, hashes and atomic
//! renames.
//!
//! | map | key | what it is for |
//! |---|---|---|
//! | **blobs** | the digest of the content | weights, artifacts, intermediate data |
//! | **names** | whatever you call it | a cache key, an artifact's id |
//!
//! Two and not one because **a cache key is not a content hash**: it is the hash
//! of the recipe, known *before* the value it names exists. Content addressing
//! alone could not answer *do I already have what this recipe produces?*
//!
//! A blob is written once and never changes; a binding is a small record written
//! by atomic rename. That is what lets a network folder be shared by everyone at
//! once with no lock server, and it turned out to work unchanged against a
//! bucket.
//!
//! | who | what it keeps | under what name |
//! |---|---|---|
//! | a worker's `Serving::store` | artifacts, so a catalog is not sent twice | `artifact:<kind>:<id>` |
//! | [`Cache`] | what a node produced | `value:<key>` |
//! | [`Recorder`] | what happened, one record per `forward` | `run/<id>/<n>` |
//!
//! [`Cache`] is the [`Keeper`](somatize_core::Keeper) and [`Recorder`] the
//! [`Watcher`](somatize_core::Watcher) — the same division from both ends. A
//! queryable index is **derived** from these records and can be thrown away;
//! making it the truth would mean a single writer, which is where NFS breaks.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod cache;
mod digest;
mod local;
mod recorder;
#[cfg(feature = "s3")]
mod s3;
mod store;

pub use cache::{Cache, bytes_of, name_of, value_of};
pub use digest::Digest;
pub use local::Local;
pub use recorder::Recorder;
#[cfg(feature = "s3")]
pub use s3::{Bucket, Credentials, UrlStyle};
pub use store::{Bound, Meta, Store, StoreError};
