//! Where what is worth keeping is kept: bytes by content, and names that point
//! at them.
//!
//! It is the fourth hole of the same shape as the other three — the core
//! provides it and does not know what a saved byte is. Here there are
//! directories, hashes and atomic renames.
//!
//! # Two maps, which is git's shape
//!
//! | map | key | what it is for |
//! |---|---|---|
//! | **blobs** | the digest of the content | weights, artifacts, intermediate data |
//! | **names** | whatever you call it | a cache key, an artifact's id |
//!
//! Two and not one because **the cache key is not a content hash**: it is the
//! hash of the recipe — the node's identity chained onto its predecessors' keys
//! — and it is known *before* the value it names exists. A store with only
//! content addressing could not answer "do I already have what this recipe
//! produces?", which is the whole question.
//!
//! # Immutable, and shared without locks
//!
//! A blob is written once and never changes; a binding is a small record written
//! by **atomic rename**. That is what lets a network folder be shared by
//! everyone at once without a lock server — the same trick optuna's
//! `JournalFileStorage` uses to claim trials, and the same one that will work
//! unchanged against S3.
//!
//! # Who asks it for things
//!
//! | who | what it keeps | under what name |
//! |---|---|---|
//! | a worker's `Serving::store` | artifacts, so a catalog is not sent twice | `artifact:<kind>:<id>` |
//! | [`Cache`] | what a node produced, so it is not computed twice | `value:<key>` |
//!
//! Two questions, one directory, and the namespace of the name is what keeps
//! them apart. [`Cache`] is the [`Keeper`](soma_next_core::Keeper) the core
//! left a hole for: the core cannot hash and has nowhere to put bytes, and both
//! of those live here.
//!
//! An index that can be queried — what do I have, from which run, from when — is
//! **derived** from these records and can be thrown away and rebuilt. Making it
//! the truth would mean a single writer, and a single writer over NFS is exactly
//! where this breaks.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod cache;
mod digest;
mod local;
#[cfg(feature = "s3")]
mod s3;
mod store;

pub use cache::{Cache, bytes_of, value_of};
pub use digest::Digest;
pub use local::Local;
#[cfg(feature = "s3")]
pub use s3::{Bucket, Credentials, UrlStyle};
pub use store::{Bound, Meta, Store, StoreError};
