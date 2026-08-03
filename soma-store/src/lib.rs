//! Remote [`DataStore`] backends.
//!
//! `soma-core` defines the [`DataStore`] trait — that is a contract, and
//! it belongs with the other contracts. These implementations are not:
//! each of them owns a `tokio::runtime::Runtime` and `block_on`s network
//! I/O, so while they lived in `soma-core` every crate that depended on
//! the contract crate inherited a runtime it had no use for.
//!
//! Both backends are feature-gated and off by default. `LocalDataStore`
//! and the trait itself stay in `soma-core`, so a caller that only needs
//! local storage never reaches this crate at all.
//!
//! [`DataStore`]: somatize_core::store::DataStore

#[cfg(feature = "s3")]
pub mod s3;
#[cfg(feature = "s3")]
pub use s3::S3DataStore;

#[cfg(feature = "zarr")]
pub mod zarr;
#[cfg(feature = "zarr")]
pub use zarr::ZarrStore;
