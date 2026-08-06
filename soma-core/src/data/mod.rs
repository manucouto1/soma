//! What flows between nodes, and where it is kept.
//!
//! [`Value`](value::Value) is the one type an edge can carry — a tensor,
//! JSON, text or bytes — and [`Schema`](schema::Schema) is what an edge
//! promises about it, checked at compile time rather than discovered at
//! run time. [`VirtualValue`](virtual_value::VirtualValue) is a value that
//! may not have been produced yet.
//!
//! Two stores sit here, and the difference between them is the reason
//! both exist. A [`DataStore`](store::DataStore) moves values *between
//! machines*: it is addressed by [`DataRef`](store::DataRef) and knows
//! nothing about what produced them. A [`StateStore`](state::StateStore)
//! holds what `fit` learned, which is **authoritative** — losing a cache
//! entry costs recomputation, losing a trained state loses the training.
//! The third store, the discardable one, is [`crate::cache`].
//!
//! [`codec`] is how a `Value` becomes bytes on the way to either.

pub mod codec;
pub mod keys;
pub mod schema;
pub mod state;
pub mod store;
pub mod value;
pub mod virtual_value;
