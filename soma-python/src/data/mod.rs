//! Values across the boundary, and where they are kept.
//!
//! [`convert`] is the translation itself: a Python object becomes a
//! [`Value`](somatize_core::Value) and back. It is the narrowest and most
//! consequential surface in the crate — `py_to_value` decides that a numeric
//! list is a tensor and a dict is JSON, and those decisions land in cache
//! keys, so changing one moves every key derived from it.
//!
//! [`store`] builds a [`DataStore`](somatize_core::DataStore) from what
//! Python passed. It is shared by `Graph.set_data_store` and
//! `Worker.set_data_store` deliberately: the argument names match because
//! they are read by one function.

pub(crate) mod convert;
pub(crate) mod store;
