//! Hyperparameter search: what to try, and how to judge what was tried.
//!
//! Two halves that are easy to confuse. [`search`] is the *space* — a
//! [`SearchDimension`](search::SearchDimension) declares that a field can
//! vary and over what range, and [`Searchable`](search::Searchable) is how
//! a filter offers its own dimensions, so the space is assembled from the
//! graph instead of written out beside it.
//!
//! [`study`] is the *campaign* over that space: a [`Study`](study::Study)
//! holds the strategy, the objective and the pruning rule, and a
//! [`Trial`](study::Trial) is one point in it with the metrics it reported.
//!
//! Neither half runs anything. The samplers that walk a space and the
//! pruners that cut a trial short live in `soma-runtime`; this crate says
//! only what a space and a campaign *are*.

pub mod search;
pub mod study;
