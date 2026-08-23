//! Who is told what happened. The fifth hole.
//!
//! The core provides it and does not fill it, like the other four: it says what
//! a [`Fact`] is and has nowhere to put one. Where a fact ends up — a file, a
//! bucket, a notebook drawing a curve while it is still being drawn — is the
//! business of whoever runs, and it is injected the way a
//! [`Keeper`](crate::Keeper) is.
//!
//! # Emitting is synchronous. Delivering is not the core's problem.
//!
//! [`saw`](Watcher::saw) is called from the walk and returns. What the
//! implementor does with the fact — write it, drop it, push it onto a channel
//! that another thread drains into a figure — is where anything asynchronous
//! belongs, and it is why this needs no runtime. That matters more than it
//! looks: an `async` here would be `async` in every caller of the engine, and
//! that is the objection that has twice kept a bus out of this project.
//!
//! # Called from several threads
//!
//! A [`Wave`](crate::Plan::Wave) runs its branches in a `std::thread::scope`, so
//! several of them call this at once — hence `Send + Sync`, and hence the order
//! facts arrive in is **not** the order they happened in. Whoever writes them
//! down decides what to do about that; the engine will not serialize a run to
//! make a log tidy.
//!
//! # There is one, and fanning out is not the core's either
//!
//! Not a list, not a registry, no `subscribe`. If you want to write **and**
//! draw, that is an implementor holding two, in whichever crate needs it. A hole
//! that starts managing its own tenants stops being a hole.

use crate::Fact;

/// Told what happened, as it happens.
pub trait Watcher: Send + Sync {
    /// One fact. Whatever this does, it does not fail: there is no useful
    /// answer to "the log could not be written" in the middle of a run, and a
    /// `Result` here would make every emit site a decision about somebody
    /// else's storage.
    fn saw(&self, fact: &Fact);
}
