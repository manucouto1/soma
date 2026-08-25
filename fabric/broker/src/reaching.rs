//! One host, reached through a broker, standing in the engine's `Transport`
//! hole.
//!
//! This is the seam: above it the engine says *carry this slice to `w1`* and
//! knows nothing else; below it a broker was asked where `w1` is and a wire was
//! opened to whatever it said. Neither half learns about the other.
//!
//! Thin on purpose. Everything true across hosts — where each one turned out to
//! be, which of them are the same place — belongs to the
//! [`Session`](crate::Session), because it is the only thing that sees more than
//! one at a time. What is left here is one name, what is staged for it, and the
//! wire once there is one.
//!
//! # The connection waits until somebody sends work
//!
//! A graph names hosts a run may never reach: a branch not taken is a worker not
//! needed. So this opens nothing when it is built. The **rendezvous** may
//! already have happened — deciding what to pack needs it — but a rendezvous is
//! tens of bytes and a connection is a socket or a process.
//!
//! It has a visible consequence worth stating: **an unreachable host now fails
//! when it is needed rather than when it is named.** Before a broker existed,
//! `Worker::at("bad:7000")` failed in the constructor; now that failure surfaces
//! from inside the run. Better behaviour, and a change rather than a side
//! effect.
//!
//! # What is provisioned, and when
//!
//! Packing an artifact is expensive and happens up front, before the first node
//! runs, because a worker has **one** catalog and half of one is a different
//! catalog. Opening a connection is not, and does not. Those two look like they
//! conflict and do not: [`Reaching::offering`] *stages* the artifact, and the
//! bytes only move inside the wire's own greeting, on the first dispatch.
//!
//! So the whole of `Worker::offering`'s rule survives the lazy boundary: the
//! same artifact twice does nothing, and changing one out from under an open
//! session fails rather than swapping a catalog with live state in it.
//!
//! # What is not honoured yet, said out loud
//!
//! A [`Reply::Met`](crate::Reply::Met) can carry a `good_for`, and **nothing
//! here enforces it**. No broker issues one today — the embedded one has no
//! policy — so enforcing it would be a mechanism with no tenant. The day one
//! does, the enforcement is this type's and not the engine's: it is the only
//! thing that knows when the rendezvous was granted.

use crate::{Host, Session};
use soma_fabric_wire::{Artifact, Worker};
use soma_next_core::{Cargo, Outcome, Plan, Transport, TransportError, Watcher};
use std::sync::{Arc, Mutex, MutexGuard};

/// One host, reached through a broker.
pub struct Reaching {
    /// The conversation this host is reached through. Shared: one session
    /// serves every host of a run, greets once for all of them, and is what
    /// notices that two of them are the same place.
    session: Arc<Session>,
    /// The name the graph gave it. Every message about this rendezvous names
    /// it, including the one sent when this is dropped.
    host: Host,
    /// What to provision the far side with, staged until there is a wire to
    /// stage it on. `None` is a worker that brings its own catalog.
    carries: Mutex<Option<(Artifact, String)>>,
    /// The wire, once there is one. Behind a lock because [`Transport`] is
    /// `Sync` and two branches of a wave arrive here at the same time — the
    /// first through opens it and the second finds it open.
    open: Mutex<Option<Arc<Worker>>>,
}

impl Reaching {
    /// A host that will be connected to through this session when somebody
    /// needs it.
    pub fn new(session: Arc<Session>, host: Host) -> Self {
        Self {
            session,
            host,
            carries: Mutex::new(None),
            open: Mutex::new(None),
        }
    }

    /// The name this reaches.
    pub fn host(&self) -> &Host {
        &self.host
    }

    /// Tells it what to provision the far side with, before the first job.
    ///
    /// Staged if the wire is not open yet, handed straight over if it is — and
    /// either way the far side only receives it if it asks, because the wire
    /// announces an artifact's **name** and sends the bytes on request.
    pub fn offering(
        &self,
        artifact: Artifact,
        runtime: impl Into<String>,
    ) -> Result<(), TransportError> {
        let open = locked(&self.open);
        if let Some(worker) = open.as_ref() {
            // Open session: the wire owns this rule and enforces it, including
            // the refusal to swap a catalog with live state behind it.
            return worker.offering(artifact, runtime);
        }
        drop(open);

        let mut carries = locked(&self.carries);
        // Nothing has greeted anybody yet, so replacing is free — and being
        // handed the same one twice is still nothing, which is what a graph run
        // in pieces does.
        match carries.as_ref() {
            Some((already, _)) if already.id == artifact.id => Ok(()),
            _ => {
                *carries = Some((artifact, runtime.into()));
                Ok(())
            }
        }
    }

    /// The wire to this host, opening it if this is the first work for it.
    fn wire(&self) -> Result<Arc<Worker>, TransportError> {
        let mut open = locked(&self.open);
        if let Some(worker) = open.as_ref() {
            return Ok(Arc::clone(worker));
        }
        let worker = self
            .session
            .wire(&self.host, locked(&self.carries).clone())?;
        *open = Some(Arc::clone(&worker));
        Ok(worker)
    }
}

impl Transport for Reaching {
    /// The connection on the first call, and after that this is the wire's
    /// dispatch with one `Arc` clone in front of it.
    fn dispatch(
        &self,
        plan: &Plan,
        cargo: &Cargo<'_>,
        seen: Option<&dyn Watcher>,
    ) -> Result<Outcome, TransportError> {
        self.wire()?.dispatch(plan, cargo, seen)
    }
}

impl Drop for Reaching {
    /// Lets the rendezvous go, so that no client has to remember to.
    ///
    /// Only one that was taken: a handle nobody sent work to never held
    /// anything. Nothing fails if this does not arrive — which is exactly why
    /// the failure is swallowed. A run that finished is not a run to report an
    /// error from.
    fn drop(&mut self) {
        if locked(&self.open).is_some() {
            self.session.done(&self.host);
        }
    }
}

fn locked<T>(what: &Mutex<T>) -> MutexGuard<'_, T> {
    match what.lock() {
        Ok(one) => one,
        Err(poisoned) => poisoned.into_inner(),
    }
}
