//! One client's conversation with one broker, and the wires it has opened.
//!
//! [`Reaching`](crate::Reaching) is one host; this is everything that has to be
//! true across all of them, and there are exactly two such things.
//!
//! # Ask eagerly, connect lazily
//!
//! These pull in opposite directions and both are real, so the split is between
//! them rather than in one direction:
//!
//! - **Asking** where a host is costs tens of bytes and has to happen before
//!   the first node runs, because what gets packed for a host depends on which
//!   hosts turn out to be the same place.
//! - **Connecting** to it costs a socket, a process, or both, and a graph names
//!   hosts a run may never reach — a branch not taken is a worker not needed.
//!
//! So a rendezvous is asked for once and remembered here, and the wire it
//! describes is opened the first time somebody actually sends work. Which also
//! means the ask happens **once** however it was triggered: a client that
//! resolved every host up front to decide what to pack finds the answers already
//! here when the run reaches them.
//!
//! # Two names for one place are one wire
//!
//! The rule [`Path::shared`] states, enforced here because here is the only
//! place that can see two hosts at once. Without it, a process named twice gets
//! provisioned twice — and since a worker has one catalog, the second half
//! replaces the first and takes every activation with it.

use crate::{Ask, Embedded, Endpoint, Host, Needs, Path, Reply, Unanswered};
use somatize_core::{Codec, TransportError};
use somatize_fabric_wire::{Artifact, Worker};
use std::collections::BTreeMap;
use std::process::Command;
use std::sync::{Arc, Mutex, MutexGuard};

/// A client's side of one conversation with one broker.
pub struct Session {
    /// Who to ask.
    broker: Arc<Embedded>,
    /// Where each host turned out to be. One `Reach` per host per session,
    /// whoever asked for it.
    found: Mutex<BTreeMap<Host, Path>>,
    /// The wires already open, by the path they were opened for. Only paths
    /// that [`Path::shared`] agrees about are looked up here; a command is run
    /// again rather than shared.
    wires: Mutex<BTreeMap<Path, Arc<Worker>>>,
    /// Who knows how to write down what would not otherwise cross. Every wire
    /// this opens gets it.
    codec: Option<Arc<dyn Codec>>,
}

impl Session {
    /// A session with this broker.
    pub fn with(broker: Arc<Embedded>) -> Self {
        Self {
            broker,
            found: Mutex::new(BTreeMap::new()),
            wires: Mutex::new(BTreeMap::new()),
            codec: None,
        }
    }

    /// The same session, with somebody who knows how to write down what an
    /// opaque carries. Set before use, so every wire gets the same one.
    pub fn packing(mut self, codec: Arc<dyn Codec>) -> Self {
        self.codec = Some(codec);
        self
    }

    /// Where this host is, asking the broker if nobody has yet.
    ///
    /// Public because deciding what to pack needs it before the run starts:
    /// what goes to a host depends on which hosts are the same place, and that
    /// is what this answers.
    pub fn find(&self, host: &Host) -> Result<Path, TransportError> {
        let mut found = locked(&self.found);
        if let Some(path) = found.get(host) {
            return Ok(path.clone());
        }
        // Not held across the ask: the greeting and the rendezvous are two
        // messages and another host may be resolving at the same time.
        drop(found);

        self.broker.greet().map_err(|why| about(host, &why))?;
        let answer = self
            .broker
            .ask(&Ask::Reach {
                host: host.clone(),
                needs: Needs::default(),
            })
            .map_err(|why| about(host, &why))?;

        let path = match answer {
            Reply::Met { path, .. } => path,
            Reply::Unreachable(why) => return Err(TransportError::new(why)),
            Reply::Refused(why) => {
                return Err(TransportError::new(format!(
                    "the broker closed the session while looking for `{host}`: {why}"
                )));
            }
            other => {
                return Err(TransportError::new(format!(
                    "asked where `{host}` is, the broker answered {other:?}"
                )));
            }
        };

        found = locked(&self.found);
        // Whoever got here first wins, so two threads asking at once still end
        // up agreeing about where `w1` is.
        Ok(found.entry(host.clone()).or_insert(path).clone())
    }

    /// The wire to this host, opening it if this is the first work for it.
    ///
    /// `carries` is what to provision the far side with. Handed to the wire when
    /// it is born and only sent if the far side asks for it — and if this wire
    /// was already open for another name of the same place, the artifact is the
    /// same one by construction, which the wire treats as nothing to do.
    pub fn wire(
        &self,
        host: &Host,
        carries: Option<(Artifact, String)>,
    ) -> Result<Arc<Worker>, TransportError> {
        let path = self.find(host)?;
        let shared = path.shared();

        if shared && let Some(worker) = locked(&self.wires).get(&path) {
            {
                let worker = Arc::clone(worker);
                // Another name for a place already open. The artifact is the
                // same by construction; saying so again is nothing, and saying
                // something different is the refusal the wire owns.
                if let Some((artifact, runtime)) = carries {
                    worker.offering(artifact, runtime)?;
                }
                return Ok(worker);
            }
        }

        let worker = Arc::new(self.open(host, &path, carries)?);
        if shared {
            locked(&self.wires).insert(path, Arc::clone(&worker));
        }
        Ok(worker)
    }

    /// A token that is **equal for two hosts that share a wire** and different
    /// for two that do not.
    ///
    /// It exists for whoever decides what to pack. A worker has one catalog, so
    /// what is packed is packed per *wire* and not per *name* — and the only
    /// thing that knows which names are one wire is this. Handing out a token
    /// rather than the path keeps the rule here: a caller can group by equality
    /// without learning what a path is or when two of them count as one.
    ///
    /// The bytes of the path itself, so two tokens are equal exactly when the
    /// paths are — and with the host appended when the path is not shared, so a
    /// command listed twice is two tokens and gets run twice.
    pub fn wire_token(&self, host: &Host) -> Result<Vec<u8>, TransportError> {
        let path = self.find(host)?;
        let mut token = rmp_serde::to_vec(&path).map_err(|e| {
            TransportError::new(format!(
                "the broker's answer about `{host}` will not write: {e}"
            ))
        })?;
        if !path.shared() {
            token.push(0);
            token.extend_from_slice(host.as_str().as_bytes());
        }
        Ok(token)
    }

    /// Lets a rendezvous go. Best effort, like the message it sends.
    pub fn done(&self, host: &Host) {
        let _ = self.broker.done(host);
    }

    /// The wire that path calls for.
    fn open(
        &self,
        host: &Host,
        path: &Path,
        carries: Option<(Artifact, String)>,
    ) -> Result<Worker, TransportError> {
        let worker = match path {
            Path::Direct {
                endpoint: Endpoint::Address(addr),
            } => Worker::connect(addr).map_err(|e| {
                TransportError::new(format!(
                    "the broker says `{host}` is at {addr}, and nobody is listening there: {e}"
                ))
            })?,
            Path::Direct {
                endpoint: Endpoint::Command(argv),
            } => {
                let (program, rest) = argv.split_first().ok_or_else(|| {
                    TransportError::new(format!(
                        "the broker says `{host}` is a command with no program in it"
                    ))
                })?;
                let mut command = Command::new(program);
                command.args(rest);
                Worker::spawn(command).map_err(|e| {
                    TransportError::new(format!(
                        "the broker says `{host}` is `{}`, which would not start: {e}",
                        argv.join(" ")
                    ))
                })?
            }
            // In the message since the first version on purpose, so that the day
            // the negotiation picks one it is new behaviour and not a new
            // protocol. Until then, saying so beats failing further down as a
            // connection that never happened.
            not_yet => {
                return Err(TransportError::new(format!(
                    "the broker put `{host}` {not_yet}, and this client only knows how to \
                     take the direct path so far; the other three arrive with the negotiation"
                )));
            }
        };

        let worker = match &self.codec {
            Some(codec) => worker.packing(Arc::clone(codec)),
            None => worker,
        };
        Ok(match carries {
            Some((artifact, runtime)) => worker.carrying(artifact, runtime),
            None => worker,
        })
    }
}

/// A poisoned lock is a panic somebody already heard about; what is under it is
/// still what it was.
fn locked<T>(what: &Mutex<T>) -> MutexGuard<'_, T> {
    match what.lock() {
        Ok(one) => one,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Whatever went wrong, said with the host in front of it: *the broker is not
/// answering* is not actionable, and *the broker is not answering about `w1`* is.
fn about(host: &Host, why: &Unanswered) -> TransportError {
    TransportError::new(format!("looking for `{host}`: {why}"))
}
