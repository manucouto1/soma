//! The far side: serving whatever arrives, over standard input or a port.
//!
//! Two axes, and they do not mix:
//!
//! - **where it listens**: [`Serving::over_stdin`] talks to a client that
//!   started it; [`Serving::listen`] opens a port and serves whoever comes. The
//!   second is what makes a worker an independent process.
//! - **where it gets what it executes**: it brings it, or it is sent it.
//!
//! On the second, there are **two kinds of worker** and neither can pretend to
//! be the other:
//!
//! | | where it gets it | who uses it |
//! |---|---|---|
//! | [`Serving::own`] | it brings it | a Rust binary with its nodes |
//! | [`Serving::provisioned`] | the client sends it | the generic worker: `pip install` and nothing else |
//!
//! Two constructors and not an optional parameter because they reject different
//! things: offering the first an artifact is an error, and not offering the
//! second one is too. An `Option` would have turned both rejections into a
//! branch that gets forgotten.
//!
//! # Where the driver comes from, which is the same place as the nodes
//!
//! A worker that brings its own catalog brings its own driver, with
//! [`Serving::driver`]. One that is provisioned gets both **in the artifact**:
//! whoever packs the nodes packs the driver, and this side rebuilds them
//! together. Declared versus injected is about the **graph** — a node is in it
//! and a driver is not — and not about how either one gets here.
//!
//! When both are there, the one that **arrived** wins: it belongs to the job,
//! and the local one is what serves clients that pack none.
//!
//! What arrives is **cached by the artifact's id**, so a second run resends
//! nothing. One is kept, not a map: collecting Python catalogs would be
//! collecting live objects with nobody saying when they are released.
//!
//! # One thread per conversation, and why
//!
//! The first version served connections **one at a time** — it seemed consistent
//! with "a worker is one process" — and deadlocked: two branches of a wave
//! against the same worker open two connections, the second sits in the `accept`
//! queue, and the first does not release its own until the `forward` finishes.
//! The integration test caught it, which is where it had to show. The lesson:
//! serializing was right, but **at message granularity, not session
//! granularity**.
//!
//! # `stdout` is the wire
//!
//! Not one `println!` in a worker: the messages go over its standard output. For
//! talking there is `stderr`, which [`Worker`](crate::Worker) leaves inherited
//! for exactly that. In Python this is more dangerous, because a stray `print`
//! in a user's node — or in a library on import — does the same thing.

use crate::frame;
use crate::{Answer, Label, Provision, Provisioned, Request};
use soma_next_core::{Catalog, Driver, Executor, Keeper};
use soma_next_store::{Store, StoreError};
use std::io::{self, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, ToSocketAddrs};
use std::sync::{Arc, Mutex, MutexGuard};

/// A worker about to serve: what it executes with, and where it listens.
pub struct Serving<'a> {
    source: Source<'a>,
    driver: Option<&'a dyn Driver>,
    store: Option<&'a dyn Store>,
    keeper: Option<&'a dyn Keeper>,
}

impl<'a> Serving<'a> {
    /// A worker that brings its own catalog.
    pub fn own(catalog: &'a Catalog) -> Self {
        Self {
            source: Source::Own(catalog),
            driver: None,
            store: None,
            keeper: None,
        }
    }

    /// A worker that starts **empty** and is sent what to execute, with this to
    /// interpret it.
    pub fn provisioned(provision: &'a dyn Provision) -> Self {
        Self {
            source: Source::Sent(provision),
            driver: None,
            store: None,
            keeper: None,
        }
    }

    /// The same worker, with whoever serves what the steps here ask for.
    ///
    /// For a provisioned one this is the fallback: a driver that arrives in an
    /// artifact wins over it.
    pub fn driver(mut self, driver: &'a dyn Driver) -> Self {
        self.driver = Some(driver);
        self
    }

    /// The same worker, with somewhere to keep the artifacts it is sent.
    ///
    /// This is the `have`/`want` finally having a `have`: on being offered an
    /// artifact it already has **in the store**, it says `Ready` and not a byte
    /// crosses. A shared folder between workers means the second one to be stood
    /// up is provisioned without the client noticing.
    pub fn store(mut self, store: &'a dyn Store) -> Self {
        self.store = Some(store);
        self
    }

    /// The same worker, able to keep what the slices it runs produce.
    ///
    /// Separate from [`Serving::store`] and not derived from it: that one keeps
    /// **artifacts**, so that a worker is not sent a catalog it already has;
    /// this one keeps **values**, so that a node whose answer is already known
    /// is not run at all. Both can be the same directory underneath, and the two
    /// questions still have nothing to do with each other.
    ///
    /// What is remembered about each node does not come from here: it arrives
    /// with the work, because it belongs to the graph and the graph is over
    /// there.
    pub fn keeping(mut self, keeper: &'a dyn Keeper) -> Self {
        self.keeper = Some(keeper);
        self
    }

    /// Serves over standard input, until the client closes.
    ///
    /// Returns `Ok(())` when the input ends **between** messages, which is how
    /// it normally finishes. A failure of what gets executed is not an error
    /// here: it travels back as an answer.
    pub fn over_stdin(self) -> io::Result<()> {
        let shared = Shared::of(self);
        attend(&shared, io::stdin().lock(), io::stdout().lock())
    }

    /// Stands on `addr` and serves whoever connects. It does not return: it
    /// stops by being shut down, and a client that cuts out does not stop it.
    pub fn listen(self, addr: impl ToSocketAddrs) -> io::Result<()> {
        self.listen_at(addr, |_| {})
    }

    /// The same, reporting which address it ended up open on, so port `0` can
    /// be asked for.
    pub fn listen_at(
        self,
        addr: impl ToSocketAddrs,
        opened: impl FnOnce(SocketAddr),
    ) -> io::Result<()> {
        let listener = TcpListener::bind(addr)?;
        opened(listener.local_addr()?);

        let shared = Shared::of(self);
        let shared = &shared;

        std::thread::scope(|scope| {
            let mut alive: Vec<std::thread::ScopedJoinHandle<'_, ()>> = Vec::new();
            for arrival in listener.incoming() {
                let socket = match arrival {
                    Ok(socket) => socket,
                    // An `accept` that fails is noted, and we keep listening.
                    Err(e) => {
                        eprintln!("could not accept a connection: {e}");
                        continue;
                    }
                };
                let _ = socket.set_nodelay(true);
                // Or a months-old worker accumulates one handle per client served.
                alive.retain(|thread| !thread.is_finished());
                alive.push(scope.spawn(move || {
                    let Ok(copy) = socket.try_clone() else {
                        return;
                    };
                    if let Err(e) = attend(shared, BufReader::new(copy), socket) {
                        eprintln!("a session was cut off: {e}");
                    }
                }));
            }
        });
        Ok(())
    }
}

/// What every session on this worker has in common. One per [`Serving`], lent
/// to each session, so the catalog that arrives serves the next client too.
struct Shared<'a> {
    source: Source<'a>,
    driver: Option<&'a dyn Driver>,
    store: Option<&'a dyn Store>,
    keeper: Option<&'a dyn Keeper>,
    loaded: Mutex<Loaded<'a>>,
}

impl<'a> Shared<'a> {
    fn of(serving: Serving<'a>) -> Self {
        let loaded = match &serving.source {
            Source::Own(catalog) => Loaded::Own(catalog),
            Source::Sent(_) => Loaded::Empty,
        };
        Self {
            source: serving.source,
            driver: serving.driver,
            store: serving.store,
            keeper: serving.keeper,
            loaded: Mutex::new(loaded),
        }
    }

    /// What is inside, even if another session broke: a poisoned `Mutex` holds
    /// a catalog, not a half-finished invariant.
    fn held(&self) -> MutexGuard<'_, Loaded<'a>> {
        match self.loaded.lock() {
            Ok(inside) => inside,
            Err(poisoned) => poisoned.into_inner(),
        }
    }
}

/// What one client's conversation remembers between messages. Per session, so a
/// client that cuts out leaves nothing behind.
#[derive(Default)]
struct Session {
    /// What was asked for, between the `Send` and the `Provision`, which
    /// arrives without its name.
    awaiting: Option<(String, String)>,
    /// The artifact this client greeted with, if it brought one. What it gets
    /// checked against is [`Shared::loaded`], on every job.
    mine: Option<String>,
}

/// Where this worker's catalog comes from.
enum Source<'a> {
    /// It brings it.
    Own(&'a Catalog),
    /// The client sends it, and this knows how to interpret it.
    Sent(&'a dyn Provision),
}

/// What this worker can execute right now.
enum Loaded<'a> {
    /// Nothing yet: nobody has greeted.
    Empty,
    /// The one it brought.
    Own(&'a Catalog),
    /// One that arrived, and which artifact it came from.
    Sent {
        id: String,
        catalog: Catalog,
        driver: Option<Arc<dyn Driver>>,
    },
}

impl Loaded<'_> {
    /// What to execute with right now: the catalog, and the driver that came
    /// with it if one did.
    fn ready(&self) -> Option<(Catalog, Option<Arc<dyn Driver>>)> {
        match self {
            Self::Empty => None,
            Self::Own(catalog) => Some(((*catalog).clone(), None)),
            Self::Sent {
                catalog, driver, ..
            } => Some((catalog.clone(), driver.clone())),
        }
    }
}

/// The loop, with both ends as arguments so it is testable without a process.
fn attend(shared: &Shared<'_>, mut input: impl Read, mut output: impl Write) -> io::Result<()> {
    let mut session = Session::default();

    while let Some(payload) = frame::recv(&mut input)? {
        let answer = match Request::from_bytes(&payload) {
            Err(e) => Answer::Refused(e.to_string()),
            Ok(request) => reply(shared, &mut session, request),
        };
        let encoded = answer.to_bytes().unwrap_or_else(|e| {
            // If not even the answer can be encoded, that fact is the answer:
            // staying quiet would leave the other side waiting forever.
            Answer::Failed(e.to_string())
                .to_bytes()
                .expect("text can always be written")
        });
        frame::send(&mut output, &encoded)?;
    }
    Ok(())
}

fn reply(shared: &Shared<'_>, session: &mut Session, request: Request) -> Answer {
    match request {
        Request::Hello { runtime, offering } => match (&shared.source, offering) {
            // A worker with its own catalog, and a client that brings nothing.
            (Source::Own(_), None) => Answer::Ready,
            (Source::Own(_), Some(label)) => Answer::Refused(format!(
                "this worker already brings its catalog and does not accept a `{}` artifact",
                label.kind
            )),
            (Source::Sent(_), None) => Answer::Refused(
                "this worker starts empty and the client brings nothing to provision it with"
                    .into(),
            ),
            (Source::Sent(provision), Some(label)) => {
                if let Err(e) = provision.accepts(&runtime, &label.kind) {
                    return Answer::Refused(e.to_string());
                }
                session.mine = Some(label.id.clone());
                // The `have`/`want`. Already open with this artifact, or in the
                // store, means not a byte crosses.
                if matches!(&*shared.held(), Loaded::Sent { id, .. } if *id == label.id) {
                    return Answer::Ready;
                }
                match kept(shared.store, *provision, &label) {
                    Err(e) => Answer::Refused(e),
                    Ok(Some(provisioned)) => {
                        *shared.held() = Loaded::Sent {
                            id: label.id,
                            catalog: provisioned.catalog,
                            driver: provisioned.driver,
                        };
                        Answer::Ready
                    }
                    Ok(None) => {
                        session.awaiting = Some((label.kind, label.id));
                        Answer::Send
                    }
                }
            }
        },
        Request::Provision { bytes } => {
            let Source::Sent(provision) = &shared.source else {
                return Answer::Refused("this worker is not provisioned".into());
            };
            let Some((kind, id)) = session.awaiting.take() else {
                return Answer::Refused("an artifact arrived that nobody had asked for".into());
            };
            match provision.provide(&kind, &bytes) {
                Ok(provisioned) => {
                    // Kept for the next worker to be stood up, and for this one
                    // if it is restarted. A failure to keep it is not a failure
                    // to work: it will just be sent again.
                    if let Some(Err(e)) = shared.store.map(|store| keep(store, &kind, &id, &bytes))
                    {
                        eprintln!("the artifact could not be kept: {e}");
                    }
                    *shared.held() = Loaded::Sent {
                        id,
                        catalog: provisioned.catalog,
                        driver: provisioned.driver,
                    };
                    Answer::Ready
                }
                Err(e) => Answer::Refused(e.to_string()),
            }
        }
        Request::Work {
            plan,
            input,
            known,
            keys,
            placement,
            memory,
        } => {
            // The lock is released before executing — a `Catalog` clones by
            // `Arc` — or every client would serialize against the run.
            let ready = {
                let loaded = shared.held();
                match (&session.mine, &*loaded) {
                    // A worker holds **one** catalog. If somebody else
                    // provisioned it with another artifact after this client
                    // greeted, executing now would run their implementations —
                    // and an id that exists in both would do it in silence. It
                    // is checked here and not at the greeting because that is
                    // where it can go wrong, and here there is no race to lose.
                    (Some(mine), Loaded::Sent { id, .. }) if mine != id => {
                        return Answer::Failed(format!(
                            "this worker was provisioned with `{id}` after you greeted \
                             with `{mine}`, and it holds one catalog: reconnect, and \
                             stand up a second worker if both are needed at once"
                        ));
                    }
                    _ => loaded.ready(),
                }
            };
            let Some((catalog, sent)) = ready else {
                return Answer::Failed(
                    "this worker has no catalog yet: work arrived before the greeting".into(),
                );
            };
            let mut executor = Executor::new(&catalog).placed(&placement);
            if let Some(keeper) = shared.keeper {
                executor = executor.keeping(keeper, &memory);
            }
            // The one that arrived wins: it belongs to the job.
            if let Some(driver) = sent.as_deref().or(shared.driver) {
                executor = executor.with_driver(driver);
            }
            // What arrived is fed in as if this run had produced it.
            match executor.resume(&plan, input, known, keys) {
                Ok(outcome) => Answer::Done(outcome),
                Err(e) => Answer::Failed(e.to_string()),
            }
        }
    }
}

/// What the store has for this artifact, opened. `None` if it does not have it.
///
/// A store that cannot be reached is **not** a refusal: it is one trip more, so
/// it is noted and we ask the client. What does refuse is an artifact that is
/// there and cannot be opened, because that is the same failure as one arriving
/// broken over the wire.
fn kept(
    store: Option<&dyn Store>,
    provision: &dyn Provision,
    label: &Label,
) -> Result<Option<Provisioned>, String> {
    let Some(store) = store else { return Ok(None) };
    let bytes = match store.resolve(&name_of(&label.kind, &label.id)) {
        Err(e) => {
            eprintln!("the store could not be asked: {e}");
            return Ok(None);
        }
        Ok(None) => return Ok(None),
        Ok(Some(bound)) => match store.get(&bound.digest) {
            Ok(Some(bytes)) => bytes,
            // Bound to bytes that are not there: the record is right and the
            // blob is missing, so it is the client's turn again.
            Ok(None) => return Ok(None),
            Err(e) => {
                eprintln!("the store could not be read: {e}");
                return Ok(None);
            }
        },
    };
    provision
        .provide(&label.kind, &bytes)
        .map(Some)
        .map_err(|e| e.to_string())
}

/// Keeps an artifact under the name it announced itself with.
fn keep(store: &dyn Store, kind: &str, id: &str, bytes: &[u8]) -> Result<(), StoreError> {
    let digest = store.put(bytes)?;
    store.bind(
        &name_of(kind, id),
        &digest,
        vec![("kind".to_string(), kind.to_string())],
    )
}

/// What an artifact is called in the store.
///
/// The kind is in the name because two artifacts of different kinds can honestly
/// be given the same id by whoever produces them — the same catalog pickled and
/// packed as a manifest — and opening one with the other's `Provision` is not a
/// mistake worth allowing.
fn name_of(kind: &str, id: &str) -> String {
    format!("artifact:{kind}:{id}")
}
