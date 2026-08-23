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
//! # Where the catalog comes from
//!
//! A worker either brings its own or is sent one in an artifact, and that is
//! the whole of it. What arrives is **cached by the artifact's id**, so a second run resends
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

use crate::codec::{self, Codec};
use crate::frame;
use crate::machine::{self, Machine};
use crate::{Answer, Label, Provision, Provisioned, Request};
use soma_next_core::{Catalog, Executor, Fact, Keeper, Outcome, Watcher};
use soma_next_store::{Store, StoreError};
use std::io::{self, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// A worker about to serve: what it executes with, and where it listens.
pub struct Serving<'a> {
    source: Source<'a>,
    store: Option<&'a dyn Store>,
    keeper: Option<&'a dyn Keeper>,
    codec: Option<&'a dyn Codec>,
    every: Option<Duration>,
}

impl<'a> Serving<'a> {
    /// A worker that brings its own catalog.
    pub fn own(catalog: &'a Catalog) -> Self {
        Self {
            source: Source::Own(catalog),
            store: None,
            every: None,
            keeper: None,
            codec: None,
        }
    }

    /// A worker that starts **empty** and is sent what to execute, with this to
    /// interpret it.
    pub fn provisioned(provision: &'a dyn Provision) -> Self {
        Self {
            source: Source::Sent(provision),
            store: None,
            every: None,
            keeper: None,
            codec: None,
        }
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

    /// Writes a reading of this machine to the store this often, whether or not
    /// anybody is asking it to do anything.
    ///
    /// The idle half, and the pipe is CU20's rule rather than a preference:
    /// *where a connection is open, facts come back down it; where there is
    /// none, they go to the store*. An idle worker's connection is one **nobody
    /// is reading** — a client only reads the socket while a job is in flight —
    /// so beating down it would fill a buffer nobody drains, block this process
    /// on the write, and hand over the **oldest** beats whenever somebody
    /// finally looked. Which is the worst available answer to *is it alive now*.
    ///
    /// Off unless asked for, and it does nothing without a
    /// [`store`](Serving::store) to write to.
    pub fn reporting(mut self, every: Duration) -> Self {
        self.every = Some(every);
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

    /// The same worker, able to read and write down what only exists in a
    /// process.
    ///
    /// **The same codecs as the client's**, or the two ends do not understand
    /// each other — which is what the error says when it happens. Whoever stands
    /// this worker up installs it; here it is one more thing that was lent, like
    /// the store, and it does not travel.
    pub fn packing(mut self, codec: &'a dyn Codec) -> Self {
        self.codec = Some(codec);
        self
    }

    /// Serves over standard input, until the client closes.
    ///
    /// Returns `Ok(())` when the input ends **between** messages, which is how
    /// it normally finishes. A failure of what gets executed is not an error
    /// here: it travels back as an answer.
    pub fn over_stdin(self) -> io::Result<()> {
        let every = self.every;
        let shared = Shared::of(self);
        let stop = AtomicBool::new(false);
        // The handle and not the lock guard: a `StdoutLock` is not `Send`, and
        // what writes down this pipe is now also whatever is watching the run,
        // from a wave's threads. `attend` holds its own lock over it.
        std::thread::scope(|scope| {
            if let Some(every) = every {
                let (shared, stop) = (&shared, &stop);
                scope.spawn(move || reporting(shared, every, stop));
            }
            let said = attend(&shared, io::stdin().lock(), io::stdout());
            // A pipe ends when the client goes, and the scope will not return
            // while the clock is still ticking in it.
            stop.store(true, Ordering::Relaxed);
            said
        })
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

        let every = self.every;
        let shared = Shared::of(self);
        let shared = &shared;
        let stop = AtomicBool::new(false);
        let stop = &stop;

        std::thread::scope(|scope| {
            if let Some(every) = every {
                scope.spawn(move || reporting(shared, every, stop));
            }
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
            // The listener only ends if it broke, and the scope will not
            // return while the reporting thread is still in it.
            stop.store(true, Ordering::Relaxed);
        });
        Ok(())
    }
}

/// What every session on this worker has in common. One per [`Serving`], lent
/// to each session, so the catalog that arrives serves the next client too.
struct Shared<'a> {
    source: Source<'a>,
    store: Option<&'a dyn Store>,
    keeper: Option<&'a dyn Keeper>,
    codec: Option<&'a dyn Codec>,
    loaded: Mutex<Loaded<'a>>,
    /// When this **process** came up, and how much it has run in total.
    ///
    /// Here and not on a `Session`, which is one client's conversation: a
    /// worker that has served three clients has served them all, and an uptime
    /// that restarted whenever somebody reconnected would be a figure of
    /// connections dressed as a figure of machines.
    since: Instant,
    served: AtomicU64,
}

impl<'a> Shared<'a> {
    fn of(serving: Serving<'a>) -> Self {
        let loaded = match &serving.source {
            Source::Own(catalog) => Loaded::Own(catalog),
            Source::Sent(_) => Loaded::Empty,
        };
        Self {
            source: serving.source,
            store: serving.store,
            keeper: serving.keeper,
            codec: serving.codec,
            loaded: Mutex::new(loaded),
            since: Instant::now(),
            served: AtomicU64::new(0),
        }
    }

    /// A reading of this machine right now.
    fn reading(&self) -> Machine {
        Machine::here(self.since.elapsed(), self.served.load(Ordering::Relaxed))
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
    Sent { id: String, catalog: Catalog },
}

impl Loaded<'_> {
    /// The catalog to execute with right now, if there is one.
    fn ready(&self) -> Option<Catalog> {
        match self {
            Self::Empty => None,
            Self::Own(catalog) => Some((*catalog).clone()),
            Self::Sent { catalog, .. } => Some(catalog.clone()),
        }
    }
}

/// A watcher that puts what it saw back down the same connection.
///
/// The far half of the live view, and it decides nothing: it does not write, it
/// does not name, it does not group. A fact goes out exactly as the engine here
/// emitted it, and the client is the one that says it came from this host —
/// because the name of this host is the graph's, and a worker does not know it.
///
/// Behind a `Mutex` because a [`Wave`](soma_next_core::Plan::Wave) emits from
/// several threads at once, and half a frame interleaved with half of another is
/// a connection that cannot be resynchronised.
struct Relaying<'a, W: Write + Send> {
    to: &'a Mutex<W>,
}

impl<W: Write + Send> Watcher for Relaying<'_, W> {
    fn saw(&self, fact: &Fact) {
        // Nothing here can be reported and nothing here should stop the run: a
        // fact that cannot be written means the connection is gone, and the
        // answer that is about to be sent down it will say so properly. Being
        // loud once per fact would be the noisiest possible way to say it.
        let Ok(encoded) = Answer::Saw(fact.clone()).to_bytes() else {
            return;
        };
        if let Ok(mut out) = self.to.lock() {
            let _ = frame::send(&mut *out, &encoded);
        }
    }
}

/// Writes a reading of this machine to the store on a clock, until told to stop.
///
/// The idle half. There is no name for a machine here — `w1` is the client's
/// word and there is no client — so it files under what the machine calls
/// itself, and whoever reads joins the two by seeing the same `id` on a reading
/// that **did** come down a wire.
///
/// One name, rewritten. The store stamps every write, so a reading that has not
/// moved is a machine that has stopped, and finding that out is a scan with no
/// fetches. It is CU18's shape and it is the only one that does not grow while
/// a worker sits there doing nothing.
fn reporting(shared: &Shared<'_>, every: Duration, stop: &AtomicBool) {
    let Some(store) = shared.store else {
        return;
    };
    while !stop.load(Ordering::Relaxed) {
        let reading = shared.reading();
        let said = reading.said();
        let (kind, mut meta) = said.flattened();
        meta.insert(0, ("fact".into(), kind.to_string()));
        // The whole of it is in the record and the blob has nothing to add,
        // which is what the price list is for: this costs a scan to read and
        // never a fetch.
        if let Ok(digest) = store.put(&[]) {
            // A store that will not take it is not something to stop serving
            // over, and there is nobody to tell: a worker's job is the work.
            let _ = store.bind(&machine::filed(&reading.id), &digest, meta);
        }
        // Slept in slices so shutting down does not wait out the interval.
        let mut left = every;
        while left > Duration::ZERO && !stop.load(Ordering::Relaxed) {
            let nap = left.min(Duration::from_millis(100));
            std::thread::sleep(nap);
            left -= nap;
        }
    }
}

/// The loop, with both ends as arguments so it is testable without a process.
fn attend(shared: &Shared<'_>, mut input: impl Read, output: impl Write + Send) -> io::Result<()> {
    let mut session = Session::default();
    // Shared with whatever is watching the run, which writes down the same
    // socket while the answer is still being worked out.
    let output = Mutex::new(output);

    while let Some(payload) = frame::recv(&mut input)? {
        let answer = match Request::from_bytes(&payload) {
            Err(e) => Answer::Refused(e.to_string()),
            Ok(request) => reply(shared, &mut session, request, &output),
        };
        let encoded = answer.to_bytes().unwrap_or_else(|e| {
            // If not even the answer can be encoded, that fact is the answer:
            // staying quiet would leave the other side waiting forever.
            Answer::Failed(e.to_string())
                .to_bytes()
                .expect("text can always be written")
        });
        let mut out = output
            .lock()
            .map_err(|_| io::Error::other("this connection was poisoned by an earlier panic"))?;
        frame::send(&mut *out, &encoded)?;
    }
    Ok(())
}

fn reply<W: Write + Send>(
    shared: &Shared<'_>,
    session: &mut Session,
    request: Request,
    output: &Mutex<W>,
) -> Answer {
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
            // What this machine looks like, said **before** the work rather
            // than after: a reading taken once the slice is over is a reading
            // of a machine that has just stopped, and the question is what it
            // was like while it was asked. It rides the connection that is
            // already open, which is CU20's rule for anything that happens
            // where somebody is listening.
            shared.served.fetch_add(1, Ordering::Relaxed);
            Relaying { to: output }.saw(&shared.reading().said());

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
            let Some(catalog) = ready else {
                return Answer::Failed(
                    "this worker has no catalog yet: work arrived before the greeting".into(),
                );
            };
            // What is remembered arrived with the work and is fed in whether or
            // not there is anywhere to keep things here: it is the graph's, and a
            // slice that carries on to a third host has to take it along.
            let relaying = Relaying { to: output };
            let mut executor = Executor::new(&catalog)
                .placed(&placement)
                .remembering(&memory)
                // The engine here is told exactly what the engine at home is
                // told, and knows no more about where it ends up. That it ends
                // up on a socket is this file's secret.
                .watching(&relaying);
            if let Some(keeper) = shared.keeper {
                executor = executor.keeping(keeper);
            }
            // Alive again before anything reads it, and not at the boundary
            // where a node is handed its argument: a value that only passes
            // through here is never handed to anybody, and the two ends have to
            // be the same one or this is impossible to explain.
            let (input, known) = match shared.codec {
                None => (input, known),
                Some(codec) => match (codec.unpacked(&input), codec::unpacking(codec, &known)) {
                    (Ok(input), Ok(known)) => (input, known),
                    (Err(e), _) | (_, Err(e)) => return Answer::Failed(e.to_string()),
                },
            };
            // What arrived is fed in as if this run had produced it.
            match executor.resume(&plan, input, known, keys) {
                Ok(outcome) => answering(shared.codec, outcome),
                Err(e) => Answer::Failed(e.to_string()),
            }
        }
    }
}

/// The answer to a slice that ran: written down, and with whatever stays here
/// left out of it.
///
/// **Packing goes first.** `travelling` drops what does not travel, and a tensor
/// with a codec does travel — asking before writing it down would leave behind
/// exactly what this exists to carry.
///
/// The two halves are not treated alike, and it is the same rule as ever:
/// `produced` is what the steps here read, so one that cannot be written down
/// **stays here** and is named by `RunError::Lost` if anybody reads it; `last`
/// is the value of the slice itself and has a reader over there by definition,
/// so there the codec's own words are the answer.
fn answering(codec: Option<&dyn Codec>, outcome: Outcome) -> Answer {
    let Some(codec) = codec else {
        return Answer::Done(outcome.travelling());
    };
    let last = match codec.packed(&outcome.last) {
        Ok(last) => last,
        Err(e) => return Answer::Failed(e.to_string()),
    };
    let produced = outcome
        .produced
        .into_iter()
        .map(|(id, value)| {
            let written = codec.packed(&value).unwrap_or(value);
            (id, written)
        })
        .collect();
    Answer::Done(
        Outcome {
            last,
            produced,
            keys: outcome.keys,
        }
        .travelling(),
    )
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
