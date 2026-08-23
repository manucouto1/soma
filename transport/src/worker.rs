//! A process that gets sent work. This side.
//!
//! The first real implementation of [`Transport`], and it does what the GIL will
//! not allow with threads: two Python nodes in the same wave interleave but do
//! not overlap, and in two processes they do.
//!
//! **The session opens by itself, and once.** The greeting is sent not when the
//! process starts but **before the first job**: a worker that stands up and
//! receives nothing should have done nothing. After that, [`dispatch`] does not
//! greet again while the conversation is alive.
//!
//! [`dispatch`]: Transport::dispatch
//!
//! **A worker serves one at a time.** [`Transport`] is `Sync`, so two branches
//! of a wave can call `dispatch` at once — and a pipe does not fit two
//! conversations. The `Mutex` queues them, which is correct and not a
//! limitation: a worker is **one** process. If you want two at once, stand up
//! two workers and place them on two hosts.
//!
//! # Two ends, and the one that matters is the second
//!
//! ```ignore
//! Worker::spawn(Command::new("./my-worker"))   // a child, over pipes
//! Worker::connect("node3:7000")                // one that was already standing
//! ```
//!
//! The first is convenient for testing and **does not satisfy the use case**: as
//! long as the client starts the process, there is no independent worker worth
//! the name. The second is the real form. That the conversation never finds out
//! which it is falls out of [`frame`](crate::frame) working over
//! `impl Read`/`impl Write`.
//!
//! [`Worker::spawn`] takes a ready-made [`Command`] rather than a path because
//! this library does not know what your binary is called, nor what environment
//! it needs, nor whether it goes inside an `srun`.

use crate::codec::{self, Codec};
use crate::frame;
use crate::{Answer, Artifact, Request};
use soma_next_core::{Cargo, Outcome, Plan, Transport, TransportError, Watcher};
use std::io::{self, BufReader};
use std::net::{TcpStream, ToSocketAddrs};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};

/// A process that executes the slices it is sent.
pub struct Worker {
    /// The process, its two ends and how far the conversation has got, held at
    /// once so two threads do not cross halfway through a message.
    open: Mutex<Open>,
    /// What to provision it with and how this client identifies itself; `None`
    /// is a worker that brings its own catalog. Behind a lock because it is set
    /// **after** opening: which nodes go here is known at run time.
    carries: Mutex<Option<(Artifact, String)>>,
    /// Who writes down what would not otherwise cross. `None` is the whole of
    /// Rust: there, an opaque carries something nobody has said how to write.
    ///
    /// Owned and not lent, unlike [`Serving`](crate::Serving)'s: this type has
    /// no lifetime and is held inside an `Arc` by whoever executes.
    codec: Option<Arc<dyn Codec>>,
}

struct Open {
    link: Link,
    /// The session opens once, not once per job.
    greeted: bool,
}

/// How the worker is spoken to. Both variants do the same thing; what changes
/// is who started the process and who ends it.
enum Link {
    /// A child process, over its pipes.
    Child {
        child: Child,
        /// `Option` so it can be closed on its own: that is the child's signal
        /// that there is no more work.
        to: Option<ChildStdin>,
        from: BufReader<ChildStdout>,
    },
    /// A worker that was already standing. Two handles on the same socket, so
    /// it can write and read without fighting over a borrow.
    Socket {
        to: TcpStream,
        from: BufReader<TcpStream>,
    },
}

impl Link {
    fn send(&mut self, payload: &[u8]) -> io::Result<()> {
        match self {
            Self::Child { to, .. } => match to {
                Some(to) => frame::send(to, payload),
                None => Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "this worker is already closed",
                )),
            },
            Self::Socket { to, .. } => frame::send(to, payload),
        }
    }

    fn recv(&mut self) -> io::Result<Option<Vec<u8>>> {
        match self {
            Self::Child { from, .. } => frame::recv(from),
            Self::Socket { from, .. } => frame::recv(from),
        }
    }
}

impl Worker {
    /// Starts the process and keeps its pipes. The worker there has to bring
    /// its own catalog; for an empty one, see [`Worker::carrying`].
    ///
    /// `stderr` is left **inherited**: its `stdout` is the wire.
    pub fn spawn(mut command: Command) -> io::Result<Self> {
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?;
        let to = child.stdin.take().expect("just asked for it with `piped`");
        let from = child.stdout.take().expect("just asked for it with `piped`");
        Ok(Self::over(Link::Child {
            child,
            to: Some(to),
            from: BufReader::new(from),
        }))
    }

    /// Connects to a worker that was already running — the form that satisfies
    /// the use case. On the other side there is [`Serving::listen`](crate::Serving::listen).
    pub fn connect(addr: impl ToSocketAddrs) -> io::Result<Self> {
        let to = TcpStream::connect(addr)?;
        // No `Nagle`: the messages are small and go ping-pong.
        to.set_nodelay(true)?;
        let from = BufReader::new(to.try_clone()?);
        Ok(Self::over(Link::Socket { to, from }))
    }

    /// The same worker, carrying what to provision it with if it started empty.
    /// `runtime` — `cpython-3.13/cloudpickle-3.1` — is what the far side's
    /// [`Provision`](crate::Provision) can reject at greeting time.
    pub fn carrying(self, artifact: Artifact, runtime: impl Into<String>) -> Self {
        let _ = self.offering(artifact, runtime);
        self
    }

    /// The same on an already-built worker, for whoever decides at run time
    /// which nodes go to this host. Setting the same artifact twice does
    /// nothing; changing an open session's fails.
    pub fn offering(
        &self,
        artifact: Artifact,
        runtime: impl Into<String>,
    ) -> Result<(), TransportError> {
        let mut carries = match self.carries.lock() {
            Ok(carries) => carries,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let Some((already, _)) = carries.as_ref() {
            if already.id == artifact.id {
                return Ok(());
            }
            if self.greeted() {
                return Err(TransportError::new(format!(
                    "this worker already opened a session with `{}` and its catalog \
                     cannot be changed to `{}` without reconnecting",
                    already.id, artifact.id
                )));
            }
        }
        *carries = Some((artifact, runtime.into()));
        Ok(())
    }

    /// The same worker, with somebody who knows how to write down what an
    /// opaque carries.
    ///
    /// Without one, a value that only exists in this process is refused at
    /// encoding time, as it always was. With one, it crosses as bytes and the
    /// refusal is left for what nobody registered a codec for.
    pub fn packing(mut self, codec: Arc<dyn Codec>) -> Self {
        self.codec = Some(codec);
        self
    }

    fn greeted(&self) -> bool {
        match self.open.lock() {
            Ok(open) => open.greeted,
            Err(poisoned) => poisoned.into_inner().greeted,
        }
    }

    fn over(link: Link) -> Self {
        Self {
            open: Mutex::new(Open {
                link,
                greeted: false,
            }),
            carries: Mutex::new(None),
            codec: None,
        }
    }
}

impl Open {
    /// Sends a message and waits for the one that answers it, handing whatever
    /// the far side says on the way to `seen`.
    ///
    /// It reads **until an answer is terminal**, which is the one change the
    /// whole live half of this needed: [`Answer::Saw`] is not an answer to
    /// anything, it is the worker talking while it works, and the loop is what
    /// turns a blocked read into a stream.
    ///
    /// A fact is passed on exactly as it was emitted. Attributing it to a host
    /// is the engine's job, not this one's: here the host is an address, and the
    /// name the graph gave it is not known.
    fn say(
        &mut self,
        request: &Request,
        seen: Option<&dyn Watcher>,
    ) -> Result<Answer, TransportError> {
        let payload = request
            .to_bytes()
            .map_err(|e| TransportError::new(e.to_string()))?;

        self.link.send(&payload).map_err(|e| broke(&e))?;
        loop {
            let answer = self
                .link
                .recv()
                .map_err(|e| broke(&e))?
                .ok_or_else(|| TransportError::new("the worker closed without answering"))?;
            match Answer::from_bytes(&answer).map_err(|e| TransportError::new(e.to_string()))? {
                // Not an answer: keep waiting for one. A client that is not
                // watching still has to read these off the socket — dropping
                // them is what it means not to watch, and leaving them there
                // would desynchronise the conversation.
                Answer::Saw(fact) => {
                    if let Some(seen) = seen {
                        seen.saw(&fact);
                    }
                }
                terminal => return Ok(terminal),
            }
        }
    }

    /// Opens the session, if it was not open: the artifact's **name** is
    /// announced and the bytes only sent if the worker asks for them.
    fn greet(
        &mut self,
        runtime: &str,
        artifact: Option<&Artifact>,
        seen: Option<&dyn Watcher>,
    ) -> Result<(), TransportError> {
        if self.greeted {
            return Ok(());
        }
        let hello = Request::Hello {
            runtime: runtime.to_string(),
            offering: artifact.map(Artifact::label),
        };
        match self.say(&hello, seen)? {
            Answer::Ready => {}
            Answer::Send => {
                let artifact = artifact.ok_or_else(|| {
                    TransportError::new(
                        "the worker asked for an artifact and this client brings none",
                    )
                })?;
                let sending = Request::Provision {
                    bytes: artifact.bytes.clone(),
                };
                match self.say(&sending, seen)? {
                    Answer::Ready => {}
                    other => return Err(unexpected("provisioning", &other)),
                }
            }
            other => return Err(unexpected("greeting", &other)),
        }
        self.greeted = true;
        Ok(())
    }
}

impl Transport for Worker {
    fn dispatch(
        &self,
        plan: &Plan,
        cargo: &Cargo<'_>,
        seen: Option<&dyn Watcher>,
    ) -> Result<Outcome, TransportError> {
        let carries = self
            .carries
            .lock()
            .map_err(|_| TransportError::new("this worker was poisoned by an earlier panic"))?;
        let mut open = self
            .open
            .lock()
            .map_err(|_| TransportError::new("this worker was poisoned by an earlier panic"))?;

        match carries.as_ref() {
            Some((artifact, runtime)) => open.greet(runtime, Some(artifact), seen)?,
            None => open.greet("rust", None, seen)?,
        }
        drop(carries);

        // Written down before the message is built, so the refusal in
        // `Request::to_bytes` is untouched and still guards: by the time it
        // looks, whatever had a codec is already bytes. Nothing to write down
        // means nothing is copied.
        let (input, known) = match self.codec.as_deref() {
            None => (cargo.input.clone(), cargo.known.to_vec()),
            Some(codec) => (
                codec.packed(cargo.input).map_err(as_transport_error)?,
                codec::packing(codec, cargo.known).map_err(as_transport_error)?,
            ),
        };
        let work = Request::Work {
            plan: plan.clone(),
            input,
            known,
            keys: cargo.keys.to_vec(),
            placement: cargo.placement.clone(),
            memory: cargo.memory.clone(),
        };
        match open.say(&work, seen)? {
            Answer::Done(outcome) => match self.codec.as_deref() {
                None => Ok(outcome),
                Some(codec) => live(codec, outcome),
            },
            Answer::Failed(why) => Err(TransportError::new(why)),
            other => Err(unexpected("working", &other)),
        }
    }
}

/// What came back, alive again.
///
/// A failure here is not a value left behind: everything in this answer was
/// written down by the other side a moment ago, so one that cannot be read back
/// means the two ends do not register the same codecs — and that is the answer,
/// not a value to work around.
fn live(codec: &dyn Codec, outcome: Outcome) -> Result<Outcome, TransportError> {
    Ok(Outcome {
        last: codec.unpacked(&outcome.last).map_err(as_transport_error)?,
        produced: codec::unpacking(codec, &outcome.produced).map_err(as_transport_error)?,
        keys: outcome.keys,
    })
}

fn as_transport_error(e: crate::CodecError) -> TransportError {
    TransportError::new(e.to_string())
}

/// What it answered does not match what it was asked: not a job failure, but
/// the two sides not speaking the same protocol.
fn unexpected(during: &str, answer: &Answer) -> TransportError {
    match answer {
        Answer::Refused(why) => {
            TransportError::new(format!("the worker does not accept the session: {why}"))
        }
        other => TransportError::new(format!(
            "while {during}, the worker answered something beside the point: {other:?}"
        )),
    }
}

/// A pipe failure almost always means the same thing — unless the error
/// already explains itself.
fn broke(e: &std::io::Error) -> TransportError {
    match e.kind() {
        std::io::ErrorKind::InvalidData | std::io::ErrorKind::InvalidInput => {
            TransportError::new(e.to_string())
        }
        _ => TransportError::new(format!(
            "the conversation with the worker was cut off ({e}); usually it has \
             died — check its stderr"
        )),
    }
}

impl Drop for Worker {
    /// Ends the conversation: a child gets its input closed and is waited for;
    /// a standing worker just loses the socket and awaits another client.
    fn drop(&mut self) {
        let mut open = match self.open.lock() {
            Ok(open) => open,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let Link::Child { child, to, .. } = &mut open.link {
            drop(to.take());
            let _ = child.wait();
        }
    }
}
