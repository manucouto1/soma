//! A broker inside the client's own process.
//!
//! The first of the three deployments and the one that makes soma work with no
//! platform, no head node and no internet: a client that has no session falls
//! back to this and the graph runs on whatever workers it can reach. Not a
//! degraded mode with its own code — the same path with another broker.
//!
//! It is a thread and the messages are really serialized, and neither is waste.
//! What crosses here is a rendezvous: a run with four workers is nine messages,
//! a few tens of microseconds all told, once, outside the loop. **The broker is
//! in the control route and steps out of the cargo one**, so the price of being
//! honest here is not measurable there — and being honest buys the messages
//! being exercised for real from the first day, by a round trip that actually
//! happens.
//!
//! One thread, and it must not become a hang. The failure this type exists not
//! to have is a client blocked forever on an answer that is never coming, so
//! every channel operation maps to [`Unanswered::Gone`] and never to an
//! `unwrap`, pinned by a test — which is why [`Embedded::served_by`] is public:
//! without a way to stand up a desk that fails, the one failure mode worth
//! testing is the one that cannot be.

use crate::{Ask, Host, Path, Reply, Unreadable};
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Mutex;
use std::sync::mpsc::{Sender, channel};
use std::thread::JoinHandle;

/// A broker running on a thread of this process.
pub struct Embedded {
    /// The way in, behind a lock for the same reason the wire's `Worker` holds
    /// one: this has to be `Sync` so two branches of a wave can reach it at
    /// once, and one channel does not fit two conversations halfway through.
    ///
    /// `Option` so that [`Drop`] can let it go **before** joining: the thread
    /// ends when its end of the channel closes, and joining first would wait
    /// for something that is waiting for us.
    desk: Mutex<Option<Sender<Errand>>>,
    /// Whether the session has been opened. **Once per broker and not once per
    /// host**: a greeting belongs to the conversation, and a run across four
    /// workers that sent four of them would be asking one question four times.
    greeted: Mutex<bool>,
    /// So the thread is joined rather than left behind. `Option` for the same
    /// reason: `Drop` has to take it.
    thread: Option<JoinHandle<()>>,
}

/// One question and where its answer goes. Bytes in both directions, because
/// the point is that this is the same conversation a socket will carry.
struct Errand {
    asked: Vec<u8>,
    back: Sender<Vec<u8>>,
}

impl Embedded {
    /// A broker that knows where these hosts are.
    ///
    /// The listing is fixed when the broker opens and the thread owns it, which
    /// is why there is no lock around it and no way for two threads to disagree
    /// about where `w1` is. A host that has to be added is a broker that has to
    /// be opened.
    pub fn open(listing: impl IntoIterator<Item = (Host, Path)>) -> Self {
        let listing: BTreeMap<Host, Path> = listing.into_iter().collect();
        Self::served_by(move |ask| answer(&listing, ask))
    }

    /// A broker whose answers come from `desk`.
    ///
    /// Public for one reason, and a present one: the worst thing this type can
    /// do is turn a panic into a hang, and there is no way to test that a desk
    /// which fails is reported as a failure without being able to stand one up.
    pub fn served_by(mut desk: impl FnMut(Ask) -> Reply + Send + 'static) -> Self {
        let (to_desk, errands) = channel::<Errand>();
        let thread = std::thread::Builder::new()
            .name("soma-broker".into())
            .spawn(move || {
                // Ends when the last sender goes, which is this broker being
                // dropped. No stop message and nothing to forget to send.
                for errand in errands {
                    let reply = match Ask::from_bytes(&errand.asked) {
                        Ok(ask) => desk(ask),
                        // Not a panic: somebody spoke a language this does not
                        // read, and saying so is the answer.
                        Err(why) => Reply::Refused(why.to_string()),
                    };
                    let said = reply
                        .to_bytes()
                        .unwrap_or_else(|why| unanswerable(&why.to_string()));
                    // The client may have stopped waiting. That is its business.
                    let _ = errand.back.send(said);
                }
            })
            .expect("a broker needs one thread, and the OS would not give one");
        Self {
            desk: Mutex::new(Some(to_desk)),
            greeted: Mutex::new(false),
            thread: Some(thread),
        }
    }

    /// Opens the session, if it was not open.
    ///
    /// Idempotent on purpose: whoever needs a rendezvous calls this first and
    /// does not have to know whether somebody else already did. A refusal is a
    /// refusal of the **session**, which is why it is not swallowed and retried.
    pub fn greet(&self) -> Result<(), Unanswered> {
        let mut greeted = match self.greeted.lock() {
            Ok(greeted) => greeted,
            Err(poisoned) => poisoned.into_inner(),
        };
        if *greeted {
            return Ok(());
        }
        match self.ask(&Ask::hello())? {
            Reply::Welcome { .. } => {
                *greeted = true;
                Ok(())
            }
            Reply::Refused(why) => Err(Unanswered::Refused(why)),
            other => Err(Unanswered::BesideThePoint(format!(
                "greeting it answered {other:?}"
            ))),
        }
    }

    /// Says something and waits for the answer.
    ///
    /// Every message has exactly one answer except [`Ask::Done`], which has
    /// none — use [`Embedded::done`] for that one. Asking it here is refused
    /// rather than waited on, because the alternative is the hang this type is
    /// built not to have.
    pub fn ask(&self, ask: &Ask) -> Result<Reply, Unanswered> {
        if let Ask::Done { .. } = ask {
            return Err(Unanswered::NoAnswerToThat);
        }
        let (back, answered) = channel();
        self.post(ask, back)?;
        let said = answered.recv().map_err(|_| Unanswered::Gone)?;
        Reply::from_bytes(&said).map_err(Unanswered::Garbled)
    }

    /// Lets a rendezvous go. Nothing answers, and nothing is waited for.
    ///
    /// An embedded broker holds nothing, so nothing is released. It is sent
    /// anyway because the same client code talks to a broker that does hold
    /// things.
    pub fn done(&self, host: &Host) -> Result<(), Unanswered> {
        let (back, _) = channel();
        self.post(&Ask::Done { host: host.clone() }, back)
    }

    fn post(&self, ask: &Ask, back: Sender<Vec<u8>>) -> Result<(), Unanswered> {
        let asked = ask.to_bytes().map_err(Unanswered::Garbled)?;
        let desk = match self.desk.lock() {
            Ok(desk) => desk,
            // A poisoned lock is a panic somebody already heard about. The
            // channel underneath is still a channel.
            Err(poisoned) => poisoned.into_inner(),
        };
        desk.as_ref()
            .ok_or(Unanswered::Gone)?
            .send(Errand { asked, back })
            .map_err(|_| Unanswered::Gone)
    }
}

/// What an embedded broker answers, given what it knows.
///
/// A free function and not a method so that it is the thread's: the listing
/// moves in when the broker opens and never comes back out.
fn answer(listing: &BTreeMap<Host, Path>, ask: Ask) -> Reply {
    match ask {
        Ask::Hello { protocol, .. } => Reply::to_greeting(protocol),
        Ask::Reach { host, .. } => match listing.get(&host) {
            Some(path) => Reply::Met {
                path: path.clone(),
                // No policy here, so nothing is taking it back.
                good_for: None,
            },
            // Naming what it does know, because the usual cause is a typo in an
            // `.at()` and the list is three names long.
            None => Reply::Unreachable(match listing.is_empty() {
                true => format!(
                    "the graph sends work to `{host}` and this broker has no hosts listed at all"
                ),
                false => format!(
                    "the graph sends work to `{host}`, which this broker does not know; it knows {}",
                    listing
                        .keys()
                        .map(|known| format!("`{known}`"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            }),
        },
        // Nothing to release, and nobody is listening for the answer.
        Ask::Done { .. } => Reply::Welcome {
            protocol: crate::PROTOCOL,
        },
    }
}

/// The last resort when even the refusal will not encode. It cannot happen with
/// today's messages, but the alternative is an `unwrap` on the one thread whose
/// panic is a client that waits forever.
fn unanswerable(why: &str) -> Vec<u8> {
    Reply::Refused(format!(
        "the broker could not put its own answer into bytes: {why}"
    ))
    .to_bytes()
    .unwrap_or_default()
}

impl Drop for Embedded {
    /// Lets the channel go, then waits for the thread. In that order: the
    /// thread's loop ends when the last sender is dropped, so joining first
    /// would be waiting for something that is waiting for us.
    fn drop(&mut self) {
        // The same shape as everywhere else a lock is taken here: a poisoned
        // one is a panic somebody already heard about, and the channel under it
        // is still a channel. `take` drops the sender there and then.
        let mut desk = match self.desk.lock() {
            Ok(desk) => desk,
            Err(poisoned) => poisoned.into_inner(),
        };
        desk.take();
        if let Some(thread) = self.thread.take() {
            // A thread that panicked is already reported: whoever was waiting
            // got `Gone`. Panicking here in turn would only lose that.
            let _ = thread.join();
        }
    }
}

/// Why an ask got no answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unanswered {
    /// The broker's thread is not there any more: it ended, or it panicked.
    /// Either way there is nobody to answer, which is a thing to be told rather
    /// than a thing to wait for.
    Gone,
    /// The bytes were not a message, in one direction or the other.
    Garbled(Unreadable),
    /// [`Ask::Done`] is the one message with no answer. Waiting for one is the
    /// hang this refuses to perform.
    NoAnswerToThat,
    /// The broker will not open a session, and here is why. Belongs to the
    /// session: after it there is nothing to retry.
    Refused(String),
    /// It answered something that does not answer what was asked. Not a failure
    /// of the errand but of the vocabulary — the two sides do not agree about
    /// what this conversation is.
    BesideThePoint(String),
}

impl fmt::Display for Unanswered {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gone => f.write_str(
                "the broker is not answering: its thread ended or panicked. Nothing was \
                 placed, so nothing is half-done — open another one",
            ),
            Self::Garbled(why) => write!(f, "{why}"),
            Self::NoAnswerToThat => f.write_str(
                "`Done` is the one message a broker does not answer; send it with `done` \
                 rather than waiting for something that is not coming",
            ),
            Self::Refused(why) => write!(f, "the broker does not open a session: {why}"),
            Self::BesideThePoint(what) => write!(
                f,
                "the broker answered something beside the point: {what}. The two sides do \
                 not agree about what this conversation is"
            ),
        }
    }
}

impl std::error::Error for Unanswered {}
