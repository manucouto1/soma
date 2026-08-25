//! A store that is a bucket: S3, MinIO, R2 — anything that speaks the protocol.
//!
//! [`Local`](crate::Local) needs a directory everybody can see, and on a cluster
//! with no shared mount there is none. This is the other way to have one, and it
//! is **behind the `s3` feature**: TLS and an XML parser are some eighty crates,
//! and a directory needs none of them.
//!
//! # A bucket and a directory are the same store
//!
//! Deliberately, and it is worth more than a browsable bucket: the same split,
//! from the same [`Digest::path`](crate::Digest), and the same JSON inside. So a
//! directory can be moved onto a bucket with `aws s3 sync` and back, and neither
//! end has to know. Naming an object after the digest **of the name** rather
//! than after the name also settles what a key may contain, which on a
//! filesystem was already settled the same way.
//!
//! # What is genuinely different
//!
//! **`claim` is a conditional PUT.** On a filesystem it is a hard link, which
//! fails when the name is taken; here it is `If-None-Match: *`, which is the
//! same promise from the other side, and the signature covers the header.
//!
//! **A scan costs a round trip per name.** `bound()` lists and then reads, and
//! the reading is fanned out over threads. The trait already expected this —
//! `resolve_many` and `get_many` are batch *"because against a store on the far
//! end of a network that is thousands of round trips unless it is one call"* —
//! and the way out, when it starts hurting, is the one
//! [`Store::bound`](crate::Store::bound) names itself: an index built from the
//! records that can be thrown away.

use crate::store::{read_record, record};
use crate::{Bound, Digest, Meta, Store, StoreError};
use rusty_s3::Bucket as Addressed;
use rusty_s3::actions::{DeleteObject, GetObject, ListObjectsV2, S3Action};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::Duration;

pub use rusty_s3::{Credentials, UrlStyle};

/// How long a signed URL is good for. Short, because it is used at once and
/// never handed to anybody: the only reason it is not shorter is a clock that
/// disagrees with the endpoint's.
const SIGNED_FOR: Duration = Duration::from_secs(300);

/// How many names a scan reads at a time. One thread per name would be one
/// thread per trial; this is the fan-out that turns a scan's latency into a
/// number that does not grow with the study.
const AT_ONCE: usize = 16;

/// So two probes from this process never pick the same key.
static PROBES: AtomicU64 = AtomicU64::new(0);

/// A store kept on a bucket.
pub struct Bucket {
    addressed: Addressed,
    credentials: Credentials,
    agent: ureq::Agent,
}

impl Bucket {
    /// The store on this bucket, checking on the way in that it can hand work
    /// out.
    ///
    /// `endpoint` is the service — `https://s3.eu-west-1.amazonaws.com`,
    /// `http://minio:9000` — and `style` says whether the bucket goes in the
    /// host or in the path. MinIO wants [`UrlStyle::Path`]; AWS wants
    /// [`UrlStyle::VirtualHost`] for anything made recently.
    ///
    /// **It talks to the endpoint before returning**, twice, to find out whether
    /// a conditional write is really conditional here. See [`Bucket::checks`]
    /// for what that costs and why it is not optional.
    pub fn at(
        endpoint: &str,
        bucket: &str,
        region: &str,
        style: UrlStyle,
        credentials: Credentials,
    ) -> Result<Self, StoreError> {
        let parsed = endpoint
            .parse()
            .map_err(|e| StoreError::Io(format!("`{endpoint}` is not a URL: {e}")))?;
        let addressed = Addressed::new(parsed, style, bucket.to_string(), region.to_string())
            .map_err(|e| StoreError::Io(format!("that is not a bucket this can address: {e}")))?;
        let agent = ureq::Agent::config_builder()
            // A 412 is an answer and not a failure: it is `claim` saying that
            // somebody else got there first. Without this, ureq turns it into an
            // error and the one operation that has to tell them apart cannot.
            .http_status_as_error(false)
            .build()
            .into();
        let store = Self {
            addressed,
            credentials,
            agent,
        };
        store.checks()?;
        Ok(store)
    }

    /// Whether a conditional write is conditional here. Two round trips, and the
    /// reason they are not optional.
    ///
    /// `claim` is how work is handed out: whoever takes the name does the work.
    /// An endpoint that accepts `If-None-Match: *` and writes anyway — some
    /// S3-compatible services do — makes every `claim` answer `true`, so every
    /// machine takes every trial and **nothing anywhere says so**. A store that
    /// cannot promise this is refused here rather than discovered in a study
    /// whose numbers are already wrong.
    fn checks(&self) -> Result<(), StoreError> {
        let key = format!(
            "probe/{}-{}",
            std::process::id(),
            PROBES.fetch_add(1, Ordering::Relaxed)
        );
        let taken = self.put_if_absent(&key, b"probe")?;
        let again = self.put_if_absent(&key, b"probe");
        let _ = self.delete(&key);
        match (taken, again?) {
            (true, false) => Ok(()),
            (true, true) => Err(StoreError::Io(
                "this endpoint took the same name twice, so `If-None-Match` is not \
                 honoured here. Handing work out over it would give the same trial \
                 to every machine and say nothing — use a service that supports \
                 conditional writes (AWS S3, R2, a recent MinIO)"
                    .into(),
            )),
            (false, _) => Err(StoreError::Io(
                "the probe key was already taken, which should be impossible: \
                 something else is writing under `probe/` in this bucket"
                    .into(),
            )),
        }
    }

    /// Where a blob's bytes live, under the same split a directory uses.
    fn blob(&self, digest: &Digest) -> String {
        let (head, rest) = digest.path();
        format!("blobs/{head}/{rest}")
    }

    /// Where a name's record lives. By the digest **of the name**, exactly as on
    /// a filesystem, so the two layouts stay the same one.
    fn record(&self, name: &str) -> String {
        let (head, rest) = Digest::of(name.as_bytes()).path();
        format!("names/{head}/{rest}")
    }

    /// Writes these bytes at that key, whatever is there.
    fn put(&self, key: &str, bytes: &[u8]) -> Result<(), StoreError> {
        let action = self.addressed.put_object(Some(&self.credentials), key);
        let url = action.sign(SIGNED_FOR);
        let answer = self
            .agent
            .put(url.as_str())
            .send(bytes)
            .map_err(|e| StoreError::Io(format!("`{key}` could not be written: {e}")))?;
        expected(answer.status().as_u16(), key, "written")
    }

    /// Writes them **only if that key is free**, and says whether it did.
    ///
    /// The header is signed as well as sent — `headers_mut` is what rusty-s3
    /// covers with the signature — so an endpoint cannot receive the request
    /// without the condition on it.
    fn put_if_absent(&self, key: &str, bytes: &[u8]) -> Result<bool, StoreError> {
        let mut action = self.addressed.put_object(Some(&self.credentials), key);
        action.headers_mut().insert("if-none-match", "*");
        let url = action.sign(SIGNED_FOR);
        let answer = self
            .agent
            .put(url.as_str())
            .header("if-none-match", "*")
            .send(bytes)
            .map_err(|e| StoreError::Io(format!("`{key}` could not be claimed: {e}")))?;
        match answer.status().as_u16() {
            // The name was free and is now this caller's.
            200..=299 => Ok(true),
            // Taken. `409` because some services answer the race that way.
            412 | 409 => Ok(false),
            other => Err(StoreError::Io(format!(
                "`{key}` could not be claimed: the endpoint answered {other}"
            ))),
        }
    }

    /// What is at that key, or `None` if nothing is.
    fn fetch(&self, key: &str) -> Result<Option<Vec<u8>>, StoreError> {
        let action: GetObject<'_> = self.addressed.get_object(Some(&self.credentials), key);
        let url = action.sign(SIGNED_FOR);
        let mut answer = self
            .agent
            .get(url.as_str())
            .call()
            .map_err(|e| StoreError::Io(format!("`{key}` could not be read: {e}")))?;
        match answer.status().as_u16() {
            404 => Ok(None),
            200..=299 => answer
                .body_mut()
                .read_to_vec()
                .map(Some)
                .map_err(|e| StoreError::Io(format!("`{key}` came back broken: {e}"))),
            other => Err(StoreError::Io(format!(
                "`{key}` could not be read: the endpoint answered {other}"
            ))),
        }
    }

    /// Removes that key. Only the probe uses it, and a probe that will not go
    /// away is not a reason to refuse the store.
    fn delete(&self, key: &str) -> Result<(), StoreError> {
        let action: DeleteObject<'_> = self.addressed.delete_object(Some(&self.credentials), key);
        let url = action.sign(SIGNED_FOR);
        self.agent
            .delete(url.as_str())
            .call()
            .map_err(|e| StoreError::Io(format!("`{key}` could not be removed: {e}")))?;
        Ok(())
    }

    /// Every key under that prefix, following the continuation until there is
    /// none left.
    fn keys(&self, prefix: &str) -> Result<Vec<String>, StoreError> {
        let mut all = Vec::new();
        let mut carry_on: Option<String> = None;
        loop {
            let mut action: ListObjectsV2<'_> =
                self.addressed.list_objects_v2(Some(&self.credentials));
            action.with_prefix(prefix.to_string());
            if let Some(token) = &carry_on {
                action.with_continuation_token(token.clone());
            }
            let url = action.sign(SIGNED_FOR);
            let mut answer = self
                .agent
                .get(url.as_str())
                .call()
                .map_err(|e| StoreError::Io(format!("the bucket could not be listed: {e}")))?;
            expected(answer.status().as_u16(), prefix, "listed")?;
            let said = answer
                .body_mut()
                .read_to_string()
                .map_err(|e| StoreError::Io(format!("the listing came back broken: {e}")))?;
            let page = ListObjectsV2::parse_response(&said)
                .map_err(|e| StoreError::Corrupt(format!("that listing cannot be read: {e}")))?;
            all.extend(page.contents.into_iter().map(|each| each.key));
            match page.next_continuation_token {
                Some(token) => carry_on = Some(token),
                None => return Ok(all),
            }
        }
    }

    /// Reads many keys at once, in the order they were asked.
    ///
    /// Threads and not one call, because there is no one call: S3 has no "give
    /// me these forty objects". What there is, is forty round trips that do not
    /// have to happen one after another.
    fn fetch_many(&self, keys: &[String]) -> Result<Vec<Option<Vec<u8>>>, StoreError> {
        let mut out = Vec::with_capacity(keys.len());
        for batch in keys.chunks(AT_ONCE) {
            let answers: Vec<_> = thread::scope(|scope| {
                let running: Vec<_> = batch
                    .iter()
                    .map(|key| scope.spawn(move || self.fetch(key)))
                    .collect();
                running
                    .into_iter()
                    .map(|one| {
                        one.join().unwrap_or_else(|_| {
                            Err(StoreError::Io("a read did not come back".into()))
                        })
                    })
                    .collect()
            });
            for answer in answers {
                out.push(answer?);
            }
        }
        Ok(out)
    }
}

impl Store for Bucket {
    fn put(&self, bytes: &[u8]) -> Result<Digest, StoreError> {
        let digest = Digest::of(bytes);
        // Content addressing: the same bytes under the same name, so writing
        // them again is writing the same thing and no condition is needed.
        Bucket::put(self, &self.blob(&digest), bytes)?;
        Ok(digest)
    }

    fn get(&self, digest: &Digest) -> Result<Option<Vec<u8>>, StoreError> {
        self.fetch(&self.blob(digest))
    }

    fn bind(&self, name: &str, digest: &Digest, meta: Meta) -> Result<(), StoreError> {
        Bucket::put(self, &self.record(name), &record(name, digest, meta)?)
    }

    fn claim(&self, name: &str, digest: &Digest, meta: Meta) -> Result<bool, StoreError> {
        self.put_if_absent(&self.record(name), &record(name, digest, meta)?)
    }

    fn resolve(&self, name: &str) -> Result<Option<Bound>, StoreError> {
        match self.fetch(&self.record(name))? {
            Some(bytes) => read_record(&bytes).map(Some),
            None => Ok(None),
        }
    }

    fn resolve_many(&self, names: &[&str]) -> Result<Vec<Option<Bound>>, StoreError> {
        let keys: Vec<String> = names.iter().map(|name| self.record(name)).collect();
        self.fetch_many(&keys)?
            .into_iter()
            .map(|found| found.as_deref().map(read_record).transpose())
            .collect()
    }

    fn get_many(&self, digests: &[&Digest]) -> Result<Vec<Option<Vec<u8>>>, StoreError> {
        let keys: Vec<String> = digests.iter().map(|digest| self.blob(digest)).collect();
        self.fetch_many(&keys)
    }

    fn bound(&self) -> Result<Vec<Bound>, StoreError> {
        let keys = self.keys("names/")?;
        let mut all: Vec<Bound> = self
            .fetch_many(&keys)?
            .into_iter()
            .flatten()
            .map(|bytes| read_record(&bytes))
            .collect::<Result<_, _>>()?;
        // By time, and by name within the same second, so two runs of this see
        // the same thing — the same order a directory answers in.
        all.sort_by(|a, b| (a.when, &a.name).cmp(&(b.when, &b.name)));
        Ok(all)
    }
}

/// Turns anything that is not a success into the error that says which key and
/// what was being done to it.
fn expected(status: u16, what: &str, doing: &str) -> Result<(), StoreError> {
    match status {
        200..=299 => Ok(()),
        other => Err(StoreError::Io(format!(
            "`{what}` could not be {doing}: the endpoint answered {other}"
        ))),
    }
}
