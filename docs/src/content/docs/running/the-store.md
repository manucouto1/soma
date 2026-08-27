---
title: The store
description: Eight methods, two implementors, and the one of them that is not a convenience — a directory and a bucket are the same store.
---

```rust
pub trait Store: Send + Sync {
    fn put(&self, bytes: &[u8]) -> Result<Digest, StoreError>;
    fn get(&self, digest: &Digest) -> Result<Option<Vec<u8>>, StoreError>;

    fn bind(&self, name: &str, digest: &Digest, meta: Meta) -> Result<(), StoreError>;
    fn claim(&self, name: &str, digest: &Digest, meta: Meta) -> Result<bool, StoreError>;
    fn resolve(&self, name: &str) -> Result<Option<Bound>, StoreError>;
    fn bound(&self) -> Result<Vec<Bound>, StoreError>;

    fn resolve_many(&self, names: &[&str]) -> Result<Vec<Option<Bound>>, StoreError> { … }
    fn get_many(&self, digests: &[&Digest]) -> Result<Vec<Option<Vec<u8>>>, StoreError> { … }
}
```

Two layers, and keeping them apart is what everything above is built on:

**Bytes by content.** `put` returns a `Digest`, which *is* the hash of what you
gave it. Nothing is ever overwritten, because nothing is ever addressed by
anything but its content.

**Names that point at digests.** `bind` and `resolve`. A name is a cache key,
an artifact's id, a trial's slot.

That split is why a source can state its version for free: a name resolves to a
digest, and the digest is already the content hash — so
[`Parquet::version`](/soma/running/where-the-data-comes-from/) costs one
`resolve` and reads no bytes.

## `claim` is the one that is not a convenience

```rust
/// Points a name at some bytes **only if nobody has**, and says whether it did.
fn claim(&self, name: &str, digest: &Digest, meta: Meta) -> Result<bool, StoreError>;
```

It is on the trait **with no default implementation**, and that is deliberate.
Written as `resolve` then `bind`, somebody else does the same thing between the
two, and two machines train the same round while nobody trains the next — a
race with a doc comment on it.

This single method is the whole coordination layer of a distributed study and
of a federated round. See
[handing trials out of a folder](/soma/searching/handing-out-trials/).

## `Meta` is text, and the vocabulary is the caller's

```rust
pub type Meta = Vec<(String, String)>;
```

Not a closed type. What this crate does with it is write it down and hand it
back — it never learns what a loss is, or a goal, or a host.

That is the same shape a `Fact` is written in: **a name and text-to-text
pairs**. So what a watcher is handed and what a store keeps are the same thing,
which is what lets a live view and a written report be one drawing function.
See [the record](/soma/looking/the-record/).

A `Bound` is a name, the digest it points at, the meta, and **when** — stamped
by the store, because a timestamp from another machine is not worth reading.

## Two implementors, and they are the same store

```python
from somatize import Store

store = Store("/mnt/shared/studies")   # a directory
store = Store.on_bucket("http://127.0.0.1:9000", "studies")   # `s3` feature, off by default
```

Same layout, same JSON. What differs is that on a bucket `claim` is a
**conditional PUT**.

The bucket implementation went in when it did for one reason: of the three uses
of a store, only one demands a genuinely shared disk. A cache degrades to a
miss and an artifact degrades to a miss, but **handing out work does not
degrade** — it silently duplicates.

So an endpoint that accepts `If-None-Match: *` and writes anyway would give
every trial to every machine and say nothing. `Bucket::at` spends **two round
trips proving it does not** before handing the store over.

Nothing above learned a new word. `take`, `report` and `gather` never asked
what kind of store they had.

## `resolve_many` and `get_many` have defaults, and one implementor overrides them

They were in the trait from the first day, which is unusual for this project —
a method nobody needs is not written. This one had a consumer immediately: a
cache working item by item asks thousands of names at once, and a bucket that
answers them one at a time is a bucket that is not usable.

## It is synchronous, and that is a decision

There is no `tokio` in this crate and none above it. An `async` `Store` would
be `async` in every caller, all the way into the engine's walk, and that is
what has twice kept a message bus out of this project.

It is also why SQL is not in the data layer: every Rust driver worth using
carries a runtime.
