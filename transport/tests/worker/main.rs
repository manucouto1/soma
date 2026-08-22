//! A worker for this crate's tests, in both its forms.
//!
//! ```text
//! test-worker                        over stdin,  own catalog
//! test-worker --empty                over stdin,  catalog sent
//! test-worker --noisy                writes on `stdout` before serving
//! test-worker --store DIR            keeps what it is sent, and looks there first
//! test-worker --store DIR --keeper   and also keeps what the nodes produce
//! test-worker --codec                writes down the one opaque it knows
//! test-worker --listen 127.0.0.1:0   standing,    own catalog
//! test-worker --listen … --empty     standing,    catalog sent
//! ```
//!
//! With `--listen` it prints the address it ended up open on to `stdout`, which
//! there really is free — the wire is the socket. That way a test can ask for
//! port `0` and find out which one it got, instead of picking a number and
//! praying.
//!
//! It is a binary and not an `example` because a test needs to know its path,
//! and `CARGO_BIN_EXE_` only exists for binaries.
//!
//! It also works as a template for what a worker is, which is not much: build
//! its catalog — or know how to interpret the one it is sent — and call `serve`.
//! Not one `println!`, since the wire runs over `stdout`.
//!
//! The `Provision` here uses no cloudpickle or anything like it, and that is on
//! purpose: its artifact is plain text, `a=1,b=10`. It serves to show that **the
//! mechanism is not a Python one** — what travels is bytes this crate does not
//! look at, and whoever interprets them can be anyone.

use soma_next_core::{Catalog, Ctx, Node, NodeError, Value};
use soma_next_store::{Cache, Local};
use soma_next_transport::{Codec, CodecError, Provision, ProvisionError, Provisioned, Serving};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Adds a constant. The same double as in the core's tests.
struct Add(f64);

impl Node for Add {
    fn forward(&self, input: &Value, _ctx: &Ctx<'_>) -> Result<Value, NodeError> {
        match input {
            Value::Number(x) => Ok(Value::number(x + self.0)),
            other => Err(NodeError::new(format!(
                "Add needs a number, it was given {}",
                other.type_name()
            ))),
        }
    }
}

/// The mean of whatever arrives along its edges.
struct Mean;

impl Node for Mean {
    fn forward(&self, input: &Value, _ctx: &Ctx<'_>) -> Result<Value, NodeError> {
        let Some(values) = input.values() else {
            return Err(NodeError::new("Mean needs a map"));
        };
        let mut total = 0.0;
        for value in &values {
            let Value::Number(x) = value else {
                return Err(NodeError::new("Mean needs numbers"));
            };
            total += x;
        }
        Ok(Value::number(total / values.len() as f64))
    }
}

/// Says which process it ran in, so it can be proved from the other side.
struct WhereIRan;

impl Node for WhereIRan {
    fn forward(&self, _input: &Value, _ctx: &Ctx<'_>) -> Result<Value, NodeError> {
        Ok(Value::number(std::process::id() as f64))
    }
}

/// Writes down the one thing it knows how to: a `u32` behind an opaque.
///
/// Deliberately not Python's, and deliberately trivial. What it shows is that
/// the hole is a hole — whoever knows what an opaque carries fills it, and this
/// crate is none the wiser — and that **both ends have to agree**, which here
/// they do by both being this.
struct U32s;

/// What a written-down `u32` looks like once it is only maps and numbers.
const WRITTEN: &str = "__a_u32__";

impl Codec for U32s {
    fn packed(&self, value: &Value) -> Result<Value, CodecError> {
        match value {
            Value::Opaque(_) => match value.downcast::<u32>() {
                Some(n) => Ok(Value::map(vec![(
                    WRITTEN.to_string(),
                    Value::number(*n as f64),
                )])),
                None => Err(CodecError::new(
                    "this opaque carries something that is not a `u32`, and a `u32` \
                     is all this codec knows how to write down",
                )),
            },
            Value::Map(pairs) => {
                let mut written = Vec::with_capacity(pairs.len());
                for (key, value) in pairs.iter() {
                    written.push((key.clone(), self.packed(value)?));
                }
                Ok(Value::map(written))
            }
            other => Ok(other.clone()),
        }
    }

    fn unpacked(&self, value: &Value) -> Result<Value, CodecError> {
        let Value::Map(pairs) = value else {
            return Ok(value.clone());
        };
        if let [(key, Value::Number(n))] = &pairs[..]
            && key == WRITTEN
        {
            return Ok(Value::opaque(*n as u32));
        }
        let mut alive = Vec::with_capacity(pairs.len());
        for (key, value) in pairs.iter() {
            alive.push((key.clone(), self.unpacked(value)?));
        }
        Ok(Value::map(alive))
    }
}

/// An opaque carrying something the codec above cannot write down, so that the
/// two halves of the answer can be told apart.
struct Unwritable;

impl Node for Unwritable {
    fn forward(&self, _input: &Value, _ctx: &Ctx<'_>) -> Result<Value, NodeError> {
        Ok(Value::opaque(String::from("alive")))
    }
}

/// Returns an opaque, which is what cannot come back over the wire.
struct Opaque;

impl Node for Opaque {
    fn forward(&self, _input: &Value, _ctx: &Ctx<'_>) -> Result<Value, NodeError> {
        Ok(Value::opaque(7u32))
    }
}

/// Reads whatever it is handed and says whether it was something that only
/// exists in this process: the second step of a stretch on one host.
struct Reads;

impl Node for Reads {
    fn forward(&self, input: &Value, _ctx: &Ctx<'_>) -> Result<Value, NodeError> {
        Ok(Value::number(match input {
            Value::Opaque(_) => 1.0,
            _ => 0.0,
        }))
    }
}

/// Always fails, to see how a failure from over there comes back.
struct Fail;

impl Node for Fail {
    fn forward(&self, _input: &Value, _ctx: &Ctx<'_>) -> Result<Value, NodeError> {
        Err(NodeError::new("I broke in the worker"))
    }
}

/// Says which device it got, to see whether the placement crossed the wire.
struct WhichDevice;

impl Node for WhichDevice {
    fn forward(&self, _input: &Value, ctx: &Ctx<'_>) -> Result<Value, NodeError> {
        Ok(match ctx.device {
            Some(device) => Value::text(device.to_string()),
            None => Value::Null,
        })
    }
}

/// Does not finish until other processes have arrived at the same meeting — a
/// file each appends a byte to. The only way to show that two workers really do
/// **overlap**, and it has to be across processes.
struct Rendezvous;

impl Node for Rendezvous {
    fn forward(&self, _input: &Value, _ctx: &Ctx<'_>) -> Result<Value, NodeError> {
        let place = std::env::var("SOMA_RENDEZVOUS")
            .map_err(|_| NodeError::new("this node needs `SOMA_RENDEZVOUS` in the environment"))?;
        let how_many: u64 = std::env::var("SOMA_RENDEZVOUS_COUNT")
            .ok()
            .and_then(|n| n.parse().ok())
            .unwrap_or(2);

        std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(&place)
            .and_then(|mut f| std::io::Write::write_all(&mut f, b"."))
            .map_err(|e| NodeError::new(format!("could not reach the meeting: {e}")))?;

        // A deadline, so a failure is a named error and not a hung test.
        let until = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while std::time::Instant::now() < until {
            if std::fs::metadata(&place).map(|m| m.len()).unwrap_or(0) >= how_many {
                return Ok(Value::number(std::process::id() as f64));
            }
            std::thread::yield_now();
        }
        Err(NodeError::new(
            "the meeting deadline ran out: the workers did not run at the same time",
        ))
    }
}

/// How many times it has been asked, in this process. What makes a cache hit
/// **over there** observable from here: a second run that answers `1` did not
/// run it.
struct Counts(AtomicUsize);

impl Node for Counts {
    fn forward(&self, _input: &Value, _ctx: &Ctx<'_>) -> Result<Value, NodeError> {
        Ok(Value::number(
            self.0.fetch_add(1, Ordering::SeqCst) as f64 + 1.0,
        ))
    }
}

/// How many times a catalog has been built here: what makes the `have`/`want`
/// observable from outside.
struct Times(Arc<AtomicUsize>);

impl Node for Times {
    fn forward(&self, _input: &Value, _ctx: &Ctx<'_>) -> Result<Value, NodeError> {
        Ok(Value::number(self.0.load(Ordering::SeqCst) as f64))
    }
}

fn catalog() -> Catalog {
    let mut catalog = Catalog::new();
    for id in ["a", "b", "c", "d", "left", "right"] {
        catalog.insert(id, Arc::new(Add(1.0)));
    }
    catalog.insert("join", Arc::new(Mean));
    catalog.insert("where", Arc::new(WhereIRan));
    catalog.insert("opaque", Arc::new(Opaque));
    catalog.insert("unwritable", Arc::new(Unwritable));
    catalog.insert("reads", Arc::new(Reads));
    catalog.insert("broken", Arc::new(Fail));
    catalog.insert("device", Arc::new(WhichDevice));
    catalog.insert("counts", Arc::new(Counts(AtomicUsize::new(0))));
    for id in ["meet_one", "meet_two"] {
        catalog.insert(id, Arc::new(Rendezvous));
    }
    catalog
}

/// Interprets artifacts of kind `adds`: `a=1,b=10,c=100`.
struct Adds {
    built: Arc<AtomicUsize>,
}

impl Provision for Adds {
    fn accepts(&self, runtime: &str, _kind: &str) -> Result<(), ProvisionError> {
        // A real worker would compare interpreter versions; here it is enough
        // that there is something to reject.
        match runtime.starts_with("rust") {
            true => Ok(()),
            false => Err(ProvisionError::Incompatible {
                client: runtime.to_string(),
                worker: "rust".into(),
            }),
        }
    }

    fn provide(&self, kind: &str, bytes: &[u8]) -> Result<Provisioned, ProvisionError> {
        if kind != "adds" {
            return Err(ProvisionError::UnknownKind(kind.to_string()));
        }
        let text =
            std::str::from_utf8(bytes).map_err(|_| ProvisionError::Broken("not UTF-8".into()))?;

        let mut catalog = Catalog::new();
        for piece in text.split(',').filter(|p| !p.is_empty()) {
            let (id, how_much) = piece
                .split_once('=')
                .ok_or_else(|| ProvisionError::Broken(format!("`{piece}` is not `id=number`")))?;
            let how_much: f64 = how_much
                .parse()
                .map_err(|_| ProvisionError::Broken(format!("`{how_much}` is not a number")))?;
            catalog.insert(id, Arc::new(Add(how_much)));
        }
        let times = Arc::clone(&self.built);
        times.fetch_add(1, Ordering::SeqCst);
        catalog.insert("times", Arc::new(Times(times)));
        catalog.insert("where", Arc::new(WhereIRan));
        Ok(Provisioned::new(catalog))
    }
}

fn empty() -> Adds {
    Adds {
        built: Arc::new(AtomicUsize::new(0)),
    }
}

/// Reports on `stdout` where it ended up open, and pushes it out.
fn opened(addr: std::net::SocketAddr) {
    use std::io::Write;
    println!("LISTEN {addr}");
    let _ = std::io::stdout().flush();
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let without_catalog = args.iter().any(|a| a == "--empty");
    let where_ = args
        .iter()
        .position(|a| a == "--listen")
        .and_then(|i| args.get(i + 1))
        .cloned();

    if args.iter().any(|a| a == "--noisy") {
        // What `frame`'s cap exists to catch: four ASCII characters on `stdout`
        // read as a length of well over a gigabyte.
        use std::io::Write;
        print!("hello, I am a stray print");
        let _ = std::io::stdout().flush();
    }
    let empty = empty();
    let catalog = catalog();
    // A directory shared with whoever else is told to use it: that is where a
    // second worker finds an artifact it was never sent.
    let store = args
        .iter()
        .position(|a| a == "--store")
        .and_then(|i| args.get(i + 1))
        .map(|where_| Local::at(where_).expect("that directory cannot be a store"));

    let mut serving = match without_catalog {
        true => Serving::provisioned(&empty),
        false => Serving::own(&catalog),
    };
    if let Some(store) = &store {
        serving = serving.store(store);
    }
    // Keeping **values** is another question from keeping artifacts, and the
    // same directory answers both: one so a catalog is not sent twice, the other
    // so a node is not run twice.
    let cache = store.as_ref().map(|store| Cache::over(store));
    if let (true, Some(cache)) = (args.iter().any(|a| a == "--keeper"), &cache) {
        serving = serving.keeping(cache);
    }
    // The same codec as the client's, which is the whole requirement: two ends
    // that do not agree on how something is written down do not understand each
    // other, and say so.
    if args.iter().any(|a| a == "--codec") {
        serving = serving.packing(&U32s);
    }
    match where_ {
        Some(addr) => serving.listen_at(addr.as_str(), opened),
        None => serving.over_stdin(),
    }
}
