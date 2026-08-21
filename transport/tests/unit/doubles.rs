//! The catalog on this side of the wire.
//!
//! It has to know **the same ids** as the worker's, which is why it is
//! duplicated in `src/bin/test-worker.rs`: they are two processes, and both
//! knowing who `a` is is exactly the contract. In a real system they would be
//! two calls to the same factory function.

use soma_next_core::{Catalog, Ctx, Node, NodeError, Transition, Value};
use std::sync::Arc;

pub struct Add(pub f64);

impl Node for Add {
    fn forward(&self, input: &Value, _ctx: &Ctx<'_>) -> Result<Transition, NodeError> {
        match input {
            Value::Number(x) => Ok(Transition::Done(Value::number(x + self.0))),
            other => Err(NodeError::new(format!(
                "Add needs a number, it was given {}",
                other.type_name()
            ))),
        }
    }
}

pub struct Mean;

impl Node for Mean {
    fn forward(&self, input: &Value, _ctx: &Ctx<'_>) -> Result<Transition, NodeError> {
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
        Ok(Transition::Done(Value::number(total / values.len() as f64)))
    }
}

/// The same as in the worker: the pid of whoever executes it.
pub struct WhereIRan;

impl Node for WhereIRan {
    fn forward(&self, _input: &Value, _ctx: &Ctx<'_>) -> Result<Transition, NodeError> {
        Ok(Transition::Done(Value::number(std::process::id() as f64)))
    }
}

pub struct Opaque;

impl Node for Opaque {
    fn forward(&self, _input: &Value, _ctx: &Ctx<'_>) -> Result<Transition, NodeError> {
        Ok(Transition::Done(Value::opaque(7u32)))
    }
}

/// Reads whatever it is handed and says whether it was something that only
/// exists in the process it ran in: the second step of a stretch on one host.
pub struct Reads;

impl Node for Reads {
    fn forward(&self, input: &Value, _ctx: &Ctx<'_>) -> Result<Transition, NodeError> {
        Ok(Transition::Done(Value::number(match input {
            Value::Opaque(_) => 1.0,
            _ => 0.0,
        })))
    }
}

pub struct Fail;

impl Node for Fail {
    fn forward(&self, _input: &Value, _ctx: &Ctx<'_>) -> Result<Transition, NodeError> {
        Err(NodeError::new("I broke"))
    }
}

pub struct WhichDevice;

impl Node for WhichDevice {
    fn forward(&self, _input: &Value, ctx: &Ctx<'_>) -> Result<Transition, NodeError> {
        Ok(Transition::Done(match ctx.device {
            Some(device) => Value::text(device.to_string()),
            None => Value::Null,
        }))
    }
}

/// The same catalog as the worker's, so `compile` finds them all.
pub fn catalog() -> Catalog {
    let mut catalog = Catalog::new();
    for id in ["a", "b", "c", "d", "left", "right"] {
        catalog.insert(id, Arc::new(Add(1.0)));
    }
    catalog.insert("join", Arc::new(Mean));
    catalog.insert("where", Arc::new(WhereIRan));
    catalog.insert("opaque", Arc::new(Opaque));
    catalog.insert("reads", Arc::new(Reads));
    catalog.insert("broken", Arc::new(Fail));
    catalog.insert("device", Arc::new(WhichDevice));
    // These only run in the worker. They are here so `compile` finds them, and
    // they fail if anyone executes them in this process — which is exactly what
    // their tests have to rule out.
    for id in ["meet_one", "meet_two", "ask", "unwritable"] {
        catalog.insert(id, Arc::new(Fail));
    }
    catalog
}

/// A directory of its own, removed when it goes out of scope.
///
/// Written here rather than taken as a dependency: it is eight lines, and a
/// test crate that pulls in a library to make a folder has lost the plot.
pub struct Dir(std::path::PathBuf);

impl Dir {
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        let at = std::env::temp_dir().join(format!(
            "soma-transport-{}-{}",
            std::process::id(),
            COUNT.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&at).expect("a temporary directory");
        Self(at)
    }

    pub fn path(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
