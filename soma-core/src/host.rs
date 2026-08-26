//! Which process a node runs in.
//!
//! A **name**, not an address, and that is the whole decision. A
//! [`Placement`](crate::Placement) is data: if it carried
//! `tcp://10.0.0.2:7000` inside, the same graph could no longer run on another
//! cluster without editing it. With a name, whoever **executes** decides what
//! `worker1` resolves to — the same boundary a `Transport` draws.
//!
//! And that is why it is not an enum, even though [`Device`](crate::Device) is.
//! A device is a closed set we decide, and a typo has to fail at declaration
//! time. Hosts are named by the user and there is no list to close.

use std::fmt;

/// The name of the process where a node runs.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(transparent)
)]
pub struct Host(String);

impl Host {
    /// A host by its name.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// The name.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for Host {
    fn from(name: &str) -> Self {
        Self(name.to_string())
    }
}

impl From<String> for Host {
    fn from(name: String) -> Self {
        Self(name)
    }
}

impl fmt::Display for Host {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
