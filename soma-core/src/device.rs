//! Where a node runs.
//!
//! An enum and not a validated `String` so that a typo is an error **at
//! declaration time**: `.on("cude:0")` fails where it was written, instead of
//! turning into a torch `RuntimeError` halfway through a run.
//!
//! The price is that the vocabulary becomes ours rather than torch's, and it can
//! be paid because **the core does not `match` on a `Device` anywhere else** —
//! it only carries it to the node. Adding a variant is three lines and nowhere
//! else stops compiling. Only the ones with a consumer today are here.

use std::fmt;
use std::str::FromStr;

/// The place where a node executes.
///
/// It travels **as text**, through [`Display`](fmt::Display) and [`FromStr`]:
/// a variant number would be shorter and would break silently the day the enum
/// grows in the middle.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(into = "String", try_from = "String")
)]
pub enum Device {
    /// The processor.
    Cpu,
    /// A CUDA GPU, by index. Mandatory: bare `"cuda"` is thread state in torch,
    /// and to whoever is placing that is not a placement.
    Cuda(usize),
    /// Torch's `meta` device: shape and dtype, without memory or compute. The
    /// only one that proves a placement is obeyed on any machine.
    Meta,
}

impl FromStr for Device {
    type Err = DeviceError;

    /// `cpu`, `cuda:0`, `meta`. Exactly as torch writes them, so what reaches
    /// the node can be handed to `.to()` without translating anything.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.is_empty() {
            return Err(DeviceError::Malformed(s.to_string()));
        }
        let (kind, index) = match s.split_once(':') {
            Some((kind, index)) => (kind, Some(index)),
            None => (s, None),
        };
        match (kind, index) {
            ("cpu", None) => Ok(Self::Cpu),
            ("meta", None) => Ok(Self::Meta),
            ("cuda", Some(index)) => index
                .parse()
                .map(Self::Cuda)
                .map_err(|_| DeviceError::Malformed(s.to_string())),
            ("cuda", None) => Err(DeviceError::NeedsIndex(kind.to_string())),
            ("cpu" | "meta", Some(_)) => Err(DeviceError::Malformed(s.to_string())),
            _ => Err(DeviceError::Unknown(kind.to_string())),
        }
    }
}

impl From<Device> for String {
    fn from(device: Device) -> Self {
        device.to_string()
    }
}

impl TryFrom<String> for Device {
    type Error = DeviceError;

    fn try_from(s: String) -> Result<Self, Self::Error> {
        s.parse()
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => f.write_str("cpu"),
            Self::Cuda(index) => write!(f, "cuda:{index}"),
            Self::Meta => f.write_str("meta"),
        }
    }
}

/// Why that does not name a place to execute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceError {
    /// We do not know that kind of device.
    Unknown(String),
    /// The kind is one of ours, but what comes with it is not.
    Malformed(String),
    /// It does not say which one.
    NeedsIndex(String),
}

impl fmt::Display for DeviceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown(kind) => write!(
                f,
                "unknown device `{kind}`; today there are `cpu`, `cuda:N` and `meta`"
            ),
            Self::Malformed(s) => write!(
                f,
                "`{s}` is not shaped like a device; write `cpu`, `cuda:N` or `meta`"
            ),
            Self::NeedsIndex(kind) => write!(
                f,
                "`{kind}` does not say which one: write `{kind}:0`. \"The current one\" \
                 is thread state, not a placement"
            ),
        }
    }
}

impl std::error::Error for DeviceError {}
