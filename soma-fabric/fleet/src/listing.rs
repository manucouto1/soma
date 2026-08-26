//! The names a graph writes, and what each of them resolves to.
//!
//! It is the local broker's, and it is here on loan: what holds a listing
//! beyond the life of one client is the **local broker**, the second of the
//! three deployments, and it is not written. So for now the listing is a file
//! this reads and writes, and the shape of these routes is what a broker will
//! answer — a name, a path, and which names are one wire. Moving it is then a
//! matter of where the answer comes from.

use serde::{Deserialize, Serialize};
use somatize_fabric_broker::{Embedded, Endpoint, Host, Path as Reach, Session};
use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::Path;
use std::sync::Arc;

/// A listing, as it sits in a file somebody may have edited by hand.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Listing {
    /// The names, in the order the file has them.
    #[serde(default)]
    pub listed: Vec<Listed>,
}

/// One name, and how to get to it. Exactly one of the two ways.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Listed {
    /// What the graph writes in `.at("…")`.
    pub host: String,
    /// A worker that is already standing: `"node3:7000"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at: Option<String>,
    /// A worker to be started here, as a child.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run: Option<Vec<String>>,
}

/// The listing grouped the way it is provisioned, with the ladder beside it.
#[derive(Debug, Clone, Serialize)]
pub struct Wires {
    /// One entry per wire, not per name.
    pub wires: Vec<Wire>,
    /// The four ways a pair of endpoints can end up talking, and which of them
    /// can be answered today.
    pub ladder: Vec<Rung>,
}

/// One wire, and every name that resolves to it.
#[derive(Debug, Clone, Serialize)]
pub struct Wire {
    /// What it resolves to: an address, or the command that would be run.
    ///
    /// The endpoint alone and not `Path`'s own sentence, which is written for a
    /// Rust error message. Saying which rung it is on is [`Wire::rung`]'s job.
    pub how: String,
    /// Which rung of the ladder it is on.
    pub rung: &'static str,
    /// Whether two names of it would share a catalog.
    pub shared: bool,
    /// The names, and what is known about each.
    pub names: Vec<Named>,
}

/// One name on a wire.
#[derive(Debug, Clone, Serialize)]
pub struct Named {
    /// What the graph calls it.
    pub host: String,
    /// What the machine behind it calls itself, when somebody has talked to it.
    /// `None` is a name nobody has met — which is not the same as broken.
    pub seen: Option<String>,
}

/// One rung of the ladder, and whether anybody can climb it yet.
#[derive(Debug, Clone, Serialize)]
pub struct Rung {
    /// Cheapest first.
    pub rung: u8,
    /// What it is, said plainly.
    pub what: &'static str,
    /// Whether a broker can answer it today.
    pub answerable: bool,
}

/// Why a listing could not be read, written, or added to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Trouble {
    /// The file is not there, or will not open.
    Io(String),
    /// It is there and is not a listing.
    Unreadable(String),
    /// What was asked for cannot be listed.
    Refused(String),
}

impl Listing {
    /// The listing in this file. A file that is not there is an empty listing
    /// and not a failure: nobody has listed anything yet.
    pub fn read(at: &Path) -> Result<Self, Trouble> {
        let said = match fs::read_to_string(at) {
            Ok(said) => said,
            Err(why) if why.kind() == std::io::ErrorKind::NotFound => return Ok(Self::default()),
            Err(why) => return Err(Trouble::Io(format!("`{}`: {why}", at.display()))),
        };
        toml::from_str(&said)
            .map_err(|why| Trouble::Unreadable(format!("`{}`: {why}", at.display())))
    }

    /// Writes it back, whole. A listing is small and a name is a question whose
    /// answer can be refreshed, which is the same shape the store uses.
    pub fn write(&self, at: &Path) -> Result<(), Trouble> {
        let said = toml::to_string_pretty(self)
            .map_err(|why| Trouble::Unreadable(format!("this listing will not write: {why}")))?;
        if let Some(directory) = at.parent().filter(|one| !one.as_os_str().is_empty()) {
            fs::create_dir_all(directory).map_err(|why| Trouble::Io(why.to_string()))?;
        }
        fs::write(at, said).map_err(|why| Trouble::Io(format!("`{}`: {why}", at.display())))
    }

    /// Adds a name, or refuses to say why.
    ///
    /// A name that is already listed is **replaced** rather than doubled: two
    /// rows for one name would be a listing that answers two things to one
    /// question, and whichever a broker picked would be arbitrary.
    pub fn add(&mut self, one: Listed) -> Result<(), Trouble> {
        one.reach()?;
        self.listed.retain(|already| already.host != one.host);
        self.listed.push(one);
        Ok(())
    }

    /// Drops a name, and says whether there was one.
    pub fn drop(&mut self, host: &str) -> bool {
        let before = self.listed.len();
        self.listed.retain(|one| one.host != host);
        before != self.listed.len()
    }

    /// The listing grouped by wire, with each name joined to the machine behind
    /// it where anybody has met one.
    ///
    /// `met` is `id → host` — the join the fleet already worked out — read the
    /// other way round here because a listing is asked about by name.
    pub fn wires(&self, met: &BTreeMap<String, Host>) -> Result<Wires, Trouble> {
        let mut reaches: Vec<(Host, Reach)> = Vec::new();
        for one in &self.listed {
            reaches.push((Host::new(&one.host), one.reach()?));
        }

        // A real broker, asked the question a client asks. It costs a thread and
        // a handful of microseconds, and it is the only way the rule stays in
        // one place.
        let session = Session::with(Arc::new(Embedded::open(reaches.clone())));
        let mut grouped: BTreeMap<Vec<u8>, Wire> = BTreeMap::new();
        for (host, reach) in reaches {
            let token = session
                .wire_token(&host)
                .map_err(|why| Trouble::Refused(why.message().to_string()))?;
            let seen = met
                .iter()
                .find(|(_, named)| **named == host)
                .map(|(id, _)| id.clone());
            grouped
                .entry(token)
                .or_insert_with(|| Wire {
                    how: match &reach {
                        Reach::Direct { endpoint } => endpoint.to_string(),
                        other => other.to_string(),
                    },
                    rung: rung_of(&reach),
                    shared: reach.shared(),
                    names: Vec::new(),
                })
                .names
                .push(Named {
                    host: host.as_str().to_string(),
                    seen,
                });
        }

        Ok(Wires {
            wires: grouped.into_values().collect(),
            ladder: LADDER.to_vec(),
        })
    }
}

impl Listed {
    /// A name for a worker that is already standing.
    pub fn at(host: impl Into<String>, address: impl Into<String>) -> Self {
        Self {
            host: host.into(),
            at: Some(address.into()),
            run: None,
        }
    }

    /// A name for a worker to be started as a child.
    pub fn run<S: Into<String>>(
        host: impl Into<String>,
        argv: impl IntoIterator<Item = S>,
    ) -> Self {
        Self {
            host: host.into(),
            at: None,
            run: Some(argv.into_iter().map(Into::into).collect()),
        }
    }

    /// What this resolves to, or why it does not.
    ///
    /// Exactly one of the two, and saying so beats picking one: a row with both
    /// is somebody who meant one of them, and guessing which would be a listing
    /// that quietly does not do what its file says.
    pub fn reach(&self) -> Result<Reach, Trouble> {
        match (&self.at, &self.run) {
            (Some(address), None) if !address.trim().is_empty() => Ok(Reach::Direct {
                endpoint: Endpoint::Address(address.clone()),
            }),
            (None, Some(argv)) if !argv.is_empty() => Ok(Reach::Direct {
                endpoint: Endpoint::Command(argv.clone()),
            }),
            (None, Some(_)) => Err(Trouble::Refused(format!(
                "`{}` is a command with no program in it",
                self.host
            ))),
            (Some(_), Some(_)) => Err(Trouble::Refused(format!(
                "`{}` is listed as an address **and** as a command; it is one or the other",
                self.host
            ))),
            _ => Err(Trouble::Refused(format!(
                "`{}` says nothing about how to get to it: give it an address or a command",
                self.host
            ))),
        }
    }
}

/// The four, cheapest first, and which of them a broker can answer today.
///
/// The three that cannot are **not hidden**. They are in the message from the
/// first version so that the day the negotiation picks one it is new behaviour
/// and not a new protocol, and a form that hid them would be teaching a ladder
/// with one rung on it.
const LADDER: [Rung; 4] = [
    Rung {
        rung: 1,
        what: "in this very process",
        answerable: false,
    },
    Rung {
        rung: 2,
        what: "through a directory both of them see",
        answerable: false,
    },
    Rung {
        rung: 3,
        what: "direct, with the broker out of it",
        answerable: true,
    },
    Rung {
        rung: 4,
        what: "relayed, through the broker",
        answerable: false,
    },
];

/// Which rung a path is on, said the way the form says it.
fn rung_of(reach: &Reach) -> &'static str {
    match reach {
        Reach::InProcess { .. } => "in this very process",
        Reach::Mount { .. } => "through a directory both of them see",
        Reach::Direct {
            endpoint: Endpoint::Address(_),
        } => "direct · an address",
        Reach::Direct {
            endpoint: Endpoint::Command(_),
        } => "direct · a command",
        Reach::Relayed { .. } => "relayed, through the broker",
    }
}

impl fmt::Display for Trouble {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(why) => write!(f, "the listing could not be reached: {why}"),
            Self::Unreadable(why) => write!(f, "what is in the listing is not a listing: {why}"),
            Self::Refused(why) => f.write_str(why),
        }
    }
}
