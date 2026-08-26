//! What is being searched over: the named knobs and what each one may be.
//!
//! Ordered, and a `Vec` rather than a map on purpose: a grid enumerates in this
//! order, a point writes itself down in this order, and both have to give the
//! same answer on two machines that never spoke.

use crate::{Point, Setting};
use std::fmt;

/// One knob and what it may be. Three kinds and not more: a bool is a `Choice`
/// of two, and a *power of two between 16 and 512* is an `Int` read as a log.
/// Not here: a **conditional** dimension, which needs a consumer first.
#[derive(Debug, Clone, PartialEq)]
pub enum Dimension {
    /// Anything between the two.
    Real {
        /// The bottom, included.
        low: f64,
        /// The top, included.
        high: f64,
        /// Drawn evenly in the **logarithm**, which is the only sane way to
        /// search a learning rate: `1e-5..1e-1` spends four fifths of a linear
        /// draw above `0.02`.
        log: bool,
    },
    /// A whole number between the two, both included.
    Int {
        /// The bottom, included.
        low: i64,
        /// The top, included.
        high: i64,
    },
    /// One of these, by name. Nothing is read into their order.
    Choice(Vec<String>),
}

impl Dimension {
    /// Whether this says anything at all, and is the right way round.
    fn sound(&self) -> bool {
        match self {
            Self::Real { low, high, log } => {
                low < high && low.is_finite() && high.is_finite() && (!log || *low > 0.0)
            }
            Self::Int { low, high } => low < high,
            Self::Choice(options) => !options.is_empty(),
        }
    }

    /// How many values a grid takes from it, given how finely it is asked to cut
    /// what is continuous. A `Choice` takes all of them; an `Int` takes all of
    /// them too unless there are more than `steps`.
    pub fn grid_of(&self, steps: usize) -> usize {
        match self {
            Self::Real { .. } => steps.max(1),
            Self::Int { low, high } => ((high - low + 1) as usize).min(steps.max(1)),
            Self::Choice(options) => options.len(),
        }
    }
}

impl fmt::Display for Dimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Real {
                low,
                high,
                log: false,
            } => write!(f, "real({low},{high})"),
            Self::Real {
                low,
                high,
                log: true,
            } => write!(f, "logreal({low},{high})"),
            Self::Int { low, high } => write!(f, "int({low},{high})"),
            Self::Choice(options) => write!(f, "choice({})", options.join("|")),
        }
    }
}

/// The knobs, in the order they were declared.
///
/// ```
/// use somatize_study::{Dimension, Space};
///
/// let space = Space::new()
///     .with("lr", Dimension::Real { low: 1e-5, high: 1e-1, log: true })?
///     .with("batch", Dimension::Int { low: 16, high: 128 })?
///     .with("optimizer", Dimension::Choice(vec!["adam".into(), "sgd".into()]))?;
/// assert_eq!(space.len(), 3);
/// # Ok::<(), somatize_study::SpaceError>(())
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Space {
    dimensions: Vec<(String, Dimension)>,
}

impl Space {
    /// Nothing to search yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// One more knob. The name has to be new, and what it may be has to be
    /// something: an empty `Choice` or a range the wrong way round is refused
    /// **here**, where it was written, and not as a search that quietly only
    /// ever tried one value.
    pub fn with(
        mut self,
        name: impl Into<String>,
        dimension: Dimension,
    ) -> Result<Self, SpaceError> {
        let name = name.into();
        if self.dimensions.iter().any(|(taken, _)| taken == &name) {
            return Err(SpaceError::Taken(name));
        }
        if !dimension.sound() {
            return Err(SpaceError::Empty(name, dimension));
        }
        // A point writes itself down as `name=value,name=value`, and a study
        // reads it back off a shared folder. A name or an option carrying one of
        // those two characters makes that text ambiguous, and the day it is read
        // wrong there is nothing left to tell which knob was meant.
        if let Some(text) = punctuated(&name, &dimension) {
            return Err(SpaceError::Unreadable(name, text));
        }
        self.dimensions.push((name, dimension));
        Ok(self)
    }

    /// The point that text names, read against these knobs.
    ///
    /// The other half of [`Point`]'s `Display`, and it needs the space: `batch=64`
    /// on its own does not say whether 64 is a whole number or a
    /// [`Choice`](Dimension::Choice) spelt `"64"`. This is what makes a study's
    /// history come back in **one scan and no fetches**.
    ///
    /// Every knob has to be there and nothing else may be: a record written
    /// against another space is a different study.
    /// ```
    /// use somatize_study::{Dimension, Space};
    ///
    /// let space = Space::new()
    ///     .with("lr", Dimension::Real { low: 1e-5, high: 1e-1, log: true })?
    ///     .with("opt", Dimension::Choice(vec!["adam".into(), "sgd".into()]))?;
    /// let point = space.read("lr=0.001,opt=adam")?;
    ///
    /// assert_eq!(point.to_string(), "lr=0.001,opt=adam");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn read(&self, said: &str) -> Result<Point, ReadError> {
        let mut given: Vec<(&str, &str)> = Vec::new();
        for piece in said.split(',').filter(|piece| !piece.trim().is_empty()) {
            let (name, value) = piece
                .split_once('=')
                .ok_or_else(|| ReadError::Shapeless(piece.trim().to_string()))?;
            given.push((name.trim(), value.trim()));
        }

        let mut settings = Vec::with_capacity(self.dimensions.len());
        for (name, dimension) in &self.dimensions {
            let value = given
                .iter()
                .find(|(said, _)| said == name)
                .ok_or_else(|| ReadError::Missing(name.clone()))?
                .1;
            let setting = understand(dimension, value).ok_or_else(|| {
                ReadError::NotIn(name.clone(), dimension.clone(), value.to_string())
            })?;
            settings.push((name.clone(), setting));
        }

        if let Some((stranger, _)) = given
            .iter()
            .find(|(name, _)| !self.dimensions.iter().any(|(taken, _)| taken == name))
        {
            return Err(ReadError::Stranger(stranger.to_string()));
        }
        Ok(Point::of(settings))
    }

    /// The knobs, in declaration order.
    pub fn dimensions(&self) -> &[(String, Dimension)] {
        &self.dimensions
    }

    /// How many knobs there are.
    pub fn len(&self) -> usize {
        self.dimensions.len()
    }

    /// Whether there is nothing to search.
    pub fn is_empty(&self) -> bool {
        self.dimensions.is_empty()
    }
}

impl fmt::Display for Space {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let said: Vec<String> = self
            .dimensions
            .iter()
            .map(|(name, dimension)| format!("{name}={dimension}"))
            .collect();
        f.write_str(&said.join(","))
    }
}

/// The name or the option that carries a `,` or an `=`, if either does.
fn punctuated(name: &str, dimension: &Dimension) -> Option<String> {
    let loud = |text: &str| text.contains(',') || text.contains('=');
    if loud(name) {
        return Some(name.to_string());
    }
    match dimension {
        Dimension::Choice(options) => options.iter().find(|option| loud(option)).cloned(),
        _ => None,
    }
}

/// That text as a setting of that knob, or `None` when it is not one of its
/// values — the wrong kind, an option nobody declared, or a number outside the
/// range. Everything a sampler produces sits inside, so this only refuses what
/// was written against a different space.
fn understand(dimension: &Dimension, value: &str) -> Option<Setting> {
    match dimension {
        Dimension::Real { low, high, .. } => value
            .parse::<f64>()
            .ok()
            .filter(|read| (low..=high).contains(&read))
            .map(Setting::Real),
        Dimension::Int { low, high } => value
            .parse::<i64>()
            .ok()
            .filter(|read| (low..=high).contains(&read))
            .map(Setting::Int),
        Dimension::Choice(options) => options
            .iter()
            .find(|option| *option == value)
            .cloned()
            .map(Setting::Choice),
    }
}

/// Why that is not a knob that can be searched.
#[derive(Debug, Clone, PartialEq)]
pub enum SpaceError {
    /// Two knobs by the same name.
    Taken(String),
    /// A knob with nothing in it, or a range the wrong way round.
    Empty(String, Dimension),
    /// A name, or one of a choice's options, carrying the punctuation a written
    /// point is made of.
    Unreadable(String, String),
}

impl fmt::Display for SpaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Taken(name) => write!(
                f,
                "`{name}` is already a dimension of this space: a point would have two \
                 values for it and no way to say which"
            ),
            Self::Empty(name, dimension) => write!(
                f,
                "`{name}` as `{dimension}` has nothing to draw from: a range needs its \
                 bottom below its top, a choice needs an option, and a logarithmic range \
                 needs to start above zero"
            ),
            Self::Unreadable(name, text) => write!(
                f,
                "`{text}`, of the knob `{name}`, has a `,` or an `=` in it, and a point \
                 writes itself down as `name=value,name=value`: kept, it would be a trial \
                 name that cannot be read back, and by then which knob was meant is gone"
            ),
        }
    }
}

impl std::error::Error for SpaceError {}

/// Why that text is not a point of this space.
#[derive(Debug, Clone, PartialEq)]
pub enum ReadError {
    /// A piece with no `=` in it.
    Shapeless(String),
    /// A knob of this space the text says nothing about.
    Missing(String),
    /// A name that is not a knob of this space.
    Stranger(String),
    /// A value that is not one this knob could take.
    NotIn(String, Dimension, String),
}

impl fmt::Display for ReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shapeless(piece) => write!(
                f,
                "`{piece}` is not `name=value`, and a point is written down as those \
                 separated by commas"
            ),
            Self::Missing(name) => write!(
                f,
                "this space has a knob `{name}` and the text sets nothing for it: a point \
                 of a space sets every one of its knobs, so this was written against \
                 another space"
            ),
            Self::Stranger(name) => write!(
                f,
                "`{name}` is not a knob of this space: the text was written against \
                 another one, and reading the part that does match would be a point of \
                 neither"
            ),
            Self::NotIn(name, dimension, value) => write!(
                f,
                "`{value}` is not a value of `{name}`, which is `{dimension}`: it is \
                 either the wrong kind, an option nobody declared, or outside the range"
            ),
        }
    }
}

impl std::error::Error for ReadError {}
