//! What is being searched over: the named knobs and what each one may be.
//!
//! Ordered, and a `Vec` rather than a map on purpose: a grid enumerates in this
//! order, a point writes itself down in this order, and both have to give the
//! same answer on two machines that never spoke.

use std::fmt;

/// One knob and what it may be.
///
/// Three kinds and not more: a bool is a `Choice` of two, and a "power of two
/// between 16 and 512" is an `Int` read as a log. What is not here is a
/// **conditional** dimension — a knob that only exists when another took a
/// particular value — which needs a consumer before it needs a design.
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
/// use soma_next_study::{Dimension, Space};
///
/// let space = Space::new()
///     .with("lr", Dimension::Real { low: 1e-5, high: 1e-1, log: true })?
///     .with("batch", Dimension::Int { low: 16, high: 128 })?
///     .with("optimizer", Dimension::Choice(vec!["adam".into(), "sgd".into()]))?;
/// assert_eq!(space.len(), 3);
/// # Ok::<(), soma_next_study::SpaceError>(())
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
        self.dimensions.push((name, dimension));
        Ok(self)
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

/// Why that is not a knob that can be searched.
#[derive(Debug, Clone, PartialEq)]
pub enum SpaceError {
    /// Two knobs by the same name.
    Taken(String),
    /// A knob with nothing in it, or a range the wrong way round.
    Empty(String, Dimension),
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
        }
    }
}

impl std::error::Error for SpaceError {}
