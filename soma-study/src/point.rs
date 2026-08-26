//! One configuration: what each knob was set to.
//!
//! It writes itself down — `batch=32,lr=0.001` — because that is a trial's
//! **name**: what a record is filed under. Derived from the values in the
//! space's order, so two machines that never spoke file it identically.

use std::fmt;

/// What one knob was set to.
#[derive(Debug, Clone, PartialEq)]
pub enum Setting {
    /// A [`Real`](crate::Dimension::Real) dimension's value.
    Real(f64),
    /// An [`Int`](crate::Dimension::Int) dimension's value.
    Int(i64),
    /// Which of a [`Choice`](crate::Dimension::Choice)'s options.
    Choice(String),
}

impl fmt::Display for Setting {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Real(value) => write!(f, "{value}"),
            Self::Int(value) => write!(f, "{value}"),
            Self::Choice(option) => f.write_str(option),
        }
    }
}

/// One point of the space: every knob, set.
#[derive(Debug, Clone, PartialEq)]
pub struct Point {
    settings: Vec<(String, Setting)>,
}

impl Point {
    /// A point from its settings, in the space's order.
    pub fn of(settings: Vec<(String, Setting)>) -> Self {
        Self { settings }
    }

    /// What that knob was set to, or `None` if this point does not have it.
    pub fn get(&self, name: &str) -> Option<&Setting> {
        self.settings
            .iter()
            .find(|(taken, _)| taken == name)
            .map(|(_, setting)| setting)
    }

    /// Every knob, in the space's order.
    pub fn settings(&self) -> &[(String, Setting)] {
        &self.settings
    }

    /// How many knobs are set.
    pub fn len(&self) -> usize {
        self.settings.len()
    }

    /// Whether nothing is set.
    pub fn is_empty(&self) -> bool {
        self.settings.is_empty()
    }
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let said: Vec<String> = self
            .settings
            .iter()
            .map(|(name, setting)| format!("{name}={setting}"))
            .collect();
        f.write_str(&said.join(","))
    }
}
