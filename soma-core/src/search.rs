//! Search spaces for hyperparameter optimization.
//!
//! Defines [`SearchSpace`] (a collection of [`SearchDimension`]s) that
//! samplers use to generate trial configurations. Dimensions can be
//! float, int, categorical, or conditional.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Scale for continuous search ranges.
///
/// Governs how samplers interpolate between `low` and `high`: which
/// values a grid places its points at and where random draws
/// concentrate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Scale {
    /// Uniform spacing across the range.
    Linear,
    /// Logarithmic spacing: samples concentrate near the low end.
    /// The right choice for learning rates and regularization
    /// strengths, where orders of magnitude matter more than
    /// absolute differences.
    Log,
    /// Mirror image of [`Scale::Log`]: samples concentrate near the
    /// *high* end. Useful for parameters like momentum or keep
    /// probability, where the interesting region is close to the
    /// upper bound.
    ReverseLog,
}

/// A single searchable parameter dimension.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "dim_type")]
#[non_exhaustive]
pub enum SearchDimension {
    /// Continuous range (f64)
    Float {
        /// Parameter name (prefixed with the filter label when spaces
        /// are merged, e.g. `"SVM.C"`).
        name: String,
        /// Inclusive lower bound. Must be strictly less than `high`.
        low: f64,
        /// Inclusive upper bound.
        high: f64,
        /// How samplers interpolate between the bounds.
        scale: Scale,
        /// Value used when the dimension is not being searched;
        /// `None` means the filter's own field default applies.
        default: Option<f64>,
    },

    /// Integer range
    Int {
        /// Parameter name (prefixed with the filter label when spaces
        /// are merged).
        name: String,
        /// Inclusive lower bound. Must be strictly less than `high`.
        low: i64,
        /// Inclusive upper bound.
        high: i64,
        /// How samplers interpolate between the bounds.
        scale: Scale,
    },

    /// Discrete set of choices
    Categorical {
        /// Parameter name (prefixed with the filter label when spaces
        /// are merged).
        name: String,
        /// The candidate values, as JSON so strings, numbers, and
        /// booleans can mix. Must not be empty.
        choices: Vec<serde_json::Value>,
    },

    /// Active only when parent parameter has specific values
    Conditional {
        /// Parameter name (prefixed with the filter label when spaces
        /// are merged).
        name: String,
        /// Name of the parameter this dimension depends on. Prefixed
        /// alongside `name` in [`SearchSpace::merge_with_prefix`], so
        /// the link survives merging.
        parent: String,
        /// Parent values that activate this dimension; for any other
        /// parent value the dimension is skipped.
        parent_values: Vec<serde_json::Value>,
        /// The dimension to sample when active.
        dimension: Box<SearchDimension>,
    },
}

impl SearchDimension {
    /// The parameter name, whichever variant this is. This is the key
    /// trial params are stored under and the handle [`SearchSpace::freeze`]
    /// matches on.
    pub fn name(&self) -> &str {
        match self {
            Self::Float { name, .. }
            | Self::Int { name, .. }
            | Self::Categorical { name, .. }
            | Self::Conditional { name, .. } => name,
        }
    }

    /// Validate the dimension configuration: ranges must satisfy
    /// `low < high`, categorical `choices` must not be empty, and a
    /// conditional is as valid as its inner dimension. The `Err`
    /// message names the offending dimension.
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::Float {
                low, high, name, ..
            } => {
                if low >= high {
                    return Err(format!(
                        "{name}: `low` ({low}) must be less than `high` ({high})"
                    ));
                }
                Ok(())
            }
            Self::Int {
                low, high, name, ..
            } => {
                if low >= high {
                    return Err(format!(
                        "{name}: `low` ({low}) must be less than `high` ({high})"
                    ));
                }
                Ok(())
            }
            Self::Categorical { choices, name } => {
                if choices.is_empty() {
                    return Err(format!("{name}: `choices` must not be empty"));
                }
                Ok(())
            }
            Self::Conditional { dimension, .. } => dimension.validate(),
        }
    }
}

impl fmt::Display for SearchDimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Float {
                name,
                low,
                high,
                scale,
                ..
            } => write!(f, "{name}: Float[{low}, {high}] {scale:?}"),
            Self::Int {
                name, low, high, ..
            } => write!(f, "{name}: Int[{low}, {high}]"),
            Self::Categorical { name, choices } => {
                let labels: Vec<String> = choices.iter().map(|c| c.to_string()).collect();
                write!(f, "{name}: Categorical[{}]", labels.join(", "))
            }
            Self::Conditional {
                name,
                parent,
                dimension,
                ..
            } => write!(f, "{name}: Conditional(if {parent}) -> {dimension}"),
        }
    }
}

/// Aggregation of search dimensions from one or more filters.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchSpace {
    /// The dimensions samplers draw from. [`SearchSpace::freeze`]
    /// removes a dimension from this list, so everything here is
    /// actively searched.
    pub dimensions: Vec<SearchDimension>,
    /// Parameters pinned to a fixed value by [`SearchSpace::freeze`],
    /// keyed by dimension name. The `StudyRunner` injects these into
    /// every trial's params so a frozen parameter still reaches the
    /// filter (and its cache key).
    pub frozen: HashMap<String, serde_json::Value>,
}

impl SearchSpace {
    /// An empty search space: no dimensions, nothing frozen.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a dimension. No name-collision check happens here; use
    /// [`SearchSpace::merge_with_prefix`] when combining spaces from
    /// several filters.
    pub fn add(&mut self, dim: SearchDimension) {
        self.dimensions.push(dim);
    }

    /// Merge another search space with a prefix to avoid name collisions.
    ///
    /// Every dimension name in `other` becomes `"{prefix}.{name}"`
    /// (conditional dimensions get their `parent` prefixed too, so the
    /// dependency still resolves). This is how a graph aggregates each
    /// filter's [`Searchable::search_space`] into one space keyed by
    /// node label.
    pub fn merge_with_prefix(&mut self, prefix: &str, other: SearchSpace) {
        for dim in other.dimensions {
            let prefixed = prefix_dimension(prefix, dim);
            self.dimensions.push(prefixed);
        }
    }

    /// Freeze a parameter to a fixed value (exclude from search).
    ///
    /// Removes the dimension named `name` from [`SearchSpace::dimensions`]
    /// and records the value in [`SearchSpace::frozen`]; the value is
    /// then injected into every trial's params by the runner. Freezing
    /// a name with no matching dimension just records the value.
    pub fn freeze(&mut self, name: &str, value: serde_json::Value) {
        self.frozen.insert(name.to_string(), value);
        self.dimensions.retain(|d| d.name() != name);
    }

    /// The dimensions still being searched. Since [`SearchSpace::freeze`]
    /// removes frozen dimensions from the list, this is every entry of
    /// [`SearchSpace::dimensions`] — the method exists so callers state
    /// their intent rather than reach into the field.
    pub fn active_dimensions(&self) -> &[SearchDimension] {
        &self.dimensions
    }

    /// Validate all dimensions, collecting every failure rather than
    /// stopping at the first — a study definition should surface all
    /// its range errors in one pass. `Ok(())` when every dimension
    /// passes [`SearchDimension::validate`].
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let errors: Vec<String> = self
            .dimensions
            .iter()
            .filter_map(|d| d.validate().err())
            .collect();
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// `true` when there is nothing left to search. Frozen values do
    /// not count: a space can be empty yet still carry frozen params.
    pub fn is_empty(&self) -> bool {
        self.dimensions.is_empty()
    }

    /// Number of searchable dimensions (frozen params excluded).
    pub fn len(&self) -> usize {
        self.dimensions.len()
    }
}

impl fmt::Display for SearchSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for dim in &self.dimensions {
            writeln!(f, "  {dim}")?;
        }
        if !self.frozen.is_empty() {
            writeln!(f, "  Frozen:")?;
            for (name, val) in &self.frozen {
                writeln!(f, "    {name} = {val}")?;
            }
        }
        Ok(())
    }
}

/// Prefix all dimension names with a filter label.
fn prefix_dimension(prefix: &str, dim: SearchDimension) -> SearchDimension {
    match dim {
        SearchDimension::Float {
            name,
            low,
            high,
            scale,
            default,
        } => SearchDimension::Float {
            name: format!("{prefix}.{name}"),
            low,
            high,
            scale,
            default,
        },
        SearchDimension::Int {
            name,
            low,
            high,
            scale,
        } => SearchDimension::Int {
            name: format!("{prefix}.{name}"),
            low,
            high,
            scale,
        },
        SearchDimension::Categorical { name, choices } => SearchDimension::Categorical {
            name: format!("{prefix}.{name}"),
            choices,
        },
        SearchDimension::Conditional {
            name,
            parent,
            parent_values,
            dimension,
        } => SearchDimension::Conditional {
            name: format!("{prefix}.{name}"),
            parent: format!("{prefix}.{parent}"),
            parent_values,
            dimension: Box::new(prefix_dimension(prefix, *dimension)),
        },
    }
}

/// Trait for filters that declare their search space.
/// Auto-generated by `#[derive(Filter)]`.
pub trait Searchable {
    /// The dimensions this filter exposes for search, one per field
    /// annotated with a `#[soma(search(...))]` range. Names are the
    /// bare field names — a graph namespaces them per node via
    /// [`SearchSpace::merge_with_prefix`].
    fn search_space() -> SearchSpace;
    /// Build an instance from a sampled configuration — how a trial's
    /// sampled point becomes a runnable filter. Params are keyed by
    /// bare field name; in the derived impl a missing or
    /// wrongly-typed key falls back to the type's `Default`, never an
    /// error (the `Result` is for hand-written impls with real
    /// construction failures).
    fn from_sample(params: &HashMap<String, serde_json::Value>) -> crate::error::Result<Self>
    where
        Self: Sized;
    /// The instance's current searchable field values, keyed by bare
    /// field name — the inverse of [`Searchable::from_sample`], used
    /// to record what configuration actually ran.
    fn current_params(&self) -> HashMap<String, serde_json::Value>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn float_dimension_display() {
        let dim = SearchDimension::Float {
            name: "lr".into(),
            low: 0.001,
            high: 0.1,
            scale: Scale::Log,
            default: None,
        };
        assert_eq!(dim.to_string(), "lr: Float[0.001, 0.1] Log");
    }

    #[test]
    fn categorical_dimension_display() {
        let dim = SearchDimension::Categorical {
            name: "kernel".into(),
            choices: vec![json!("linear"), json!("rbf")],
        };
        assert_eq!(dim.to_string(), "kernel: Categorical[\"linear\", \"rbf\"]");
    }

    #[test]
    fn validate_rejects_inverted_range() {
        let dim = SearchDimension::Float {
            name: "lr".into(),
            low: 1.0,
            high: 0.1,
            scale: Scale::Linear,
            default: None,
        };
        assert!(dim.validate().is_err());
    }

    #[test]
    fn validate_rejects_empty_choices() {
        let dim = SearchDimension::Categorical {
            name: "kernel".into(),
            choices: vec![],
        };
        assert!(dim.validate().is_err());
    }

    #[test]
    fn validate_accepts_valid_dimensions() {
        let float = SearchDimension::Float {
            name: "lr".into(),
            low: 0.001,
            high: 0.1,
            scale: Scale::Log,
            default: None,
        };
        let int = SearchDimension::Int {
            name: "epochs".into(),
            low: 10,
            high: 100,
            scale: Scale::Linear,
        };
        assert!(float.validate().is_ok());
        assert!(int.validate().is_ok());
    }

    #[test]
    fn search_space_merge_with_prefix() {
        let mut space1 = SearchSpace::new();
        space1.add(SearchDimension::Float {
            name: "scale".into(),
            low: 0.1,
            high: 10.0,
            scale: Scale::Log,
            default: None,
        });

        let mut space2 = SearchSpace::new();
        space2.add(SearchDimension::Float {
            name: "C".into(),
            low: 0.01,
            high: 100.0,
            scale: Scale::Log,
            default: None,
        });

        let mut combined = SearchSpace::new();
        combined.merge_with_prefix("Scaler", space1);
        combined.merge_with_prefix("SVM", space2);

        assert_eq!(combined.len(), 2);
        assert_eq!(combined.dimensions[0].name(), "Scaler.scale");
        assert_eq!(combined.dimensions[1].name(), "SVM.C");
    }

    #[test]
    fn search_space_freeze() {
        let mut space = SearchSpace::new();
        space.add(SearchDimension::Float {
            name: "lr".into(),
            low: 0.001,
            high: 0.1,
            scale: Scale::Log,
            default: None,
        });
        space.add(SearchDimension::Categorical {
            name: "kernel".into(),
            choices: vec![json!("rbf"), json!("linear")],
        });

        assert_eq!(space.len(), 2);
        space.freeze("kernel", json!("rbf"));
        assert_eq!(space.len(), 1);
        assert_eq!(space.dimensions[0].name(), "lr");
        assert_eq!(space.frozen["kernel"], json!("rbf"));
    }

    #[test]
    fn search_space_validate() {
        let mut space = SearchSpace::new();
        space.add(SearchDimension::Float {
            name: "good".into(),
            low: 0.0,
            high: 1.0,
            scale: Scale::Linear,
            default: None,
        });
        assert!(space.validate().is_ok());

        space.add(SearchDimension::Float {
            name: "bad".into(),
            low: 10.0,
            high: 1.0,
            scale: Scale::Linear,
            default: None,
        });
        assert!(space.validate().is_err());
    }

    #[test]
    fn search_space_serde_roundtrip() {
        let mut space = SearchSpace::new();
        space.add(SearchDimension::Float {
            name: "lr".into(),
            low: 0.001,
            high: 0.1,
            scale: Scale::Log,
            default: Some(0.01),
        });
        space.add(SearchDimension::Int {
            name: "epochs".into(),
            low: 10,
            high: 100,
            scale: Scale::Linear,
        });
        space.add(SearchDimension::Categorical {
            name: "kernel".into(),
            choices: vec![json!("rbf"), json!("linear")],
        });

        let json = serde_json::to_string(&space).unwrap();
        let deserialized: SearchSpace = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.len(), 3);
    }

    #[test]
    fn conditional_dimension() {
        let dim = SearchDimension::Conditional {
            name: "momentum".into(),
            parent: "optimizer".into(),
            parent_values: vec![json!("sgd")],
            dimension: Box::new(SearchDimension::Float {
                name: "momentum".into(),
                low: 0.0,
                high: 0.99,
                scale: Scale::Linear,
                default: None,
            }),
        };
        assert!(dim.validate().is_ok());
    }
}
