use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Scale for continuous search ranges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Scale {
    Linear,
    Log,
    ReverseLog,
}

/// A single searchable parameter dimension.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "dim_type")]
pub enum SearchDimension {
    /// Continuous range (f64)
    Float {
        name: String,
        low: f64,
        high: f64,
        scale: Scale,
        default: Option<f64>,
    },

    /// Integer range
    Int {
        name: String,
        low: i64,
        high: i64,
        scale: Scale,
    },

    /// Discrete set of choices
    Categorical {
        name: String,
        choices: Vec<serde_json::Value>,
    },

    /// Active only when parent parameter has specific values
    Conditional {
        name: String,
        parent: String,
        parent_values: Vec<serde_json::Value>,
        dimension: Box<SearchDimension>,
    },
}

impl SearchDimension {
    pub fn name(&self) -> &str {
        match self {
            Self::Float { name, .. }
            | Self::Int { name, .. }
            | Self::Categorical { name, .. }
            | Self::Conditional { name, .. } => name,
        }
    }

    /// Validate the dimension configuration.
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::Float { low, high, name, .. } => {
                if low >= high {
                    return Err(format!("{name}: `low` ({low}) must be less than `high` ({high})"));
                }
                Ok(())
            }
            Self::Int { low, high, name, .. } => {
                if low >= high {
                    return Err(format!("{name}: `low` ({low}) must be less than `high` ({high})"));
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
                name, parent, dimension, ..
            } => write!(f, "{name}: Conditional(if {parent}) -> {dimension}"),
        }
    }
}

/// Aggregation of search dimensions from one or more filters.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchSpace {
    pub dimensions: Vec<SearchDimension>,
    pub frozen: HashMap<String, serde_json::Value>,
}

impl SearchSpace {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, dim: SearchDimension) {
        self.dimensions.push(dim);
    }

    /// Merge another search space with a prefix to avoid name collisions.
    pub fn merge_with_prefix(&mut self, prefix: &str, other: SearchSpace) {
        for dim in other.dimensions {
            let prefixed = prefix_dimension(prefix, dim);
            self.dimensions.push(prefixed);
        }
    }

    /// Freeze a parameter to a fixed value (exclude from search).
    pub fn freeze(&mut self, name: &str, value: serde_json::Value) {
        self.frozen.insert(name.to_string(), value);
        self.dimensions.retain(|d| d.name() != name);
    }

    /// Get only the active (non-frozen) dimensions.
    pub fn active_dimensions(&self) -> &[SearchDimension] {
        &self.dimensions
    }

    /// Validate all dimensions.
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

    pub fn is_empty(&self) -> bool {
        self.dimensions.is_empty()
    }

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
    fn search_space() -> SearchSpace;
    fn from_sample(params: &HashMap<String, serde_json::Value>) -> crate::error::Result<Self>
    where
        Self: Sized;
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
        assert_eq!(
            dim.to_string(),
            "kernel: Categorical[\"linear\", \"rbf\"]"
        );
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
