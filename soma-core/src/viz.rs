//! Rendering overlays for graph visualization.
//!
//! A [`GraphOverlay`] carries per-node execution facts (status, timing,
//! cache tier, health flags) that [`Graph::to_mermaid_with`] and
//! [`Graph::to_graphviz_with`] fold into the rendered diagram. The
//! overlay is pure data — computed elsewhere (e.g. `soma-runtime`'s
//! `RunReader` aggregates it from a run's event log) and passed in, so
//! rendering stays a dependency-free data→string transform.
//!
//! [`Graph::to_mermaid_with`]: crate::graph::Graph::to_mermaid_with
//! [`Graph::to_graphviz_with`]: crate::graph::Graph::to_graphviz_with

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Execution outcome of a node, for status coloring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum NodeStatus {
    /// Executed successfully.
    Completed,
    /// Served from cache without executing.
    Cached,
    /// Execution failed.
    Failed,
    /// Started but not finished (live run, or died mid-node).
    Running,
}

/// Per-node annotation folded into the rendered label and style.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NodeOverlay {
    #[serde(default)]
    pub status: Option<NodeStatus>,
    /// Total compute time across this node's executions.
    #[serde(default)]
    pub duration_ms: Option<u64>,
    /// Cache tier that served a hit (`memory`, `local`, `remote`).
    #[serde(default)]
    pub cache_tier: Option<String>,
    /// Health flags raised on this node (`DEAD_CHANNELS`, `LEAKAGE`, …).
    #[serde(default)]
    pub flags: Vec<String>,
    /// Free-form extra label line (appended after the derived parts).
    #[serde(default)]
    pub sublabel: Option<String>,
}

/// Per-node annotations for one rendering, keyed by node id.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GraphOverlay {
    #[serde(default)]
    pub nodes: BTreeMap<String, NodeOverlay>,
}

impl GraphOverlay {
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

impl NodeOverlay {
    /// The extra label line derived from this overlay, e.g.
    /// `"1.2s · mem hit · ⚠ LEAKAGE"`. `None` when there is nothing
    /// to show.
    pub fn sublabel_text(&self) -> Option<String> {
        let mut parts: Vec<String> = Vec::new();
        if let Some(ms) = self.duration_ms {
            parts.push(format_duration_ms(ms));
        }
        if let Some(tier) = &self.cache_tier {
            let short = match tier.as_str() {
                "memory" => "mem",
                other => other,
            };
            parts.push(format!("{short} hit"));
        } else if self.status == Some(NodeStatus::Failed) {
            parts.push("failed".into());
        }
        for flag in &self.flags {
            parts.push(format!("⚠ {flag}"));
        }
        if let Some(extra) = &self.sublabel {
            parts.push(extra.clone());
        }
        if parts.is_empty() {
            None
        } else {
            Some(parts.join(" · "))
        }
    }

    /// The style class this node gets (mermaid classDef / graphviz
    /// fillcolor). Flags win over status: an unhealthy node must stand
    /// out even when it completed.
    pub fn style_class(&self) -> Option<&'static str> {
        if !self.flags.is_empty() {
            return Some("soma_flagged");
        }
        match self.status? {
            NodeStatus::Completed => Some("soma_completed"),
            NodeStatus::Cached => Some("soma_cached"),
            NodeStatus::Failed => Some("soma_failed"),
            NodeStatus::Running => Some("soma_running"),
        }
    }
}

/// Mermaid `classDef` body for a status class.
pub(crate) fn mermaid_class_style(class: &str) -> &'static str {
    match class {
        "soma_completed" => "fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20;",
        "soma_cached" => "fill:#e3f2fd,stroke:#1565c0,color:#0d47a1;",
        "soma_failed" => "fill:#ffebee,stroke:#c62828,color:#b71c1c;",
        "soma_running" => "fill:#fff8e1,stroke:#f9a825,color:#f57f17;",
        _ => "fill:#fff3e0,stroke:#ef6c00,stroke-width:3px,color:#e65100;",
    }
}

/// Graphviz node attributes for a status class.
pub(crate) fn dot_class_style(class: &str) -> String {
    let (fill, border, extra) = match class {
        "soma_completed" => ("#e8f5e9", "#2e7d32", ""),
        "soma_cached" => ("#e3f2fd", "#1565c0", ""),
        "soma_failed" => ("#ffebee", "#c62828", ""),
        "soma_running" => ("#fff8e1", "#f9a825", ""),
        _ => ("#fff3e0", "#ef6c00", " penwidth=3"),
    };
    format!(" style=filled fillcolor=\"{fill}\" color=\"{border}\"{extra}")
}

/// Compact human duration: `340ms`, `1.2s`, `3.5m`, `2.1h`.
pub fn format_duration_ms(ms: u64) -> String {
    if ms < 1_000 {
        format!("{ms}ms")
    } else if ms < 120_000 {
        format!("{:.1}s", ms as f64 / 1_000.0)
    } else if ms < 7_200_000 {
        format!("{:.1}m", ms as f64 / 60_000.0)
    } else {
        format!("{:.1}h", ms as f64 / 3_600_000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sublabel_composes_parts_in_order() {
        let ov = NodeOverlay {
            status: Some(NodeStatus::Cached),
            duration_ms: Some(1_200),
            cache_tier: Some("memory".into()),
            flags: vec!["LEAKAGE".into()],
            sublabel: Some("×3".into()),
        };
        assert_eq!(
            ov.sublabel_text().unwrap(),
            "1.2s · mem hit · ⚠ LEAKAGE · ×3"
        );
    }

    #[test]
    fn empty_overlay_has_no_sublabel_or_class() {
        let ov = NodeOverlay::default();
        assert!(ov.sublabel_text().is_none());
        assert!(ov.style_class().is_none());
    }

    #[test]
    fn failed_status_shows_in_sublabel_and_class() {
        let ov = NodeOverlay {
            status: Some(NodeStatus::Failed),
            ..Default::default()
        };
        assert_eq!(ov.sublabel_text().unwrap(), "failed");
        assert_eq!(ov.style_class(), Some("soma_failed"));
    }

    #[test]
    fn flags_take_style_precedence_over_status() {
        let ov = NodeOverlay {
            status: Some(NodeStatus::Completed),
            flags: vec!["DEAD_CHANNELS".into()],
            ..Default::default()
        };
        assert_eq!(ov.style_class(), Some("soma_flagged"));
    }

    #[test]
    fn duration_formatting_ranges() {
        assert_eq!(format_duration_ms(340), "340ms");
        assert_eq!(format_duration_ms(1_234), "1.2s");
        assert_eq!(format_duration_ms(150_000), "2.5m");
        assert_eq!(format_duration_ms(9_000_000), "2.5h");
    }

    #[test]
    fn overlay_deserializes_from_partial_json() {
        // The Python side passes overlays as JSON dicts — every field
        // must be optional.
        let ov: GraphOverlay =
            serde_json::from_str(r#"{"nodes": {"a": {"status": "completed", "duration_ms": 42}}}"#)
                .unwrap();
        assert_eq!(ov.nodes["a"].status, Some(NodeStatus::Completed));
        assert_eq!(ov.nodes["a"].duration_ms, Some(42));
        assert!(ov.nodes["a"].flags.is_empty());
    }
}
