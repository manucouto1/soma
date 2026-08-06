//! Drawing the graph.
//!
//! Pure data → string: no runtime, no I/O. `to_svg` exists because a
//! notebook sanitizes `<script>`, so the Mermaid diagram a terminal reader
//! is happy with cannot be what `_repr_html_` returns.

use super::PyGraph;
use crate::prelude::*;
use crate::tracking::readers::py_overlay;

/// The graph as a Mermaid diagram, optionally annotated per node.
pub(super) fn to_mermaid(
    g: &PyGraph,
    py: Python<'_>,
    overlay: Option<&Bound<'_, PyDict>>,
) -> PyResult<String> {
    match overlay {
        None => Ok(g.graph.to_mermaid()),
        Some(ov) => Ok(g.graph.to_mermaid_with(&py_overlay(py, ov)?)),
    }
}

/// The graph as a self-contained SVG, optionally annotated per node.
pub(super) fn to_svg(
    g: &PyGraph,
    py: Python<'_>,
    overlay: Option<&Bound<'_, PyDict>>,
) -> PyResult<String> {
    match overlay {
        None => Ok(g.graph.to_svg()),
        Some(ov) => Ok(g.graph.to_svg_with(&py_overlay(py, ov)?)),
    }
}

/// The graph as an ASCII text tree.
pub(super) fn to_text(g: &PyGraph) -> String {
    g.graph.to_text()
}

/// Notebook display: the architecture inline, as SVG.
///
/// Past 80 nodes the diagram is unreadable and slow to lay out, so the
/// text tree is the honest answer instead.
pub(super) fn repr_html(g: &PyGraph) -> String {
    if g.graph.nodes.is_empty() {
        return "<i>empty graph — add nodes with g.node(...)</i>".to_string();
    }
    if g.graph.nodes.len() > 80 {
        return format!(
            "<pre style='font-family:ui-monospace,monospace'>{}</pre>",
            g.graph.to_text().replace('&', "&amp;").replace('<', "&lt;")
        );
    }
    g.graph.to_svg()
}

/// The topology as JSON — written into run directories so a front-end can
/// draw the architecture without the extension.
pub(super) fn graph_json(g: &PyGraph) -> PyResult<String> {
    serde_json::to_string_pretty(&g.graph).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}
