//! Self-contained SVG rendering of a [`Graph`] — no JavaScript, no
//! external tools.
//!
//! Mermaid needs a JS runtime and notebook front-ends sanitize
//! `<script>` out of cell outputs, so diagrams that must show up
//! *inline* (notebook reprs, offline HTML reports, GitHub previews)
//! render through this pure data→string layer instead. Layout is a
//! simple longest-path layering (left→right), which fits Soma's small
//! chain/fork DAGs; styling reuses the same status palette as the
//! mermaid/graphviz overlay classes so a run reads identically in
//! every rendering.

use crate::graph::{EdgeKind, Graph, NodeKind};
use crate::viz::GraphOverlay;
use std::collections::HashMap;

const NODE_H: f32 = 34.0;
const SUB_EXTRA_H: f32 = 16.0;
const X_GAP: f32 = 56.0;
const Y_GAP: f32 = 22.0;
const MARGIN: f32 = 16.0;
const PAD_X: f32 = 14.0;
const CHAR_W: f32 = 7.6; // ≈13px system-ui
const SUB_CHAR_W: f32 = 6.4; // ≈11px

/// (fill, stroke, label ink, stroke width) per overlay style class.
fn class_colors(class: Option<&str>) -> (&'static str, &'static str, &'static str, f32) {
    match class {
        Some("soma_completed") => ("#e8f5e9", "#2e7d32", "#1b5e20", 1.4),
        Some("soma_cached") => ("#e3f2fd", "#1565c0", "#0d47a1", 1.4),
        Some("soma_failed") => ("#ffebee", "#c62828", "#b71c1c", 1.4),
        Some("soma_running") => ("#fff8e1", "#f9a825", "#f57f17", 1.4),
        Some(_) => ("#fff3e0", "#ef6c00", "#e65100", 2.4), // flagged
        None => ("#fcfcfb", "#c3c2b7", "#0b0b0b", 1.4),
    }
}

fn esc(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

struct NodeBox {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    label: String,
    sublabel: Option<String>,
    class: Option<&'static str>,
    kind_tag: &'static str,
}

impl Graph {
    /// Render as a self-contained SVG diagram.
    pub fn to_svg(&self) -> String {
        self.to_svg_with(&GraphOverlay::default())
    }

    /// Render as a self-contained SVG diagram with per-node execution
    /// annotations (status colors + a duration/cache/flags sublabel),
    /// same overlay semantics as [`Graph::to_mermaid_with`].
    pub fn to_svg_with(&self, overlay: &GraphOverlay) -> String {
        use std::fmt::Write;

        let order: Vec<String> = self
            .topological_sort()
            .unwrap_or_else(|_| self.nodes.iter().map(|n| n.id.as_str()).collect())
            .into_iter()
            .map(str::to_string)
            .collect();

        // Longest-path layering, left→right.
        let mut layer: HashMap<&str, usize> = HashMap::new();
        for id in &order {
            let l = self
                .predecessors(id)
                .iter()
                .filter_map(|p| layer.get(*p))
                .max()
                .map(|l| l + 1)
                .unwrap_or(0);
            layer.insert(id.as_str(), l);
        }
        let n_layers = layer.values().max().map(|l| l + 1).unwrap_or(0);
        let mut layers: Vec<Vec<&str>> = vec![Vec::new(); n_layers];
        for id in &order {
            layers[layer[id.as_str()]].push(id);
        }

        // Boxes: size from label/sublabel, stacked per layer, layers
        // centered vertically.
        let mut boxes: HashMap<String, NodeBox> = HashMap::new();
        let mut layer_widths = Vec::with_capacity(n_layers);
        let mut layer_heights = Vec::with_capacity(n_layers);
        for ids in &layers {
            let mut width: f32 = 0.0;
            let mut height: f32 = 0.0;
            for (i, id) in ids.iter().enumerate() {
                let node = self.node(id).expect("node in topo order");
                let ov = overlay.nodes.get(*id);
                let sublabel = ov.and_then(|o| o.sublabel_text());
                let label = match &node.kind {
                    NodeKind::Loop {
                        max_iterations: Some(n),
                        ..
                    } => {
                        format!("{} (max {n})", node.label)
                    }
                    _ => node.label.clone(),
                };
                let kind_tag = match &node.kind {
                    NodeKind::Filter { .. } => "filter",
                    NodeKind::SubGraph { .. } => "subgraph",
                    NodeKind::Loop { .. } => "loop",
                    NodeKind::Branch { .. } => "branch",
                    NodeKind::Step { .. } => "step",
                };
                let w = (label.chars().count() as f32 * CHAR_W)
                    .max(
                        sublabel
                            .as_deref()
                            .map_or(0.0, |s| s.chars().count() as f32 * SUB_CHAR_W),
                    )
                    .max(44.0)
                    + 2.0 * PAD_X;
                let h = NODE_H + if sublabel.is_some() { SUB_EXTRA_H } else { 0.0 };
                if i > 0 {
                    height += Y_GAP;
                }
                boxes.insert(
                    (*id).to_string(),
                    NodeBox {
                        x: 0.0,
                        y: height,
                        w,
                        h,
                        label,
                        sublabel,
                        class: ov.and_then(|o| o.style_class()),
                        kind_tag,
                    },
                );
                height += h;
                width = width.max(w);
            }
            layer_widths.push(width);
            layer_heights.push(height);
        }
        let max_height = layer_heights.iter().cloned().fold(0.0, f32::max);
        let mut x = MARGIN;
        for (l, ids) in layers.iter().enumerate() {
            let y0 = MARGIN + (max_height - layer_heights[l]) / 2.0;
            for id in ids {
                let b = boxes.get_mut(*id).expect("box exists");
                b.x = x;
                b.y += y0;
            }
            x += layer_widths[l] + X_GAP;
        }
        let canvas_w = x - X_GAP + MARGIN;
        let canvas_h = max_height + 2.0 * MARGIN;

        let mut out = String::new();
        let _ = write!(
            out,
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0}" height="{h:.0}" viewBox="0 0 {w:.0} {h:.0}" font-family="system-ui, -apple-system, 'Segoe UI', sans-serif">"#,
            w = canvas_w,
            h = canvas_h,
        );
        out.push_str(
            r##"<defs><marker id="soma-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 1 L 9 5 L 0 9 z" fill="#898781"/></marker></defs>"##,
        );

        // Edges first (under the nodes).
        for edge in &self.edges {
            let (Some(src), Some(dst)) = (boxes.get(&edge.source), boxes.get(&edge.target)) else {
                continue;
            };
            let (x1, y1) = (src.x + src.w, src.y + src.h / 2.0);
            let (x2, y2) = (dst.x, dst.y + dst.h / 2.0);
            let dx = ((x2 - x1) / 2.0).max(18.0);
            let dash = match edge.kind {
                EdgeKind::Data => "",
                EdgeKind::Control => r#" stroke-dasharray="5 4""#,
            };
            let _ = write!(
                out,
                r##"<path d="M {x1:.1} {y1:.1} C {c1:.1} {y1:.1}, {c2:.1} {y2:.1}, {x2:.1} {y2:.1}" fill="none" stroke="#898781" stroke-width="1.5"{dash} marker-end="url(#soma-arrow)"/>"##,
                c1 = x1 + dx,
                c2 = x2 - dx,
            );
            if let Some(label) = &edge.label {
                let _ = write!(
                    out,
                    r##"<text x="{x:.1}" y="{y:.1}" font-size="10" fill="#898781" text-anchor="middle">{t}</text>"##,
                    x = (x1 + x2) / 2.0,
                    y = (y1 + y2) / 2.0 - 5.0,
                    t = esc(label),
                );
            }
        }

        // Nodes.
        for id in &order {
            let b = &boxes[id.as_str()];
            let (fill, stroke, ink, sw) = class_colors(b.class);
            let rx = match b.kind_tag {
                "loop" => b.h / 2.0,
                _ => 6.0,
            };
            let _ = write!(
                out,
                r#"<rect x="{x:.1}" y="{y:.1}" width="{w:.1}" height="{h:.1}" rx="{rx:.1}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>"#,
                x = b.x,
                y = b.y,
                w = b.w,
                h = b.h,
            );
            match b.kind_tag {
                // SubGraph: double border, like mermaid's [[...]].
                "subgraph" => {
                    let _ = write!(
                        out,
                        r#"<rect x="{x:.1}" y="{y:.1}" width="{w:.1}" height="{h:.1}" rx="4" fill="none" stroke="{stroke}" stroke-width="1"/>"#,
                        x = b.x + 3.0,
                        y = b.y + 3.0,
                        w = b.w - 6.0,
                        h = b.h - 6.0,
                    );
                }
                // Branch: decision notches on the vertical edges.
                "branch" => {
                    let _ = write!(
                        out,
                        r#"<path d="M {x1:.1} {ym:.1} l 6 -6 M {x1:.1} {ym:.1} l 6 6 M {x2:.1} {ym:.1} l -6 -6 M {x2:.1} {ym:.1} l -6 6" stroke="{stroke}" stroke-width="1.2" fill="none"/>"#,
                        x1 = b.x,
                        x2 = b.x + b.w,
                        ym = b.y + b.h / 2.0,
                    );
                }
                _ => {}
            }
            let label_y = if b.sublabel.is_some() {
                b.y + 20.0
            } else {
                b.y + b.h / 2.0 + 4.5
            };
            let _ = write!(
                out,
                r#"<text x="{x:.1}" y="{y:.1}" font-size="13" fill="{ink}" text-anchor="middle">{t}</text>"#,
                x = b.x + b.w / 2.0,
                y = label_y,
                t = esc(&b.label),
            );
            if let Some(sub) = &b.sublabel {
                let _ = write!(
                    out,
                    r##"<text x="{x:.1}" y="{y:.1}" font-size="11" fill="#52514e" text-anchor="middle">{t}</text>"##,
                    x = b.x + b.w / 2.0,
                    y = b.y + 36.0,
                    t = esc(sub),
                );
            }
        }
        out.push_str("</svg>");
        out
    }
}

#[cfg(test)]
mod tests {
    use crate::graph::{Edge, Graph, Node};
    use crate::viz::{GraphOverlay, NodeOverlay, NodeStatus};

    fn sample() -> Graph {
        let mut g = Graph::new();
        g.add_node(Node::new("a", "Scaler", "Scaler"));
        g.add_node(Node::new("b", "PCA", "PCA"));
        g.add_node(Node::new("c", "SVM", "SVM"));
        g.add_edge(Edge::data("e0", "a", "b"));
        g.add_edge(Edge::data("e1", "b", "c"));
        g
    }

    #[test]
    fn svg_contains_nodes_edges_and_valid_envelope() {
        let svg = sample().to_svg();
        assert!(svg.starts_with("<svg xmlns=\"http://www.w3.org/2000/svg\""));
        assert!(svg.ends_with("</svg>"));
        for label in ["Scaler", "PCA", "SVM"] {
            assert!(svg.contains(&format!(">{label}</text>")), "{svg}");
        }
        assert_eq!(
            svg.matches("marker-end=\"url(#soma-arrow)\"").count(),
            2,
            "two edges"
        );
        // No unescaped angle brackets from labels.
        assert!(!svg.contains("<<"));
    }

    #[test]
    fn svg_overlay_colors_and_sublabels() {
        let g = sample();
        let mut ov = GraphOverlay::default();
        ov.nodes.insert(
            "a".into(),
            NodeOverlay {
                status: Some(NodeStatus::Completed),
                duration_ms: Some(1200),
                ..Default::default()
            },
        );
        ov.nodes.insert(
            "b".into(),
            NodeOverlay {
                flags: vec!["LEAKAGE".into()],
                ..Default::default()
            },
        );
        let svg = g.to_svg_with(&ov);
        assert!(svg.contains("fill=\"#e8f5e9\""), "completed fill");
        assert!(svg.contains(">1.2s</text>"), "duration sublabel");
        assert!(svg.contains("fill=\"#fff3e0\""), "flagged fill");
        assert!(svg.contains("⚠ LEAKAGE"));
        // Unannotated node keeps the neutral surface.
        assert!(svg.contains("fill=\"#fcfcfb\""));
    }

    #[test]
    fn svg_layers_forks_side_by_side() {
        let mut g = Graph::new();
        g.add_node(Node::new("src", "src", "src"));
        g.add_node(Node::new("l", "left", "left"));
        g.add_node(Node::new("r", "right", "right"));
        g.add_node(Node::new("sink", "sink", "sink"));
        g.add_edge(Edge::data("e0", "src", "l"));
        g.add_edge(Edge::data("e1", "src", "r"));
        g.add_edge(Edge::data("e2", "l", "sink"));
        g.add_edge(Edge::data("e3", "r", "sink"));
        let svg = g.to_svg();
        assert_eq!(svg.matches("<rect").count(), 4);
        assert_eq!(svg.matches("marker-end").count(), 4);

        // Escaping: a hostile label cannot break out of the SVG.
        let mut g2 = Graph::new();
        g2.add_node(Node::new("x", "<script>\"&\"</script>", "x"));
        let svg2 = g2.to_svg();
        assert!(!svg2.contains("<script>"));
        assert!(svg2.contains("&lt;script&gt;"));
    }
}
