//! Ranked retrieval over the experiment pool.
//!
//! The pool exists to answer "what have I already tried that bears on
//! this?", which is four questions at once: does the text match, does
//! the architecture look like mine, is it recent enough to still be
//! about the same code, and is it worth reading at all. The score adds
//! them:
//!
//! ```text
//! 0.40 · lexical + 0.25 · structural + 0.15 · recency + 0.20 · importance
//! ```
//!
//! **Additive, not multiplicative.** A product lets any single term veto
//! a record: an experiment from last year scores ~0 on recency and
//! therefore ~0 overall, when a year-old dead end is exactly the kind of
//! thing the pool exists to surface. Terms that do not apply (no query
//! architecture, no timestamps to compare) have their weight
//! redistributed over the rest, so scores stay comparable across
//! queries instead of silently shrinking.
//!
//! Lexical relevance is BM25 (`k1 = 1.2`, `b = 0.75`) over a document
//! built by repeating each field: a term in the experiment's *name*
//! counts three times, in its conclusion headline twice, in its notes
//! once. No stemming — experiment vocabulary is mostly identifiers and
//! acronyms, which stemmers mangle.
//!
//! **Failures rank.** [`importance`] puts a floor under any record that
//! failed, crashed or regressed *and* carries a conclusion. Cutting
//! experimental cost means not repeating dead ends, not only repeating
//! wins.
//!
//! Everything here is deterministic: `now` is a parameter, ordering
//! ties break on id, and no clock or RNG is read.

use crate::record::ExperimentRecord;
use chrono::{DateTime, Utc};
use somatize_core::fingerprint::{ArchitectureFingerprint, structural_similarity};
use somatize_core::summary::RunOutcome;
use std::collections::HashMap;

/// BM25 term-frequency saturation.
const K1: f64 = 1.2;
/// BM25 length normalization.
const B: f64 = 0.75;

/// Composite weights. See the module docs for why they are added.
const W_LEXICAL: f64 = 0.40;
const W_STRUCTURAL: f64 = 0.25;
const W_RECENCY: f64 = 0.15;
const W_IMPORTANCE: f64 = 0.20;

/// Age at which recency has halved.
pub const DEFAULT_HALF_LIFE_DAYS: f64 = 30.0;

/// Turns text into a vector. Not implemented by soma: the seam exists
/// so an embedding model can be plugged in from outside (a
/// sentence-transformer behind the Python worker, an HTTP endpoint)
/// without soma taking a dependency on one.
///
/// `id` identifies the model. It is stored next to every vector so
/// vectors from different models are never silently compared — a
/// cosine between two different embedding spaces is a number, not a
/// similarity.
pub trait Embedder: Send + Sync {
    /// Stable identifier of the model, e.g. `"minilm-l6-v2"`.
    fn id(&self) -> &str;

    /// Embed one text.
    fn embed(&self, text: &str) -> somatize_core::error::Result<Vec<f32>>;
}

/// The text an embedder should be given for a record: the same fields
/// BM25 indexes, without the repetition weighting.
pub fn embedding_text(record: &ExperimentRecord) -> String {
    let mut parts: Vec<&str> = vec![&record.name, &record.pipeline_summary];
    if let Some(h) = &record.hypothesis {
        parts.push(h);
    }
    if let Some(c) = &record.conclusion {
        parts.push(&c.headline);
    }
    if let Some(d) = &record.derivation {
        parts.push(&d.summary);
    }
    if let Some(n) = &record.notes {
        parts.push(n);
    }
    parts.extend(record.tags.iter().map(String::as_str));
    parts
        .into_iter()
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join(" — ")
}

/// What to retrieve, and what to measure it against.
#[derive(Debug, Clone)]
pub struct RetrievalQuery {
    /// Free text. Empty is allowed: the other three terms still rank.
    pub text: String,
    /// Architecture to compare against, when the caller has one.
    pub architecture: Option<ArchitectureFingerprint>,
    /// Reference time for recency. A parameter, not `Utc::now()`, so
    /// ranking is reproducible.
    pub now: DateTime<Utc>,
    /// Recency half-life in days; [`DEFAULT_HALF_LIFE_DAYS`] unless set.
    pub half_life_days: f64,
    /// Query vector, from the same embedder that produced the records'.
    /// When present, it blends into the lexical term for records
    /// carrying a vector from the *same* model.
    pub embedding: Option<(String, Vec<f32>)>,
    /// Maximum number of results returned.
    pub limit: usize,
    /// Restrict to one research line.
    pub research_line: Option<String>,
    /// Every listed tag must be present.
    pub tags: Vec<String>,
}

impl RetrievalQuery {
    /// A text query against `now`, with the default half-life.
    pub fn new(text: impl Into<String>, now: DateTime<Utc>) -> Self {
        Self {
            text: text.into(),
            architecture: None,
            now,
            half_life_days: DEFAULT_HALF_LIFE_DAYS,
            embedding: None,
            limit: 10,
            research_line: None,
            tags: Vec::new(),
        }
    }

    /// Compare candidates structurally against this fingerprint,
    /// activating the structural term.
    pub fn with_architecture(mut self, architecture: ArchitectureFingerprint) -> Self {
        self.architecture = Some(architecture);
        self
    }

    /// Cap the number of results.
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Restrict candidates to one research line, before scoring.
    pub fn in_line(mut self, line: impl Into<String>) -> Self {
        self.research_line = Some(line.into());
        self
    }

    /// Require every listed tag on a candidate, before scoring.
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }
}

/// The four terms behind a score, so a result can explain itself.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ScoreComponents {
    /// BM25 relevance normalized against the best hit, optionally
    /// blended with cosine similarity — in `[0, 1]`.
    pub lexical: f64,
    /// Architecture similarity to the query's fingerprint, in `[0, 1]`.
    pub structural: f64,
    /// Exponential age decay (see [`recency`]), in `(0, 1]`.
    pub recency: f64,
    /// How much the record is worth reading (see [`importance`]).
    pub importance: f64,
}

/// A record and why it came back.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ScoredRecord {
    /// The matching record itself.
    pub record: ExperimentRecord,
    /// Composite score in `[0, 1]`.
    pub score: f64,
    /// The four terms the score was added from.
    pub components: ScoreComponents,
}

impl ScoredRecord {
    /// One-line, deterministic explanation of the ranking.
    pub fn why(&self) -> String {
        let c = &self.components;
        format!(
            "score {:.2} (text {:.2}, structure {:.2}, recency {:.2}, importance {:.2})",
            self.score, c.lexical, c.structural, c.recency, c.importance
        )
    }
}

/// Rank `records` against `query`, best first.
///
/// Ties break on record id, so the order is total and reproducible.
pub fn rank(records: &[ExperimentRecord], query: &RetrievalQuery) -> Vec<ScoredRecord> {
    let candidates: Vec<&ExperimentRecord> = records
        .iter()
        .filter(|r| {
            query
                .research_line
                .as_ref()
                .is_none_or(|line| r.research_line.as_deref() == Some(line.as_str()))
        })
        .filter(|r| query.tags.iter().all(|t| r.tags.contains(t)))
        .collect();
    if candidates.is_empty() {
        return Vec::new();
    }

    let bm25 = Bm25Index::build(&candidates);
    let query_terms = tokenize(&query.text);
    let raw: Vec<f64> = candidates
        .iter()
        .enumerate()
        .map(|(i, _)| bm25.score(i, &query_terms))
        .collect();
    // BM25 is unbounded; normalizing against the best hit in this
    // candidate set makes the term comparable with the other three.
    let peak = raw.iter().cloned().fold(0.0_f64, f64::max);

    // Which terms this query can actually speak to.
    let has_text = !query_terms.is_empty() && peak > 0.0;
    let has_structure = query
        .architecture
        .as_ref()
        .is_some_and(|a| !a.nodes.is_empty());
    let applicable = [
        (has_text, W_LEXICAL),
        (has_structure, W_STRUCTURAL),
        (true, W_RECENCY),
        (true, W_IMPORTANCE),
    ];
    let total: f64 = applicable
        .iter()
        .filter(|(on, _)| *on)
        .map(|(_, w)| w)
        .sum();
    let weight = |on: bool, w: f64| if on && total > 0.0 { w / total } else { 0.0 };
    let (w_lex, w_struct) = (
        weight(has_text, W_LEXICAL),
        weight(has_structure, W_STRUCTURAL),
    );
    let (w_rec, w_imp) = (weight(true, W_RECENCY), weight(true, W_IMPORTANCE));

    let mut scored: Vec<ScoredRecord> = candidates
        .iter()
        .enumerate()
        .map(|(i, record)| {
            let lexical =
                blend_semantic(if peak > 0.0 { raw[i] / peak } else { 0.0 }, record, query);
            let structural = match (&query.architecture, &record.architecture) {
                (Some(a), Some(b)) => structural_similarity(a, b),
                _ => 0.0,
            };
            let recency = recency(record.timestamp, query.now, query.half_life_days);
            let importance = importance(record);
            let components = ScoreComponents {
                lexical,
                structural,
                recency,
                importance,
            };
            ScoredRecord {
                record: (*record).clone(),
                score: w_lex * lexical
                    + w_struct * structural
                    + w_rec * recency
                    + w_imp * importance,
                components,
            }
        })
        .collect();

    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.record.id.cmp(&b.record.id))
    });
    scored.truncate(query.limit);
    scored
}

/// Fold cosine similarity into the lexical term when — and only when —
/// both sides carry a vector from the same model.
fn blend_semantic(lexical: f64, record: &ExperimentRecord, query: &RetrievalQuery) -> f64 {
    let (Some((query_id, query_vec)), Some(embedding)) = (&query.embedding, &record.embedding)
    else {
        return lexical;
    };
    if &embedding.embedder_id != query_id {
        return lexical;
    }
    let cosine = cosine(query_vec, &embedding.vector);
    0.5 * lexical + 0.5 * cosine
}

/// Cosine similarity clamped to `[0, 1]` — a negative cosine is "not
/// similar", not "less than nothing".
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na.sqrt() * nb.sqrt())).clamp(0.0, 1.0)
}

/// Exponential decay with a half-life, in `(0, 1]`. A record from the
/// future (clock skew) scores 1.0 rather than exploding.
pub fn recency(timestamp: DateTime<Utc>, now: DateTime<Utc>, half_life_days: f64) -> f64 {
    if half_life_days <= 0.0 {
        return 1.0;
    }
    let age_days = (now - timestamp).num_seconds() as f64 / 86_400.0;
    if age_days <= 0.0 {
        return 1.0;
    }
    (-std::f64::consts::LN_2 * age_days / half_life_days).exp()
}

/// How much this record is worth reading, in `[0, 1]`.
///
/// Rewards recorded human intent (a hypothesis, notes), a machine
/// conclusion, and a deliberate derivation. Puts a **floor of 0.6**
/// under dead ends that carry a conclusion: a failure someone bothered
/// to explain is one of the most valuable things in the pool.
pub fn importance(record: &ExperimentRecord) -> f64 {
    let mut score: f64 = 0.3;
    if record.conclusion.as_ref().is_some_and(|c| !c.is_empty()) {
        score += 0.2;
    }
    if record.hypothesis.is_some() || record.notes.is_some() {
        score += 0.2;
    }
    if record.derivation.is_some() {
        score += 0.1;
    }
    if improved(record) {
        score += 0.2;
    }
    if is_dead_end(record) && record.has_conclusion() {
        score = score.max(0.6);
    }
    score.clamp(0.0, 1.0)
}

/// Whether the move that produced this record improved any metric.
///
/// Direction is not knowable in general (lower loss is better, higher
/// f1 is better), so "improved" means: the objective metric went up
/// when one is declared, else any metric went up.
fn improved(record: &ExperimentRecord) -> bool {
    let Some(derivation) = &record.derivation else {
        return false;
    };
    match record.objective.as_deref() {
        Some(objective) => derivation
            .metric_delta
            .get(objective)
            .is_some_and(|d| d.delta > 0.0),
        None => derivation.metric_delta.values().any(|d| d.delta > 0.0),
    }
}

/// A run that failed, crashed, or moved every metric the wrong way.
pub fn is_dead_end(record: &ExperimentRecord) -> bool {
    let failed = record.conclusion.as_ref().is_some_and(|c| {
        matches!(
            c.outcome,
            Some(RunOutcome::Failed) | Some(RunOutcome::Crashed)
        )
    });
    let regressed = record.derivation.as_ref().is_some_and(|d| {
        !d.metric_delta.is_empty() && d.metric_delta.values().all(|m| m.delta < 0.0)
    });
    failed || regressed
}

// ── BM25 ────────────────────────────────────────────────────────────

/// Field weights, expressed as how many times a field's tokens are
/// repeated in the document. A name match should beat a note match.
const FIELD_WEIGHTS: &[(Field, usize)] = &[
    (Field::Name, 3),
    (Field::Hypothesis, 3),
    (Field::Headline, 2),
    (Field::Tags, 2),
    (Field::Pipeline, 2),
    (Field::Derivation, 2),
    (Field::Notes, 1),
    (Field::ResearchLine, 1),
    (Field::Keys, 1),
    (Field::Architecture, 1),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Field {
    Name,
    Hypothesis,
    Headline,
    Tags,
    Pipeline,
    Derivation,
    Notes,
    ResearchLine,
    Keys,
    Architecture,
}

/// The tokens one field contributes, before repetition.
fn field_tokens(record: &ExperimentRecord, field: Field) -> Vec<String> {
    match field {
        Field::Name => tokenize(&record.name),
        Field::Hypothesis => tokenize(record.hypothesis.as_deref().unwrap_or("")),
        Field::Headline => tokenize(record.conclusion.as_ref().map_or("", |c| &c.headline)),
        Field::Tags => record.tags.iter().flat_map(|t| tokenize(t)).collect(),
        Field::Pipeline => tokenize(&record.pipeline_summary),
        Field::Derivation => tokenize(record.derivation.as_ref().map_or("", |d| &d.summary)),
        Field::Notes => tokenize(record.notes.as_deref().unwrap_or("")),
        Field::ResearchLine => tokenize(record.research_line.as_deref().unwrap_or("")),
        Field::Keys => record
            .params
            .keys()
            .chain(record.metrics.keys())
            .flat_map(|k| tokenize(k))
            .collect(),
        Field::Architecture => record
            .architecture
            .iter()
            .flat_map(|a| a.node_tokens())
            .flat_map(|t| tokenize(&t))
            .collect(),
    }
}

/// Bag of terms for one record, with fields weighted by repetition.
fn document_terms(record: &ExperimentRecord) -> HashMap<String, usize> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for (field, weight) in FIELD_WEIGHTS {
        for token in field_tokens(record, *field) {
            *counts.entry(token).or_default() += weight;
        }
    }
    counts
}

/// Lowercase alphanumeric terms, splitting `snake_case`, `kebab-case`,
/// dotted paths and `camelCase`. No stemming: experiment vocabulary is
/// identifiers and acronyms, which stemmers only damage.
///
/// Two camel boundaries, both needed by real filter names: `aB`
/// (`valF1` → `val f1`) and `ABc`, the end of an acronym (`MoSHead` →
/// `mo s head`).
pub fn tokenize(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut tokens = Vec::new();
    let mut current = String::new();

    for (i, &ch) in chars.iter().enumerate() {
        if !ch.is_alphanumeric() {
            if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
            continue;
        }
        let prev = i.checked_sub(1).map(|j| chars[j]);
        let next = chars.get(i + 1).copied();
        let after_lower_or_digit = prev.is_some_and(|p| p.is_lowercase() || p.is_numeric());
        let acronym_end = prev.is_some_and(char::is_uppercase)
            && ch.is_uppercase()
            && next.is_some_and(char::is_lowercase);
        if ((ch.is_uppercase() && after_lower_or_digit) || acronym_end) && !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
        current.extend(ch.to_lowercase());
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

/// A BM25 index over one candidate set. Built per query: the pool is a
/// journal of at most tens of thousands of lines, and rebuilding keeps
/// scoring stateless and obviously correct.
struct Bm25Index {
    docs: Vec<HashMap<String, usize>>,
    lengths: Vec<f64>,
    avg_length: f64,
    /// term → number of documents containing it.
    doc_freq: HashMap<String, usize>,
}

impl Bm25Index {
    fn build(records: &[&ExperimentRecord]) -> Self {
        let docs: Vec<HashMap<String, usize>> = records.iter().map(|r| document_terms(r)).collect();
        let lengths: Vec<f64> = docs
            .iter()
            .map(|d| d.values().sum::<usize>() as f64)
            .collect();
        let avg_length = if lengths.is_empty() {
            0.0
        } else {
            lengths.iter().sum::<f64>() / lengths.len() as f64
        };
        let mut doc_freq: HashMap<String, usize> = HashMap::new();
        for doc in &docs {
            for term in doc.keys() {
                *doc_freq.entry(term.clone()).or_default() += 1;
            }
        }
        Self {
            docs,
            lengths,
            avg_length,
            doc_freq,
        }
    }

    fn score(&self, doc: usize, query_terms: &[String]) -> f64 {
        if self.avg_length == 0.0 {
            return 0.0;
        }
        let n = self.docs.len() as f64;
        let length_norm = K1 * (1.0 - B + B * self.lengths[doc] / self.avg_length);
        query_terms
            .iter()
            .map(|term| {
                let tf = *self.docs[doc].get(term).unwrap_or(&0) as f64;
                if tf == 0.0 {
                    return 0.0;
                }
                let df = *self.doc_freq.get(term).unwrap_or(&0) as f64;
                // Standard BM25 idf, floored at zero: a term in every
                // document is uninformative, never evidence against.
                let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln().max(0.0);
                idf * (tf * (K1 + 1.0)) / (tf + length_norm)
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::derivation::{DerivationMove, MetricDelta};
    use chrono::Duration;
    use somatize_core::summary::RunConclusion;
    use std::collections::BTreeMap;

    fn now() -> DateTime<Utc> {
        DateTime::parse_from_rfc3339("2026-07-30T12:00:00Z")
            .unwrap()
            .with_timezone(&Utc)
    }

    fn record(id: &str, name: &str) -> ExperimentRecord {
        let mut r = ExperimentRecord::new(id, name);
        r.timestamp = now();
        r
    }

    fn with_conclusion(mut r: ExperimentRecord, outcome: RunOutcome) -> ExperimentRecord {
        r.conclusion = Some(RunConclusion {
            headline: format!("{} something", outcome.verb()),
            outcome: Some(outcome),
            ..RunConclusion::default()
        });
        r
    }

    fn derivation(deltas: &[(&str, f64)]) -> DerivationMove {
        DerivationMove {
            from: "p".into(),
            to: "c".into(),
            changes: Vec::new(),
            metric_delta: deltas
                .iter()
                .map(|(name, delta)| {
                    (
                        (*name).to_string(),
                        MetricDelta {
                            before: 0.5,
                            after: 0.5 + delta,
                            delta: *delta,
                        },
                    )
                })
                .collect::<BTreeMap<_, _>>(),
            summary: String::new(),
        }
    }

    // ── Tokenizer ───────────────────────────────────────────────────

    #[test]
    fn tokenizer_splits_the_shapes_experiment_names_actually_use() {
        assert_eq!(tokenize("val_f1"), vec!["val", "f1"]);
        assert_eq!(tokenize("StandardScaler"), vec!["standard", "scaler"]);
        assert_eq!(tokenize("rocket-znorm"), vec!["rocket", "znorm"]);
        assert_eq!(
            tokenize("filter:MoSHead"),
            vec!["filter", "mo", "s", "head"]
        );
        assert_eq!(tokenize("encoder/layers.0"), vec!["encoder", "layers", "0"]);
        assert_eq!(tokenize("  "), Vec::<String>::new());
        // No stemming: "training" must not collapse into "train".
        assert_eq!(tokenize("training trains"), vec!["training", "trains"]);
    }

    // ── Lexical ─────────────────────────────────────────────────────

    #[test]
    fn a_name_match_outranks_a_note_match() {
        let mut in_name = record("a", "dropout sweep");
        in_name.notes = Some("nothing to see".into());
        let mut in_notes = record("b", "unrelated run");
        in_notes.notes = Some("we tried dropout here".into());

        let hits = rank(&[in_name, in_notes], &RetrievalQuery::new("dropout", now()));
        assert_eq!(hits[0].record.id, "a");
        assert!(hits[0].components.lexical > hits[1].components.lexical);
    }

    #[test]
    fn a_record_matching_nothing_scores_zero_lexically() {
        let hits = rank(
            &[record("a", "dropout sweep"), record("b", "batch norm")],
            &RetrievalQuery::new("dropout", now()),
        );
        let miss = hits.iter().find(|h| h.record.id == "b").unwrap();
        assert_eq!(miss.components.lexical, 0.0);
        assert!(miss.score > 0.0, "recency and importance still count");
    }

    #[test]
    fn the_derivation_summary_is_searchable() {
        let mut moved = record("a", "run");
        let mut d = derivation(&[("f1", 0.1)]);
        d.summary = "swapped SVM for RandomForest".into();
        moved.derivation = Some(d);

        let hits = rank(
            &[moved, record("b", "run")],
            &RetrievalQuery::new("random forest", now()),
        );
        assert_eq!(hits[0].record.id, "a");
        assert!(hits[0].components.lexical > 0.0);
    }

    #[test]
    fn an_empty_query_ranks_on_the_other_terms_alone() {
        let old = {
            let mut r = record("old", "ancient");
            r.timestamp = now() - Duration::days(365);
            r
        };
        let hits = rank(
            &[record("new", "fresh"), old],
            &RetrievalQuery::new("", now()),
        );
        assert_eq!(hits[0].record.id, "new");
        assert!(hits.iter().all(|h| h.components.lexical == 0.0));
        // The lexical weight was redistributed, not left on the floor.
        assert!(hits[0].score > 0.5, "{}", hits[0].score);
    }

    // ── Structural ──────────────────────────────────────────────────

    #[test]
    fn architecture_similarity_breaks_a_text_tie() {
        use somatize_core::graph::{Node, linear_pipeline};

        let mine = ArchitectureFingerprint::of(&linear_pipeline(vec![
            Node::filter("Scaler"),
            Node::filter("SVM"),
        ]))
        .unwrap();
        let same = mine.clone();
        let other = ArchitectureFingerprint::of(&linear_pipeline(vec![
            Node::filter("Tokenizer"),
            Node::filter("Transformer"),
        ]))
        .unwrap();

        let mut a = record("same", "run");
        a.architecture = Some(same);
        let mut b = record("other", "run");
        b.architecture = Some(other);

        let hits = rank(
            &[b, a],
            &RetrievalQuery::new("run", now()).with_architecture(mine),
        );
        assert_eq!(hits[0].record.id, "same");
        assert_eq!(hits[0].components.structural, 1.0);
        assert_eq!(hits[1].components.structural, 0.0);
    }

    #[test]
    fn without_a_query_architecture_the_structural_weight_is_redistributed() {
        let query = RetrievalQuery::new("run", now());
        let hits = rank(&[record("a", "run")], &query);
        assert_eq!(hits[0].components.structural, 0.0);
        // 0.40 + 0.15 + 0.20 = 0.75 of weight, renormalized to 1.
        let c = hits[0].components;
        let expected = (0.40 * c.lexical + 0.15 * c.recency + 0.20 * c.importance) / 0.75;
        assert!((hits[0].score - expected).abs() < 1e-9);
    }

    // ── Recency ─────────────────────────────────────────────────────

    #[test]
    fn recency_halves_every_half_life() {
        assert!((recency(now(), now(), 30.0) - 1.0).abs() < 1e-9);
        assert!((recency(now() - Duration::days(30), now(), 30.0) - 0.5).abs() < 1e-6);
        assert!((recency(now() - Duration::days(60), now(), 30.0) - 0.25).abs() < 1e-6);
        // Never zero: an old record is faint, not invisible.
        assert!(recency(now() - Duration::days(3650), now(), 30.0) > 0.0);
        // Clock skew must not produce a score above 1.
        assert_eq!(recency(now() + Duration::days(5), now(), 30.0), 1.0);
    }

    #[test]
    fn an_old_record_still_beats_a_new_irrelevant_one() {
        // The whole reason the score is additive: a year-old dead end
        // that matches the query must not be buried by today's noise.
        let mut old = record("old", "dropout collapse investigation");
        old.timestamp = now() - Duration::days(365);
        old = with_conclusion(old, RunOutcome::Failed);
        old.notes = Some("dropout above 0.5 collapses the encoder".into());

        let fresh = record("fresh", "unrelated batch norm run");

        let hits = rank(
            &[old, fresh],
            &RetrievalQuery::new("dropout collapse", now()),
        );
        assert_eq!(hits[0].record.id, "old");
    }

    // ── Importance ──────────────────────────────────────────────────

    #[test]
    fn a_failure_with_a_conclusion_has_a_floor() {
        let bare = record("bare", "nothing recorded");
        assert!((importance(&bare) - 0.3).abs() < 1e-9);

        let failed = with_conclusion(record("failed", "boom"), RunOutcome::Failed);
        assert!(
            importance(&failed) >= 0.6,
            "a dead end with a conclusion must stay retrievable: {}",
            importance(&failed)
        );

        let crashed = with_conclusion(record("crashed", "oom"), RunOutcome::Crashed);
        assert!(importance(&crashed) >= 0.6);
    }

    #[test]
    fn a_regression_counts_as_a_dead_end() {
        let mut regressed = with_conclusion(record("r", "worse"), RunOutcome::Completed);
        regressed.derivation = Some(derivation(&[("f1", -0.1), ("auc", -0.05)]));
        assert!(is_dead_end(&regressed));
        assert!(importance(&regressed) >= 0.6);

        let mut mixed = with_conclusion(record("m", "mixed"), RunOutcome::Completed);
        mixed.derivation = Some(derivation(&[("f1", -0.1), ("auc", 0.05)]));
        assert!(!is_dead_end(&mixed), "one metric up is not a dead end");
    }

    #[test]
    fn improvement_is_judged_against_the_declared_objective() {
        let mut up = with_conclusion(record("up", "better"), RunOutcome::Completed);
        up.objective = Some("f1".into());
        up.derivation = Some(derivation(&[("f1", 0.1), ("loss", 0.2)]));
        assert!(improved(&up));

        // The objective went down; another metric going up is not a win.
        let mut down = with_conclusion(record("down", "worse"), RunOutcome::Completed);
        down.objective = Some("f1".into());
        down.derivation = Some(derivation(&[("f1", -0.1), ("loss", 0.2)]));
        assert!(!improved(&down));
        assert!(importance(&up) > importance(&down));
    }

    #[test]
    fn importance_stays_in_range() {
        let mut everything = with_conclusion(record("e", "all"), RunOutcome::Completed);
        everything.hypothesis = Some("h".into());
        everything.notes = Some("n".into());
        everything.derivation = Some(derivation(&[("f1", 0.1)]));
        let score = importance(&everything);
        assert!((0.0..=1.0).contains(&score), "{score}");
        assert!(score > importance(&record("bare", "bare")));
    }

    // ── Ranking as a whole ──────────────────────────────────────────

    #[test]
    fn scores_are_bounded_and_ordering_is_total() {
        let records: Vec<ExperimentRecord> = (0..20)
            .map(|i| {
                let mut r = record(&format!("e{i:02}"), "dropout sweep");
                r.timestamp = now() - Duration::days(i);
                r
            })
            .collect();
        let hits = rank(
            &records,
            &RetrievalQuery::new("dropout", now()).with_limit(20),
        );
        assert_eq!(hits.len(), 20);
        for hit in &hits {
            assert!((0.0..=1.0).contains(&hit.score), "{}", hit.score);
        }
        for pair in hits.windows(2) {
            assert!(pair[0].score >= pair[1].score);
        }
    }

    #[test]
    fn ranking_is_insertion_order_independent_and_repeatable() {
        let mut records = vec![
            record("a", "dropout sweep"),
            with_conclusion(record("b", "dropout collapse"), RunOutcome::Failed),
            record("c", "batch norm"),
        ];
        let query = RetrievalQuery::new("dropout", now());
        let forward: Vec<String> = rank(&records, &query)
            .into_iter()
            .map(|h| h.record.id)
            .collect();
        records.reverse();
        let reversed: Vec<String> = rank(&records, &query)
            .into_iter()
            .map(|h| h.record.id)
            .collect();
        assert_eq!(forward, reversed);
        assert_eq!(
            forward,
            rank(&records, &query)
                .into_iter()
                .map(|h| h.record.id)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn identical_records_tie_break_on_id() {
        let a = record("zzz", "same");
        let b = record("aaa", "same");
        let hits = rank(&[a, b], &RetrievalQuery::new("same", now()));
        assert_eq!(hits[0].record.id, "aaa");
    }

    #[test]
    fn filters_apply_before_scoring() {
        let mut in_line = record("a", "run");
        in_line.research_line = Some("mos".into());
        in_line.tags = vec!["gpu".into(), "v2".into()];
        let mut other_line = record("b", "run");
        other_line.research_line = Some("other".into());

        let query = RetrievalQuery::new("run", now()).in_line("mos");
        let hits = rank(&[in_line.clone(), other_line.clone()], &query);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].record.id, "a");

        let query = RetrievalQuery::new("run", now()).with_tags(vec!["gpu".into()]);
        assert_eq!(rank(&[in_line.clone(), other_line], &query).len(), 1);

        // Every listed tag must be present, not just one.
        let query =
            RetrievalQuery::new("run", now()).with_tags(vec!["gpu".into(), "missing".into()]);
        assert!(rank(&[in_line], &query).is_empty());
    }

    #[test]
    fn limit_is_honored_and_an_empty_pool_is_empty() {
        let records: Vec<ExperimentRecord> =
            (0..10).map(|i| record(&format!("e{i}"), "run")).collect();
        assert_eq!(
            rank(&records, &RetrievalQuery::new("run", now()).with_limit(3)).len(),
            3
        );
        assert!(rank(&[], &RetrievalQuery::new("run", now())).is_empty());
    }

    #[test]
    fn why_explains_the_score() {
        let hits = rank(
            &[record("a", "dropout")],
            &RetrievalQuery::new("dropout", now()),
        );
        let why = hits[0].why();
        assert!(why.starts_with("score "), "{why}");
        assert!(why.contains("importance"), "{why}");
    }

    // ── Embeddings ──────────────────────────────────────────────────

    #[test]
    fn a_matching_embedder_blends_in_but_a_mismatched_one_is_ignored() {
        use crate::record::Embedding;

        let embedded = |id: &str, model: &str, vector: Vec<f32>| {
            // Identical text everywhere, so the only thing that can
            // move the lexical term is the vector.
            let mut r = record(id, "run");
            r.embedding = Some(Embedding {
                embedder_id: model.into(),
                vector,
            });
            r
        };
        let records = vec![
            embedded("aligned", "minilm", vec![1.0, 0.0]),
            embedded("orthogonal", "minilm", vec![0.0, 1.0]),
            // Numerically aligned, but from another embedding space:
            // comparing it would be meaningless, so it must not count.
            embedded("other-model", "a-different-model", vec![1.0, 0.0]),
        ];

        let mut query = RetrievalQuery::new("run", now());
        query.embedding = Some(("minilm".to_string(), vec![1.0, 0.0]));
        let hits = rank(&records, &query);
        let lexical = |id: &str| {
            hits.iter()
                .find(|h| h.record.id == id)
                .unwrap()
                .components
                .lexical
        };

        assert_eq!(lexical("aligned"), 1.0, "a matching vector agrees");
        assert_eq!(lexical("orthogonal"), 0.5, "a matching vector disagrees");
        assert_eq!(
            lexical("other-model"),
            1.0,
            "a vector from another model is left out of the blend entirely"
        );
        assert_eq!(hits.last().unwrap().record.id, "orthogonal");
    }

    #[test]
    fn cosine_handles_degenerate_vectors() {
        assert_eq!(cosine(&[], &[]), 0.0);
        assert_eq!(cosine(&[1.0], &[1.0, 2.0]), 0.0, "length mismatch");
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0, "zero vector");
        assert_eq!(cosine(&[1.0, 0.0], &[-1.0, 0.0]), 0.0, "opposite clamps");
        assert!((cosine(&[1.0, 1.0], &[2.0, 2.0]) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn embedding_text_gathers_the_prose() {
        let mut r = record("e", "mos baseline");
        r.pipeline_summary = "a → b".into();
        r.hypothesis = Some("wider helps".into());
        r.tags = vec!["mos".into()];
        let text = embedding_text(&r);
        for expected in ["mos baseline", "a → b", "wider helps", "mos"] {
            assert!(text.contains(expected), "{text}");
        }
    }
}
