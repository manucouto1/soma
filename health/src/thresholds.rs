//! Where every arguable number lives, together and in one place.

/// The bounds a [`verdict`](crate::verdict) is taken at.
///
/// **The whole of the opinion, and it is data.** Nothing else in this crate
/// holds a number, which is what lets a record be judged again tomorrow with
/// other bounds and lets two people disagree about a network by comparing two
/// of these rather than two codebases.
///
/// The defaults come from the original soma, which tuned them for
/// LayerNorm-ish activations and Adam-sized steps, plus the literature for the
/// three it did not have. They are a starting point and they are meant to be
/// argued with.
#[derive(Debug, Clone, PartialEq)]
pub struct Thresholds {
    /// Below this parameter-gradient norm, nothing is being learnt here.
    pub grad_low: f64,
    /// Above it, the next step will not be a step.
    pub grad_high: f64,
    /// A value this close to zero counts as off.
    pub dead_eps: f64,
    /// More of the output than this being off, on any one step, is dead.
    pub dead_frac: f64,
    /// A value this large counts as pinned.
    pub saturated_at: f64,
    /// More of the output than this being pinned, on any one step, is saturated.
    pub saturated_frac: f64,
    /// Below this update-to-weight ratio a node is not going to arrive.
    ///
    /// Practice puts a healthy one near `1e-3`; a decade either side of that is
    /// where the two bounds sit, because the useful signal is an order of
    /// magnitude and not a percentage.
    pub update_low: f64,
    /// And above it, each step throws away where it was.
    pub update_high: f64,
    /// A channel whose mean normalised activation is under this is dormant
    /// (Sokar et al., ICML 2023).
    pub dormant_tau: f64,
    /// This much of a layer dormant is part of what says it is losing
    /// plasticity.
    pub dormant_frac: f64,
    /// Linear CKA above this, between two groups meant to stay apart, is
    /// leakage (Kornblith et al., 2019).
    pub leakage_cka: f64,
    /// The update's stable rank falling below this fraction of its own recent
    /// median is narrowing. **`0.0` by default, which never fires.**
    ///
    /// It is off because it was measured and the measurement did not support
    /// it. Huang et al. (2026) monitor the spectrum of `W_t - W_{t-d}` and find
    /// it collapses thousands of steps before the loss — but their certificate
    /// is the **deviation from a healthy baseline run**, and a single training
    /// run has no baseline. Against its own recent median, on a 4-layer GELU
    /// net learning a fixed teacher, three healthy runs dipped to 0.69-0.71 of
    /// their own median and six destabilised ones ranged 0.43-0.86: the two
    /// overlap in both directions, so no bound separates them. The numbers are
    /// in `docs/use-cases.md`.
    ///
    /// What is kept is the **metric**, which is recorded and drawn: the
    /// collapse is visible to a person looking at the curve, and that is a
    /// weaker and honest claim. Set this yourself if you have a baseline to
    /// compare against, which is what the paper actually asks for.
    pub narrowing_of_usual: f64,
    /// How fast a thing has to be moving, per step and relative to itself, to
    /// count as growing or shrinking rather than wobbling.
    pub plasticity_growth: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            grad_low: 1e-7,
            grad_high: 1e3,
            dead_eps: 1e-7,
            dead_frac: 0.95,
            saturated_at: 50.0,
            saturated_frac: 0.5,
            update_low: 1e-4,
            update_high: 1e-2,
            dormant_tau: 0.1,
            dormant_frac: 0.5,
            leakage_cka: 0.95,
            narrowing_of_usual: 0.0,
            plasticity_growth: 1e-3,
        }
    }
}
