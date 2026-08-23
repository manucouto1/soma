//! What was measured about one node over one window.

/// The numbers a verdict is taken over.
///
/// **`None` means nobody measured it**, which is not zero and not healthy. A
/// node with no parameters has no gradient norm; a window with no snapshot in
/// it has no effective rank; per-channel statistics are opt-in because they
/// cost a reduction per step. A metric that was not taken cannot raise a flag,
/// and it must not silently pass for one that was taken and came out fine.
///
/// It is `Option<f64>` and not a `NaN` sentinel, which is what the original
/// used. `NaN < x` being false does make an unobserved metric quietly not
/// flag — elegant, and it works right up until somebody sums a column.
///
/// # A window, not a step
///
/// Every field here is already reduced over however many steps the caller
/// decided to look at. Which steps those were, and how many, is the caller's:
/// a verdict that quietly needed a particular cadence would be a threshold
/// hiding in a schedule.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Seen {
    /// Anything in the window was not a number.
    pub nan: bool,
    /// Anything in the window was not finite.
    pub inf: bool,
    /// The mean L2 norm of this node's parameter gradients.
    pub grad_norm: Option<f64>,
    /// The **largest** fraction of the output that was off, over the window.
    pub zero_frac_max: Option<f64>,
    /// The **largest** fraction that was pinned, over the window.
    pub sat_frac_max: Option<f64>,
    /// The mean ratio of the size of a step to the size of the weights it
    /// moved.
    pub update_ratio: Option<f64>,
    /// How many channels were off across the whole window.
    pub dead_channels: usize,
    /// How many were alive and never asked for.
    pub ignored_channels: usize,
    /// What fraction of the channels are dormant on Sokar's normalised score.
    pub dormancy_frac: Option<f64>,
    /// The largest linear CKA between two groups the caller declared separate.
    pub group_cka: Option<f64>,
    /// The effective rank of the representation at the end of the window.
    pub eff_rank: Option<f64>,
    /// How the effective rank is moving, per step and relative to itself.
    pub eff_rank_slope: Option<f64>,
    /// How the norm of the parameters is moving, likewise.
    pub param_norm_slope: Option<f64>,
    /// The stable rank of the update over the window: how many directions this
    /// node actually moved in.
    pub update_rank: Option<f64>,
    /// And what that usually was for this run, which is the only reference a
    /// single training run has.
    pub update_rank_usual: Option<f64>,
}
