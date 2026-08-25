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
    /// The scale of the signal here against where the last normalisation
    /// upstream left it.
    ///
    /// Measured before a step is taken, by a probe rather than by an audit: a
    /// probe knows what feeds what, because it traced it. Where the last
    /// normalisation **is** is structure and not a bound, which is why it is
    /// baked into the number and the threshold stays on this side of the wall.
    pub signal_gain: Option<f64>,
    /// The factor a gradient at the output arrives here by:
    /// `sqrt(E||J^T v||^2)` over random probes, from this layer to the output.
    ///
    /// Deliberately not a parameter-gradient norm. At initialisation there is
    /// no loss, so a gradient norm would be taken against a target somebody
    /// made up and would land in [`Seen::grad_norm`] at a different scale, to
    /// be judged by the same bound. This one is a ratio and needs no target.
    ///
    /// **It raises nothing, and that was measured.** Walked across criticality,
    /// the worst network that still trained read 1.41 and the best one that did
    /// not read 1.95: a factor of 1.4 is where the sampling landed and not a
    /// bound. See `health/tests/isometry.py`. The number is recorded and drawn,
    /// because its profile over depth is the vanishing picture and a person can
    /// read one.
    pub jacobian_gain: Option<f64>,
    /// How spread that Jacobian's spectrum is: `s_max / s_rms` of a random
    /// sketch of it.
    ///
    /// Dynamical isometry (Pennington et al., 2017) is a claim about the
    /// spectrum's **shape** and not its size — a flat one trains dramatically
    /// faster than one with the same mean and a long tail — and a mean cannot
    /// see the difference.
    ///
    /// **It raises nothing either, and the inversion is the reason**: a network
    /// reading 1.87 trained and one reading 1.76 did not, so the failing one
    /// had the tighter spectrum.
    ///
    /// There is a rule underneath both of these and it is worth more than
    /// either. [`Flag::MissingNormalisation`](crate::Flag::MissingNormalisation)
    /// separates because the forward scale is a **runaway** — a geometric
    /// process either stays put or leaves by decades, and there is nothing in
    /// between to be wrong about. These two vary **continuously** with how well
    /// a network turns out, and something continuous is a ranking. A ranking
    /// belongs at level 3, beside the proxies, where a number only ever means
    /// something next to another candidate's.
    pub jacobian_spread: Option<f64>,
}
