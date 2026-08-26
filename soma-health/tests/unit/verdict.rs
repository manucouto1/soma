//! What is said about a set of numbers, and — as much — what is not.

use somatize_health::{Flag, Seen, Thresholds};

fn said(seen: &Seen) -> Vec<String> {
    somatize_health::verdict(seen, &Thresholds::default())
        .iter()
        .map(ToString::to_string)
        .collect()
}

#[test]
fn a_node_nobody_measured_is_not_called_healthy_and_is_not_flagged() {
    // The distinction the whole crate turns on: no flags is not a clean bill.
    // A metric nobody took cannot raise one, and it must not pass for one that
    // was taken and came out fine.
    assert!(said(&Seen::default()).is_empty());
}

#[test]
fn a_gradient_too_small_to_train_on_says_so() {
    let seen = Seen {
        grad_norm: Some(1e-9),
        ..Seen::default()
    };

    assert_eq!(said(&seen), ["VANISHING"]);
}

#[test]
fn and_one_too_big_to_step_on() {
    let seen = Seen {
        grad_norm: Some(1e5),
        ..Seen::default()
    };

    assert_eq!(said(&seen), ["EXPLODING"]);
}

#[test]
fn a_gradient_cannot_be_both() {
    // They are the two ends of one number, so a `match` and not two `if`s: the
    // day somebody moves a bound, this is what stops both firing.
    for norm in [1e-9, 1.0, 1e5] {
        let seen = Seen {
            grad_norm: Some(norm),
            ..Seen::default()
        };
        assert!(said(&seen).len() <= 1, "{norm} said {:?}", said(&seen));
    }
}

#[test]
fn a_layer_that_dies_one_step_in_four_is_dead() {
    // The original's finding, and the reason this reads the maximum: the mean
    // is exactly what hides it. Four steps, one of them entirely off.
    let seen = Seen {
        zero_frac_max: Some(1.0),
        ..Seen::default()
    };

    assert_eq!(said(&seen), ["DEAD"]);
}

#[test]
fn and_one_that_is_merely_sparse_is_not() {
    let seen = Seen {
        zero_frac_max: Some(0.6),
        ..Seen::default()
    };

    assert!(said(&seen).is_empty(), "half a ReLU layer is off by design");
}

#[test]
fn a_layer_pinned_where_the_derivative_is_nothing_says_so() {
    let seen = Seen {
        sat_frac_max: Some(0.8),
        ..Seen::default()
    };

    assert_eq!(said(&seen), ["SATURATED"]);
}

#[test]
fn a_node_moving_too_little_next_to_its_own_weights_says_so() {
    // The cheapest signal there is, and the original measured it without ever
    // saying anything about it.
    let seen = Seen {
        update_ratio: Some(1e-7),
        ..Seen::default()
    };

    assert_eq!(said(&seen), ["STALLED"]);
}

#[test]
fn and_one_moving_so_much_it_forgets_where_it_was() {
    let seen = Seen {
        update_ratio: Some(0.5),
        ..Seen::default()
    };

    assert_eq!(said(&seen), ["OVERSTEPPING"]);
}

#[test]
fn a_healthy_ratio_is_near_a_thousandth_and_says_nothing() {
    for ratio in [3e-4, 1e-3, 3e-3] {
        let seen = Seen {
            update_ratio: Some(ratio),
            ..Seen::default()
        };
        assert!(said(&seen).is_empty(), "{ratio} said {:?}", said(&seen));
    }
}

#[test]
fn dead_channels_are_counted_and_are_not_the_same_as_a_dead_layer() {
    // A layer can be perfectly alive with a quarter of its width doing
    // nothing, and that is a width problem rather than a layer problem.
    let seen = Seen {
        dead_channels: 4,
        zero_frac_max: Some(0.25),
        ..Seen::default()
    };

    assert_eq!(said(&seen), ["DEAD_CHANNELS(4)"]);
}

#[test]
fn a_channel_alive_and_never_asked_for_is_its_own_finding() {
    let seen = Seen {
        ignored_channels: 3,
        ..Seen::default()
    };

    assert_eq!(said(&seen), ["IGNORED_CHANNELS(3)"]);
}

#[test]
fn a_signal_growing_where_nothing_normalises_it_says_so() {
    let seen = Seen {
        signal_gain: Some(100.0),
        ..Seen::default()
    };

    assert_eq!(said(&seen), ["MISSING_NORMALISATION"]);
}

#[test]
fn a_signal_that_shrank_says_nothing_at_all() {
    // Measured, and it is the finding rather than an omission: a plain stack
    // whose output arrives five ten-thousandths of the size it went in trained
    // as well as a healthy one, because Adam is scale-invariant per parameter.
    // See `health/tests/normalisation.py`.
    for gain in [5.5e-4, 4.0e-6, 1e-12] {
        let seen = Seen {
            signal_gain: Some(gain),
            ..Seen::default()
        };
        assert!(said(&seen).is_empty(), "{gain} said {:?}", said(&seen));
    }
}

#[test]
fn the_drift_a_residual_trunk_has_anyway_is_not_a_finding() {
    // Eighty unnormalised residual blocks reach 3.96x, and the notebook that
    // dropped the normalisation from three of them scored **better** for it. A
    // bound that fires here teaches somebody else's lesson.
    for gain in [0.61, 1.01, 2.81, 3.96] {
        let seen = Seen {
            signal_gain: Some(gain),
            ..Seen::default()
        };
        assert!(said(&seen).is_empty(), "{gain} said {:?}", said(&seen));
    }
}

#[test]
fn and_whoever_normalises_differently_moves_the_bound() {
    let seen = Seen {
        signal_gain: Some(4.0),
        ..Seen::default()
    };
    let tighter = Thresholds {
        gain_drift: 3.0,
        ..Thresholds::default()
    };

    assert!(said(&seen).is_empty());
    assert_eq!(
        somatize_health::verdict(&seen, &tighter)
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>(),
        ["MISSING_NORMALISATION"]
    );
}

#[test]
fn a_probe_that_measured_no_signal_is_not_called_healthy() {
    // The same rule as everywhere else, and it matters most here: a static
    // probe measures three things and an audit measures fifteen, so most of a
    // `Seen` is `None` on either side of the wall.
    let seen = Seen {
        jacobian_gain: Some(1e-6),
        jacobian_spread: Some(40.0),
        ..Seen::default()
    };

    assert!(said(&seen).is_empty());
}

#[test]
fn two_groups_carrying_the_same_information_leak() {
    let seen = Seen {
        group_cka: Some(0.99),
        ..Seen::default()
    };

    assert_eq!(said(&seen), ["LEAKAGE"]);
    let apart = Seen {
        group_cka: Some(0.4),
        ..Seen::default()
    };
    assert!(said(&apart).is_empty());
}

#[test]
fn a_collapsing_update_says_nothing_at_the_default_bound_because_it_is_off() {
    // Measured, and the measurement did not support it: healthy runs dipped to
    // 0.69 of their own median and destabilised ones ranged 0.43-0.86, which
    // overlaps in both directions. The metric is kept and drawn; the alarm is
    // not made up.
    let seen = Seen {
        update_rank: Some(2.0),
        update_rank_usual: Some(20.0),
        ..Seen::default()
    };

    assert!(said(&seen).is_empty());
}

#[test]
fn and_says_so_for_whoever_has_a_baseline_to_set_it_against() {
    let seen = Seen {
        update_rank: Some(2.0),
        update_rank_usual: Some(20.0),
        ..Seen::default()
    };
    let theirs = Thresholds {
        narrowing_of_usual: 0.6,
        ..Thresholds::default()
    };

    assert_eq!(somatize_health::verdict(&seen, &theirs), [Flag::Narrowing]);
}

#[test]
fn a_run_with_no_history_yet_is_not_said_to_have_departed_from_it() {
    // The published monitor compares against a healthy baseline run. With one
    // run to hand its own past is the reference, and before there is a past
    // there is nothing to say.
    let seen = Seen {
        update_rank: Some(2.0),
        update_rank_usual: None,
        ..Seen::default()
    };

    assert!(said(&seen).is_empty());
}

#[test]
fn an_update_that_is_merely_low_rank_all_along_is_not_narrowing() {
    // Healthy updates already carry low-rank structure — it is what LoRA and
    // the Muon family exploit. The finding is the **departure**, not the rank.
    let seen = Seen {
        update_rank: Some(2.0),
        update_rank_usual: Some(2.1),
        ..Seen::default()
    };
    let theirs = Thresholds {
        narrowing_of_usual: 0.6,
        ..Thresholds::default()
    };

    assert!(somatize_health::verdict(&seen, &theirs).is_empty());
}

#[test]
fn losing_plasticity_needs_all_three_signs_at_once() {
    let losing = Seen {
        param_norm_slope: Some(0.01),
        eff_rank_slope: Some(-0.01),
        dormancy_frac: Some(0.8),
        ..Seen::default()
    };

    assert_eq!(said(&losing), ["LOSING_PLASTICITY"]);
}

#[test]
fn and_any_one_of_them_alone_is_ordinary() {
    // A network whose weights grow is training. One with dormant units is a
    // ReLU network. Flagging either alone is crying wolf.
    let growing = Seen {
        param_norm_slope: Some(0.01),
        eff_rank_slope: Some(0.001),
        dormancy_frac: Some(0.8),
        ..Seen::default()
    };
    let quiet = Seen {
        param_norm_slope: Some(0.0),
        eff_rank_slope: Some(-0.01),
        dormancy_frac: Some(0.8),
        ..Seen::default()
    };

    assert!(said(&growing).is_empty());
    assert!(said(&quiet).is_empty());
}

#[test]
fn what_stops_a_run_is_read_first() {
    // A NaN makes every number below it meaningless, so it is read before them
    // rather than sorted in beside them.
    let seen = Seen {
        nan: true,
        grad_norm: Some(1e-9),
        zero_frac_max: Some(1.0),
        ..Seen::default()
    };

    assert_eq!(said(&seen)[0], "NAN");
}

#[test]
fn the_same_numbers_answer_differently_under_other_thresholds() {
    // The invariant of the whole layer: a diagnosis is reproducible from the
    // record, so an argument about a bound is settled by re-asking rather than
    // by training again.
    let seen = Seen {
        grad_norm: Some(1e-8),
        ..Seen::default()
    };
    let strict = Thresholds::default();
    let lenient = Thresholds {
        grad_low: 1e-12,
        ..Thresholds::default()
    };

    assert_eq!(somatize_health::verdict(&seen, &strict), [Flag::Vanishing]);
    assert!(somatize_health::verdict(&seen, &lenient).is_empty());
}
