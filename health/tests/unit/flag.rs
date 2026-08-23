//! What a flag is called, and what it says to do about itself.

use soma_next_health::Flag;

#[test]
fn a_flag_that_counts_something_says_how_many() {
    assert_eq!(Flag::DeadChannels(7).to_string(), "DEAD_CHANNELS(7)");
    assert_eq!(Flag::IgnoredChannels(2).to_string(), "IGNORED_CHANNELS(2)");
    assert_eq!(Flag::Vanishing.to_string(), "VANISHING");
}

#[test]
fn the_name_is_stable_whatever_it_counts() {
    // What is written down and what somebody greps for. Two runs of a
    // diagnosis have to say the same word for the same thing, or nothing can be
    // filtered on.
    assert_eq!(Flag::DeadChannels(7).name(), Flag::DeadChannels(2).name());
}

#[test]
fn every_flag_says_what_to_do_about_it() {
    // Part of the flag and not of whoever draws it: the thresholds and the
    // advice are the same opinion, and splitting them is how a dashboard ends
    // up saying something this crate never said.
    //
    // The list is kept by hand and a `match` would keep itself, which is the
    // rule everywhere else in this project. It is written out anyway because
    // the enum carries values — `DEAD_CHANNELS(n)` — and the point of the test
    // is that every **variant** answers, not that every value does. Adding one
    // and not adding it here is the mistake to watch for; it had already
    // happened twice when this comment was written.
    for flag in [
        Flag::Nan,
        Flag::Inf,
        Flag::Vanishing,
        Flag::Exploding,
        Flag::Dead,
        Flag::Saturated,
        Flag::Stalled,
        Flag::Overstepping,
        Flag::DeadChannels(1),
        Flag::IgnoredChannels(1),
        Flag::MissingNormalisation,
        Flag::Leakage,
        Flag::Narrowing,
        Flag::LosingPlasticity,
        Flag::IgnoredInput("audio".into()),
        Flag::SoleReliance("audio".into()),
    ] {
        assert!(!flag.about().is_empty(), "{flag} says nothing about itself");
        assert!(
            flag.name()
                .chars()
                .all(|c| c.is_ascii_uppercase() || c == '_')
        );
    }
}
