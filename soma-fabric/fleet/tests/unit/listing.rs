//! The names a graph writes, and what each of them resolves to.

use somatize_fabric_fleet::{Listed, Listing, Trouble};
use std::collections::BTreeMap;
use tempfile::TempDir;

fn nobody() -> BTreeMap<String, somatize_fabric_fleet::Host> {
    BTreeMap::new()
}

#[test]
fn a_listing_goes_to_a_file_and_comes_back_the_same_listing() {
    let dir = TempDir::new().expect("a directory");
    let at = dir.path().join("listing.toml");
    let mut paper = Listing::default();
    paper.add(Listed::at("w1", "node3:7000")).expect("listed");
    paper
        .add(Listed::run("tok", ["python", "-m", "somatize.worker"]))
        .expect("listed");

    paper.write(&at).expect("written");
    let back = Listing::read(&at).expect("read");

    assert_eq!(back.listed.len(), 2);
    assert_eq!(back.listed[0].host, "w1");
    assert_eq!(back.listed[0].at.as_deref(), Some("node3:7000"));
    assert_eq!(back.listed[1].run.as_deref().unwrap().len(), 3);
}

#[test]
fn a_listing_nobody_has_written_yet_is_empty_and_not_a_failure() {
    let dir = TempDir::new().expect("a directory");

    let paper = Listing::read(&dir.path().join("nothing.toml")).expect("an empty listing");

    assert!(paper.listed.is_empty());
}

#[test]
fn a_name_listed_again_is_replaced_and_not_doubled() {
    // Two rows for one name is a listing that answers two things to one
    // question, and whichever a broker picked would be arbitrary.
    let mut paper = Listing::default();
    paper.add(Listed::at("w1", "node3:7000")).expect("listed");
    paper
        .add(Listed::at("w1", "node9:7000"))
        .expect("listed again");

    assert_eq!(paper.listed.len(), 1);
    assert_eq!(paper.listed[0].at.as_deref(), Some("node9:7000"));
}

#[test]
fn two_names_at_one_address_are_one_wire() {
    // The rule that decides whether a run keeps its state, and it is asked of a
    // real broker rather than worked out here.
    let mut paper = Listing::default();
    paper.add(Listed::at("w1", "node3:7000")).expect("listed");
    paper
        .add(Listed::at("principal", "node3:7000"))
        .expect("listed");

    let wires = paper.wires(&nobody()).expect("grouped");

    assert_eq!(wires.wires.len(), 1, "one address is one wire");
    assert_eq!(wires.wires[0].names.len(), 2);
    assert!(wires.wires[0].shared, "so it is packed once");
}

#[test]
fn two_names_with_the_same_command_are_two_wires() {
    // A command is not an identity: it is a thing to run, and running it twice
    // gives two of them — two processes, two catalogs, never one wire.
    let mut paper = Listing::default();
    paper
        .add(Listed::run("tok", ["python", "-m", "w"]))
        .expect("listed");
    paper
        .add(Listed::run("tok2", ["python", "-m", "w"]))
        .expect("listed");

    let wires = paper.wires(&nobody()).expect("grouped");

    assert_eq!(
        wires.wires.len(),
        2,
        "identical argv was taken for one place"
    );
    assert!(wires.wires.iter().all(|one| !one.shared));
}

#[test]
fn a_name_that_is_an_address_and_a_command_is_refused_saying_which() {
    // Somebody meant one of them, and guessing which is a listing that quietly
    // does not do what its file says.
    let mut paper = Listing::default();
    let both = Listed {
        host: "w1".into(),
        at: Some("node3:7000".into()),
        run: Some(vec!["python".into()]),
    };

    let why = paper.add(both).expect_err("it was accepted");

    assert!(matches!(why, Trouble::Refused(_)), "{why:?}");
    assert!(why.to_string().contains("w1"), "{why}");
}

#[test]
fn a_name_that_says_nothing_about_how_to_get_to_it_is_refused() {
    let mut paper = Listing::default();
    let neither = Listed {
        host: "w1".into(),
        at: None,
        run: None,
    };

    assert!(matches!(paper.add(neither), Err(Trouble::Refused(_))));
}

#[test]
fn a_name_nobody_has_met_is_listed_with_no_machine_behind_it() {
    // Not an error and not missing data: it is a name nobody has sent work to.
    let mut paper = Listing::default();
    paper.add(Listed::at("w2", "node4:7000")).expect("listed");

    let wires = paper.wires(&nobody()).expect("grouped");

    assert_eq!(wires.wires[0].names[0].seen, None);
}

#[test]
fn a_name_somebody_has_met_carries_what_that_machine_calls_itself() {
    let mut paper = Listing::default();
    paper.add(Listed::at("w1", "node3:7000")).expect("listed");
    let met = BTreeMap::from([(
        "node3-4127".to_string(),
        somatize_fabric_fleet::Host::new("w1"),
    )]);

    let wires = paper.wires(&met).expect("grouped");

    assert_eq!(wires.wires[0].names[0].seen.as_deref(), Some("node3-4127"));
}

#[test]
fn the_ladder_is_answered_whole_with_two_rungs_that_cannot_be_climbed() {
    // Which rungs can be answered is a fact about the code, not about a screen,
    // so it is said here: the day the negotiation makes another one answerable,
    // no view has to be edited to stop greying it out.
    let wires = Listing::default().wires(&nobody()).expect("grouped");

    assert_eq!(
        wires.ladder.len(),
        4,
        "the vocabulary is four and stays four"
    );
    assert_eq!(
        wires.ladder.iter().filter(|one| one.answerable).count(),
        1,
        "only the direct rung can be answered today"
    );
    assert!(wires.ladder[2].answerable, "and it is the third");
}

#[test]
fn dropping_a_name_says_whether_there_was_one() {
    let mut paper = Listing::default();
    paper.add(Listed::at("w1", "node3:7000")).expect("listed");

    assert!(paper.drop("w1"));
    assert!(!paper.drop("w1"), "it said it dropped one twice");
    assert!(paper.listed.is_empty());
}
