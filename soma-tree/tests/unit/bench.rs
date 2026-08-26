//! Where the probe comes from, which is now nowhere but the binary itself.

use somatize_tree::bench::probe_laid_down;
use std::path::PathBuf;

#[test]
fn the_probe_laid_down_is_the_one_the_recipe_hashed() {
    // A snapshot is a pure function of a commit **given a fixed probe**, so the
    // probe's own source goes into the name it is remembered under. That only
    // holds while the bytes hashed and the bytes `python` runs are the same
    // ones, which is the whole reason it is `include_str!` and not a file
    // looked for at run time — beside the binary, where nobody puts it, or at
    // the `CARGO_MANIFEST_DIR` of whoever compiled it, which a `cargo install`
    // does not own.
    let cache = tempfile::tempdir().expect("a temporary directory");
    let laid = probe_laid_down(cache.path()).expect("the probe is laid down");

    let contract = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/soma_tree_probe.py");
    assert_eq!(
        std::fs::read(&laid).expect("it is readable"),
        std::fs::read(&contract).expect("the contract is readable"),
        "what runs is what the repository holds"
    );
}

#[test]
fn and_it_is_written_once_and_named_by_what_is_in_it() {
    // Content-addressed, so a probe that changed is a different name and the
    // old file is never asked for — no staleness to invalidate, and no second
    // write on every walk somebody runs.
    let cache = tempfile::tempdir().expect("a temporary directory");
    let first = probe_laid_down(cache.path()).expect("laid down");
    let written = std::fs::metadata(&first).expect("it is there").modified();

    let again = probe_laid_down(cache.path()).expect("laid down again");

    assert_eq!(first, again, "one name");
    assert_eq!(
        written.ok(),
        std::fs::metadata(&again)
            .expect("still there")
            .modified()
            .ok(),
        "and it was not written a second time"
    );
    assert!(
        first
            .file_name()
            .and_then(|it| it.to_str())
            .is_some_and(|it| { it.starts_with("soma_tree_probe-") && it.ends_with(".py") }),
        "the name a traceback shows still says what the file is: {first:?}"
    );
}
