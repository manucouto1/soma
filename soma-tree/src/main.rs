//! What an edit did to a graph, said before anybody runs it.
//!
//! `diff` compares two commits; `log` walks a whole line of exploration and
//! says what each step did. Neither executes a node: every name is worked out
//! from the recipe, which is what makes the question cheap enough to ask about
//! ten commits at once.
//!
//! It holds no graph itself — see `somatize_tree::snapshot`.

use clap::{Parser, Subcommand};
use somatize_tree::bench::{Bench, probed, walking};
use somatize_tree::data;
use somatize_tree::findings::{DOWNSTREAM, Findings, RESETTLED, SALTED, STALE, SUSPECT};
use somatize_tree::journal::Verdict;
use somatize_tree::revision;
use somatize_tree::snapshot::Snapshot;
use somatize_tree::trials::{Goal, Trials};
use somatize_tree::walk::Walk;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser)]
#[command(name = "somatize-tree", about, version)]
struct Cli {
    #[command(subcommand)]
    doing: Doing,
}

/// Where the answers are looked for and kept, shared by every command.
#[derive(Parser)]
struct Where {
    /// The repository. Defaults to the current directory.
    #[arg(long, default_value = ".")]
    repo: PathBuf,
    /// Where probes are remembered, and where values are looked up. Defaults
    /// to `$XDG_CACHE_HOME/somatize-tree`. Point it at the store your runs use and
    /// a diff also says what is already computed — that half needs `--input`,
    /// the remembering does not.
    #[arg(long)]
    store: Option<PathBuf>,
    /// A JSON file holding the graph's real input. Without it the root is
    /// named by hashing `Null` — enough to compare commits with each other,
    /// not enough to look anything up.
    #[arg(long = "input")]
    given: Option<PathBuf>,
    /// The probes themselves, unread, for whoever is building on this.
    #[arg(long)]
    json: bool,
}

#[derive(Subcommand)]
enum Doing {
    /// What changed between two commits, node by node.
    Diff {
        /// The commit to compare from. Any revspec: a sha, a tag, `HEAD~3`.
        before: String,
        /// The commit to compare to. Defaults to the working tree's `HEAD`.
        #[arg(default_value = "HEAD")]
        after: String,
        #[command(flatten)]
        at: Where,
    },
    /// A whole line of exploration: every commit in a range and what its step
    /// did. `somatize-tree log main~10..main`.
    Log {
        /// Draw the pruned lines too. They fold by default.
        #[arg(long)]
        all_lines: bool,
        /// A range git understands — `main~10..main` — or a revspec, meaning
        /// the history back from there. Defaults to every branch, because
        /// three variants of one idea are three branches and a walk from one
        /// tip cannot see its own siblings.
        #[arg(default_value = somatize_tree::revision::ALL)]
        range: String,
        /// How far back, when what was asked is not a range.
        #[arg(long, default_value_t = 10)]
        most: usize,
        #[command(flatten)]
        at: Where,
    },
    /// Writes down what you saw. Markdown, LaTeX, whatever you read later.
    Note {
        /// The commit it is about.
        #[arg(default_value = "HEAD")]
        rev: String,
        /// The note. Without it, read from stdin — which is how a long one
        /// gets in without fighting a shell over quoting.
        #[arg(short = 'm', long)]
        message: Option<String>,
        #[command(flatten)]
        at: Where,
    },
    /// Marks a commit invalid: something here was wrong.
    ///
    /// The only judgement left that is about the code rather than about where
    /// to go next, and the only one whose consequence is mechanical —
    /// everything under it becomes suspect. Deciding a line is dead or
    /// promising is a decision, it belongs in the reasoning with its scope and
    /// its reason, and it is written there.
    Verdict {
        /// `invalid`. The other three left; it will tell you where.
        verdict: String,
        #[arg(default_value = "HEAD")]
        rev: String,
        /// Why. Worth more than the verdict in six months.
        #[arg(short = 'm', long)]
        message: Option<String>,
        #[command(flatten)]
        at: Where,
    },
    /// What was run with one version: its trials, and one curve if asked.
    ///
    /// The version is the commit and does not change; the trials grow without
    /// end and are **associated** to it rather than versioned. They are written
    /// by whoever runs the study, from soma, and only read here.
    Trials {
        #[arg(default_value = "HEAD")]
        rev: String,
        /// Which one's curve to draw. Costs a fetch; the list costs a scan.
        #[arg(long)]
        curve: Option<u32>,
        #[command(flatten)]
        at: Where,
    },
    /// What data the store holds under each version, and what is nobody's.
    ///
    /// Iterating five versions of one question in an afternoon leaves five sets
    /// of intermediates, and a month later that is a pile of hashes nobody can
    /// attribute. This attributes them.
    ///
    /// **It deletes nothing, and it will not.** What can be said here is whose
    /// each thing is and how much room it takes; what is spare is a decision,
    /// and decisions are written down, not inferred.
    Data {
        /// Which versions to look at. Every branch by default, which is the
        /// shape an investigation has.
        #[arg(default_value = revision::ALL)]
        range: String,
        /// How far back, when what is asked for is not a range.
        #[arg(long, default_value_t = 10)]
        most: usize,
        #[command(flatten)]
        at: Where,
    },
    /// Everything anybody said about one commit, prose included.
    Show {
        #[arg(default_value = "HEAD")]
        rev: String,
        #[command(flatten)]
        at: Where,
    },
}

fn main() -> ExitCode {
    let done = match &Cli::parse().doing {
        Doing::Diff { before, after, at } => diffing(before, after, at),
        Doing::Log {
            all_lines,
            range,
            most,
            at,
        } => logging(range, *most, *all_lines, at),
        Doing::Note { rev, message, at } => saying(rev, None, message.as_deref(), at),
        Doing::Verdict {
            verdict,
            rev,
            message,
            at,
        } => match Verdict::read(verdict) {
            Some(verdict) => saying(rev, Some(verdict), message.as_deref(), at),
            // Said in full and not *that is not one*: whoever writes it has
            // the old habit, and what they need is not that they were wrong
            // but where what they meant to say went.
            None if matches!(verdict.as_str(), "promising" | "dead-end" | "superseded") => {
                Err(format!(
                    "`{verdict}` is no longer a verdict. It was never something that \
                     happened to the code: it was a decision about where to go next, and \
                     it is now written in the reasoning — with its scope, which says which \
                     line it is about, and its reason — from the view. `invalid` stays here."
                )
                .into())
            }
            None => Err(format!("`{verdict}` is not one: invalid").into()),
        },
        Doing::Trials { rev, curve, at } => trialling(rev, *curve, at),
        Doing::Data { range, most, at } => dataing(range, *most, at),
        Doing::Show { rev, at } => showing(rev, at),
    };
    match done {
        Ok(quiet) => match quiet {
            true => ExitCode::SUCCESS,
            // Something moved. A separate code so a script can branch on it
            // without parsing the text, the way `git diff --quiet` does.
            false => ExitCode::from(1),
        },
        Err(why) => {
            eprintln!("{why}");
            ExitCode::from(2)
        }
    }
}

fn diffing(before: &str, after: &str, at: &Where) -> Result<bool, Box<dyn std::error::Error>> {
    let bench = Bench::set_up(&at.repo, at.store.as_deref(), at.given.as_deref())?;
    let probing = bench.probing(at.store.as_deref(), at.given.as_deref());
    let commits = [
        revision::named(&bench.repo, after)?,
        revision::named(&bench.repo, before)?,
    ];
    let known = probed(&bench, &probing, &commits)?;
    let (after, before) = (&known[commits[0].as_str()], &known[commits[1].as_str()]);

    if at.json {
        println!("{}", serde_json::to_string_pretty(&[before, after])?);
        return Ok(true);
    }
    let found = probing.compared(&known, &[(commits[1].clone(), commits[0].clone())])?;
    let found = found.first().cloned().unwrap_or_default();
    report(before, after, &found);
    Ok(found.is_quiet())
}

fn logging(
    range: &str,
    most: usize,
    all_lines: bool,
    at: &Where,
) -> Result<bool, Box<dyn std::error::Error>> {
    let walk = walking(
        &at.repo,
        at.store.as_deref(),
        at.given.as_deref(),
        range,
        most,
    )?;
    if at.json {
        // Unfolded: whoever asks for JSON is going to process it, and hiding
        // rows for readability would hide them from a program.
        println!("{}", serde_json::to_string_pretty(&walk)?);
        return Ok(true);
    }
    Ok(printed(&walk, all_lines))
}

/// The line of exploration, newest first, as git prints history.
///
/// Pruned lines fold unless asked for. Pruning is not drawing and never
/// deleting — it is all still in git, in the journal and in the reasoning, and
/// `--all-lines` brings it back.
fn printed(walk: &Walk, all_lines: bool) -> bool {
    let every: Vec<_> = walk.stops.iter().filter(|stop| !stop.context).collect();
    // What somebody judged does not fold: an `invalid` commit is what casts
    // doubt on the measurement the decision to abandon leaned on, and hiding it
    // would hide the reason to look again.
    let folded = every.iter().filter(|stop| stop.pruned).count();
    let shown: Vec<_> = if all_lines {
        every
    } else {
        every.into_iter().filter(|stop| !stop.pruned).collect()
    };

    println!("{}   ·   {} commits", walk.built_from, shown.len());
    // And never in silence.
    if folded > 0 && !all_lines {
        println!(
            "{folded} more on pruned lines. Nothing is deleted: `--all-lines` brings them back."
        );
    }
    println!();

    let mut restless = 0;
    for (n, stop) in shown.iter().enumerate() {
        // Both layers on one line, and in this order: that something was
        // wrong reads before where somebody decided to stop, because the first
        // casts doubt on the measurement the second leaned on.
        let mut said = Vec::new();
        match stop.verdict {
            Some(verdict) => said.push(verdict.to_string()),
            // Said out loud rather than left blank: inheriting doubt from an
            // ancestor is not the same as nobody having looked at this.
            None if stop.doubted => said.push("under something invalid".to_string()),
            None => {}
        }
        if let Some(course) = stop.decided {
            said.push(course.to_string());
        }
        // What was run, which is the other half of what a version is. What is
        // under way apart from what finished: a half-done study reads
        // differently from one nobody touched.
        if stop.trials.trials > 0 {
            let mut how = format!("{} trials", stop.trials.trials);
            if stop.trials.running > 0 {
                how.push_str(&format!(", {} running", stop.trials.running));
            }
            if let Some(best) = stop.trials.best {
                how.push_str(&format!(", best {best:.4}"));
            } else if let (Some(low), Some(high)) = (stop.trials.lowest, stop.trials.highest) {
                how.push_str(&format!(", between {low:.4} and {high:.4}"));
            }
            said.push(how);
        }
        let judged = if said.is_empty() {
            String::new()
        } else {
            format!("  [{}]", said.join(" · "))
        };
        println!("{}  {}{judged}", stop.short, stop.subject);
        let Some(step) = walk.step_to(&stop.commit) else {
            continue;
        };
        restless += usize::from(!step.found.not_comparable().is_empty());
        // Which commit this is a step **from**, whenever that is not the line
        // underneath. With three branches off one base, git's order interleaves
        // them, and a bare `│` would say a step came from a line of exploration
        // it has nothing to do with.
        let below = shown
            .get(n + 1)
            .is_some_and(|next| next.commit == step.from);
        let from = match below {
            true => String::new(),
            false => {
                // Said when the parent is folded. A hash pointing at a row
                // that is not drawn is a dead end for whoever reads: either it
                // says why it is not there, or it looks lost.
                let cut = walk
                    .stops
                    .iter()
                    .find(|one| one.commit == step.from)
                    .is_some_and(|one| one.pruned && !all_lines);
                format!(
                    "from {}{} · ",
                    &step.from[..12.min(step.from.len())],
                    if cut { " (pruned)" } else { "" }
                )
            }
        };
        println!("     │    {from}{}", stepped(&step.found));
        // Compact here, where the tree's shape is what carries the reading. A
        // block of versions would break the one column a walk has.
        if !step.drift.is_empty() {
            println!(
                "     │    ⚠ another environment: {}",
                step.drift
                    .iter()
                    .map(|(what, was, is)| format!("{what} {was} → {is}"))
                    .collect::<Vec<_>>()
                    .join(", "),
            );
        }
    }

    println!();
    match restless {
        0 => println!("No step leaves results that cannot be compared with the one before."),
        // Steps and not a sum of nodes: the same node counted at three steps is
        // one node looked at three times, and adding those said nothing.
        n => println!("{n} of the steps leave results NOT comparable with the one before."),
    }
    restless == 0
}

/// One step of a walk, as a line: where the edit is, and what it left behind.
fn stepped(found: &Findings) -> String {
    if found.is_quiet() {
        return "no changes".into();
    }
    let edit = found.the_edit();
    // Said either way. A step where names moved and nobody typed anything is a
    // trial of the same variant, and leaving that to be inferred from the
    // absence of a word is how somebody reads it as an edit they forgot.
    let mut said = vec![match edit.is_empty() {
        true => "no edit".to_string(),
        false => format!("edit: {}", edit.join(", ")),
    }];
    for (finding, called) in [
        (STALE, "⚠ STALE"),
        (SUSPECT, "⚠ suspect"),
        ("UNKNOWN", "cannot be foreseen"),
        ("UNVERSIONED", "unversioned"),
        (RESETTLED, "resettled"),
        (SALTED, "another salt"),
    ] {
        let these = found.saying(finding);
        if !these.is_empty() {
            said.push(format!("{called}: {}", these.join(", ")));
        }
    }
    if edit.is_empty() && said.len() == 1 {
        // Names moved, nobody typed anything, and no weights or salt account
        // for it either: what is left is something above them. Only said here,
        // because listing what inherited an edit is the noise `the_edit`
        // exists to leave out.
        said.push(format!(
            "inherited: {}",
            found.saying(DOWNSTREAM).join(", ")
        ));
    }
    said.join(" · ")
}

fn report(before: &Snapshot, after: &Snapshot, found: &Findings) {
    println!(
        "{} → {}   {}",
        before.commit, after.commit, before.built_from
    );
    if before.input == "sentinel" {
        println!(
            "names computed against a sentinel input: comparable with each other, \
             not the names a run would produce"
        );
    }
    drifted(before, after);
    println!();

    if found.is_quiet() {
        println!("  Nothing to say about any node.");
        return;
    }
    let widest = found
        .findings
        .keys()
        .map(String::len)
        .max()
        .unwrap_or_default();
    for (node, said) in &found.findings {
        let mut line = said.join(" · ");
        // The readable form of the one axis the model was not given: two
        // digests nobody can act on, said as what somebody actually typed.
        if let Some([was, is]) = found.declared.get(node) {
            line.push_str(&format!("  ({was} → {is})"));
        }
        println!("  {node:<widest$}  {line}");
    }

    println!();
    let edit = found.the_edit();
    if !edit.is_empty() {
        println!("The edit is in: {}", edit.join(", "));
    }
    let stale = found.saying(STALE);
    match stale.is_empty() {
        true => println!("No node will be handed a cached value that is no longer its own."),
        false => println!(
            "⚠ {} node(s) with the SAME key and other code: the cache will HIT.",
            stale.len(),
        ),
    }
    let not_comparable = found.not_comparable().len();
    if not_comparable > 0 {
        println!("{not_comparable} node(s) with results NOT comparable with the ones before.");
    }
    if !after.unneeded.is_empty() {
        println!(
            "Already computed, would not have to run: {}",
            after.unneeded.join(", "),
        );
    }
}

/// Says so when the two sides were not built against the same thing.
///
/// A whole-graph fact and not a finding per node: an interpreter that moved
/// moved under all of them, and attributing it forty times would be noise.
fn drifted(before: &Snapshot, after: &Snapshot) {
    let drift = before.drifted_from(after);
    if drift.is_empty() {
        return;
    }
    println!(
        "⚠ the two sides were probed against different environments, and that is in \
         no commit:"
    );
    for (what, was, is) in drift {
        println!("    {what}: {was} → {is}");
    }
}

/// Writes one thing down about a commit, verdict or not.
fn saying(
    rev: &str,
    verdict: Option<Verdict>,
    message: Option<&str>,
    at: &Where,
) -> Result<bool, Box<dyn std::error::Error>> {
    let bench = Bench::set_up(&at.repo, at.store.as_deref(), at.given.as_deref())?;
    // Resolved to a full hash, never stored as `HEAD~2`: what somebody wrote
    // about a commit has to still be about that commit tomorrow.
    let commit = revision::named(&bench.repo, rev)?;
    let prose = match message {
        Some(said) => said.to_string(),
        // No `-m`, so it comes from stdin — which is how a note long enough to
        // be worth reading gets in without fighting a shell over quoting.
        None => std::io::read_to_string(std::io::stdin())?,
    };
    if prose.trim().is_empty() && verdict.is_none() {
        return Err("a note with nothing in it says nothing".into());
    }

    let journal = bench.journal();
    let nth = journal.say(
        &commit,
        verdict,
        &revision::whoami(&bench.repo),
        prose.trim(),
    )?;

    let short = &commit[..12.min(commit.len())];
    match verdict {
        Some(verdict) => println!("{short} · {verdict} · said {nth}"),
        None => println!("{short} · note · said {nth}"),
    }
    Ok(true)
}

/// What data sits under each version, and what belongs to none of them.
fn dataing(range: &str, most: usize, at: &Where) -> Result<bool, Box<dyn std::error::Error>> {
    let bench = Bench::set_up(&at.repo, at.store.as_deref(), at.given.as_deref())?;
    let Some(store) = at.store.as_deref() else {
        // Said in full: with no store there is nothing to attribute, and an
        // empty table reads as *nothing is kept* rather than as *you have not
        // said where to look*.
        println!("With no `--store` there is nothing to look at: what this attributes");
        println!("are the values a run left kept, and the store is where they are.");
        return Ok(true);
    };
    let kept = somatize_store::Local::at(store)?;
    let commits = revision::commits_in(&bench.repo, range, most)?;
    let probing = bench.probing(at.store.as_deref(), at.given.as_deref());
    let known = probed(&bench, &probing, &commits)?;
    let said = data::under(&kept, &known)?;

    if said.is_empty() {
        println!("The store holds no value from a run yet.");
        return Ok(true);
    }

    // Said before the table and not after. Without `--given` a probe names
    // against a sentinel, so no key matches one from a run over real data and
    // **everything** is attributed by fingerprint. The table is right; what is
    // missing is why it never says *that one is its own*, which from outside
    // reads as half the mechanism not working.
    if known.values().any(|taken| taken.input == "sentinel") {
        println!("Probed against a sentinel, so this attributes by code and not by name:");
        println!("`--given` with the real input is what makes the keys match.");
    }

    // By version, and what is nobody's last: it is the last thing looked at
    // and the first anybody would want to hide.
    let mut by_commit: BTreeMap<&str, Vec<&data::Belongs>> = BTreeMap::new();
    let mut nobodys: Vec<&data::Belongs> = Vec::new();
    for one in &said {
        match one.is_nobodys() {
            true => nobodys.push(one),
            false => {
                for commit in one.of.keys() {
                    by_commit.entry(commit.as_str()).or_default().push(one);
                }
            }
        }
    }

    let told = revision::told(&bench.repo, &commits);
    for commit in &commits {
        let Some(mine) = by_commit.get(commit.as_str()) else {
            continue;
        };
        let subject = told
            .get(commit)
            .map(|(_, said)| said.as_str())
            .unwrap_or("");
        println!(
            "
{}  {subject}",
            &commit[..12.min(commit.len())]
        );
        for one in mine {
            let how = match one.of.get(commit.as_str()) {
                Some(data::How::Named) => "its own",
                // Said differently because it **is** different: the code is
                // the same and the name does not match, which nearly always
                // means another input or another environment. Folding them
                // into *it is its own* would hide the case somebody wants.
                _ => "same code",
            };
            println!(
                "  {:<10} {:<10} {how:<13} {}",
                one.node.as_deref().unwrap_or("—"),
                one.fingerprint.as_deref().unwrap_or("—"),
                one.environment.as_deref().unwrap_or("—"),
            );
        }
    }

    if !nobodys.is_empty() {
        println!(
            "
{} from no version looked at:",
            nobodys.len()
        );
        for one in nobodys.iter().take(most) {
            println!(
                "  {:<10} {:<10} {}",
                one.node.as_deref().unwrap_or("—"),
                one.fingerprint.as_deref().unwrap_or("—"),
                one.environment.as_deref().unwrap_or("—"),
            );
        }
        println!();
        println!("Which is not the same as being spare: it may be from a branch nobody");
        println!("looked at, from a commit that is gone, or from an environment that");
        println!("cannot be reproduced.");
    }
    Ok(true)
}

/// A version's trials, and one curve if asked for.
fn trialling(
    rev: &str,
    curve: Option<u32>,
    at: &Where,
) -> Result<bool, Box<dyn std::error::Error>> {
    let bench = Bench::set_up(&at.repo, at.store.as_deref(), at.given.as_deref())?;
    let commit = revision::named(&bench.repo, rev)?;
    let tree = bench.config.tree(&bench.repo);
    let goal = bench.config.towards()?;
    let trials = Trials::of(&tree, &bench.remembering).towards(goal);
    let seen = trials.of_commit(&commit)?;

    println!("{}   ·   {} trials", trials.study(&commit), seen.len());
    if seen.is_empty() {
        println!(
            "\nNobody has run anything with this version yet. They are written from \
             somatize,\nwith `study=\"{}\"`.",
            trials.study(&commit)
        );
        return Ok(true);
    }
    println!();
    for one in &seen {
        let state = one.state.as_deref().unwrap_or("?");
        let score = match one.score {
            // Said and not left quiet: a pruned score is real and **not**
            // comparable with a finished one — measured after fewer epochs —
            // and putting them in one column silently invites comparing them.
            Some(score) if one.comparable() => format!("{score:>10.4}"),
            Some(score) => format!("{score:>10.4} (pruned)"),
            None => format!("{:>10}", "—"),
        };
        let rescued = if one.attempt > 0 {
            format!("  ·  attempt {}", one.attempt)
        } else {
            String::new()
        };
        println!(
            "  {:>3}  {state:<8}{score}  {}{rescued}",
            one.trial,
            one.point.as_deref().unwrap_or("")
        );
    }

    let done: Vec<f64> = seen
        .iter()
        .filter(|one| one.comparable())
        .filter_map(|one| one.score)
        .collect();
    if !done.is_empty() {
        println!();
        match goal {
            Some(goal) => println!(
                "  The best of {} comparable: {:.4}",
                done.len(),
                goal.best_of(done.iter().copied()).unwrap_or(f64::NAN)
            ),
            // And not *the best*: which one that is depends on whether the
            // metric is maximised or minimised, and that is in no record.
            None => println!(
                "  {} comparable, between {:.4} and {:.4}. Declare `goal = \"min\"` or \
                 `goal = \"max\"`\n  in soma-tree.toml and it will say which is best.",
                done.len(),
                Goal::Min.best_of(done.iter().copied()).unwrap_or(f64::NAN),
                Goal::Max.best_of(done.iter().copied()).unwrap_or(f64::NAN),
            ),
        }
    }

    if let Some(which) = curve {
        let one = seen
            .iter()
            .find(|one| one.trial == which)
            .ok_or_else(|| format!("there is no trial {which} in {}", trials.study(&commit)))?;
        match trials.curve(one)? {
            Some(drawn) => {
                println!("\n  Trial {which}, {} reports:", drawn.reports.len());
                for (n, value) in drawn.reports.iter().enumerate() {
                    println!("    {n:>4}  {value:.4}");
                }
                if let Some(because) = drawn.because {
                    println!("    stopped because {because}");
                }
            }
            None => println!("\n  The record of trial {which} points at a blob that is not there."),
        }
    }
    Ok(true)
}

/// Everything anybody said about one commit, prose and all.
fn showing(rev: &str, at: &Where) -> Result<bool, Box<dyn std::error::Error>> {
    let bench = Bench::set_up(&at.repo, at.store.as_deref(), at.given.as_deref())?;
    let commit = revision::named(&bench.repo, rev)?;
    let journal = bench.journal();

    let short = &commit[..12.min(commit.len())];
    let told = revision::told(&bench.repo, std::slice::from_ref(&commit));
    let said = told
        .get(&commit)
        .map(|(_, said)| said.as_str())
        .unwrap_or_default();
    println!("{short}  {said}");
    let said: Vec<_> = journal
        .all()?
        .into_iter()
        .filter(|saying| saying.commit == commit)
        .collect();
    if said.is_empty() {
        println!("\n  Nobody has said anything about this commit.");
        return Ok(true);
    }
    // Here the prose is fetched, and only here: a scan is what the log pays
    // for, and one blob per saying is what reading them costs.
    for saying in &said {
        println!();
        match saying.verdict {
            Some(verdict) => println!("  [{}] {} · {}", saying.nth, verdict, saying.who),
            None => println!("  [{}] note · {}", saying.nth, saying.who),
        }
        for line in journal.read(saying)?.lines() {
            println!("      {line}");
        }
    }
    Ok(true)
}
