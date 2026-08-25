//! What an edit did to a graph, said before anybody runs it.
//!
//! `diff` compares two commits; `log` walks a whole line of exploration and
//! says what each step did. Neither executes a node: every name is worked out
//! from the recipe, which is what makes the question cheap enough to ask about
//! ten commits at once.
//!
//! It holds no graph itself — see `soma_tree::snapshot`.

use clap::{Parser, Subcommand};
use soma_tree::bench::{Bench, probed, walking};
use soma_tree::data;
use soma_tree::findings::{DOWNSTREAM, Findings, RESETTLED, SALTED, STALE, SUSPECT};
use soma_tree::journal::Verdict;
use soma_tree::revision;
use soma_tree::serving::{Serving, routes};
use soma_tree::snapshot::Snapshot;
use soma_tree::trials::{Goal, Trials};
use soma_tree::walk::Walk;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser)]
#[command(name = "soma-tree", about, version)]
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
    /// to `$XDG_CACHE_HOME/soma-tree`. Point it at the store your runs use and
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
    /// did. `soma-tree log main~10..main`.
    Log {
        /// Dibujar también las líneas podadas. Se pliegan por defecto.
        #[arg(long)]
        all_lines: bool,
        /// A range git understands — `main~10..main` — or a revspec, meaning
        /// the history back from there. Defaults to every branch, because
        /// three variants of one idea are three branches and a walk from one
        /// tip cannot see its own siblings.
        #[arg(default_value = soma_tree::revision::ALL)]
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
    /// Serves the line over HTTP, for whoever draws it.
    Serve {
        /// Where to listen. Loopback by default: it serves one person's own
        /// exploration off their own machine.
        #[arg(long, default_value = "127.0.0.1:7373")]
        at: String,
        #[command(flatten)]
        where_: Where,
    },
    /// What was run with one version: its trials, and one curve if asked.
    ///
    /// The version is the commit and does not change; the trials grow without
    /// end and are **associated** to it rather than versioned. They are written
    /// by whoever runs the study, from soma-next, and only read here.
    Trials {
        #[arg(default_value = "HEAD")]
        rev: String,
        /// Which one's curve to draw. Costs a fetch; the list costs a scan.
        #[arg(long)]
        curve: Option<u32>,
        #[command(flatten)]
        at: Where,
    },
    /// Qué datos hay en el store bajo cada versión, y cuáles no son de ninguna.
    ///
    /// Dentro de un movimiento se iteran cinco versiones en una tarde, y cada
    /// una deja intermedios. Al mes siguiente eso es un montón de hashes que
    /// nadie puede atribuir. Esto los atribuye.
    ///
    /// **No borra nada, y no va a hacerlo.** Lo que se puede decir aquí es de
    /// quién es cada cosa y cuánto ocupa; qué sobra es una decisión, y las
    /// decisiones se escriben, no se deducen.
    Data {
        /// Qué versiones mirar. Todas las ramas por defecto, que es la forma
        /// que tiene una investigación.
        #[arg(default_value = revision::ALL)]
        range: String,
        /// Hasta dónde, cuando lo que se pide no es un rango.
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
            // Dicho entero y no «no es uno»: quien lo escribe tiene la costumbre
            // vieja, y lo que necesita saber no es que se equivocó sino a dónde
            // se fue lo que quería decir.
            None if matches!(verdict.as_str(), "promising" | "dead-end" | "superseded") => {
                Err(format!(
                    "`{verdict}` ya no es un veredicto. No era algo que le pasara al \
                     código: era una decisión sobre por dónde seguir, y ahora se \
                     escribe en el razonamiento —con su alcance, que dice de qué \
                     línea habla, y su motivo— desde la vista. Aquí queda `invalid`."
                )
                .into())
            }
            None => Err(format!("`{verdict}` no es uno: invalid").into()),
        },
        Doing::Trials { rev, curve, at } => trialling(rev, *curve, at),
        Doing::Serve { at, where_ } => serving(at, where_),
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
        // Sin plegar: quien pide JSON lo va a procesar, y esconderle filas por
        // legibilidad sería esconderlas de un programa que no las echa de menos.
        println!("{}", serde_json::to_string_pretty(&walk)?);
        return Ok(true);
    }
    Ok(printed(&walk, all_lines))
}

/// The line of exploration, newest first, as git prints history.
///
/// Las líneas podadas se pliegan salvo que se pidan. Podar es dejar de dibujar
/// y nunca borrar: sigue todo en git, en el diario y en el razonamiento, y
/// `--all-lines` lo devuelve. Un árbol de cuarenta variantes no se lee, y ése
/// es el problema, no que sobre nada.
fn printed(walk: &Walk, all_lines: bool) -> bool {
    let every: Vec<_> = walk.stops.iter().filter(|stop| !stop.context).collect();
    // Lo que ha juzgado alguien no se pliega: un commit `invalid` es lo que
    // pone en duda la medida en la que se basó la decisión de abandonar la
    // línea, y esconderlo sería esconder la razón para volver a mirarla.
    let folded = every.iter().filter(|stop| stop.pruned).count();
    let shown: Vec<_> = if all_lines {
        every
    } else {
        every.into_iter().filter(|stop| !stop.pruned).collect()
    };

    println!("{}   ·   {} commits", walk.built_from, shown.len());
    // Y nunca en silencio.
    if folded > 0 && !all_lines {
        println!("{folded} más en líneas podadas. Nada se borra: `--all-lines` los trae.");
    }
    println!();

    let mut restless = 0;
    for (n, stop) in shown.iter().enumerate() {
        // Las dos capas en una línea, y en este orden: que algo estuviera mal
        // se lee antes que dónde se decidió no seguir, porque lo primero pone
        // en duda la medida en la que se basó lo segundo.
        let mut said = Vec::new();
        match stop.verdict {
            Some(verdict) => said.push(verdict.to_string()),
            // Said out loud rather than left blank: inheriting doubt from an
            // ancestor is not the same as nobody having looked at this.
            None if stop.doubted => said.push("bajo algo inválido".to_string()),
            None => {}
        }
        if let Some(course) = stop.decided {
            said.push(course.to_string());
        }
        // Lo que se corrió, que es la otra mitad de qué es una versión. Con lo
        // que va en marcha aparte de lo que terminó: un estudio a medias se lee
        // distinto de uno que nadie tocó.
        if stop.trials.trials > 0 {
            let mut how = format!("{} ensayos", stop.trials.trials);
            if stop.trials.running > 0 {
                how.push_str(&format!(", {} corriendo", stop.trials.running));
            }
            if let Some(best) = stop.trials.best {
                how.push_str(&format!(", mejor {best:.4}"));
            } else if let (Some(low), Some(high)) = (stop.trials.lowest, stop.trials.highest) {
                how.push_str(&format!(", entre {low:.4} y {high:.4}"));
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
                // Dicho cuando el padre está plegado. Un hash que apunta a una
                // fila que no se dibuja es un callejón para quien lee: o se
                // dice por qué no está, o parece que se ha perdido.
                let cut = walk
                    .stops
                    .iter()
                    .find(|one| one.commit == step.from)
                    .is_some_and(|one| one.pruned && !all_lines);
                format!(
                    "desde {}{} · ",
                    &step.from[..12.min(step.from.len())],
                    if cut { " (podado)" } else { "" }
                )
            }
        };
        println!("     │    {from}{}", stepped(&step.found));
        // Compact here, where the tree's shape is what carries the reading. A
        // block of versions would break the one column a walk has.
        if !step.drift.is_empty() {
            println!(
                "     │    ⚠ otro entorno: {}",
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
        0 => println!("Ningún paso deja resultados no comparables con los del anterior."),
        // Steps and not a sum of nodes: the same node counted at three steps is
        // one node looked at three times, and adding those said nothing.
        n => println!("{n} de los pasos dejan resultados NO comparables con los del anterior."),
    }
    restless == 0
}

/// One step of a walk, as a line: where the edit is, and what it left behind.
fn stepped(found: &Findings) -> String {
    if found.is_quiet() {
        return "sin cambios".into();
    }
    let edit = found.the_edit();
    // Said either way. A step where names moved and nobody typed anything is a
    // trial of the same variant, and leaving that to be inferred from the
    // absence of a word is how somebody reads it as an edit they forgot.
    let mut said = vec![match edit.is_empty() {
        true => "sin edición".to_string(),
        false => format!("edición: {}", edit.join(", ")),
    }];
    for (finding, called) in [
        (STALE, "⚠ RANCIO"),
        (SUSPECT, "⚠ sospechoso"),
        ("UNKNOWN", "no previsible"),
        ("UNVERSIONED", "sin versionar"),
        (RESETTLED, "repesado"),
        (SALTED, "otro salt"),
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
        said.push(format!("heredado: {}", found.saying(DOWNSTREAM).join(", ")));
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
        println!("  Nada que decir de ningún nodo.");
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
        println!("La edición está en: {}", edit.join(", "));
    }
    let stale = found.saying(STALE);
    match stale.is_empty() {
        true => println!("Ningún nodo recibirá un valor cacheado que ya no le corresponde."),
        false => println!(
            "⚠ {} nodo(s) con la MISMA clave y otro código: la caché dará HIT.",
            stale.len(),
        ),
    }
    let not_comparable = found.not_comparable().len();
    if not_comparable > 0 {
        println!("{not_comparable} nodo(s) con resultados NO comparables con los de antes.");
    }
    if !after.unneeded.is_empty() {
        println!(
            "Ya calculado, no haría falta ejecutar: {}",
            after.unneeded.join(", "),
        );
    }
}

/// Says so when the two sides were not built against the same thing.
///
/// A whole-graph fact and not a finding per node: an interpreter that moved
/// moved under all of them, and attributing it forty times would be noise
/// rather than an answer.
fn drifted(before: &Snapshot, after: &Snapshot) {
    let drift = before.drifted_from(after);
    if drift.is_empty() {
        return;
    }
    println!(
        "⚠ los dos lados se sondearon en entornos distintos, y eso no está en \
         ningún commit:"
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
        Some(verdict) => println!("{short} · {verdict} · dicho {nth}"),
        None => println!("{short} · nota · dicho {nth}"),
    }
    Ok(true)
}

/// Everything said about one commit, prose and all.
/// Los ensayos de una versión, y una curva si se pide.
/// Qué datos hay bajo cada versión, y cuáles no son de ninguna.
fn dataing(range: &str, most: usize, at: &Where) -> Result<bool, Box<dyn std::error::Error>> {
    let bench = Bench::set_up(&at.repo, at.store.as_deref(), at.given.as_deref())?;
    let Some(store) = at.store.as_deref() else {
        // Dicho entero: sin store no hay datos que atribuir, y una tabla vacía
        // se lee como «no hay nada guardado» en vez de como «no me has dicho
        // dónde mirar».
        println!("Sin `--store` no hay nada que mirar: lo que atribuye esto son los");
        println!("valores que una corrida dejó guardados, y el store es donde están.");
        return Ok(true);
    };
    let kept = soma_next_store::Local::at(store)?;
    let commits = revision::commits_in(&bench.repo, range, most)?;
    let probing = bench.probing(at.store.as_deref(), at.given.as_deref());
    let known = probed(&bench, &probing, &commits)?;
    let said = data::under(&kept, &known)?;

    if said.is_empty() {
        println!("El store no tiene ningún valor de una corrida todavía.");
        return Ok(true);
    }

    // Dicho antes de la tabla y no después. Sin `--given`, un sondeo nombra
    // contra un centinela, así que ninguna clave coincide con la de una corrida
    // sobre datos de verdad y **todo** se atribuye por la huella. La tabla sale
    // bien; lo que no sale es por qué no dice nunca «es la suya», y eso desde
    // fuera se lee como que la mitad del mecanismo no funciona.
    if known.values().any(|taken| taken.input == "sentinel") {
        println!("Sondeado contra un centinela, así que se atribuye por el código y no");
        println!("por el nombre: `--given` con la entrada real es lo que hace coincidir");
        println!("las claves.");
    }

    // Por versión, y lo huérfano al final: es lo último que se mira y lo
    // primero que se querría esconder.
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
                Some(data::How::Named) => "es la suya",
                // Dicho distinto porque **es** distinto: el código es el mismo
                // y el nombre no coincide, que casi siempre significa otra
                // entrada o otro entorno. Fundirlos en «es suyo» escondería
                // justo el caso que alguien está buscando.
                _ => "mismo código",
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
{} de ninguna versión de las miradas:",
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
        println!("Que no es lo mismo que sobrar: puede ser de una rama que no se ha");
        println!("mirado, de un commit que ya no está, o de un entorno que no se");
        println!("puede reproducir.");
    }
    Ok(true)
}

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

    println!("{}   ·   {} ensayos", trials.study(&commit), seen.len());
    if seen.is_empty() {
        println!(
            "\nNadie ha corrido nada con esta versión todavía. Se escriben desde \
             soma-next,\ncon `study=\"{}\"`.",
            trials.study(&commit)
        );
        return Ok(true);
    }
    println!();
    for one in &seen {
        let state = one.state.as_deref().unwrap_or("¿?");
        let score = match one.score {
            // Dicho y no callado: una puntuación podada es real y **no** es
            // comparable con una terminada — se midió tras menos épocas — y
            // ponerlas en la misma columna sin decirlo invita a compararlas.
            Some(score) if one.comparable() => format!("{score:>10.4}"),
            Some(score) => format!("{score:>10.4} (podado)"),
            None => format!("{:>10}", "—"),
        };
        let rescued = if one.attempt > 0 {
            format!("  ·  intento {}", one.attempt)
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
                "  El mejor de {} comparables: {:.4}",
                done.len(),
                goal.best_of(done.iter().copied()).unwrap_or(f64::NAN)
            ),
            // Y no «el mejor»: cuál lo es depende de si esa métrica se maximiza
            // o se minimiza, y esa dirección no está en ningún registro.
            None => println!(
                "  {} comparables, entre {:.4} y {:.4}. Declara `goal = \"min\"` o \
                 `goal = \"max\"`\n  en soma-tree.toml y te diré cuál es el mejor.",
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
            .ok_or_else(|| format!("no hay ensayo {which} en {}", trials.study(&commit)))?;
        match trials.curve(one)? {
            Some(drawn) => {
                println!("\n  Ensayo {which}, {} informes:", drawn.reports.len());
                for (n, value) in drawn.reports.iter().enumerate() {
                    println!("    {n:>4}  {value:.4}");
                }
                if let Some(because) = drawn.because {
                    println!("    paró porque {because}");
                }
            }
            None => println!("\n  El registro del ensayo {which} apunta a un blob que no está."),
        }
    }
    Ok(true)
}

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
        println!("\n  Nadie ha dicho nada de este commit.");
        return Ok(true);
    }
    // Here the prose is fetched, and only here: a scan is what the log pays
    // for, and one blob per saying is what reading them costs.
    for saying in &said {
        println!();
        match saying.verdict {
            Some(verdict) => println!("  [{}] {} · {}", saying.nth, verdict, saying.who),
            None => println!("  [{}] nota · {}", saying.nth, saying.who),
        }
        for line in journal.read(saying)?.lines() {
            println!("      {line}");
        }
    }
    Ok(true)
}

/// Listens, and hands the line to whoever asks for it.
fn serving(at: &str, where_: &Where) -> Result<bool, Box<dyn std::error::Error>> {
    // Read once and thrown away: it is only here so a misconfigured repository
    // is a message now rather than a 400 on the first request.
    Bench::set_up(
        &where_.repo,
        where_.store.as_deref(),
        where_.given.as_deref(),
    )?;
    let serving = Serving {
        repo: where_.repo.canonicalize()?,
        store: where_.store.clone(),
        given: where_.given.clone(),
    };

    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async {
        let listening = tokio::net::TcpListener::bind(at).await?;
        println!("soma-tree en http://{at}");
        println!("  GET  /api/walk?range=HEAD~10..HEAD");
        println!("  GET  /api/said/<rev>");
        println!("  POST /api/said/<rev>   {{\"verdict\": \"invalid\", \"prose\": \"...\"}}");
        axum::serve(listening, routes(serving)).await?;
        Ok::<_, Box<dyn std::error::Error>>(())
    })?;
    Ok(true)
}
