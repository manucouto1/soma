//! The line of exploration, over HTTP, for whoever draws it.
//!
//! axum 0.8 on tokio, which is not a taste: `chatty-the-lab`'s backend is on
//! exactly those and its integration plan already names the panel this feeds,
//! so landing there should be moving routes rather than rewriting a server.
//!
//! Everything here blocks and none of it blocks the runtime. Answering one
//! request runs `git`, sometimes a Python interpreter, and a scan of a store,
//! and on an async runtime that is not a slow handler but a **stalled
//! executor** — so the work goes to
//! [`spawn_blocking`](tokio::task::spawn_blocking) and the async side only
//! hands over a result.
//!
//! Nothing is held between requests: no cached walk, no open repository. The
//! expensive half — a probe of a commit — is already remembered in the store,
//! content-addressed and for ever, so a second request is a scan and not an
//! interpreter. State here would buy little and would have to be invalidated
//! by somebody committing, which is a thing this cannot see.

use crate::bench::{Bench, journalling, probed, walking};
use crate::journal::{Journal, Verdict};
use crate::moves::{Cited, Course, Kind, MoveId, Said as Verb, Says, Scope};
use crate::revision;
use crate::trials::Trials;
use axum::extract::{Path as At, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::json;
use somatize_store::{Digest, Store};
use std::path::PathBuf;
use std::sync::Arc;

/// What every request needs and no request changes.
#[derive(Clone)]
pub struct Serving {
    pub repo: PathBuf,
    pub store: Option<PathBuf>,
    pub given: Option<PathBuf>,
}

/// The routes. Mounted at the root here; under a prefix wherever this lands.
pub fn routes(serving: Serving) -> Router {
    Router::new()
        .route("/api/walk", get(walk))
        .route("/api/said/{commit}", get(said).post(say))
        .route("/api/graph/{commit}", get(graph))
        .route("/api/file/{commit}", get(file))
        .route("/api/trials/{commit}", get(trials))
        .route("/api/trials/{commit}/{trial}", get(curve))
        .route("/api/fork", post(fork))
        .route("/api/check", post(check))
        .route("/api/format", post(prettify))
        .route("/api/moves", get(moves).post(add_move))
        .route("/api/moves/{id}", axum::routing::patch(reword))
        .route("/api/moves/{id}/under/{parent}", post(hang))
        .route("/api/moves/{id}/cites", post(cite))
        .route("/api/kept", post(keep))
        .route("/api/kept/{digest}", get(read_kept))
        .route("/api/moves/says", post(speak))
        .route("/api/health", get(|| async { "ok" }))
        // Wide open, because it serves a line of somebody's own exploration
        // from their own machine. Whatever mounts this in front of a network
        // brings its own policy.
        .layer(tower_http::cors::CorsLayer::permissive())
        .with_state(Arc::new(serving))
}

#[derive(Deserialize)]
struct Asked {
    /// A range git understands, a revspec meaning the history back from there,
    /// or `--all`. Every branch when nobody says: three variants of one idea
    /// are three branches, and a walk from one tip cannot see its siblings.
    /// Never a range by default, either — one asking for ten commits is an
    /// error and not an answer in a repository with four.
    #[serde(default = "from_the_top")]
    range: String,
    /// How far back, when what was asked is not a range.
    #[serde(default = "ten")]
    most: usize,
}

fn from_the_top() -> String {
    crate::revision::ALL.to_string()
}

fn ten() -> usize {
    10
}

/// The whole line: its stops, their verdicts, and what each step did.
async fn walk(State(serving): State<Arc<Serving>>, Query(asked): Query<Asked>) -> Response {
    // `git`, a Python subprocess and a store scan, none of which an async
    // runtime can wait on without stalling a thread somebody else needs.
    blocking(move || {
        walking(
            &serving.repo,
            serving.store.as_deref(),
            serving.given.as_deref(),
            &asked.range,
            asked.most,
        )
    })
    .await
}

/// The compute graph at one commit: its shape, and the class behind each node.
///
/// Its own route because it is asked for **on selection** and not for the whole
/// walk. Forty commits' worth of node sources would be most of the answer and
/// none of it read.
async fn graph(State(serving): State<Arc<Serving>>, At(commit): At<String>) -> Response {
    blocking(move || {
        let bench = Bench::set_up(
            &serving.repo,
            serving.store.as_deref(),
            serving.given.as_deref(),
        )?;
        let probing = bench.probing(serving.store.as_deref(), serving.given.as_deref());
        let commit = revision::named(&serving.repo, &commit)?;
        let known = probed(&bench, &probing, std::slice::from_ref(&commit))?;
        let taken = &known[commit.as_str()];
        Ok(json!({
            "commit": commit,
            "shape": taken.snapshot.get("shape"),
            // The edges go apart from `shape` even though `shape` carries
            // each node's parents: they are what says what feeds what, and a
            // figure without them lies the moment a graph stops being a chain.
            "edges": taken.snapshot.get("edges"),
            "declared": taken.snapshot.get("declared"),
            "architecture": taken.architecture,
            "inside": taken.inside,
            "declaring": taken.declaring,
            "code": taken.code,
            "reaches": taken.reaches,
        }))
    })
    .await
}

/// One file as that commit had it.
///
/// Apart from `graph` for the same reason `graph` is apart from `walk`: it is
/// asked for **on opening** and not before. A node reaches four files, a graph
/// forty nodes and a walk forty commits, so putting the content in the graph's
/// answer would send nearly all of it for none of it to be read.
///
/// `git show` and not a read off the disk: what is shown is the file **at that
/// commit**, and it does not even need a worktree, which is the expensive part.
async fn file(
    State(serving): State<Arc<Serving>>,
    At(commit): At<String>,
    Query(asked): Query<Reading>,
) -> Response {
    blocking(move || {
        let commit = revision::named(&serving.repo, &commit)?;
        Ok(json!({
            "commit": commit,
            "path": asked.path,
            "source": revision::read(&serving.repo, &commit, &asked.path)?,
        }))
    })
    .await
}

/// Which file, of the ones a node reaches.
#[derive(Deserialize)]
struct Reading {
    path: String,
}

/// Whether an edit survives, before anybody commits it.
///
/// Four questions, cheapest first: it parses, a linter is quiet, the graph
/// still builds, and the node runs on the value its predecessors actually left
/// in the store. Asked in a **detached** checkout — finding out that an edit
/// does not work should not leave a branch behind saying it did.
async fn check(State(serving): State<Arc<Serving>>, Json(asked): Json<Checking>) -> Response {
    blocking(move || {
        let bench = Bench::set_up(
            &serving.repo,
            serving.store.as_deref(),
            serving.given.as_deref(),
        )?;
        let from = revision::named(&serving.repo, &asked.from)?;
        let (_held, working) = revision::laid_out(
            &serving.repo,
            &from,
            None,
            &asked.file,
            revision::Splice {
                line: asked.line,
                lines: asked.lines,
            },
            &asked.source,
        )?;
        let probing = bench.probing(serving.store.as_deref(), serving.given.as_deref());
        let said = probing.checked(&working, &asked.node);
        // The worktree goes whatever the answer was.
        let _ = revision::forget(&serving.repo, &working);
        Ok(said?)
    })
    .await
}

/// The whole reasoning: the moves, who hangs off whom, what has been said, and
/// how each question and hypothesis stands.
///
/// One answer and not four routes, because none of it is readable alone: a
/// standing is a count over the sayings, and a scope is a walk over the
/// hanging. Handing them out separately would be asking whoever draws it to
/// reassemble the model, and to keep up when it changes.
async fn moves(State(serving): State<Arc<Serving>>) -> Response {
    blocking(move || {
        let bench = Bench::set_up(
            &serving.repo,
            serving.store.as_deref(),
            serving.given.as_deref(),
        )?;
        let moves = bench.moves();
        let under = moves.under()?;
        let all = moves.all()?;
        Ok(json!({
            "moves": all.values().collect::<Vec<_>>(),
            "under": all
                .keys()
                .map(|id| (id.to_string(), json!(under.parents_of(*id))))
                .collect::<serde_json::Map<_, _>>(),
            "says": moves.says()?,
            "standing": moves
                .standing()?
                .into_iter()
                .map(|(id, how)| (id.to_string(), json!(how)))
                .collect::<serde_json::Map<_, _>>(),
        }))
    })
    .await
}

async fn add_move(State(serving): State<Arc<Serving>>, Json(asked): Json<Adding>) -> Response {
    blocking(move || {
        let bench = Bench::set_up(
            &serving.repo,
            serving.store.as_deref(),
            serving.given.as_deref(),
        )?;
        let kind = Kind::read(&asked.kind)
            .ok_or_else(|| format!("`{}` is not a kind of move", asked.kind))?;
        let course =
            match asked.course.as_deref() {
                None => None,
                Some(said) => Some(Course::read(said).ok_or_else(|| {
                    format!("`{said}` is not a course: pursue, abandon, superseded")
                })?),
            };
        let who = asked
            .who
            .clone()
            .unwrap_or_else(|| revision::whoami(&serving.repo));
        let id = bench.moves().add(
            kind,
            &asked.prose,
            &who,
            Scope::of(asked.scope.clone()),
            asked.cites.clone(),
            course,
        )?;
        for parent in &asked.under {
            bench.moves().hang(id, *parent)?;
        }
        Ok(json!({"id": id}))
    })
    .await
}

async fn reword(
    State(serving): State<Arc<Serving>>,
    At(id): At<MoveId>,
    Json(asked): Json<Rewording>,
) -> Response {
    blocking(move || {
        let bench = Bench::set_up(
            &serving.repo,
            serving.store.as_deref(),
            serving.given.as_deref(),
        )?;
        let who = asked
            .who
            .clone()
            .unwrap_or_else(|| revision::whoami(&serving.repo));
        let course =
            match asked.course.as_deref() {
                None => None,
                Some(said) => Some(Course::read(said).ok_or_else(|| {
                    format!("`{said}` is not a course: pursue, abandon, superseded")
                })?),
            };
        let nth = bench.moves().reword(
            id,
            asked.prose.as_deref(),
            asked.scope.clone().map(Scope::of),
            course,
            &who,
        )?;
        Ok(json!({"nth": nth}))
    })
    .await
}

async fn hang(
    State(serving): State<Arc<Serving>>,
    At((id, parent)): At<(MoveId, MoveId)>,
) -> Response {
    blocking(move || {
        let bench = Bench::set_up(
            &serving.repo,
            serving.store.as_deref(),
            serving.given.as_deref(),
        )?;
        bench.moves().hang(id, parent)?;
        Ok(json!({"id": id, "under": parent}))
    })
    .await
}

/// Keeps a text and says where to find it again.
///
/// What makes a **configuration** citable, which is the half of an experiment
/// git does not have: `run_experiment.py --decorr-weight 0.1` runs the same
/// commit as `--decorr-weight 0.5` and is another experiment, and the resolved
/// invocation is in no git tree.
///
/// Generic and not `POST /api/config`, for the same reason `Cited.what` is
/// open: what somebody wants tied to an attempt is their vocabulary, and this
/// layer keeps it without learning it.
async fn keep(State(serving): State<Arc<Serving>>, body: String) -> Response {
    blocking(move || {
        let bench = Bench::set_up(
            &serving.repo,
            serving.store.as_deref(),
            serving.given.as_deref(),
        )?;
        // Content-addressed, so keeping the same thing twice is the same: two
        // attempts with one configuration cite a single blob, and that they
        // ran the same thing shows without comparing anything.
        let digest = bench
            .remembering
            .put(body.as_bytes())
            .map_err(|why| why.to_string())?;
        Ok(json!({"digest": digest.to_string()}))
    })
    .await
}

/// What is under a digest. A fetch, and so only when asked for.
async fn read_kept(State(serving): State<Arc<Serving>>, At(digest): At<String>) -> Response {
    blocking(move || {
        let bench = Bench::set_up(
            &serving.repo,
            serving.store.as_deref(),
            serving.given.as_deref(),
        )?;
        let digest = Digest::parse(digest);
        let bytes = bench
            .remembering
            .get(&digest)
            .map_err(|why| why.to_string())?
            .ok_or("this store does not have that")?;
        Ok(json!({"text": String::from_utf8_lossy(&bytes)}))
    })
    .await
}

/// Adds a piece of evidence to a move: the trial that ran, the artifact that
/// came out. After it was written, because the trials run afterwards.
async fn cite(
    State(serving): State<Arc<Serving>>,
    At(id): At<MoveId>,
    Json(asked): Json<Cited>,
) -> Response {
    blocking(move || {
        let bench = Bench::set_up(
            &serving.repo,
            serving.store.as_deref(),
            serving.given.as_deref(),
        )?;
        let who = revision::whoami(&serving.repo);
        let nth = bench.moves().cite(id, asked, &who)?;
        Ok(json!({"id": id, "nth": nth}))
    })
    .await
}

async fn speak(State(serving): State<Arc<Serving>>, Json(asked): Json<Speaking>) -> Response {
    blocking(move || {
        let bench = Bench::set_up(
            &serving.repo,
            serving.store.as_deref(),
            serving.given.as_deref(),
        )?;
        let says = Says::read(&asked.says)
            .ok_or_else(|| format!("`{}` is not something that can be said", asked.says))?;
        bench.moves().say(Verb {
            from: asked.from,
            to: asked.to,
            says,
            scope: Scope::of(asked.scope.clone()),
            in_part: asked.in_part,
        })?;
        Ok(json!({"from": asked.from, "to": asked.to, "says": asked.says}))
    })
    .await
}

/// Leaves an attempt for the commit just made, hung where somebody was already
/// looking.
///
/// Under whatever the attempt citing the starting commit hung under: forking is
/// carrying on exploring the same question. If that commit had no attempt this
/// one is left **loose** rather than inventing a question for it — a node under
/// something nobody asked is worse than one waiting for a place.
///
/// None of its troubles interrupt a fork: the branch exists, and not being able
/// to annotate it is no reason to pretend it was not created.
fn placed(bench: &Bench, from: &str, made: &str, branch: &str) -> Option<MoveId> {
    let moves = bench.moves();
    let known = moves.all().ok()?;
    let before = known.values().find(|one| {
        one.kind == Kind::Attempt
            && one
                .cites
                .iter()
                .any(|cited| cited.what == "commit" && cited.id == from)
    });

    let id = moves
        .add(
            Kind::Attempt,
            branch,
            &revision::whoami(&bench.repo),
            Scope::everything(),
            vec![Cited {
                what: "commit".into(),
                id: made.to_string(),
            }],
            None,
        )
        .ok()?;
    if let Some(before) = before {
        for parent in moves.under().ok()?.parents_of(before.id) {
            let _ = moves.hang(id, parent);
        }
    }
    Some(id)
}

#[derive(Deserialize)]
struct Adding {
    kind: String,
    prose: String,
    /// The roots of what it covers. Empty is everything, which is what makes a
    /// general question general.
    #[serde(default)]
    scope: Vec<MoveId>,
    #[serde(default)]
    under: Vec<MoveId>,
    #[serde(default)]
    cites: Vec<Cited>,
    /// Only a decision carries one: `pursue`, `abandon`, `superseded`.
    #[serde(default)]
    course: Option<String>,
    #[serde(default)]
    who: Option<String>,
}

/// What is corrected of a move. What does not arrive stays as it was, which is
/// what stops rewriting the prose from wiping the scope.
#[derive(Deserialize)]
struct Rewording {
    #[serde(default)]
    prose: Option<String>,
    #[serde(default)]
    scope: Option<Vec<MoveId>>,
    #[serde(default)]
    course: Option<String>,
    #[serde(default)]
    who: Option<String>,
}

#[derive(Deserialize)]
struct Speaking {
    from: MoveId,
    to: MoveId,
    says: String,
    #[serde(default)]
    scope: Vec<MoveId>,
    #[serde(default)]
    in_part: bool,
}

/// The same source, formatted. No checkout: it is text in and text out.
async fn prettify(State(serving): State<Arc<Serving>>, Json(asked): Json<Prettifying>) -> Response {
    blocking(move || {
        let bench = Bench::set_up(
            &serving.repo,
            serving.store.as_deref(),
            serving.given.as_deref(),
        )?;
        Ok(bench
            .probing(serving.store.as_deref(), serving.given.as_deref())
            .prettified(&asked.source)?)
    })
    .await
}

#[derive(Deserialize)]
struct Prettifying {
    source: String,
}

/// Cuts a variant from a commit: a branch, one file rewritten, one commit.
///
/// **Editing is forking.** A commit has already been measured, so nothing here
/// changes one; what somebody wants when they edit a node is the next variant,
/// and this is it.
async fn fork(State(serving): State<Arc<Serving>>, Json(asked): Json<Forking>) -> Response {
    blocking(move || {
        let from = revision::named(&serving.repo, &asked.from)?;
        let said = match asked.message.trim().is_empty() {
            true => format!("variant from {}", &from[..12.min(from.len())]),
            false => asked.message.trim().to_string(),
        };
        let made = revision::forked(
            &serving.repo,
            &from,
            &asked.branch,
            &asked.file,
            revision::Splice {
                line: asked.line,
                lines: asked.lines,
            },
            &asked.source,
            &said,
        )?;

        // What the fork **did**, said back at once. A `STALE` here is not a
        // complaint about the edit: it is the one thing somebody has to know
        // before running it, because the cache will hit and hand back what the
        // old code produced. Worked out now rather than left for them to
        // notice in a row of the walk.
        let bench = Bench::set_up(
            &serving.repo,
            serving.store.as_deref(),
            serving.given.as_deref(),
        )?;
        let probing = bench.probing(serving.store.as_deref(), serving.given.as_deref());
        let both = [from.clone(), made.clone()];
        let stale = probed(&bench, &probing, &both)
            .and_then(|known| Ok(probing.compared(&known, &[(from.clone(), made.clone())])?))
            .map(|found| {
                found
                    .first()
                    .map(|one| {
                        one.saying(crate::findings::STALE)
                            .into_iter()
                            .map(str::to_string)
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default()
            })
            // Not knowing costs nothing here: the walk says it a second later.
            .unwrap_or_default();

        // The skeleton is generated, the content is written. Forking **is** an
        // attempt, so the tool puts the move down and what is left to write is
        // the only part a machine cannot: what was seen and what is decided. A
        // lab notebook you have to remember to open is one nobody keeps.
        let attempt = placed(&bench, &from, &made, &asked.branch);

        Ok(json!({
            "from": from,
            "branch": asked.branch,
            "commit": made,
            "stale": stale,
            "attempt": attempt,
        }))
    })
    .await
}

/// An edit, asked about rather than committed.
#[derive(Deserialize)]
struct Checking {
    from: String,
    /// Which node of the graph the edited class is behind — what gets run.
    node: String,
    file: String,
    line: u32,
    lines: u32,
    source: String,
}

#[derive(Deserialize)]
struct Forking {
    from: String,
    branch: String,
    file: String,
    /// Which lines the class occupies, so the rest of the file survives.
    line: u32,
    lines: u32,
    source: String,
    #[serde(default)]
    message: String,
}

/// Everything anybody said about one commit, prose included.
async fn said(State(serving): State<Arc<Serving>>, At(commit): At<String>) -> Response {
    blocking(move || {
        let (tree, kept) = journalling(&serving.repo, serving.store.as_deref())?;
        let commit = revision::named(&serving.repo, &commit)?;
        let journal = Journal::of(tree, &kept);
        let said: Vec<Said> = journal
            .all()?
            .into_iter()
            .filter(|saying| saying.commit == commit)
            .map(|saying| {
                Ok(Said {
                    nth: saying.nth,
                    verdict: saying.verdict,
                    who: saying.who.clone(),
                    when: saying.when,
                    // Here the prose is fetched, and only here: a walk pays for
                    // a scan, and reading the words costs a blob apiece.
                    prose: journal.read(&saying)?,
                })
            })
            .collect::<Result<_, Box<dyn std::error::Error>>>()?;
        Ok(json!({"commit": commit, "said": said}))
    })
    .await
}

/// What was run with this version. **One scan and no fetches**: each trial's
/// curve is not here, because it grows and this is a list.
async fn trials(State(serving): State<Arc<Serving>>, At(commit): At<String>) -> Response {
    blocking(move || {
        let bench = Bench::set_up(
            &serving.repo,
            serving.store.as_deref(),
            serving.given.as_deref(),
        )?;
        let commit = revision::named(&serving.repo, &commit)?;
        let tree = bench.config.tree(&bench.repo);
        let trials = Trials::of(&tree, &bench.remembering).towards(bench.config.towards()?);
        Ok(json!({
            "study": trials.study(&commit),
            "goal": bench.config.goal,
            "trials": trials.of_commit(&commit)?,
        }))
    })
    .await
}

/// A trial's curve, which is what costs a fetch and is why it is apart from
/// the list.
async fn curve(
    State(serving): State<Arc<Serving>>,
    At((commit, trial)): At<(String, u32)>,
) -> Response {
    blocking(move || {
        let bench = Bench::set_up(
            &serving.repo,
            serving.store.as_deref(),
            serving.given.as_deref(),
        )?;
        let commit = revision::named(&serving.repo, &commit)?;
        let tree = bench.config.tree(&bench.repo);
        let trials = Trials::of(&tree, &bench.remembering);
        let seen = trials.of_commit(&commit)?;
        let which = seen
            .iter()
            .find(|one| one.trial == trial)
            .ok_or_else(|| format!("there is no trial {trial} in {}", trials.study(&commit)))?;
        Ok(json!({"trial": which, "curve": trials.curve(which)?}))
    })
    .await
}

/// Writes one thing down: a note, a verdict, or a verdict with its reason.
async fn say(
    State(serving): State<Arc<Serving>>,
    At(commit): At<String>,
    Json(saying): Json<Saying>,
) -> Response {
    blocking(move || {
        let (tree, kept) = journalling(&serving.repo, serving.store.as_deref())?;
        let commit = revision::named(&serving.repo, &commit)?;
        let verdict = match &saying.verdict {
            None => None,
            Some(said) => Some(Verdict::read(said).ok_or_else(|| match said.as_str() {
                "promising" | "dead-end" | "superseded" => format!(
                    "`{said}` is no longer a verdict: it was a decision about where to go \
                     next, and it is written in the reasoning with its scope and its reason"
                ),
                _ => format!("`{said}` is not one: invalid, sound"),
            })?),
        };
        if saying.prose.trim().is_empty() && verdict.is_none() {
            return Err("a note with nothing in it says nothing".into());
        }
        let who = saying
            .who
            .clone()
            .unwrap_or_else(|| revision::whoami(&serving.repo));
        let nth = Journal::of(tree, &kept).say(&commit, verdict, &who, saying.prose.trim())?;
        Ok(json!({"commit": commit, "nth": nth}))
    })
    .await
}

#[derive(Serialize)]
struct Said {
    nth: u32,
    verdict: Option<Verdict>,
    who: String,
    when: u64,
    prose: String,
}

#[derive(Deserialize)]
struct Saying {
    /// One of the verdicts, or absent for a note that judges nothing.
    #[serde(default)]
    verdict: Option<String>,
    #[serde(default)]
    prose: String,
    /// Who, when whoever is asking knows better than git config does.
    #[serde(default)]
    who: Option<String>,
}

/// Runs blocking work off the runtime and turns whatever it says into a
/// response.
///
/// One place, so no handler forgets: a `git` call on an async thread is a
/// request nobody else's request can get past.
async fn blocking<T, F>(work: F) -> Response
where
    T: Serialize + Send + 'static,
    F: FnOnce() -> Result<T, Box<dyn std::error::Error>> + Send + 'static,
{
    // Flattened to text before it crosses the thread: a `Box<dyn Error>` is not
    // `Send`, and what a response carries is the sentence anyway.
    let work = move || work().map_err(|why| why.to_string());
    match tokio::task::spawn_blocking(work).await {
        Ok(Ok(said)) => Json(said).into_response(),
        // Said out loud rather than as a bare 500: what goes wrong here is
        // usually a revspec nobody has or an interpreter that cannot import
        // soma, and both are worth reading.
        Ok(Err(why)) => (StatusCode::BAD_REQUEST, Json(json!({"trouble": why}))).into_response(),
        Err(why) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"trouble": format!("the work did not finish: {why}")})),
        )
            .into_response(),
    }
}
