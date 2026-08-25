//! The fleet over HTTP, for whoever draws it.
//!
//! axum 0.8 on tokio, which is not a taste: it is what somatize-tree runs and what
//! chatty-the-lab's backend runs, so landing there is moving routes rather than
//! rewriting a server. Mounted at the root here; under a prefix wherever this
//! goes.
//!
//! # Nothing is held between requests
//!
//! No cached scan, no open worker, no registry in memory. What makes that
//! affordable is that everything this answers is already written down — one
//! reading per machine, rewritten, and the record — so a second request is a
//! scan and not a conversation.
//!
//! It is also what leaves the platform's shape undecided: a handler that keeps
//! nothing is the same code as a module of one backend and as a service behind
//! a URL. The decision can be made later because it was not made here.
//!
//! # Everything here blocks, and none of it blocks the runtime
//!
//! Reading a store is a blocking call — a directory, or a round trip to a
//! bucket. On an async runtime that is not a slow handler, it is a **stalled
//! thread** nobody else's request can get past, so the work goes to
//! [`spawn_blocking`](tokio::task::spawn_blocking) and the async side only ever
//! hands back a result.

use crate::listing::{Listed, Listing, Trouble};
use crate::{Fleet, naming, ran, runs};
use axum::extract::{Path as At, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::json;
use somatize_store::Store;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

/// What every request needs and no request changes.
#[derive(Clone)]
pub struct Serving {
    /// Where the readings are.
    pub store: Arc<dyn Store>,
    /// After how many seconds without writing a machine is called quiet.
    ///
    /// **The server's rule and not the view's.** Written in two places it would
    /// live in two languages, and the day it changed a terminal and a browser
    /// would both look right and quietly disagree about which machines are
    /// there — which is the mistake somatize-tree wrote down about pruning.
    pub quiet_after: u64,
    /// How many records to read to learn what the graphs call these machines.
    pub read_records: usize,
    /// Where the listing is.
    ///
    /// A file, because what holds a listing beyond one client is the local
    /// broker and it is not written. The routes over it are the shape a broker
    /// will answer, so landing there is where the answer comes from and not what
    /// it looks like.
    pub listing: PathBuf,
}

/// The routes. Mounted at the root here; under a prefix wherever this lands.
pub fn routes(serving: Serving) -> Router {
    Router::new()
        .route("/api/fleet", get(fleet))
        .route("/api/listing", get(listing).post(list_one))
        .route("/api/listing/{host}", delete(unlist))
        .route("/api/runs", get(which_runs))
        .route("/api/ran", get(what_ran))
        .route("/api/health", get(|| async { "ok" }))
        // Wide open, because it serves somebody their own machines from their
        // own network. Whatever mounts this in front of anything else brings
        // its own policy.
        .layer(tower_http::cors::CorsLayer::permissive())
        .with_state(Arc::new(serving))
}

/// What a caller may narrow. Both default to the server's, which is what a
/// screen that has not asked for anything gets.
#[derive(Deserialize)]
struct Asked {
    /// Override the quiet bound, to argue with it without restarting.
    quiet_after: Option<u64>,
    /// How many records to read for the names. The join's whole price.
    records: Option<usize>,
    /// Which run to look at. The newest when nobody says.
    run: Option<String>,
    /// How many of its `forward`s to read.
    last: Option<usize>,
}

async fn fleet(State(serving): State<Arc<Serving>>, Query(asked): Query<Asked>) -> Response {
    let quiet_after = asked.quiet_after.unwrap_or(serving.quiet_after);
    let records = asked.records.unwrap_or(serving.read_records);
    let store = Arc::clone(&serving.store);
    blocking(move || Fleet::read(store.as_ref(), quiet_after, records)).await
}

/// The names, grouped by the wire they resolve to.
///
/// The join comes from the same place the fleet's does, so a name and a machine
/// that met on one screen have met on the other.
async fn listing(State(serving): State<Arc<Serving>>, Query(asked): Query<Asked>) -> Response {
    let records = asked.records.unwrap_or(serving.read_records);
    let (store, at) = (Arc::clone(&serving.store), serving.listing.clone());
    blocking(move || {
        let met = naming::names(store.as_ref(), records).map_err(|why| why.to_string())?;
        let paper = Listing::read(&at).map_err(|why| why.to_string())?;
        paper.wires(&met).map_err(|why| why.to_string())
    })
    .await
}

/// Adds a name. It starts nothing and connects to nothing — that is the whole
/// point of a listing, and it is why an unreachable host fails when it is needed
/// rather than when it is named.
async fn list_one(State(serving): State<Arc<Serving>>, Json(one): Json<Listed>) -> Response {
    let (store, at) = (Arc::clone(&serving.store), serving.listing.clone());
    let records = serving.read_records;
    blocking(move || {
        let mut paper = Listing::read(&at).map_err(said)?;
        paper.add(one).map_err(said)?;
        paper.write(&at).map_err(said)?;
        let met = naming::names(store.as_ref(), records).map_err(|why| why.to_string())?;
        paper.wires(&met).map_err(said)
    })
    .await
}

/// Drops a name. Nothing is disconnected: a name is not a connection.
async fn unlist(State(serving): State<Arc<Serving>>, At(host): At<String>) -> Response {
    let (store, at) = (Arc::clone(&serving.store), serving.listing.clone());
    let records = serving.read_records;
    blocking(move || {
        let mut paper = Listing::read(&at).map_err(said)?;
        if !paper.drop(&host) {
            return Err(format!("`{host}` was not listed"));
        }
        paper.write(&at).map_err(said)?;
        let met = naming::names(store.as_ref(), records).map_err(|why| why.to_string())?;
        paper.wires(&met).map_err(said)
    })
    .await
}

/// Every run this store has a record of. One scan and no fetches.
async fn which_runs(State(serving): State<Arc<Serving>>) -> Response {
    let store = Arc::clone(&serving.store);
    blocking(move || runs(store.as_ref()).map_err(|why| why.to_string())).await
}

/// One run, by where its work happened. Without `run`, the newest one there is.
async fn what_ran(State(serving): State<Arc<Serving>>, Query(asked): Query<Asked>) -> Response {
    let store = Arc::clone(&serving.store);
    let last = asked.last.unwrap_or(serving.read_records);
    blocking(move || {
        let which = match asked.run {
            Some(named) => named,
            None => runs(store.as_ref())
                .map_err(|why| why.to_string())?
                .pop()
                .map(|one| one.run)
                .ok_or_else(|| "no hay ningún run escrito en este store".to_string())?,
        };
        ran(store.as_ref(), &which, last).map_err(|why| why.to_string())
    })
    .await
}

/// A listing's own complaint, which already reads as a sentence.
fn said(why: Trouble) -> String {
    why.to_string()
}

/// Runs the work off the runtime and answers with it.
///
/// A failure is the store's and is said as the store said it: *the store could
/// not be reached* is actionable and *500* is not.
async fn blocking<T, E, F>(work: F) -> Response
where
    T: serde::Serialize + Send + 'static,
    E: fmt::Display + Send + 'static,
    F: FnOnce() -> Result<T, E> + Send + 'static,
{
    match tokio::task::spawn_blocking(work).await {
        Ok(Ok(answer)) => Json(answer).into_response(),
        Ok(Err(why)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": why.to_string() })),
        )
            .into_response(),
        // The thread carrying it went away. Saying so beats a timeout.
        Err(why) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("that answer was never finished: {why}") })),
        )
            .into_response(),
    }
}
