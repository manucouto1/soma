//! `soma-fabric-fleet` — serve the fleet, or print it.
//!
//! Two ways in for the same answer, which is the point: the terminal and the
//! browser read the same function, so they cannot come to disagree about which
//! machines are there.

use clap::{Parser, Subcommand};
use soma_fabric_fleet::{Fleet, Serving, routes, seed};
use soma_next_store::{Local, Store};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

#[derive(Parser)]
#[command(
    name = "soma-fabric-fleet",
    about = "Where the workers are, and what they are doing."
)]
struct Cli {
    #[command(subcommand)]
    what: What,
}

#[derive(Subcommand)]
enum What {
    /// Serve it over HTTP for whoever draws it.
    Serve {
        /// The store the workers report into.
        #[arg(long)]
        store: String,
        /// Where to listen.
        #[arg(long, default_value = "127.0.0.1:7380")]
        listen: String,
        /// After how many seconds without writing a machine is called quiet.
        ///
        /// Told and not derived: the store keeps one reading per machine and
        /// rewrites it, so there is no cadence in there to work out. Three
        /// times a common `--reporting 30`, so a worker that reports slowly is
        /// not called dead for it.
        #[arg(long, default_value_t = 90)]
        quiet_after: u64,
        /// How many records to read to learn what the graphs call these
        /// machines. The join's whole price, one fetch each.
        #[arg(long, default_value_t = 40)]
        records: usize,
        /// Where the listing lives. Held in a file until the local broker is
        /// written, which is what will hold it.
        #[arg(long, default_value = "listing.toml")]
        listing: PathBuf,
    },
    /// Print it once and leave.
    Now {
        /// The store the workers report into.
        #[arg(long)]
        store: String,
        /// As above.
        #[arg(long, default_value_t = 90)]
        quiet_after: u64,
        /// As above.
        #[arg(long, default_value_t = 40)]
        records: usize,
    },
    /// Writes a fleet to look at: machines in every state there is, a listing,
    /// and a run across two of them.
    ///
    /// What it writes is what a worker and a run write, through the store's own
    /// types — so the screens draw from it exactly what they draw from a
    /// cluster. It is a fixture and it says so; nothing else in this binary
    /// invents anything.
    Seed {
        /// Where to write it. Anything already there is left alone.
        #[arg(long)]
        store: String,
        /// And where to write the listing.
        #[arg(long, default_value = "listing.toml")]
        listing: PathBuf,
    },
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(why) => {
            eprintln!("{why}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<(), String> {
    match cli.what {
        What::Now {
            store,
            quiet_after,
            records,
        } => {
            let store = opened(&store)?;
            let fleet =
                Fleet::read(store.as_ref(), quiet_after, records).map_err(|why| why.to_string())?;
            println!(
                "{}",
                serde_json::to_string_pretty(&fleet).map_err(|why| why.to_string())?
            );
            Ok(())
        }
        What::Seed { store, listing } => {
            let sown = seed::sow(std::path::Path::new(&store), Some(&listing))
                .map_err(|why| why.to_string())?;
            println!(
                "{} máquinas y el run `{}` en `{store}`, {} nombres en `{}`.",
                sown.machines,
                sown.run,
                sown.names,
                listing.display()
            );
            println!(
                "Mirarlo: soma-fabric-fleet serve --store {store} --listing {}",
                listing.display()
            );
            Ok(())
        }
        What::Serve {
            store,
            listen,
            quiet_after,
            records,
            listing,
        } => {
            let serving = Serving {
                store: opened(&store)?,
                quiet_after,
                read_records: records,
                listing,
            };
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .map_err(|why| format!("there is no runtime to serve on: {why}"))?
                .block_on(async move {
                    let listener = tokio::net::TcpListener::bind(&listen)
                        .await
                        .map_err(|why| format!("nothing can listen on `{listen}`: {why}"))?;
                    eprintln!("the fleet is at http://{listen}");
                    axum::serve(listener, routes(serving))
                        .await
                        .map_err(|why| format!("it stopped serving: {why}"))
                })
        }
    }
}

/// The store, said the way somebody who mistyped a path needs to hear it.
fn opened(where_: &str) -> Result<Arc<dyn Store>, String> {
    Local::at(where_)
        .map(|one| Arc::new(one) as Arc<dyn Store>)
        .map_err(|why| format!("there is no store at `{where_}`: {why}"))
}
