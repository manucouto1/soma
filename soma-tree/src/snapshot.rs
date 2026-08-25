//! What a graph was, at one commit. The probe's answer, typed.
//!
//! Nothing here holds a graph — a graph is Python and lives for the length of a
//! subprocess. What crosses back is this, and the fields are the probe's to
//! add: `python/soma_tree_probe.py` is the contract, and this is one reader of
//! it.

use crate::findings::Findings;
use serde::Deserialize;
use somatize_store::{Digest, Meta, Store};
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::path::Path;
use std::process::Command;

/// A graph as one checkout had it.
#[derive(Debug, Deserialize, serde::Serialize)]
pub struct Snapshot {
    /// What checkout this was, for saying so afterwards.
    pub commit: String,
    /// The `module:function` that built it.
    pub built_from: String,
    /// `"sentinel"` when no real input was hashed. Two snapshots are only
    /// comparable if they were taken the same way: the names come out of the
    /// snapshot, so one taken with an input against one taken without has
    /// **everything** moved and nothing saying why.
    pub input: String,
    /// What the graph was built against: the interpreter, and the version of
    /// every distribution it reached for.
    ///
    /// The axis git does not cover — a checkout pins its own code and not the
    /// interpreter outside it — and deliberately **not** part of the recipe a
    /// snapshot is remembered under. Two probes months apart are meant to
    /// disagree here out loud.
    #[serde(default)]
    pub environment: BTreeMap<String, String>,
    /// `foreseen.snapshot`'s own answer, carried **opaque**.
    ///
    /// Never read on this side and never reshaped: what a name is made of is
    /// the model's business, and a reader here that understood its insides
    /// would be a second model with a delay on it.
    pub snapshot: serde_json::Value,
    /// The class behind each node: where it lives and what it says.
    ///
    /// Read while the graph existed, because a snapshot outlives the process
    /// that made it — and the checkout it was read from is a worktree that was
    /// removed minutes later.
    #[serde(default)]
    pub code: BTreeMap<String, Written>,
    /// De qué está hecho cada nodo por dentro: `{nodo: [pieza, ...]}`.
    ///
    /// Un nodo es una caja y lo que lleva dentro suele ser de lo que va el
    /// experimento — `Pure` es un envoltorio y el enrutador que tiene dentro es
    /// la pieza. Dibujar la caja y callarse lo de dentro es dibujar el
    /// envoltorio.
    ///
    /// Leído sin correr nada, así que es la composición **declarada**: lo que
    /// `__init__` construyó. `somatize.torch.architecture` responde mejor
    /// —ve hasta lo que no es un módulo— y para eso ejecuta el grafo, que es
    /// justo lo que este lado no hace nunca.
    ///
    /// Opaco, como el resto: el vocabulario es de quien lo escribió.
    #[serde(default)]
    pub inside: serde_json::Value,
    /// De qué **ficheros** está hecho cada nodo, y dónde para la cuenta.
    ///
    /// `code` enseña una clase, que es lo que quiere quien pincha un nodo: un
    /// nodo **es** su clase. Pero una red se escribe muchas veces en cuatro
    /// módulos que se juntan en un `__init__`, y de esos cuatro
    /// `inspect.getsourcefile` sólo sabe uno. Los otros tres no estaban en
    /// ninguna parte de la respuesta.
    ///
    /// No es un segundo modelo de qué depende de qué: es el cierre transitivo
    /// que la huella de soma ya recorría para hashearlo, dicho en voz
    /// alta. Si cambia lo que entra en una huella, cambia esto con ella.
    ///
    /// **Sin fuente dentro**, a propósito: cuarenta commits de ficheros
    /// enteros son casi toda la respuesta y nada de ella leída. Lo que hay son
    /// rutas, y el contenido se pide por la suya al abrir uno.
    ///
    /// Opaco como `inside` y por lo mismo: el vocabulario es de quien lo
    /// escribió, y el contrato está en `python/soma_tree_probe.py`.
    #[serde(default)]
    pub reaches: serde_json::Value,
    /// Los cinco hechos ortogonales de un grafo aparte de qué calcula cada
    /// nodo: quién lo implementa, dónde corre, en qué dispositivo, qué se
    /// guarda, qué está congelado, y en qué orden correría.
    ///
    /// Opaco igual que `snapshot`, y por lo mismo: el vocabulario es de
    /// soma, y un lector de este lado que lo entendiera sería un segundo
    /// modelo con retraso. Lo único que hace falta aquí es que llegue entero a
    /// quien dibuja.
    #[serde(default)]
    pub architecture: serde_json::Value,
    /// El código que **declara** el grafo: el cuerpo de `build`, con los `>>`
    /// y los `|`.
    ///
    /// Es lo único de un grafo que no se puede leer nodo a nodo. Cada clase
    /// dice qué hace y ninguna dice cómo se conectan, así que sin esto la
    /// topología sólo se ve dibujada y nunca escrita — y quien la va a editar
    /// tiene que ir a buscarla al repositorio.
    ///
    /// `None` cuando no hay fuente que leer, que es la misma ausencia que
    /// `UNVERSIONED` nombra un nivel más abajo.
    #[serde(default)]
    pub declaring: Option<Written>,
    /// The nodes named by the content of their items, which nobody has before
    /// a run. Carried so a report can say *cannot tell* out loud.
    #[serde(default)]
    pub mapped: Vec<String>,
    /// What would not have to run at all, because something under it is kept.
    /// Empty without a real store, and that is not the same as "nothing".
    #[serde(default)]
    pub unneeded: Vec<String>,
}

impl Snapshot {
    /// What each node's answer will be called, `{nodo: clave}`.
    ///
    /// # Leer dentro de lo opaco, dos campos y sólo dos
    ///
    /// La regla de aquí es que `snapshot` no se abre: de qué está hecho un
    /// nombre es cosa del modelo, y un lector de este lado que lo entendiera
    /// sería un segundo modelo con retraso. Esto no lo hace. Usar un nombre
    /// **como nombre** —buscarlo en un store, ver si está— no es entender de
    /// qué está hecho, y es justamente lo que el modelo publica esto para.
    ///
    /// La línea es esa: si algún día hiciera falta *descomponer* una clave para
    /// sacar algo de dentro, eso sí sería la otra cosa, y la respuesta estaría
    /// en `foreseen` y no aquí.
    ///
    /// Faltan nodos y no es un olvido: un `.mapped()` se nombra por el
    /// contenido de sus items, que nadie tiene antes de correr. Esa ausencia se
    /// lee «no se puede saber» y nunca «no hay datos».
    pub fn names(&self) -> BTreeMap<String, String> {
        read_map(&self.snapshot, "names")
    }

    /// Qué versión del código tenía cada nodo, `{nodo: huella}`.
    ///
    /// El otro lado de la atribución, y el que aguanta lo que el primero no.
    /// Una clave se calcula contra el entorno del intérprete que sondea, así
    /// que sondear hoy un commit de hace tres meses da otras claves y no casan
    /// con lo que se guardó entonces. La huella la escribió quien corrió, al
    /// lado del valor, y sigue ahí.
    pub fn fingerprints(&self) -> BTreeMap<String, String> {
        read_map(&self.snapshot, "fingerprints")
    }

    /// What was built differently around the two of them: name, before, after.
    ///
    /// Usually nothing, because two commits probed in one sitting share an
    /// interpreter. It is a cached probe from months ago against a fresh one
    /// that answers something here — which is the whole reason the environment
    /// is left out of what a snapshot is remembered under.
    pub fn drifted_from<'a>(&'a self, other: &'a Self) -> Vec<(&'a str, String, String)> {
        let absent = "—".to_string();
        let mut said: Vec<(&str, String, String)> = Vec::new();
        for name in self.environment.keys().chain(other.environment.keys()) {
            let (was, is) = (self.environment.get(name), other.environment.get(name));
            if was != is && !said.iter().any(|(said, _, _)| said == name) {
                said.push((
                    name.as_str(),
                    was.cloned().unwrap_or(absent.clone()),
                    is.cloned().unwrap_or(absent.clone()),
                ));
            }
        }
        said
    }
}

/// Un `{texto: texto}` de dentro de la respuesta del modelo, o nada.
///
/// Nada y no un fallo: un sondeo viejo, guardado antes de que el modelo
/// publicara este campo, sigue siendo una respuesta buena a todo lo demás. Que
/// se caiga por leer algo que no le pedimos entonces sería tirar el registro de
/// una investigación por una función que se añadió después.
fn read_map(said: &serde_json::Value, what: &str) -> BTreeMap<String, String> {
    said.get(what)
        .and_then(|found| found.as_object())
        .map(|found| {
            found
                .iter()
                .filter_map(|(node, told)| Some((node.clone(), told.as_str()?.to_string())))
                .collect()
        })
        .unwrap_or_default()
}

/// One node's class, as it was at that commit.
#[derive(Debug, Deserialize, serde::Serialize)]
pub struct Written {
    /// Relative to the checkout. `None` for a class with no file behind it.
    pub file: Option<String>,
    /// Where the class starts, so an editor can open at it.
    pub line: u32,
    /// `None` when it is long enough that reading it is opening a file, not
    /// glancing at a panel.
    pub source: Option<String>,
    pub lines: u32,
}

/// Everything a probe needs that is the same for every commit it is asked
/// about. Held together so a call says only what varies: the checkout.
pub struct Probing<'a> {
    pub python: &'a Path,
    pub probe: &'a Path,
    pub build: &'a str,
    /// Handed to the probe so it can say what is already computed. Not the
    /// store snapshots are remembered in, though it is usually the same one.
    pub store: Option<&'a Path>,
    pub given: Option<&'a Path>,
    /// What identifies this probing, everything but the commit: the build, the
    /// input, and **the probe's own source**.
    ///
    /// That last one is not belt and braces. A snapshot is a pure function of a
    /// commit only *given a fixed probe*, so the day `declared` learned to read
    /// an object's attributes, every snapshot taken before it became wrong — in
    /// exactly the way this tool exists to catch. Same lesson one level up, and
    /// the reason it is paid for here.
    pub recipe: Digest,
}

impl Probing<'_> {
    /// The name this checkout's snapshot is kept under. Content-addressed and
    /// immutable: a commit does not change, so neither does the answer.
    pub fn named(&self, commit: &str) -> String {
        format!("snapshot:{commit}:{}", self.recipe)
    }

    /// What was already probed for this commit, without touching a checkout.
    ///
    /// Its own method because a walk asks this of **every** commit first and
    /// only then lays out the ones nobody has an answer for. On a line of
    /// exploration that has been looked at once, that is no worktrees at all.
    pub fn recalled(&self, kept: &dyn Store, commit: &str) -> Option<Snapshot> {
        match recall(kept, &self.named(commit)) {
            Ok(snapshot) => snapshot,
            Err(why) => {
                eprintln!("what was already probed could not be looked up: {why}");
                None
            }
        }
    }

    /// The snapshot for this checkout, from the store if it is there.
    ///
    /// A store that cannot answer is **not** the end of it, exactly as a keeper
    /// that cannot answer is not the end of a run: the probe is asked instead
    /// and the trouble is said out loud. A cache gone cold is slow; a cache that
    /// stops the tool is broken.
    pub fn remembered(
        &self,
        kept: &dyn Store,
        working: &Path,
        commit: &str,
    ) -> Result<Snapshot, Trouble> {
        let name = self.named(commit);
        match recall(kept, &name) {
            Ok(Some(snapshot)) => return Ok(snapshot),
            Ok(None) => {}
            Err(why) => eprintln!("what was already probed could not be looked up: {why}"),
        }

        let (snapshot, bytes) = self.taken(working, commit)?;
        if let Err(why) = keep(kept, &name, &bytes, commit, self.build) {
            eprintln!("this probe could not be kept for next time: {why}");
        }
        Ok(snapshot)
    }

    /// What the edit did, for each of these pairs of `(older, newer)`.
    ///
    /// Pairs and not an ordered list, because **a step is an edge**. Three
    /// variants of one idea are three branches off one commit, and two entries
    /// next to each other in a walk are then two different lines of
    /// exploration: comparing them would answer confidently about an edit
    /// nobody ever made.
    ///
    /// One subprocess for the whole walk. Comparing needs no checkout and no
    /// graph, only the model and the snapshots, so nine comparisons paying for
    /// nine interpreters would be the slowest part of a walk of ten commits.
    ///
    /// The model is `somatize.foreseen`'s. Nothing here decides what a finding
    /// means — one implementation of something still being designed beats two
    /// that drift.
    pub fn compared(
        &self,
        taken: &HashMap<&str, Snapshot>,
        pairs: &[(String, String)],
    ) -> Result<Vec<Findings>, Trouble> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }
        let garbled = |why: serde_json::Error| Trouble::Garbled {
            commit: "comparing".into(),
            why: why.to_string(),
        };
        let asked = serde_json::json!({"snapshots": taken, "pairs": pairs})
            .to_string()
            .into_bytes();
        let written = tempfile::tempdir().map_err(|why| Trouble::Garbled {
            commit: "comparing".into(),
            why: why.to_string(),
        })?;
        let at = written.path().join("asked.json");
        std::fs::write(&at, &asked).map_err(|why| Trouble::Garbled {
            commit: "comparing".into(),
            why: why.to_string(),
        })?;

        let said = Command::new(self.python)
            .arg(self.probe)
            .arg("--compare")
            .arg(&at)
            .output()
            .map_err(|why| Trouble::Unreachable {
                python: self.python.display().to_string(),
                why: why.to_string(),
            })?;
        if !said.status.success() {
            return Err(Trouble::Refused {
                commit: "comparing".into(),
                said: String::from_utf8_lossy(&said.stderr).trim().to_string(),
            });
        }
        serde_json::from_slice(&said.stdout).map_err(garbled)
    }

    /// Whether an edit survives: it parses, a linter is quiet, the graph still
    /// builds, and the node runs on what its predecessors left in the store.
    ///
    /// Asked in a checkout that is **the same tree a fork would commit**, so a
    /// green light is about the thing that would land and not about something
    /// near it.
    pub fn checked(&self, working: &Path, node: &str) -> Result<serde_json::Value, Trouble> {
        let mut asking = Command::new(self.python);
        asking
            .arg(self.probe)
            .arg("--build")
            .arg(self.build)
            .arg("--check")
            .arg(node)
            .current_dir(working);
        if let Some(store) = self.store {
            asking.arg("--store").arg(store);
        }
        if let Some(given) = self.given {
            asking.arg("--input").arg(given);
        }
        let said = asking.output().map_err(|why| Trouble::Unreachable {
            python: self.python.display().to_string(),
            why: why.to_string(),
        })?;
        if !said.status.success() {
            return Err(Trouble::Refused {
                commit: node.to_string(),
                said: String::from_utf8_lossy(&said.stderr).trim().to_string(),
            });
        }
        serde_json::from_slice(&said.stdout).map_err(|why| Trouble::Garbled {
            commit: node.to_string(),
            why: why.to_string(),
        })
    }

    /// The same source, formatted — or the same source back and a reason.
    ///
    /// `ruff format` if this environment has one. Refused rather than
    /// half-done when it does not: handing back something that looks formatted
    /// and is not is worse than a button that says it cannot.
    pub fn prettified(&self, source: &str) -> Result<serde_json::Value, Trouble> {
        use std::io::Write as _;
        let mut running = Command::new(self.python)
            .arg(self.probe)
            .args(["--build", "x:y", "--format"])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|why| Trouble::Unreachable {
                python: self.python.display().to_string(),
                why: why.to_string(),
            })?;
        let asked = |why: String| Trouble::Garbled {
            commit: "formatting".into(),
            why,
        };
        running
            .stdin
            .take()
            .ok_or_else(|| asked("no stdin".into()))?
            .write_all(source.as_bytes())
            .map_err(|why| asked(why.to_string()))?;
        let said = running
            .wait_with_output()
            .map_err(|why| asked(why.to_string()))?;
        serde_json::from_slice(&said.stdout).map_err(|why| asked(why.to_string()))
    }

    /// Runs the probe in a checkout and reads what it wrote, with the bytes it
    /// wrote — which are what gets kept.
    ///
    /// A subprocess and not a library call, and it is the whole reason this
    /// tool has two languages in it: the graph only exists once the checkout's
    /// own code has been imported and run, against the soma *that*
    /// checkout pins. Reaching into it from here would be running one version
    /// of the engine over another version's declarations.
    pub fn taken(&self, working: &Path, commit: &str) -> Result<(Snapshot, Vec<u8>), Trouble> {
        let mut asking = Command::new(self.python);
        asking
            .arg(self.probe)
            .arg("--build")
            .arg(self.build)
            .arg("--commit")
            .arg(commit)
            .current_dir(working);
        if let Some(store) = self.store {
            asking.arg("--store").arg(store);
        }
        if let Some(given) = self.given {
            asking.arg("--input").arg(given);
        }
        let said = asking.output().map_err(|why| Trouble::Unreachable {
            python: self.python.display().to_string(),
            why: why.to_string(),
        })?;
        if !said.status.success() {
            return Err(Trouble::Refused {
                commit: commit.to_string(),
                said: String::from_utf8_lossy(&said.stderr).trim().to_string(),
            });
        }
        let snapshot = serde_json::from_slice(&said.stdout).map_err(|why| Trouble::Garbled {
            commit: commit.to_string(),
            why: why.to_string(),
        })?;
        Ok((snapshot, said.stdout))
    }
}

/// What is kept under that name, if anything readable is.
fn recall(kept: &dyn Store, name: &str) -> Result<Option<Snapshot>, String> {
    let Some(bound) = kept.resolve(name).map_err(|why| why.to_string())? else {
        return Ok(None);
    };
    let Some(bytes) = kept.get(&bound.digest).map_err(|why| why.to_string())? else {
        // Named, but the blob is not there. That is what a half-copied store
        // looks like and not a reason to stop: the probe still works.
        return Ok(None);
    };
    serde_json::from_slice(&bytes)
        .map(Some)
        .map_err(|why| why.to_string())
}

/// Puts a probe's answer where the next one will find it.
fn keep(
    kept: &dyn Store,
    name: &str,
    bytes: &[u8],
    commit: &str,
    build: &str,
) -> Result<(), String> {
    let digest = kept.put(bytes).map_err(|why| why.to_string())?;
    // Said beside it so that a scan of the store reads as something. The
    // records are the truth, and any index over them is built from these.
    let meta: Meta = vec![
        ("what".into(), "snapshot".into()),
        ("commit".into(), commit.into()),
        ("built_from".into(), build.into()),
    ];
    kept.bind(name, &digest, meta)
        .map_err(|why| why.to_string())
}

/// What can go wrong between here and a graph.
#[derive(Debug)]
pub enum Trouble {
    Unreachable { python: String, why: String },
    Refused { commit: String, said: String },
    Garbled { commit: String, why: String },
}

impl fmt::Display for Trouble {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            // The commonest failure by far, and worth naming precisely: the
            // interpreter that can import soma is rarely the one on PATH.
            Self::Unreachable { python, why } => {
                write!(f, "`{python}` could not be run: {why}")
            }
            Self::Refused { commit, said } => {
                write!(f, "building the graph at {commit} failed:\n{said}")
            }
            Self::Garbled { commit, why } => {
                write!(f, "the probe at {commit} said something unreadable: {why}")
            }
        }
    }
}

impl std::error::Error for Trouble {}
