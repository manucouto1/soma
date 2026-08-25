//! Lo que se corrió con una versión: los ensayos, sus estados y sus curvas.
//!
//! # Por qué esto se lee y no se escribe
//!
//! Un commit es la versión y no cambia. Lo que se prueba con esa versión crece
//! sin parar —cien ensayos, tres análisis, un informe— y nada de eso puede
//! tocar el hash de un commit. Así que van **asociados** a la versión, no
//! versionados: son dos mecanismos distintos, y éste es el segundo.
//!
//! Quien los escribe es soma, desde la máquina que corre el estudio.
//! Aquí sólo se leen. Eso no es una simplificación: quien reclama un ensayo es
//! el único que escribe en él, y meter un segundo escritor sería inventar una
//! carrera que hoy no existe.
//!
//! # El nombre, que es todo el acoplamiento
//!
//! soma ata cada ensayo a `<study>/trial/<n>/<attempt>`, y el nombre de un
//! estudio es una cadena cualquiera. Así que:
//!
//! ```text
//! exp/<tree>/<commit>              ← el estudio de esa versión
//! exp/<tree>/<commit>/trial/3/0    ← su cuarto ensayo, primer intento
//! exp/<tree>/<commit>/said/2       ← lo que alguien dijo de ese commit
//! ```
//!
//! El estudio de un commit **es** el prefijo bajo el que ya vive su diario, así
//! que los ensayos caen debajo sin que soma cambie una línea. No hay
//! registro de correspondencia, ni tabla, ni índice que mantener: el commit es
//! la versión y el nombre es el vínculo.
//!
//! # Un scan y no una lectura
//!
//! La regla de coste del store, que aquí manda: un registro vuelve gratis en un
//! recorrido y un blob es una lectura. soma puso el estado, el punto y la
//! puntuación en el **registro** por eso mismo. De modo que contar los ensayos
//! de cuarenta commits cuesta un recorrido, y la curva —que crece— se paga sólo
//! cuando alguien pide verla.
//!
//! # Lo que no se puede decir desde aquí
//!
//! **Cuál es el mejor.** Si `0.0837` es bueno o malo depende de si esa métrica
//! se maximiza o se minimiza, y esa dirección vive en el `Goal` que se le pasa
//! al sampler: no se escribe en ningún registro. Adivinarla sería exactamente
//! la clase de mentira callada que esta herramienta existe para no dejar pasar
//! —«mejor» es la palabra que más se copia a un informe sin comprobar—. Así que
//! o está declarada en `soma-tree.toml` o no se dice, y en su lugar se enseña
//! el rango, que sí es cierto sin saber la dirección.

use serde::{Deserialize, Serialize};
use somatize_store::{Digest, Store};
use std::collections::BTreeMap;
use std::fmt;

/// Hacia dónde es mejor. No está en el store: se declara.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Goal {
    /// Menos es mejor: una pérdida, un error, un tiempo.
    Min,
    /// Más es mejor: una exactitud, un F1, una recompensa.
    Max,
}

impl Goal {
    pub fn read(said: &str) -> Option<Self> {
        match said {
            "min" | "minimize" => Some(Self::Min),
            "max" | "maximize" => Some(Self::Max),
            _ => None,
        }
    }

    /// El mejor de unos cuantos, en la dirección declarada.
    pub fn best_of(&self, values: impl IntoIterator<Item = f64>) -> Option<f64> {
        values
            .into_iter()
            .filter(|one| !one.is_nan())
            .reduce(|best, one| match self {
                Self::Min => best.min(one),
                Self::Max => best.max(one),
            })
    }
}

/// Un ensayo, tal como vuelve de un recorrido.
///
/// Los estados son de soma y no de aquí —`running`, `done`, `pruned`,
/// `failed`—, y por eso viajan como texto: el vocabulario es de quien lo
/// escribe, y aprendérselo sería tener que migrar dos sitios el día que crezca.
#[derive(Debug, Clone, Serialize)]
pub struct Trial {
    pub trial: u32,
    /// Cuál de los intentos. Gana el más alto: reclamar es un enlace, así que
    /// un ensayo cuya máquina murió se rescata reclamando el siguiente.
    pub attempt: u32,
    pub state: Option<String>,
    /// La configuración que corrió, tal como la escribió `str(point)`.
    pub point: Option<String>,
    /// Ausente mientras corre. Presente en un `pruned`, y **no comparable** con
    /// la de un `done`: se midió tras menos épocas.
    pub score: Option<f64>,
    pub who: Option<String>,
    pub when: u64,
    /// Dónde está la curva. No la curva: eso es una lectura y esto no.
    #[serde(skip)]
    pub kept: Digest,
}

impl Trial {
    /// Si su puntuación se puede comparar con la de otro `done`.
    pub fn comparable(&self) -> bool {
        self.state.as_deref() == Some("done") && self.score.is_some()
    }
}

/// La curva de un ensayo, que es lo que cuesta una lectura.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Curve {
    #[serde(default)]
    pub point: String,
    #[serde(default)]
    pub reports: Vec<f64>,
    #[serde(default)]
    pub state: Option<String>,
    /// Por qué paró. Lo que un `pruned` tiene y una lista de números no.
    #[serde(default)]
    pub because: Option<String>,
    #[serde(default)]
    pub took: Option<f64>,
}

/// Lo que se ve de los ensayos de un commit sin leer ni un blob.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Tally {
    pub trials: u32,
    pub running: u32,
    pub done: u32,
    pub pruned: u32,
    pub failed: u32,
    /// El rango de lo comparable, que es cierto sin saber la dirección.
    pub lowest: Option<f64>,
    pub highest: Option<f64>,
    /// El mejor **sólo si alguien declaró hacia dónde es mejor**. `None` cuando
    /// no se declaró, y entonces se enseña el rango en su lugar.
    pub best: Option<f64>,
}

/// Los ensayos de una investigación, guardados en un store.
pub struct Trials<'a> {
    kept: &'a dyn Store,
    tree: String,
    goal: Option<Goal>,
}

impl<'a> Trials<'a> {
    pub fn of(tree: impl Into<String>, kept: &'a dyn Store) -> Self {
        Self {
            kept,
            tree: tree.into(),
            goal: None,
        }
    }

    /// Con la dirección declarada, si la hay.
    pub fn towards(mut self, goal: Option<Goal>) -> Self {
        self.goal = goal;
        self
    }

    /// El nombre del estudio de un commit, que es el vínculo entero.
    pub fn study(&self, commit: &str) -> String {
        format!("exp/{}/{commit}", self.tree)
    }

    /// Los ensayos de un commit, el intento más alto de cada uno, en orden.
    ///
    /// Un recorrido y ninguna lectura.
    pub fn of_commit(&self, commit: &str) -> Result<Vec<Trial>, Trouble> {
        let mut best: BTreeMap<u32, Trial> = BTreeMap::new();
        let under = format!("{}/trial/", self.study(commit));
        for bound in self.kept.bound().map_err(Trouble::Store)? {
            let Some((trial, attempt)) = numbered(&bound.name, &under) else {
                continue;
            };
            match best.get(&trial) {
                Some(had) if had.attempt >= attempt => {}
                _ => {
                    best.insert(
                        trial,
                        Trial {
                            trial,
                            attempt,
                            state: beside(&bound.meta, "state").map(str::to_string),
                            point: beside(&bound.meta, "point").map(str::to_string),
                            // `repr(float(score))` de Python, que es un número
                            // que Rust lee igual. Si algún día no lo fuera, un
                            // ensayo sin puntuación es preferible a uno con una
                            // inventada.
                            score: beside(&bound.meta, "score").and_then(|one| one.parse().ok()),
                            who: beside(&bound.meta, "who").map(str::to_string),
                            when: bound.when,
                            kept: bound.digest,
                        },
                    );
                }
            }
        }
        Ok(best.into_values().collect())
    }

    /// Cuántos ensayos tiene cada commit y cómo van, en **un solo recorrido**.
    ///
    /// Lo que el raíl necesita: preguntarlo commit a commit serían cuarenta
    /// recorridos del store para dibujar una lista de cuarenta filas.
    pub fn counted(&self) -> Result<BTreeMap<String, Tally>, Trouble> {
        let under = format!("exp/{}/", self.tree);
        // El intento más alto de cada `(commit, trial)` antes de contar: contar
        // los registros sin más contaría dos veces un ensayo que se rescató.
        let mut best: BTreeMap<(String, u32), Highest> = BTreeMap::new();
        for bound in self.kept.bound().map_err(Trouble::Store)? {
            let Some(rest) = bound.name.strip_prefix(&under) else {
                continue;
            };
            let Some((commit, numbers)) = rest.split_once("/trial/") else {
                continue;
            };
            let Some((trial, attempt)) = numbered(numbers, "") else {
                continue;
            };
            let mine = (commit.to_string(), trial);
            match best.get(&mine) {
                Some(had) if had.attempt >= attempt => {}
                _ => {
                    best.insert(
                        mine,
                        Highest {
                            attempt,
                            state: beside(&bound.meta, "state").map(str::to_string),
                            score: beside(&bound.meta, "score").and_then(|one| one.parse().ok()),
                        },
                    );
                }
            }
        }

        let mut counted: BTreeMap<String, Tally> = BTreeMap::new();
        let mut comparable: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        for ((commit, _), one) in best {
            let tally = counted.entry(commit.clone()).or_default();
            tally.trials += 1;
            match one.state.as_deref() {
                Some("running") => tally.running += 1,
                Some("done") => tally.done += 1,
                Some("pruned") => tally.pruned += 1,
                Some("failed") => tally.failed += 1,
                _ => {}
            }
            // Sólo las de un `done` entran en el rango: la de un `pruned` es
            // real y no es comparable —se midió tras menos épocas—, y meterla
            // haría el rango más ancho de lo que nadie midió.
            if let (Some("done"), Some(score)) = (one.state.as_deref(), one.score) {
                comparable.entry(commit).or_default().push(score);
            }
        }
        for (commit, scores) in comparable {
            let Some(tally) = counted.get_mut(&commit) else {
                continue;
            };
            tally.lowest = Goal::Min.best_of(scores.iter().copied());
            tally.highest = Goal::Max.best_of(scores.iter().copied());
            tally.best = self.goal.and_then(|goal| goal.best_of(scores));
        }
        Ok(counted)
    }

    /// La curva de un ensayo. **Esto sí es una lectura**, y por eso está aparte.
    pub fn curve(&self, of: &Trial) -> Result<Option<Curve>, Trouble> {
        let Some(bytes) = self.kept.get(&of.kept).map_err(Trouble::Store)? else {
            return Ok(None);
        };
        serde_json::from_slice(&bytes)
            .map(Some)
            .map_err(|why| Trouble::Garbled(why.to_string()))
    }
}

/// El intento más alto visto de un ensayo, mientras se recorre.
struct Highest {
    attempt: u32,
    state: Option<String>,
    score: Option<f64>,
}

/// El `(ensayo, intento)` que ese nombre es, o `None` si no es uno.
///
/// Una pregunta y no una suposición, como en soma y por lo mismo: un store
/// guarda lo que le echen —una caché, otra investigación, un artefacto—.
fn numbered(name: &str, under: &str) -> Option<(u32, u32)> {
    let rest = name.strip_prefix(under)?;
    let (trial, attempt) = rest.split_once('/')?;
    Some((trial.parse().ok()?, attempt.parse().ok()?))
}

fn beside<'a>(meta: &'a somatize_store::Meta, what: &str) -> Option<&'a str> {
    meta.iter()
        .find(|(said, _)| said == what)
        .map(|(_, value)| value.as_str())
}

#[derive(Debug)]
pub enum Trouble {
    Store(somatize_store::StoreError),
    Garbled(String),
}

impl fmt::Display for Trouble {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Store(why) => write!(f, "los ensayos no se pudieron alcanzar: {why}"),
            Self::Garbled(why) => write!(f, "una curva no se pudo leer: {why}"),
        }
    }
}

impl std::error::Error for Trouble {}
