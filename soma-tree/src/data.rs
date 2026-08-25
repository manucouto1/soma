//! Qué datos hay bajo cada versión, y cuáles no son de ninguna.
//!
//! # El problema, dicho como se nota
//!
//! Dentro de un movimiento —una pregunta, una hipótesis que se está probando—
//! se iteran cinco versiones en una tarde, y cada una deja intermedios en el
//! store. Al mes siguiente eso es un montón de hashes y nadie puede decir de
//! cuál de las cinco era ninguno. No están mal: están mudos, que con el tiempo
//! es peor.
//!
//! # Por qué esto se deriva y no se escribe
//!
//! La misma regla que la poda. Lo que se puede recalcular es registro y no se
//! guarda, así que aquí no hay índice, ni tabla, ni un fichero al lado del
//! commit que alguien tenga que mantener al día. Hay dos preguntas al store y
//! un sondeo que ya estaba hecho.
//!
//! # Dos maneras de atribuir, y la segunda es la que aguanta
//!
//! **Por la clave.** Un sondeo dice cómo se va a llamar la respuesta de cada
//! nodo antes de que exista, así que las claves de un commit se saben sin
//! correr nada y se cruzan con lo que el store tiene. Exacto — y frágil en un
//! sitio: una clave se calcula contra el entorno del intérprete que sondea, así
//! que sondear hoy un commit de hace tres meses da otras claves y no casa con
//! nada de lo que se guardó entonces.
//!
//! **Por la huella.** Cada valor lleva escrito al lado qué nodo y qué versión
//! del código lo produjo, y eso lo escribió quien corrió, entonces. No hace
//! falta reproducir nada: se compara con las huellas del sondeo. Es la que
//! contesta de los datos viejos, que son justo los que nadie puede atribuir de
//! memoria.
//!
//! Las dos, porque dicen cosas distintas y las dos son ciertas: una clave que
//! casa es *este dato es exactamente el que esta versión pediría*; una huella
//! que casa es *esto lo hizo este código*.
//!
//! # Y lo que no es de nadie no se calla
//!
//! Un valor cuya huella no es la de ninguna versión que se pueda nombrar aquí
//! sale igualmente, diciendo eso. Es una frase verdadera —«lo hizo `embed`, con
//! código `a1b2c3d4`, que no es ninguna versión que yo sepa nombrar»— y hoy no
//! hay ninguna. Callarlo sería devolver a los hashes mudos por la puerta de
//! atrás.

use crate::snapshot::Snapshot;
use somatize_core::Key;
use somatize_store::{Bound, Store};
use std::collections::{BTreeMap, HashMap};

/// Cómo se supo que un valor es de una versión.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum How {
    /// Se llama como esa versión va a llamar a ese nodo. Lo más fuerte que se
    /// puede decir: es el dato que esta versión pediría, no uno parecido.
    Named,
    /// Lo produjo el código de esa versión, según lo que quien corrió escribió
    /// al lado. Sobrevive a que el entorno de entonces ya no exista.
    Written,
}

/// Un valor del store, y de qué versión resultó ser.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Belongs {
    /// El nombre bajo el que está en el store.
    pub name: String,
    /// Qué nodo lo produjo, si se dijo.
    pub node: Option<String>,
    /// Qué versión del código, si se dijo.
    pub fingerprint: Option<String>,
    /// Con qué entrada, por el nombre que tiene su contenido.
    pub input: Option<String>,
    /// Contra qué entorno, por su nombre corto.
    pub environment: Option<String>,
    /// Cuándo se ató, en segundos desde la época.
    pub when: u64,
    /// De qué commits es, y por qué se sabe. Vacío es una respuesta: de ninguno
    /// que se pueda nombrar aquí.
    pub of: BTreeMap<String, How>,
}

impl Belongs {
    /// Si no resultó ser de ninguna versión de las que se preguntó.
    ///
    /// **No es lo mismo que sobrar.** Puede ser de una rama que no se miró, de
    /// un commit que ya no existe, o de un entorno que no se puede reproducir.
    /// Lo único que dice es que aquí no se sabe.
    pub fn is_nobodys(&self) -> bool {
        self.of.is_empty()
    }
}

/// Lo que hay en el store, atribuido a las versiones que se pasaron.
///
/// Un recorrido del store y ni una lectura de un blob: lo que hace falta va en
/// el registro, que es la regla de coste de este store desde el primer día.
pub fn under(
    store: &dyn Store,
    known: &HashMap<&str, Snapshot>,
) -> Result<Vec<Belongs>, Box<dyn std::error::Error>> {
    // Los dos índices al revés, una vez, en vez de recorrer las versiones por
    // cada valor: con cuarenta commits y unos miles de valores, lo segundo es
    // el mismo trabajo hecho miles de veces.
    let mut by_name: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut by_code: HashMap<(&str, &str), Vec<&str>> = HashMap::new();
    let names: Vec<(&str, BTreeMap<String, String>)> = known
        .iter()
        .map(|(commit, taken)| (*commit, taken.names()))
        .collect();
    let codes: Vec<(&str, BTreeMap<String, String>)> = known
        .iter()
        .map(|(commit, taken)| (*commit, taken.fingerprints()))
        .collect();
    // La clave es cómo se llama la **receta**; el nombre es dónde se ata el
    // valor de esa receta, y no son la misma cadena. Quien traduce de una a
    // otro es el store, y se le pregunta en vez de copiarle el `format!`: dos
    // sitios diciendo lo mismo y ninguna forma de saber cuál manda el día que
    // dejen de coincidir.
    let bound_as: Vec<(&str, Vec<String>)> = names
        .iter()
        .map(|(commit, said)| {
            (
                *commit,
                said.values()
                    .map(|key| somatize_store::name_of(&Key::new(key.clone())))
                    .collect(),
            )
        })
        .collect();
    for (commit, said) in &bound_as {
        for name in said {
            by_name.entry(name.as_str()).or_default().push(commit);
        }
    }
    for (commit, said) in &codes {
        for (node, written) in said {
            by_code
                .entry((node.as_str(), written.as_str()))
                .or_default()
                .push(commit);
        }
    }

    let mut said: Vec<Belongs> = store
        .bound()?
        .into_iter()
        .filter(|bound| !bookkeeping(bound))
        .map(|bound| {
            let meta = |what: &str| {
                bound
                    .meta
                    .iter()
                    .find(|(said, _)| said == what)
                    .map(|(_, told)| told.clone())
            };
            let (node, fingerprint) = (meta(somatize_core::NODE), meta(somatize_core::FINGERPRINT));
            let mut of: BTreeMap<String, How> = BTreeMap::new();
            // La huella primero y la clave después, para que `Named` gane donde
            // valen las dos: es lo más fuerte que se puede decir, y decir lo
            // más débil pudiendo decir lo otro es perder información.
            if let (Some(node), Some(written)) = (&node, &fingerprint) {
                for commit in by_code
                    .get(&(node.as_str(), written.as_str()))
                    .into_iter()
                    .flatten()
                {
                    of.insert((*commit).to_string(), How::Written);
                }
            }
            for commit in by_name.get(bound.name.as_str()).into_iter().flatten() {
                of.insert((*commit).to_string(), How::Named);
            }
            Belongs {
                name: bound.name.clone(),
                node,
                fingerprint,
                input: meta(somatize_core::INPUT),
                environment: meta(ENVIRONMENT),
                when: bound.when,
                of,
            }
        })
        .collect();
    // Por fecha, que es como se lee un store: lo último que se hizo arriba.
    said.sort_by(|a, b| (b.when, &a.name).cmp(&(a.when, &b.name)));
    Ok(said)
}

/// Cómo se llama, en el `meta`, el entorno contra el que se produjo un valor.
///
/// La palabra es de `soma_next._environment` y no del motor, así que no está en
/// las constantes del core. Escrita aquí una vez y no en cada sitio que la
/// mira, que es la mitad de la deriva que este acuerdo evita.
pub const ENVIRONMENT: &str = "env";

/// Lo que no son datos de una corrida, sino la contabilidad de quien mira.
///
/// Tres escritores comparten este store y sólo uno deja intermedios:
///
/// - `exp/…` es de esta herramienta —el diario, los veredictos, los
///   movimientos, los ensayos— y ya se atribuye por su nombre, que lleva el
///   commit dentro. Contarlo aquí sería enseñarle a alguien su propio cuaderno
///   de laboratorio como si fuera un intermedio que a lo mejor sobra.
/// - `snapshot:…` es la caché de sondeos de esta herramienta. Se recalcula
///   desde un commit y no es de nadie.
/// - `env/…` es la lectura de un entorno, que escribe soma-next para que el
///   nombre corto que llevan los valores se pueda entender. Es lo que explica
///   la atribución, no algo que atribuir.
///
/// Y todo lo demás sí es un dato, incluido lo que no se sepa de quién es. Un
/// filtro que se quedara sólo con lo reconocido sería un listado que no puede
/// enseñar nunca el caso que importa.
fn bookkeeping(bound: &Bound) -> bool {
    ["exp/", "snapshot:", "env/"]
        .iter()
        .any(|prefix| bound.name.starts_with(prefix))
}
