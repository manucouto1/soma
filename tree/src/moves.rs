//! El razonamiento: preguntas, hipótesis, intentos, hallazgos y decisiones.
//!
//! La capa 1 —commits, snapshots, hallazgos por nodo, trials— responde a *qué
//! se ejecutó y qué salió*. Ésta responde a *qué se estaba intentando averiguar*,
//! y no comparte su unidad: un commit no es una decisión de nadie, una pregunta
//! sin intentar no tiene commit, y un movimiento puede producir tres ramas.
//!
//! # Lo que decide a qué capa pertenece algo
//!
//! Si se puede recalcular, es registro. Si alguien lo pensó, es razonamiento.
//!
//! # Es un DAG, y el caso que lo obliga
//!
//! Dos preguntas vivas —¿más capacidad mejora la interpretabilidad? ¿mejora el
//! rendimiento?—, una variante que valida cada una, y entonces la pregunta que
//! ninguna contenía: ¿y si las junto? Ese intento cuelga de **las dos**. Con un
//! solo padre habría que elegir, o duplicar el nodo, y un nodo duplicado son dos
//! que se desincronizan. Por eso [`Under`] es multivaluado y por eso hay que
//! rechazar ciclos al escribir: un recorrido sobre uno no termina.
//!
//! # Todo lleva alcance, también lo que se dice
//!
//! Una pregunta habla de unos movimientos y no de la investigación entera; y una
//! respuesta vale **donde vale**. Sin eso, «validada» y «refutada» sobre la misma
//! hipótesis parecen una contradicción cuando lo normal es que sean dos hechos
//! sobre dos situaciones: A sola funcionaba, A+B se anulan. Sólo hay disputa
//! cuando dos aristas de signo contrario tienen alcances que **se tocan**.

use serde::{Deserialize, Serialize};
use soma_next_store::{Bound as Record, Digest, Meta, Store};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fmt;

/// Cuántas ranuras probar antes de rendirse. Sólo se pasa de una cuando alguien
/// reclamó la misma en el mismo instante: acota una carrera, no una cola.
const PATIENCE: u32 = 32;

/// Qué identifica a un movimiento. Su ranura, porque un movimiento es mutable
/// —le editas la prosa— y no puede direccionarse por su contenido.
pub type MoveId = u32;

/// Las cinco clases, y no hay más.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Kind {
    /// Lo que no se sabe. Se le **responde**. La única clase que puede existir
    /// sin nada debajo: una pregunta sin intentar es trabajo pendiente.
    Question,
    /// Una respuesta propuesta y falsable. Se la **valida** o se la **refuta**
    /// — verbos que una pregunta no tiene, y por eso no es una pregunta con
    /// otra redacción.
    Hypothesis,
    /// Lo que se probó, citando la capa 1. La única clase que la toca.
    Attempt,
    /// Lo que dice la evidencia. De aquí salen las aristas de verbo, y es lo
    /// único exportable a un lago de conocimiento.
    Finding,
    /// Qué se hace al respecto. Separada del hallazgo porque dos personas
    /// pueden coincidir en uno y discrepar en la otra.
    Decision,
}

impl Kind {
    pub fn read(said: &str) -> Option<Self> {
        match said {
            "question" => Some(Self::Question),
            "hypothesis" => Some(Self::Hypothesis),
            "attempt" => Some(Self::Attempt),
            "finding" => Some(Self::Finding),
            "decision" => Some(Self::Decision),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Question => "question",
            Self::Hypothesis => "hypothesis",
            Self::Attempt => "attempt",
            Self::Finding => "finding",
            Self::Decision => "decision",
        }
    }
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// De qué habla algo: unos movimientos y lo que cuelga de ellos.
///
/// **Raíces y no un conjunto libre**, que es lo que lo hace pagable. «Toda la
/// rama del encoder» es una raíz; «este paso» es una raíz; la investigación
/// entera es ninguna. Un conjunto arbitrario sería más fiel y convertiría
/// «¿se solapan?» en algo que hay que materializar en vez de recorrer.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Scope(pub BTreeSet<MoveId>);

impl Scope {
    /// De todo. Lo que hace general a una pregunta general.
    pub fn everything() -> Self {
        Self(BTreeSet::new())
    }

    pub fn of(roots: impl IntoIterator<Item = MoveId>) -> Self {
        Self(roots.into_iter().collect())
    }

    pub fn is_everything(&self) -> bool {
        self.0.is_empty()
    }

    /// Los movimientos que abarca: sus raíces y todo lo que cuelga.
    pub fn covers(&self, under: &Undernath) -> HashSet<MoveId> {
        let mut reached = HashSet::new();
        let mut asking: Vec<MoveId> = self.0.iter().copied().collect();
        while let Some(one) = asking.pop() {
            if !reached.insert(one) {
                continue;
            }
            asking.extend(under.children_of(one));
        }
        reached
    }

    /// Si dos alcances se tocan. Lo que separa una contradicción de dos hechos
    /// sobre dos situaciones distintas.
    pub fn touches(&self, other: &Self, under: &Undernath) -> bool {
        // Lo que abarca todo toca todo, incluido otro que abarque todo.
        if self.is_everything() || other.is_everything() {
            return true;
        }
        let mine = self.covers(under);
        other.covers(under).iter().any(|one| mine.contains(one))
    }
}

/// Qué dice un hallazgo, y hacia dónde.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Says {
    /// Hacia una pregunta.
    Answers,
    /// Hacia una hipótesis.
    Validates,
    /// Hacia una hipótesis.
    Refutes,
    /// De un intento hacia los intentos que compone. No es `under`: dice que
    /// este intento **es** la composición de aquellos, que es lo que permite
    /// leer «cada una funcionaba sola, juntas se anulan» como lo que es.
    Combines,
}

impl Says {
    pub fn read(said: &str) -> Option<Self> {
        match said {
            "answers" => Some(Self::Answers),
            "validates" => Some(Self::Validates),
            "refutes" => Some(Self::Refutes),
            "combines" => Some(Self::Combines),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Answers => "answers",
            Self::Validates => "validates",
            Self::Refutes => "refutes",
            Self::Combines => "combines",
        }
    }

    /// Quién puede decirlo y a quién, porque un `valida` apuntando a un intento
    /// no significa nada y aceptarlo es guardar una frase que nadie puede leer.
    fn between(&self) -> (&'static [Kind], &'static [Kind]) {
        match self {
            Self::Answers => (&[Kind::Finding], &[Kind::Question]),
            Self::Validates | Self::Refutes => (&[Kind::Finding], &[Kind::Hypothesis]),
            Self::Combines => (&[Kind::Attempt], &[Kind::Attempt]),
        }
    }
}

impl fmt::Display for Says {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Una cosa dicha de un movimiento hacia otro.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Said {
    pub from: MoveId,
    pub to: MoveId,
    pub says: Says,
    /// Dónde vale. Casi nunca es todo, y ahí está la gracia.
    #[serde(default)]
    pub scope: Scope,
    /// Si cierra la cuestión o sólo la empuja. «¿Si aumento la capacidad
    /// mejora?» no se responde de una vez: tres intentos responden en parte.
    #[serde(default)]
    pub in_part: bool,
}

/// Qué se decidió hacer con la línea de la que habla una decisión.
///
/// Esto era un veredicto pegado a un commit —`promising`, `dead-end`,
/// `superseded`— y no era una propiedad del código: era lo que alguien decidió
/// sobre por dónde seguir. Aquí sí tiene lo que le faltaba allí: un **alcance**
/// que dice de qué línea habla, un **motivo** en la prosa, y un sitio en el DAG
/// bajo la cuestión que estaba respondiendo. `invalid` no está y no lo estará:
/// eso sí es del código, y se queda en el diario.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Course {
    /// Seguir por aquí. La lectura por defecto de una línea que nadie juzgó,
    /// y por eso decirlo sólo hace falta para desdecir un abandono.
    Pursue,
    /// Explorada y no vale la pena seguir. Se guarda, nunca se borra: una línea
    /// que no funcionó es lo más reutilizable que produce una investigación, y
    /// lo único que evita volver a descubrirla.
    Abandon,
    /// Alguien lo hizo mejor en otro sitio. No está mal, no es el camino.
    Superseded,
}

impl Course {
    pub fn read(said: &str) -> Option<Self> {
        match said {
            "pursue" => Some(Self::Pursue),
            "abandon" => Some(Self::Abandon),
            "superseded" => Some(Self::Superseded),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pursue => "pursue",
            Self::Abandon => "abandon",
            Self::Superseded => "superseded",
        }
    }
}

impl fmt::Display for Course {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Un movimiento, sin sus aristas.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Move {
    pub id: MoveId,
    pub kind: Kind,
    /// De qué habla. Sólo lo llevan preguntas e hipótesis; en las demás es de
    /// todo y no se lee.
    #[serde(default)]
    pub scope: Scope,
    pub prose: String,
    /// Lo que cita de la capa 1: commits, ensayos, artefactos.
    ///
    /// Lo llevan un intento —el commit que corrió, y los ensayos que corrió con
    /// él— y un hallazgo —el ensayo donde se vio—. Una pregunta, una hipótesis
    /// y una decisión no: hablan de movimientos, no de piezas de la capa 1, y
    /// dejarlas citar sería dejar que una pregunta apunte a un commit sin que
    /// nadie sepa qué significa eso.
    #[serde(default)]
    pub cites: Vec<Cited>,
    /// Qué se decidió. Sólo lo lleva una [`Kind::Decision`]; en las demás es
    /// `None` y no se lee.
    #[serde(default)]
    pub course: Option<Course>,
    pub who: String,
    pub when: u64,
}

/// Una pieza de evidencia de la capa 1.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cited {
    /// `commit`, `trial`, `artifact`. Abierto a propósito: el vocabulario es de
    /// quien cita, y esta capa lo guarda sin aprendérselo.
    pub what: String,
    pub id: String,
}

/// Cómo está una cuestión, contando lo que le han dicho.
///
/// **Derivado, nunca guardado.** Un campo «estado» que alguien sobrescribe
/// pierde el hecho anterior, y aquí el hecho anterior es lo que hace que una
/// hipótesis vuelva sola a estar abierta cuando se invalida lo que la refutaba.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Standing {
    /// Nadie ha dicho nada todavía.
    Open,
    /// Respondida, y del todo.
    Answered,
    /// Empujada: todo lo que le llegó dijo «en parte».
    Partly,
    Validated,
    PartlyValidated,
    Refuted,
    PartlyRefuted,
    /// Le llegan aristas de signo contrario **con alcances que se tocan**. El
    /// estado interesante, y el que no se puede expresar con un campo.
    Disputed,
    /// Validada en unas situaciones y refutada en otras, sin que se toquen.
    ///
    /// No es «en parte» y no es disputa: es que la respuesta **depende**. «A
    /// sola mejora, A+B se anulan» no es media validación ni un conflicto, es el
    /// desenlace más informativo que da una investigación — y llamarlo `Partly`
    /// lo escondía debajo de la misma palabra que usa una pregunta a medio
    /// responder.
    Depends,
}

/// Quién cuelga de quién. Un índice sobre las aristas `under`, construido para
/// poder recorrer hacia arriba y hacia abajo sin volver a escanear.
#[derive(Debug, Default)]
pub struct Undernath {
    over: BTreeMap<MoveId, BTreeSet<MoveId>>,
    below: BTreeMap<MoveId, BTreeSet<MoveId>>,
}

impl Undernath {
    pub fn add(&mut self, child: MoveId, parent: MoveId) {
        self.over.entry(child).or_default().insert(parent);
        self.below.entry(parent).or_default().insert(child);
    }

    pub fn parents_of(&self, child: MoveId) -> Vec<MoveId> {
        self.over
            .get(&child)
            .into_iter()
            .flatten()
            .copied()
            .collect()
    }

    pub fn children_of(&self, parent: MoveId) -> Vec<MoveId> {
        self.below
            .get(&parent)
            .into_iter()
            .flatten()
            .copied()
            .collect()
    }

    /// Si `maybe` está por encima de `one`, mirando hacia arriba.
    ///
    /// Lo que hace falta para rechazar un ciclo antes de escribirlo: con
    /// `under` multivaluado ya no basta con confiar en la forma, y un ciclo
    /// cuelga cualquier recorrido posterior — incluido el que lo dibujaría.
    pub fn is_over(&self, maybe: MoveId, one: MoveId) -> bool {
        let mut seen = HashSet::new();
        let mut asking = vec![one];
        while let Some(which) = asking.pop() {
            if which == maybe && !seen.is_empty() {
                return true;
            }
            if !seen.insert(which) {
                continue;
            }
            asking.extend(self.parents_of(which));
        }
        false
    }
}

/// El razonamiento de una investigación, guardado en un store.
pub struct Moves<'a> {
    kept: &'a dyn Store,
    tree: String,
}

impl<'a> Moves<'a> {
    pub fn of(tree: impl Into<String>, kept: &'a dyn Store) -> Self {
        Self {
            kept,
            tree: tree.into(),
        }
    }

    fn named(&self, id: MoveId, what: &str, nth: u32) -> String {
        format!("exp/{}/move/{id}/{what}/{nth}", self.tree)
    }

    /// Escribe un movimiento nuevo y devuelve su id.
    ///
    /// Reclama la ranura igual que un trial: sin coordinador, y quien la
    /// encuentra ocupada pide la siguiente. Dos personas escribiendo a la vez
    /// obtienen dos movimientos, no uno perdido.
    pub fn add(
        &self,
        kind: Kind,
        prose: &str,
        who: &str,
        scope: Scope,
        cites: Vec<Cited>,
        course: Option<Course>,
    ) -> Result<MoveId, Trouble> {
        if course.is_some() && kind != Kind::Decision {
            return Err(Trouble::NotADecision { kind });
        }
        let first = self.all()?.keys().copied().max().map_or(0, |last| last + 1);
        for id in first..first + PATIENCE {
            let body = Move {
                id,
                kind,
                scope: scope.clone(),
                prose: prose.trim().to_string(),
                cites: cites.clone(),
                course,
                who: who.to_string(),
                when: 0,
            };
            let bytes =
                serde_json::to_vec(&body).map_err(|why| Trouble::Garbled(why.to_string()))?;
            let digest = self.kept.put(&bytes).map_err(Trouble::Store)?;
            let mut meta: Meta = vec![
                ("what".into(), "move".into()),
                ("kind".into(), kind.to_string()),
                ("who".into(), who.to_string()),
            ];
            if let Some(course) = course {
                meta.push(("course".into(), course.to_string()));
            }
            if self
                .kept
                .claim(&self.named(id, "said", 0), &digest, meta)
                .map_err(Trouble::Store)?
            {
                return Ok(id);
            }
        }
        Err(Trouble::Crowded)
    }

    /// Vuelve a redactar un movimiento. Ranura nueva, gana la última: lo
    /// anterior sigue ahí, como en el diario.
    ///
    /// Lo que llega como `None` se queda como estaba, que es lo que hace que
    /// corregir la prosa no borre el alcance ni al revés. El alcance **tiene**
    /// que poder corregirse: en una decisión es lo que dice de qué línea habla,
    /// y equivocarse en él —alcanzar un hallazgo, que no es una línea, en vez
    /// del intento del que salió— deja la decisión sin llegar a ningún commit,
    /// sin que nada avise. Un alcance que no se puede corregir es una trampa.
    ///
    /// Un rumbo se cambia pero no se quita: una decisión que ya no decide nada
    /// es `pursue`, y decirlo es más honesto que dejarla muda.
    pub fn reword(
        &self,
        id: MoveId,
        prose: Option<&str>,
        scope: Option<Scope>,
        course: Option<Course>,
        who: &str,
    ) -> Result<u32, Trouble> {
        let mut body = self.all()?.remove(&id).ok_or(Trouble::NoSuchMove { id })?;
        if course.is_some() && body.kind != Kind::Decision {
            return Err(Trouble::NotADecision { kind: body.kind });
        }
        if let Some(prose) = prose {
            body.prose = prose.trim().to_string();
        }
        if let Some(scope) = scope {
            body.scope = scope;
        }
        if course.is_some() {
            body.course = course;
        }
        self.redrafted(id, body, who)
    }

    /// Escribe una redacción de un movimiento en la ranura siguiente.
    fn redrafted(&self, id: MoveId, mut body: Move, who: &str) -> Result<u32, Trouble> {
        body.who = who.to_string();
        let bytes = serde_json::to_vec(&body).map_err(|why| Trouble::Garbled(why.to_string()))?;
        let digest = self.kept.put(&bytes).map_err(Trouble::Store)?;
        let first = self.slots(id, "said")? + 1;
        for nth in first..first + PATIENCE {
            let mut meta: Meta = vec![
                ("what".into(), "move".into()),
                ("kind".into(), body.kind.to_string()),
                ("who".into(), who.to_string()),
            ];
            // El mismo meta que escribe `add`, y no uno más pobre: un registro
            // que dice menos que el anterior es un registro que miente sobre lo
            // que hay debajo.
            if let Some(course) = body.course {
                meta.push(("course".into(), course.to_string()));
            }
            if self
                .kept
                .claim(&self.named(id, "said", nth), &digest, meta)
                .map_err(Trouble::Store)?
            {
                return Ok(nth);
            }
        }
        Err(Trouble::Crowded)
    }

    /// Añade una pieza de evidencia a un movimiento.
    ///
    /// Redacción nueva y gana la última, como todo aquí: la evidencia se junta
    /// después de escribir el intento, porque los ensayos se corren después.
    /// Citar dos veces lo mismo no la duplica —lo pedirían dos personas mirando
    /// la misma pantalla, y una lista con el mismo ensayo dos veces no dice
    /// nada más que una con él una vez.
    pub fn cite(&self, id: MoveId, cited: Cited, who: &str) -> Result<u32, Trouble> {
        let known = self.all()?;
        let body = known.get(&id).ok_or(Trouble::NoSuchMove { id })?;
        if !matches!(body.kind, Kind::Attempt | Kind::Finding) {
            return Err(Trouble::CannotCite { kind: body.kind });
        }
        if body.cites.contains(&cited) {
            return self.slots(id, "said");
        }
        let mut body = body.clone();
        body.cites.push(cited);
        self.redrafted(id, body, who)
    }

    /// Cuelga un movimiento de otro.
    ///
    /// Rechaza el ciclo aquí, que es el único sitio donde sale barato: leerlo
    /// después significa descubrirlo colgando un recorrido.
    pub fn hang(&self, child: MoveId, parent: MoveId) -> Result<(), Trouble> {
        if child == parent {
            return Err(Trouble::Circular { child, parent });
        }
        let known = self.all()?;
        for one in [child, parent] {
            if !known.contains_key(&one) {
                return Err(Trouble::NoSuchMove { id: one });
            }
        }
        if self.under()?.is_over(child, parent) {
            return Err(Trouble::Circular { child, parent });
        }
        self.bind(
            child,
            "under",
            &parent.to_string(),
            &[("parent", &parent.to_string())],
        )
    }

    /// Dice algo de un movimiento hacia otro.
    pub fn say(&self, said: Said) -> Result<(), Trouble> {
        let known = self.all()?;
        let (from, to) = (
            known
                .get(&said.from)
                .ok_or(Trouble::NoSuchMove { id: said.from })?,
            known
                .get(&said.to)
                .ok_or(Trouble::NoSuchMove { id: said.to })?,
        );
        let (says_from, says_to) = said.says.between();
        if !says_from.contains(&from.kind) || !says_to.contains(&to.kind) {
            return Err(Trouble::Nonsense {
                says: said.says,
                from: from.kind,
                to: to.kind,
            });
        }
        let body = serde_json::to_vec(&said).map_err(|why| Trouble::Garbled(why.to_string()))?;
        let digest = self.kept.put(&body).map_err(Trouble::Store)?;
        let target = said.to.to_string();
        let says = said.says.to_string();
        self.bound(
            said.from,
            "says",
            &digest,
            &[("says", says.as_str()), ("to", target.as_str())],
        )
    }

    /// Todos los movimientos, por id, con su última redacción.
    pub fn all(&self) -> Result<BTreeMap<MoveId, Move>, Trouble> {
        let under = format!("exp/{}/move/", self.tree);
        let mut latest: BTreeMap<MoveId, (u32, Digest, u64)> = BTreeMap::new();
        for bound in self.kept.bound().map_err(Trouble::Store)? {
            // Un store guarda lo que le echen —una caché, otra investigación,
            // un artefacto— así que esto es una pregunta y no una suposición.
            let Some(rest) = bound.name.strip_prefix(&under) else {
                continue;
            };
            let Some((id, nth)) = rest.split_once("/said/") else {
                continue;
            };
            let (Ok(id), Ok(nth)) = (id.parse::<MoveId>(), nth.parse::<u32>()) else {
                continue;
            };
            match latest.get(&id) {
                Some((had, _, _)) if *had >= nth => {}
                _ => {
                    latest.insert(id, (nth, bound.digest, bound.when));
                }
            }
        }

        let mut said = BTreeMap::new();
        for (id, (_, digest, when)) in latest {
            let Some(bytes) = self.kept.get(&digest).map_err(Trouble::Store)? else {
                continue;
            };
            if let Ok(mut body) = serde_json::from_slice::<Move>(&bytes) {
                body.when = when;
                said.insert(id, body);
            }
        }
        Ok(said)
    }

    /// El índice de quién cuelga de quién.
    pub fn under(&self) -> Result<Undernath, Trouble> {
        let mut said = Undernath::default();
        for (child, bound) in self.records("under")? {
            // Del registro y no del nombre: el último segmento de un nombre es
            // la **ranura**, y leerlo como si fuera el padre construye un índice
            // que parece correcto y apunta a movimientos que no existen.
            if let Some(parent) = beside(&bound.meta, "parent").and_then(|one| one.parse().ok()) {
                said.add(child, parent);
            }
        }
        Ok(said)
    }

    /// Todo lo que alguien ha dicho de un movimiento hacia otro.
    ///
    /// Gana la última por cada terna `(de, a, verbo)`, que es la misma regla
    /// que sigue una redacción en `all`. Sin ella no hay forma de corregir un
    /// alcance: decirlo otra vez dejaría las dos aristas y el recuento las
    /// contaría a las dos, así que ampliar un alcance parecería estar
    /// diciéndolo dos veces. Volver a decirlo **es** el gesto de cambiar de
    /// opinión sobre el alcance; retirar el verbo entero sigue sin gesto.
    ///
    /// La terna sale del meta y no del cuerpo: para quedarse con la última no
    /// hace falta leer las anteriores.
    pub fn says(&self) -> Result<Vec<Said>, Trouble> {
        let under = format!("exp/{}/move/", self.tree);
        let mut latest: BTreeMap<(MoveId, String, String), (u32, Digest)> = BTreeMap::new();
        for bound in self.kept.bound().map_err(Trouble::Store)? {
            let Some(rest) = bound.name.strip_prefix(&under) else {
                continue;
            };
            let Some((from, nth)) = rest.split_once("/says/") else {
                continue;
            };
            let (Ok(from), Ok(nth)) = (from.parse::<MoveId>(), nth.parse::<u32>()) else {
                continue;
            };
            let (Some(verb), Some(to)) = (
                beside(&bound.meta, "says").map(str::to_string),
                beside(&bound.meta, "to").map(str::to_string),
            ) else {
                continue;
            };
            match latest.get(&(from, verb.clone(), to.clone())) {
                Some((had, _)) if *had >= nth => {}
                _ => {
                    latest.insert((from, verb, to), (nth, bound.digest));
                }
            }
        }

        let mut said = Vec::new();
        for (_, digest) in latest.into_values() {
            let Some(bytes) = self.kept.get(&digest).map_err(Trouble::Store)? else {
                continue;
            };
            if let Ok(one) = serde_json::from_slice::<Said>(&bytes) {
                said.push(one);
            }
        }
        Ok(said)
    }

    /// Qué se decidió sobre cada commit, derivado del razonamiento.
    ///
    /// El puente entre las dos capas, y va en este sentido: un commit no
    /// guarda que esté abandonado. Se llega a él bajando —decisión, su alcance,
    /// los intentos que ese alcance abarca, los commits que esos intentos
    /// citan— y por eso un commit creado mañana bajo una línea abandonada sale
    /// abandonado sin que nadie vuelva a escribir nada.
    ///
    /// **Una decisión sin alcance habla de donde cuelga**, y aquí es donde se
    /// aparta de una pregunta o una hipótesis, para las que no tener alcance
    /// significa hablar de todo. En una decisión eso sería una trampa callada:
    /// escribir «esta línea está muerta» mirando un intento marcaría el árbol
    /// entero. Para abandonarlo todo hay que colgarla de la raíz o nombrarla.
    ///
    /// Gana la última: cambiar de opinión es decidir otra vez, y el abandono de
    /// ayer sigue escrito con su motivo.
    pub fn decided(&self) -> Result<BTreeMap<String, Course>, Trouble> {
        let known = self.all()?;
        let under = self.under()?;
        let mut said: BTreeMap<String, Course> = BTreeMap::new();
        // Por antigüedad, que aquí es el orden de los ids: el último que hable
        // de un commit es el que vale.
        for (id, body) in &known {
            let Some(course) = body.course else { continue };
            let scope = if body.scope.is_everything() {
                Scope::of(under.parents_of(*id))
            } else {
                body.scope.clone()
            };
            // Colgada de nada y sin alcance: no habla de ninguna línea en
            // concreto, así que no tiñe ninguna en vez de teñirlas todas.
            if scope.is_everything() {
                continue;
            }
            for one in scope.covers(&under) {
                let Some(reached) = known.get(&one) else {
                    continue;
                };
                for cited in &reached.cites {
                    if cited.what == "commit" {
                        said.insert(cited.id.clone(), course);
                    }
                }
            }
        }
        Ok(said)
    }

    /// Cómo está cada pregunta y cada hipótesis, contando lo que le llegó.
    pub fn standing(&self) -> Result<BTreeMap<MoveId, Standing>, Trouble> {
        let known = self.all()?;
        let under = self.under()?;
        let says = self.says()?;
        Ok(known
            .iter()
            .filter(|(_, body)| matches!(body.kind, Kind::Question | Kind::Hypothesis))
            .map(|(id, body)| {
                let mine: Vec<&Said> = says.iter().filter(|one| one.to == *id).collect();
                (*id, stands(body.kind, &mine, &under))
            })
            .collect())
    }

    /// La prosa que hay bajo una cita, o lo que sea que se guardó.
    pub fn read(&self, digest: &Digest) -> Result<Option<Vec<u8>>, Trouble> {
        self.kept.get(digest).map_err(Trouble::Store)
    }

    /// Los registros bajo `exp/<tree>/move/<id>/<what>/…`, con su id.
    ///
    /// El registro entero y no sólo el nombre: lo que hace falta de una arista
    /// —a quién apunta— está en su meta, que es lo que un escaneo trae gratis.
    fn records(&self, what: &str) -> Result<Vec<(MoveId, Record)>, Trouble> {
        let under = format!("exp/{}/move/", self.tree);
        let mark = format!("/{what}/");
        Ok(self
            .kept
            .bound()
            .map_err(Trouble::Store)?
            .into_iter()
            .filter_map(|bound| {
                let rest = bound.name.strip_prefix(&under)?;
                let (id, _) = rest.split_once(&mark)?;
                Some((id.parse().ok()?, bound))
            })
            .collect())
    }

    /// Cuántas ranuras hay ocupadas de un tipo bajo un movimiento.
    fn slots(&self, id: MoveId, what: &str) -> Result<u32, Trouble> {
        let mark = format!("/move/{id}/{what}/");
        Ok(self
            .records(what)?
            .iter()
            .filter(|(which, bound)| *which == id && bound.name.contains(&mark))
            .filter_map(|(_, bound)| bound.name.rsplit('/').next()?.parse::<u32>().ok())
            .max()
            .unwrap_or(0))
    }

    /// Reclama una ranura para un hecho sin cuerpo propio: el nombre es el dato.
    fn bind(
        &self,
        id: MoveId,
        what: &str,
        body: &str,
        meta: &[(&str, &str)],
    ) -> Result<(), Trouble> {
        let digest = self.kept.put(body.as_bytes()).map_err(Trouble::Store)?;
        self.bound(id, what, &digest, meta)
    }

    fn bound(
        &self,
        id: MoveId,
        what: &str,
        digest: &Digest,
        meta: &[(&str, &str)],
    ) -> Result<(), Trouble> {
        let first = self.slots(id, what)? + 1;
        for nth in first..first + PATIENCE {
            let meta: Meta = meta
                .iter()
                .map(|(a, b)| (a.to_string(), b.to_string()))
                .collect();
            if self
                .kept
                .claim(&self.named(id, what, nth), digest, meta)
                .map_err(Trouble::Store)?
            {
                return Ok(());
            }
        }
        Err(Trouble::Crowded)
    }
}

/// Cómo queda una cuestión dado lo que le han dicho.
///
/// Los alcances hacen el trabajo: dos aristas de signo contrario sólo son una
/// contradicción si hablan de situaciones que se tocan. «A sola funcionaba» y
/// «A+B se anulan» son dos hechos, no un conflicto.
fn stands(kind: Kind, said: &[&Said], under: &Undernath) -> Standing {
    if said.is_empty() {
        return Standing::Open;
    }
    if kind == Kind::Question {
        let answers: Vec<&&Said> = said
            .iter()
            .filter(|one| one.says == Says::Answers)
            .collect();
        return match answers.iter().any(|one| !one.in_part) {
            true => Standing::Answered,
            false if answers.is_empty() => Standing::Open,
            false => Standing::Partly,
        };
    }

    let (yes, no): (Vec<&&Said>, Vec<&&Said>) = said
        .iter()
        .filter(|one| matches!(one.says, Says::Validates | Says::Refutes))
        .partition(|one| one.says == Says::Validates);
    if yes.is_empty() && no.is_empty() {
        return Standing::Open;
    }
    // La disputa se mide por solape, no por presencia: si nadie habla de lo
    // mismo, no hay nada que disputar.
    let disputed = yes
        .iter()
        .any(|a| no.iter().any(|b| a.scope.touches(&b.scope, under)));
    if disputed {
        return Standing::Disputed;
    }
    match (yes.is_empty(), no.is_empty()) {
        (false, true) if yes.iter().any(|one| !one.in_part) => Standing::Validated,
        (false, true) => Standing::PartlyValidated,
        (true, false) if no.iter().any(|one| !one.in_part) => Standing::Refuted,
        (true, false) => Standing::PartlyRefuted,
        // Los dos signos sin tocarse: la respuesta depende de dónde se mire.
        _ => Standing::Depends,
    }
}

/// Un campo del registro, si está.
fn beside<'a>(meta: &'a Meta, what: &str) -> Option<&'a str> {
    meta.iter()
        .find(|(said, _)| said == what)
        .map(|(_, value)| value.as_str())
}

#[derive(Debug)]
pub enum Trouble {
    Store(soma_next_store::StoreError),
    Garbled(String),
    NoSuchMove { id: MoveId },
    Circular { child: MoveId, parent: MoveId },
    Nonsense { says: Says, from: Kind, to: Kind },
    NotADecision { kind: Kind },
    CannotCite { kind: Kind },
    Crowded,
}

impl fmt::Display for Trouble {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Store(why) => write!(f, "el razonamiento no se pudo alcanzar: {why}"),
            Self::Garbled(why) => write!(f, "algo no se pudo escribir ni leer: {why}"),
            Self::NoSuchMove { id } => write!(f, "no hay ningún movimiento {id}"),
            Self::Circular { child, parent } => write!(
                f,
                "colgar {child} de {parent} haría un ciclo, y un recorrido sobre un ciclo no termina"
            ),
            Self::Nonsense { says, from, to } => {
                write!(f, "un `{says}` de un {from} a un {to} no significa nada")
            }
            Self::NotADecision { kind } => {
                write!(f, "un rumbo lo lleva una decisión, y esto es un {kind}")
            }
            Self::CannotCite { kind } => write!(
                f,
                "un {kind} habla de movimientos y no de commits ni de ensayos: citar es de \
                 un intento o de un hallazgo"
            ),
            Self::Crowded => write!(f, "demasiada gente escribiendo a la vez"),
        }
    }
}

impl std::error::Error for Trouble {}
