//! El contrato de lo que ejecuta un nodo. Uno solo.
//!
//! Un nodo avanza un turno y dice cómo sigue la cosa: [`Transition::Done`] si
//! ya está, [`Transition::Await`] si necesita algo del mundo antes de seguir.
//!
//! **La diferencia entre un filtro y un step es esa variante, no un tipo.** Un
//! filtro es un nodo que siempre contesta `Done`; un step, uno que a veces
//! contesta `Await`. Tener dos traits duplicaba en el sistema de tipos una
//! distinción que ya estaba en el retorno, y con ella se propagaba hacia arriba
//! —catálogo, plan, motor, errores— la obligación de saber cuál de los dos era
//! cada nodo.
//!
//! Un efecto secundario que vale por sí solo: un nodo puede **evolucionar**.
//! Empiezas devolviendo `Done` siempre, y el día que necesite consultar algo le
//! añades una rama `Await` en el mismo cuerpo. Con dos traits eso obligaba a
//! reescribir el tipo entero.
//!
//! Lo que un nodo pide es **opaco para el núcleo**: un [`Value`] que el
//! [`Driver`](crate::Driver) sabe interpretar. Por eso aquí no hay ni LLMs, ni
//! herramientas, ni diario de efectos.

use crate::{Device, Value};

/// Algo que un nodo sabe hacer.
///
/// `Send + Sync` no es decoración: un `Graph` de Python es un `#[pyclass]`, y
/// PyO3 exige que un pyclass sea `Send`. Como el grafo lleva dentro el
/// catálogo, y el catálogo `Arc<dyn Node>`, la cota sube hasta aquí.
pub trait Node: Send + Sync {
    /// Avanza un turno.
    ///
    /// Se llama con `ctx.turn == 0` y sin resultados; después, con lo que el
    /// driver devolvió de lo que se pidió en el turno anterior, en el mismo
    /// orden. `input` es siempre el mismo: lo que llegó por las aristas.
    ///
    /// # Errores
    /// Lo que el nodo quiera decir; el motor lo envuelve con el id del nodo.
    fn forward(&self, input: &Value, ctx: &Ctx<'_>) -> Result<Transition, NodeError>;
}

/// Lo que un nodo sabe además de su entrada.
///
/// Un nodo que termina a la primera no lo mira nunca — de ahí que `input` vaya
/// aparte y no aquí dentro: es lo único que le importa al caso común, y no
/// tiene por qué atravesar una struct para llegar a ello.
#[derive(Debug, Clone, Copy)]
pub struct Ctx<'a> {
    /// Cuántas veces se le ha preguntado ya; empieza en 0.
    pub turn: usize,
    /// Lo que devolvió el driver de lo pedido en el turno anterior, en orden.
    /// Vacío en el turno 0.
    pub results: &'a [Value],
    /// Dónde se dijo que corriera este nodo, si se dijo.
    ///
    /// Es el único canal por el que una colocación llega a una
    /// implementación, y llega como **información**: el núcleo no sabe mover
    /// nada a una GPU, así que quien obedece es el nodo. Lo que sí hay es una
    /// postcondición al otro lado de la costura de Python, para que
    /// desobedecer no salga gratis ni en silencio.
    pub device: Option<&'a Device>,
}

/// Cómo sigue la cosa después de un turno.
///
/// Deliberadamente **sin** `#[non_exhaustive]`: quien ejecuta un nodo tiene que
/// decidir qué hacer con cada variante, y un brazo comodín ahí es una respuesta
/// equivocada en silencio. Añadir una variante *debe* romper a todo el mundo.
#[derive(Debug, Clone, PartialEq)]
pub enum Transition {
    /// Terminado, con esta salida.
    Done(Value),
    /// Necesita que alguien haga esto antes de seguir. Se le volverá a
    /// preguntar con los resultados.
    Await(Vec<Value>),
}

/// Lo que un nodo puede contestar cuando no puede avanzar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeError(String);

impl NodeError {
    /// Un fallo descrito con un mensaje.
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }

    /// El mensaje.
    pub fn message(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for NodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for NodeError {}
