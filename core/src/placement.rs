//! Dónde corre cada nodo.
//!
//! Es un tipo aparte, y no un campo del [`Graph`](crate::Graph) ni del
//! [`Catalog`](crate::Catalog), porque colocar es un cuarto hecho —distinto de
//! qué hay, de quién lo ejecuta y de en qué orden—:
//!
//! | pieza | contesta |
//! |---|---|
//! | [`Graph`](crate::Graph) | **qué** hay y cómo se conecta |
//! | [`Catalog`](crate::Catalog) | **quién** lo ejecuta |
//! | `Placement` | **dónde** |
//! | [`Plan`](crate::Plan) | **cuándo**, y con qué concurrencia |
//!
//! En el grafo no cabe: un `Graph` es solo topología, y además el motor no lo
//! mira —cada paso del plan es autónomo desde CU3—, así que meterlo ahí
//! obligaría a pasarle el grafo al [`Executor`](crate::Executor). En el
//! catálogo tampoco: el catálogo es la mitad que **no** es dato, y una
//! colocación sí lo es —el día que un subgrafo viaje a otra máquina, la
//! colocación viaja con él y las implementaciones no—.
//!
//! Y en el plan menos que en ninguno. El plan decide el orden y la
//! concurrencia; colocar no cambia ni uno ni otra. Que `compile` no vea esto
//! no es una omisión: es lo que hace que colocar no pueda alterar el recorrido.
//!
//! Un mapa suelto es lo que es, sin comprobar que los ids existan: eso se
//! comprueba donde hay un grafo delante —al declarar con
//! [`Wire::on`](crate::Wire::on), que solo puede nombrar nodos suyos, y en el
//! `place()` de los bindings, que valida contra el grafo—.

use crate::{Device, NodeId};
use std::collections::HashMap;

/// Dónde corre cada nodo. Los que no estén, donde caigan.
///
/// «No colocado» y «colocado en `cpu`» son cosas distintas: el primero es
/// «donde ya esté», el segundo es una orden de moverlo.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Placement {
    nodes: HashMap<NodeId, Device>,
}

impl Placement {
    /// Sin colocar nada.
    pub fn new() -> Self {
        Self::default()
    }

    /// Coloca un nodo, devolviendo dónde estuviera antes.
    pub fn place(&mut self, id: impl Into<NodeId>, device: Device) -> Option<Device> {
        self.nodes.insert(id.into(), device)
    }

    /// Dónde corre este nodo, si se dijo.
    pub fn of(&self, id: &NodeId) -> Option<&Device> {
        self.nodes.get(id)
    }

    /// Cuántos nodos están colocados.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Si no hay ninguno.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}
