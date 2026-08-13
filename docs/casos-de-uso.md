# Casos de uso

El proyecto avanza por rebanadas verticales. Cada caso de uso llega hasta
Python, y se da por cerrado cuando contesta a todas las garantías de su
cuestionario.

---

## CU1 — Crear un grafo

```python
g = soma_next.Graph()
g.node("limpiar", Limpiar())
g.edge("limpiar", "vectorizar")
```

Estado: **esqueleto**. `Graph()` existe y cruza la costura; nada más.

### La decisión de diseño previa

Antes de `node()` hay que contestar a **qué es un nodo**. En el original esa
pregunta se contestó con `NodeKind` (5 variantes estructurales), `NodeMeta`
(metadatos comunes a filtros y steps, con `cacheable`/`deterministic` como
datos en vez de un `if is_step`) y un `NodeCatalog` que es el registro único.
Es una respuesta razonable; no es la única, y no se hereda.

Lo que sí conviene mirar del original antes de decidir, porque son cicatrices
de errores reales: `soma-core/src/graph/node.rs` (172 líneas) y sus tests
`graph_node.rs` — `a_filter_keeps_its_caching_contract`,
`a_step_is_not_output_cacheable`, `schemas_survive_both_directions`.

### Cuestionario (de `soma-core/tests/unit/graph*.rs`)

**Construcción**
- [ ] un grafo vacío es válido
- [ ] un grafo de un solo nodo es válido
- [ ] se puede añadir un nodo con id explícito
- [ ] se puede añadir un nodo sin id y el sistema le pone uno
- [ ] añadir dos veces lo mismo no duplica *(¿bajo qué criterio de identidad?)*
- [ ] una tubería lineal tiene la estructura que dice tener

**Consultas de topología**
- [ ] raíces y hojas
- [ ] predecesores y sucesores de un nodo
- [ ] orden topológico de una cadena lineal
- [ ] orden topológico con ramas paralelas
- [ ] un ciclo se detecta y es un error, no un cuelgue

**Validación** — la pregunta de diseño real es *cuándo*: ¿`validate()` explícito
como en el original, o typestate que impida construir el grafo inválido?
- [ ] ids duplicados se rechazan
- [ ] una arista a un nodo inexistente se rechaza

**Diferido a casos de uso posteriores** — está en los mismos ficheros de test,
no lo arrastres a CU1: serialización (`graph_serde_roundtrip`), render
(`to_mermaid*`, `to_text`, overlays), nodos de control (`loop_and_branch_nodes`,
`subgraph_node`, todo `graph_control.rs`), y el contrato de `Filter`/`Step`
(`graph_filter.rs`, `graph_step.rs`).

---

## Casos de uso siguientes (sin abrir)

Orden tentativo; se decide al cerrar cada uno, no ahora.

- CU2 — ejecutar un grafo lineal de filtros
- CU3 — cachear la salida de un nodo por contenido
- CU4 — validar tipos entre nodos conectados (schemas)
- CU5 — control de flujo: rama y bucle
