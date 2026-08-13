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

Estado: **cerrado**. 16 tests en Rust, 13 en Python.

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
- [x] un grafo vacío es válido
- [x] un grafo de un solo nodo es válido
- [x] se puede añadir un nodo con id explícito
- [x] se puede añadir un nodo sin id y el sistema le pone uno (snake_case de la clase)
- [x] añadir dos veces lo mismo no duplica — **decidido**: identidad = id, y el id
      derivado sufija `_2`, `_3`. Dos filtros idénticos son dos nodos; deduplicar por
      contenido es una decisión de caché, no de topología, y ese caso de uso no existe
- [x] una tubería lineal tiene la estructura que dice tener

**Consultas de topología**
- [x] raíces y hojas
- [x] predecesores y sucesores de un nodo
- [x] orden topológico de una cadena lineal
- [x] orden topológico con ramas paralelas
- [x] un ciclo es un error — **decidido**: al poner la arista, no al recorrer

**Validación** — **decidido**: no hay `validate()`. Los constructores devuelven
`Result` y el invariante se sostiene en todo momento, así que un `Graph` inválido no
es un valor que exista. Lo que compra: `topological_sort()` no devuelve `Result`,
porque no puede fallar.
- [x] ids duplicados se rechazan, en `add_node`
- [x] una arista a un nodo inexistente se rechaza, en `add_edge`

**Diferido a casos de uso posteriores** — está en los mismos ficheros de test,
no lo arrastres a CU1: serialización (`graph_serde_roundtrip`), render
(`to_mermaid*`, `to_text`, overlays), nodos de control (`loop_and_branch_nodes`,
`subgraph_node`, todo `graph_control.rs`), y el contrato de `Filter`/`Step`
(`graph_filter.rs`, `graph_step.rs`).

### Decisiones tomadas en CU1

1. **El `Graph` del núcleo es solo topología.** Ids y aristas. Qué hace un nodo no
   es asunto suyo, porque crear un grafo no necesita saberlo. El mapa
   id → objeto Python vive en `python/`. Por eso `core` no depende de nada.
2. **Errores en la inserción, no en un `validate()`.** No hay instante en el que el
   grafo esté mal formado.
3. **DAG por construcción.** El ciclo se rechaza en `add_edge`. *Riesgo asumido*: si
   un caso de uso futuro necesitara aristas de vuelta, habría que revisarlo — en el
   original los bucles son nodos, no aristas hacia atrás, así que la apuesta es que
   no hace falta.
4. **`NodeId` es un tipo, no un `String`.** Hay más ids por venir.
5. **O(n) donde podría ser O(1).** La adyacencia se calcula al vuelo. El código se lee
   de un vistazo y ningún caso de uso ha pedido otra cosa.

### Lo que NO entró, y por qué

`target=` en `node()`. Lo escribí como azúcar para crear la arista en el mismo paso,
y al comprobarlo contra el original resultó que allí `target` **no es una arista**:
es el objetivo de supervisión que lee el `step()` del optimizador. Reutilizar el
nombre con otro significado es justo el tipo de cosa que hace un sistema
incomprensible, así que fuera: no es CU1 y no tiene consumidor hoy.

---

## Casos de uso siguientes (sin abrir)

Orden tentativo; se decide al cerrar cada uno, no ahora.

- CU2 — ejecutar un grafo lineal de filtros
- CU3 — cachear la salida de un nodo por contenido
- CU4 — validar tipos entre nodos conectados (schemas)
- CU5 — control de flujo: rama y bucle
