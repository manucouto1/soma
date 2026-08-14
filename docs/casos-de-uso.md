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

## CU2 — Ejecutar un grafo de filtros

```python
g.node("sumar", Sumar(1))
g.node("doblar", Doblar())
g.edge("sumar", "doblar")
g.forward(41)          # → 84.0
```

Estado: **cerrado**. 29 tests en Rust, 25 en Python.

### La decisión de partida

**El motor va en Rust, Python es envoltorio.** Es decisión tuya y tiene una
consecuencia que conviene tener presente: obliga a que exista `Value` ya. Si el
núcleo ejecuta, los datos tienen que tener una forma que Rust entienda.

Los cuatro papeles, que es fácil confundir:

| pieza | papel | dónde |
|---|---|---|
| `Graph` | la **estructura** | `core/src/graph.rs` |
| `Catalog` | el **almacén** de implementaciones | `core/src/filter.rs` |
| `Filter` | el **contrato** de una unidad ejecutable | `core/src/filter.rs` |
| `Graph::run` | el **motor** | `core/src/execution.rs` |

### Decisiones tomadas

1. **`Value` con cuatro variantes**: `Null`, `Text`, `Bytes`, `Tensor`. No hay
   `Json` porque pediría `serde_json` y el núcleo no depende de nada; no hay
   `Object` opaco porque solo sirve para mandar algo por un cable y no hay cable.
   El error de conversión dice qué falta en vez de inventarse una representación.
2. **`Filter` tiene un método**: `forward(&Value) -> Result<Value, FilterError>`.
   Sin `fit` (entrenar es otro caso de uso), sin `config_hash` (caché), sin
   `meta` (compilador), sin `composite_fit` (autograd).
3. **Sin parámetro `state`.** El original pasa `state` en cada `forward` incluso
   a los filtros sin estado. El estado llega con `fit`.
4. **`Send + Sync` en el trait** — y no es decoración: PyO3 exige que un
   `#[pyclass]` sea `Send`, el grafo lleva el catálogo dentro, y la cota sube
   hasta el trait. El compilador lo descubrió solo.
5. **Un objeto sin `forward` falla al registrarlo**, no a mitad de un run.
6. **Un bool no cruza la frontera.** `True` como el tensor `1.0` es la clase de
   conversión silenciosa que después nadie entiende.

### Las dos decisiones que el motor NO tomaba

Las dos se cerraron en CU4. `Fanin` y `ManyLeaves` ya no existen como errores.

## CU3 — La forma de la ejecución, y los steps

```python
g.step("agente", Agente())          # un objeto con poll(ctx)
g.forward(x, driver=MiDriver())     # quien atiende lo que pida
g.plan()                            # cómo se va a recorrer
```

Estado: **cerrado**. 39 tests en Rust, 38 en Python.

### La pregunta: ¿hay varias formas de ejecutar un grafo?

Sí — local, remota, por turnos, en paralelo. Pero la respuesta del original **no
es un trait de ejecutores**: es un enum de diez variantes (`Sequence | Parallel |
Execute | Step | Loop | Branch | Remote | Composite | Stream | Empty`) y un solo
`match`. `Remote` no es otro ejecutor: es una variante que *envuelve* un
sub-plan.

Es el mismo principio que ya habíamos encontrado —la variación como dato, no
como subtipo— y tiene una ventaja concreta sobre una función suelta o un trait:
al añadir `Parallel` a mitad de este caso de uso, el compilador señaló el único
sitio que tenía que decidir qué hacer con ella. Un brazo comodín no habría dicho
nada.

### La pieza que faltaba: compilar

Entre la estructura y el motor hay ahora un paso: `compile(&Graph, &Catalog) ->
Plan`. Decide la forma, y de paso **todo lo estructural se detecta antes de
ejecutar nada**. El motor ya no resuelve de dónde sale la entrada de cada nodo:
eso lo dice el plan.

Necesita el catálogo porque la forma depende de qué es cada nodo — un filtro se
llama una vez, un step se conduce por turnos.

### Decisiones tomadas

1. **`Plan` es un enum**, no un trait. Cerrado, exhaustivo, sin comodines.
2. **`Parallel` significa "las ramas no dependen entre sí"**, no "corre en
   hilos". Repartirlas es una decisión que no cambia el resultado y no la ha
   pedido nadie.
3. **Un abanico produce una `Value::List`** con lo de cada rama, en orden.
4. **`Executor` es un tipo**, no una función suelta: ejecutar necesita contexto
   (hoy el almacén y el driver; mañana caché y eventos). Ese "mañana" es lo que
   en el original se llama `GraphSession`.
5. **Lo que un step pide es opaco para el núcleo**: un `Value` que el `Driver`
   interpreta. Por eso no hay ni LLMs, ni herramientas, ni diario de efectos —
   eso es biblioteca y persistencia, no el contrato.
6. **`Transition` tiene dos variantes**: `Done` y `Await`. `Spawn`, `Goto` y
   `Suspend` entrarán con su caso de uso. Sin `#[non_exhaustive]`, a propósito.
7. **Tope de 64 turnos.** Un step que no termina es un bug del step; el tope
   hace que se note como un error con nombre y no como un proceso parado.
8. **`Value` pierde `Tensor` y gana `Number` y `List`.** Nadie producía un
   tensor con forma, y la ida y vuelta a Python tiene que ser simétrica: lo que
   entra como lista sale como lista.

### Lo que NO entró

**`Plan::Remote`.** No hay transporte, así que sería una variante que nadie
puede ejecutar. Lo que compra el enum es justo que añadirla el día que haya
worker sea una variante más, y que el compilador señale cada sitio que tenga que
decidir.

> **Nota de CU4**: `Plan::Parallel` y los errores `Fanin`/`ManyLeaves` que se
> describen arriba ya no existen. Ver abajo por qué.

## CU4 — Abanicos en las dos direcciones

```python
Graph.somatize(Izq().named("izq") | Der().named("der")) >> Media()
# a `Media` le llega {"izq": …, "der": …}
```

Estado: **cerrado**. 46 tests en Rust, 52 en Python.

### La pregunta: ¿dónde vive la agregación?

El original la contesta **dos veces**, y las dos enseñan:

- **En la arista** (forward): junta lo que llega en un `serde_json::Map` con la
  clave del nodo origen, y el agregador es un nodo corriente — `MajorityVote`
  es un `Filter`.
- **En el entrenamiento** (federated): `FederatedAggregation::{FedAvg, FedProx,
  FedYogi}` y `GradientAggregation::{AllReduce, ParameterServer, …}` — enums de
  algoritmos, envueltos en traits `StateAggregator`/`GradientAggregator` con
  **un solo implementador cada uno: el propio enum**. Los dos están en la lista
  de traits huérfanos. Cuando enumeraron los algoritmos reales de FL les salió
  un enum; el trait de encima no compró nada.

Y la trampa que hay que ver: **la agregación federada no es fan-in.** En FedAvg
lo que se promedia son estados de N workers al cerrar una ronda — ahí no hay
arista ni predecesores. Es una operación dentro de `fit`, y llegará con él.

### Decisiones tomadas

1. **No hay trait `Aggregator`.** Un agregador es un filtro que lee un mapa.
   `Media`, `MajorityVote`, `Concat`, `WeightedMean` son biblioteca.
2. **`Value::Map`, ordenado.** Un `HashMap` itera distinto en cada proceso, así
   que pasarlo a lista daría un orden distinto cada vez y el hash por contenido
   —cuando llegue la caché— sería inservible. Los pares van en orden de
   declaración de las aristas, que es además lo simétrico con un `dict` de
   Python: la ida y la vuelta dan el mismo dict.
3. **Las dos direcciones tienen la misma forma.** Varias entradas → un mapa con
   la clave de cada origen. Varias hojas → un mapa con la clave de cada hoja.
   Un diamante da la vuelta.
4. **El peso viaja con el valor.** FedAvg pondera por muestras de cada cliente;
   ni una lista ni un mapa de salidas crudas dan ese peso. Cada rama produce
   algo como `{"update": …, "n": 128}` — otra razón independiente para tener
   `Value::Map`.

### Lo que se quitó

**`Plan::Parallel`**, añadido en CU3. Se rompía con el fan-in: en un diamante
las dos ramas reclamaban el nodo de unión y se ejecutaba dos veces. La forma
correcta es que **cada paso lleve escrito de dónde sale su entrada**
(`Execute { node, from }`). Con eso el plan sigue siendo autónomo —el motor no
vuelve a mirar el grafo— y los abanicos salen sin ninguna variante especial.
`Parallel` volverá cuando signifique algo que hoy no significa: repartir entre
hilos.

Con él se fueron los errores `CompileError::Fanin` y `ManyLeaves`. `CompileError`
se queda con una sola variante.

## CU5 — El DSL

```python
from soma_next import Filter, Graph

g = Graph.somatize(Fuente() >> (Izq().named("izq") | Der().named("der")) >> Media())
g.forward(0)
```

```rust
let (graph, catalog) = (filter("fuente", Sumar(1.0))
    >> (filter("izq", Sumar(10.0)) | filter("der", Sumar(100.0)))
    >> filter("juntar", Media))
.somatize()?;
```

Estado: **cerrado**. Mismos tests, más `build.rs` y `test_dsl.py`.

`>>` encadena, `|` abre en ramas, y un `>>` detrás de unas ramas abiertas las
cierra — que es el fan-in de CU4: el nodo de la derecha recibe el mapa. La
expresión de arriba *es* el diamante.

### Decisiones tomadas

1. **La misma sintaxis en los dos lenguajes.** En Rust sale de implementar
   `std::ops::Shr` y `BitOr` sobre un tipo propio; no hace falta un macro
   (`macro_rules!` daría sintaxis que los operadores no dan, pero para esto no
   hace falta). La precedencia también coincide: `>>` aprieta más que `|`, así
   que las ramas van entre paréntesis en los dos.
2. **Se llama `somatize`, que es el verbo del proyecto.** En Python es un
   classmethod de `Graph`; en Rust, un método de `Wire`, porque devuelve **dos**
   cosas —la estructura y el almacén— y ninguna contiene a la otra.
3. **Hay una clase Python encima de la de Rust.** `somatize` recorre una
   expresión de objetos Python, así que no puede estar en Rust; y un
   `#[pyclass]` es un tipo inmutable, así que tampoco se le puede colgar al
   importar. Se declara en el cuerpo de una subclase, que además es lo único
   que ven `help()`, un IDE y mypy. Es la misma estructura que el Soma original
   (`soma/_graph.py`), y por las mismas razones.
4. **`Filter` y `Step` son abstractas, y la herencia es lo que decide.** Cada
   una exige su método con `@abstractmethod`, así que un `class X(Filter)` sin
   `forward` no se puede ni instanciar, y `isinstance` es la única pregunta que
   se hace el DSL.

   Fue una corrección: nacieron como mixins vacíos que preguntaban por duck
   typing si el objeto *tenía* `poll`. Eso dejaba pasar dos cosas feas —un
   `class X(Step)` con solo `forward` acababa registrado como filtro sin un
   aviso, y un objeto con los dos métodos daba tres respuestas distintas según
   entrara por `node()`, por `step()` o por el DSL—. Los nombres prometían un
   contrato que nadie exigía.

   `node()` y `step()` siguen siendo la puerta de abajo y aceptan un objeto de
   fuera que no herede de nada, porque ahí el tipo lo elige quien llama. Lo que
   no aceptan es la contradicción: `node()` con algo que hereda de `Step` es un
   error que dice cuál es la llamada correcta.
5. **`Wire` no materializa hasta `somatize`.** Guarda por dónde se entra y por
   dónde se sale, más las listas de nodos y aristas. Así juntar dos trozos es
   concatenar listas y no fusionar dos grafos, y un id repetido se cuenta al
   final, una sola vez.
6. **El DSL no es otra cosa que `node` y `edge`.** Hay un test que construye el
   mismo grafo de las dos formas y compara nodos, aristas y plan.

## CU6 — Un solo contrato

```rust
pub trait Node: Send + Sync {
    fn forward(&self, input: &Value, ctx: &Ctx<'_>) -> Result<Transition, NodeError>;
}
```

Estado: **cerrado**. 48 tests en Rust, 61 en Python.

### La pregunta: ¿por qué dos tipos?

La diferencia entre un filtro y un step era una sola cosa: si puede terminar
solo. Pero **eso ya estaba dicho en `Transition`** — un filtro es un nodo que
siempre contesta `Done`. Tener dos traits duplicaba en el sistema de tipos una
distinción que vivía en el retorno, y con ella propagaba hacia arriba la
obligación de saber cuál era cada nodo: catálogo, plan, motor, errores,
adaptadores, DSL. **35 sitios.**

Antes de decidir se probaron dos alternativas más, y las dos se descartaron con
razones concretas:

- **Un trait de azúcar con blanket impl** (`impl<T: Filter> Node for T`).
  Compilado: `error[E0034]` — con dos traits a la vista el nombre `forward`
  queda ambiguo *aunque las aridades sean distintas*, porque Rust resuelve el
  nombre antes que los argumentos. Y `error[E0119]` — un tipo que implementa
  `Filter` ya no puede implementar `Node` a mano, así que un nodo no podía
  evolucionar de terminar siempre a pedir un turno sin reescribirse entero.
- **El estado como continuación** (`Pending { requests, resume }`). Más simple
  en superficie —se lleva `Ctx` entero— pero rompe el replay determinista: el
  diario se apoya en que `forward` sea determinista dado `(turn, results)`, y
  reanudar una continuación exigiría serializar un `Box<dyn Node>`.
  La variante typestate muere antes: el `Catalog` es un mapa heterogéneo que
  borra el parámetro de tipo, y no cruza a Python en absoluto.

### Decisiones tomadas

1. **Un trait, un método, `forward` en los dos lenguajes.** Sin el segundo
   trait no hay ambigüedad de nombres, así que no hace falta llamarlo
   `advance` en Rust.
2. **`Pure` es una struct, no un trait.** Azúcar para envolver una función.
   Cumple nuestra propia regla: si no puedes nombrar dos implementadores, es
   una struct. Y no reintroduce ninguno de los dos errores de arriba.
3. **`input` va aparte del contexto.** Un nodo que termina a la primera no mira
   `ctx` nunca; no tiene por qué atravesar una struct para llegar a lo único
   que le importa.
4. **En Python se quedan `Filter` y `Step`, como fachada.** Son las dos
   convenciones de llamada cómodas —`forward(x)` devuelve un valor,
   `forward(x, ctx)` devuelve una transición— y la herencia sigue decidiendo
   cuál se usa. Debajo hay un solo contrato. La separación pasó de ser del
   sistema a ser de la puerta.

### Lo que desapareció

`trait Step`, `FilterError`, `StepError`, `StepCtx`, `NodeImpl`, `Plan::Step`,
`RunError::{Filter, Step, WrongKind}`, `insert_filter`/`insert_step`,
`run_filter`/`drive_step`, y `PyStep` como adaptador aparte.

`RunError` pasa de 7 variantes a 5. `Plan` de 4 a 3. `compile` ya no necesita el
catálogo para saber de qué tipo es cada nodo — solo para comprobar que lo hay.

### Lo que se ganó, y no estaba en el plan

**Un nodo puede evolucionar.** Empieza devolviendo `Done` siempre, y el día que
necesite consultar algo se le añade una rama `Await` en el mismo cuerpo, sin
cambiar de tipo ni de registro. Con dos traits eso era `error[E0119]`. Hay test.

## CU7 — El mismo mecanismo en los dos lenguajes

```python
from soma_next import Await, Done, Graph, Node

class Limpiar(Node):
    def forward(self, x, ctx):
        return Done(x.strip())

class Preguntar(Node):
    def forward(self, x, ctx):
        if ctx.turn == 0:
            return Await([f"¿y {x}?"])
        return Done(ctx.results[0])

Graph.somatize(Limpiar() >> Preguntar()).forward("  hola  ", driver=MiDriver())
```

Estado: **cerrado**. 47 tests en Rust, 56 en Python.

### Lo que faltaba de CU6

CU6 unificó el núcleo pero dejó Python con dos clases (`Filter` y `Step`) y dos
convenciones de llamada. Era una asimetría sin razón: si debajo hay un contrato,
arriba no hay por qué elegir puerta.

### Decisiones tomadas

1. **Una sola clase `Node` en Python**, con `forward(input, ctx)` que devuelve
   una transición. `Filter` y `Step` desaparecen, y con ellas `g.step()`,
   `kind_of`, `ensure_kind` y los dos `override` de `Graph`.
2. **`Ctx`, `Done` y `Await` son `#[pyclass]`**, no diccionarios. Son los mismos
   conceptos del núcleo cruzando la costura, así que el adaptador los reconoce
   por su tipo en vez de adivinar por las claves de un dict, y `ctx.turn` se lee
   como en Rust en lugar de `ctx["turn"]`.
3. **Un adaptador, no dos.** `PyFilterNode` y `PyStepNode` se funden en `PyNode`.
4. **Fuera `Pure`.** Era azúcar en el núcleo para envolver una función, y cada
   implementación decide cómo transiciona sin necesitar un atajo.

### El precio, dicho claro

Un nodo que solo transforma escribe ahora `return Done(x.strip())` y acepta un
`ctx` que no mira. Es más ceremonia que `return x.strip()`, y es deliberado:
compra que **no haya dos formas de escribir un nodo**, que el DSL tenga una sola
puerta, y que un nodo pueda ganar un turno añadiendo una rama en vez de
cambiando de clase.

## Casos de uso siguientes (sin abrir)

Orden tentativo; se decide al cerrar cada uno, no ahora.

- CU8 — cachear la salida de un nodo por contenido
- CU9 — validar tipos entre nodos conectados (schemas)
- CU10 — control de flujo: rama y bucle
