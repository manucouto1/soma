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

## CU8 — Un valor que cruza sin convertirse

```python
class Capa(Node, nn.Module):
    def __init__(self, m):
        nn.Module.__init__(self); self.m = m
    def forward(self, x, ctx):
        return Done(Opaque(self.m(x)))

g = Graph.somatize(Capa(l1) >> Capa(nn.ReLU()) >> Capa(l2))
y = g.forward(Opaque(x))
y.pow(2).sum().backward()      # atraviesa los tres nodos
```

Estado: **cerrado**. 53 tests en Rust, 64 en Python.

### El problema

`Value` es una frontera de **conversión**, y hay valores que no sobreviven a
convertirse. El caso que lo motivó: un tensor de torch a mitad de una gráfica de
autograd. Medido — pasarlo a listas y de vuelta da `requires_grad = False,
grad_fn = None`. El grafo de gradientes se rompe.

### La decisión

Una variante, y solo una:

```rust
Opaque(Arc<dyn Any + Send + Sync>)
```

No es un `PyObject` porque el núcleo no depende de PyO3 y no va a empezar.
`Arc<dyn Any + Send + Sync>` deja que el crate de Python guarde dentro un
`Py<PyAny>` y lo recupere con `downcast_ref`, sin que el núcleo sepa que hay
Python detrás.

**Lo que significa la variante, y de donde sale todo lo demás**: este valor solo
existe en este proceso y en este run.

| propiedad | consecuencia | ¿correcto? |
|---|---|---|
| no se hashea por contenido | el nodo no se memoiza | sí — memoizar un tensor a mitad de autograd sería un error |
| no se serializa | ese subgrafo no viaja a otra máquina | sí — por eso el original manda gradientes por el cable, no la gráfica |
| solo se compara por identidad (`Arc::ptr_eq`) | dos envoltorios del mismo objeto son distintos | es lo único que el núcleo puede afirmar |

Las fronteras de la futura caché y de la ejecución remota quedan **visibles en el
tipo** en vez de ser una regla que alguien tenga que recordar.

### Se pide a mano, a propósito

`Opaque(x)` se escribe. Se descartó que un objeto desconocido se volviera opaco
solo: se perdería la honestidad de "un `set` no cruza", y un hueco por el que
cabe todo se convierte en el camino por defecto — dejando el grafo sin caché,
sin schemas y sin distribución a la vez, sin que nadie se entere.

También se descartó un **registro de tipos opacos** que `soma_next.torch`
rellenaría al importarse: añade estado global mutable y dependencia del orden de
importación, para ahorrar una palabra.

El nodo que lo recibe lo ve **desenvuelto**, así que solo se escribe al
devolverlo (y una vez en la entrada del grafo).

### Limitaciones, medidas

- **`torch.compile` no funde entre nodos.** Tres nodos → 3 grafos y 2 roturas;
  lo mismo sin Rust → 1 grafo, 0 roturas. Es correcto (el backward llega) pero
  **el nodo es la unidad de compilación**. Mitigación del usuario, una línea:
  `torch.compile(mi_modulo)` dentro del nodo.
- **Sin caché por contenido en esas aristas.** Para entrenar es lo correcto;
  para inferencia es una pérdida real. Se recupera convirtiendo a propósito en
  el borde: `Done(y.detach().tolist())`.
- **Fuera del alcance de los schemas** cuando lleguen: no hay dtype ni shape.
- **El GIL** serializa el despacho por nodo; torch lo libera durante los
  kernels.

### El patrón, probado de extremo a extremo

`python/tests/test_pipeline_torch.py` monta un pipeline de cuatro nodos —
lematizador (sin gradientes) → encoder → cuello de botella → clasificador LSTM —
y lo **entrena**: 12.571 parámetros, la pérdida baja de 1.09 a 0.005 en 40 pasos.
Está entero en los tests porque además de comprobar, documenta el patrón.

Tres cosas que enseña, y que no son evidentes:

- **Los dos regímenes conviven.** El lematizador devuelve texto, que cruza
  convertido; los tres nodos con parámetros devuelven `Opaque`. La frontera cae
  sola donde empieza la gráfica de gradientes, sin declararla.
- **El nodo tiene los módulos, no hereda de `nn.Module`.** Heredar registra los
  parámetros solo, pero rompe llamar al nodo como módulo: nuestro `forward`
  lleva `ctx` y torch lo llama sin él (`TypeError`). Comprobado.
- **El bucle de entrenamiento va fuera**, y la línea que recoge los parámetros
  recorriendo `g.nodes()` es exactamente el dolor que un
  `soma_next.torch.parameters(g)` borraría. Está a la vista para que se decida
  con el ejemplo delante.

### Lo que NO entró

`soma_next.torch` —`module()`, `parameters()`, el bucle de entrenamiento— queda
para cuando esté claro cómo debe funcionar. **El núcleo aporta el hueco; quien
sabe qué hay dentro es biblioteca**, y esa separación es la que permite que esto
se cerrara sin decidir aquello.

## Lo siguiente, sin decidir (16 de agosto de 2026)

La discusión quedó abierta aquí, con **CU12 en duda**. «Micro-lotes» tapa tres
problemas distintos que no tienen ni el mismo dueño ni el mismo valor:

| problema | qué lo resuelve | de quién es | ¿consumidor hoy? |
|---|---|---|---|
| el lote no cabe en memoria | partirlo y acumular gradientes | del **Trainer**, cinco líneas | sí, y es el 80% de los casos |
| la burbuja: `cuda:1` parado mientras computa `cuda:0` | encadenar micro-lotes | del **grafo** | dudoso |
| acotar las activaciones vivas | el planificador 1F1B de verdad | **de nadie**, y ése es el problema | no |

Las dos razones de la duda:

**La burbuja puede que ya no exista.** CUDA lanza asíncrono: un bucle de
micro-lotes en el host ya solapa los dispositivos sin planificador, porque nada
sincroniza por el camino —`Opaque` envuelve el tensor y no hay ningún `.item()`
en la costura—. Lo que un planificador añadiría **hay que medirlo antes de
escribirlo**, y con una sola GPU aquí no se puede medir.

**El 1F1B de verdad no es nuestro.** Su valor no es la burbuja sino acotar
cuántos micro-lotes tienen las activaciones vivas, y para eso hay que
intercalar los backward entre los forward. El backward lo dispara el Trainer,
no el motor. Un planificador 1F1B obligaría a que el plan supiera del backward
— o sea, a meter el entrenamiento dentro del grafo, que es justo lo que CU11
decidió que no. La versión que sí encaja en los niveles —micro-lotes solo hacia
delante— es la que menos valor tiene.

### Los tres candidatos

- **A. Acumulación de gradiente en el Trainer.** El problema real del 80%,
  nivel 2, cero Rust. Honesto y pequeño; no enseña nada nuevo del diseño.
- **B. Un worker local: `Plan::Remote` con destino un proceso de esta misma
  máquina.** ← *recomendado.* Es el único con beneficio **medible aquí y
  ahora**: dos nodos Python en una wave se serializan contra el GIL, y
  `test_waves.py` ya lo deja documentado como limitación conocida; en dos
  procesos, no. Todo Rust —el trait `Transport`, la serialización, `Opaque`
  como `CompileError`—, primer trait nuevo desde CU2, y prepara CU13 sin
  necesitar una segunda máquina.
- **C. Lo que no cabe en la GPU**: poder pedir que dos ramas estructuralmente
  paralelas no se solapen, y liberar lo que ya no lee nadie. Todo Rust, y toca
  lo más delicado: el compilador y el motor.

## Casos de uso siguientes (sin abrir)

El orden sale de la investigación bibliográfica de agosto de 2026, y el
contenido de los dos últimos cambió al cerrar CU11: la separación en tres
niveles —grafo, entrenamiento, estudio— reordena qué mecanismo resuelve qué.

- CU12 — micro-lotes: solapar dentro de una rama (GPipe, 1F1B). **Nivel del
  grafo**: es un forward por dentro
- CU13 — `Plan::Remote`: transporte (el único trait nuevo, `Transport`), y
  `Opaque` cruzando el cable pasa a ser un `CompileError`. Solo para repartir
  **un grafo** entre hosts —una red que no cabe en una máquina—, que es model
  parallel y **split learning** a la vez. Repartir *entrenamientos enteros* es
  otra cosa y no lo necesita
- CU14 — federado: `map` sobre los clientes y `reduce` con FedAvg, en el nivel
  del estudio. FedAvg, FedProx y compañía son biblioteca — **funciones**, no
  nodos. Aquí es donde toca contestar qué exporta un entrenamiento como estado
- *(candidata)* liberar lo que ya no lee nadie, y poder pedir que dos ramas
  estructuralmente paralelas no se solapen. Las dos nacen de «no me cabe en la
  GPU» y las dos son del nivel del grafo
- *(candidata, con condición de entrada)* **entrenar desde Rust**. No se abre
  hasta que haya consumidor, y el consumidor tiene nombre: un cliente federado
  que entrene **sin un CPython cargado**. Investigado el 16 de agosto de 2026,
  y estos cuatro resultados ahorran tener que averiguarlo otra vez:
  - `tch::Tensor` es `Send` pero **no `Sync`** (`unsafe impl Send for Tensor`
    en `wrappers/tensor.rs`, y ningún `Sync` en todo el crate), así que **no
    cabe en `Value::Opaque`**, cuya cota es `Arc<dyn Any + Send + Sync>`. `tch`
    queda descartado salvo envolviendo cada tensor en un `Mutex`
  - `candle_core::Tensor` **sí** es `Send + Sync` —lleva `Arc<RwLock<Storage>>`
    dentro, y su propio código explica que eligieron el `RwLock` justo para
    eso—, así que cabría hoy **sin tocar el núcleo**. Comprobado compilando
  - el límite que no se mueve: **un grafo es todo-Python o todo-Rust para los
    tensores**. Un `Opaque` puesto por Python lleva un `PyObject` dentro, y un
    nodo de Rust que hiciera `downcast_ref::<candle::Tensor>()` obtendría
    `None`. No hay puente barato: convertir de verdad sería copiar los datos
    crudos y perder la gráfica de autograd, que es lo que `Opaque` evita
  - el orden, si llega el día: **primero un nodo de Rust con parámetros**,
    después la recogida, y el Trainer al final. Nunca empezando por el Trainer

## CU9 — Las ramas corren a la vez

```python
g = Graph.somatize(
    Fuente()
    >> ((Encoder() >> Cuello()) | (Otro() >> Otro2()))
    >> Juntar()
)
g.plan()      # Sequence([Execute, Wave([Sequence, Sequence]), Execute])
g.forward(x)  # las dos ramas, en dos hilos, de principio a fin
```

Estado: **cerrado**. 88 tests en Rust, 86 en Python.

### La pregunta: ¿qué agrupa una wave?

`Plan::Parallel` se añadió en CU3 y se quitó en CU4 porque se rompía en el
diamante: sus ramas se solapaban —las dos reclamaban el nodo de unión— y se
ejecutaba dos veces. Se dijo entonces que volvería «cuando signifique algo que
hoy no significa: repartir entre hilos». Este es ese día, y la pregunta que
faltaba por contestar es **qué va dentro de cada rama**.

Se probaron dos respuestas y la primera se descartó con un contraejemplo:

- **Por nivel topológico** (Kahn por niveles). Cada wave es una anticadena, y
  sus miembros son pasos sueltos. Es correcto y ningún nodo puede duplicarse.
  Pero con `a >> (b >> b2 >> b3 | c >> c2) >> d` sale
  `Seq([a, Wave([b,c]), Wave([b2,c2]), b3, d])`: **lockstep**. `b2` no arranca
  hasta que `c` termina aunque no dependa de ella, y `c2` acaba y se queda
  mirando mientras `b3` corre sola. Peor para lo que viene: el dispositivo de
  torch es *thread-local*, así que una rama que salta de hilo en cada wave no
  puede fijarlo una sola vez.

- **Por rama**, que es lo que quedó. `Seq([a, Wave([Seq([b,b2,b3]), Seq([c,c2])]), d])`.
  Un hilo por rama, de principio a fin, y una sola junta.

### La pieza que faltaba: descomponer, no aplanar

`compile` deja de recorrer el orden topológico y **recupera el árbol**. Y tiene
que recuperarlo del grafo, no de la expresión: la decisión 6 de CU5 dice que el
mismo grafo construido con `node()`/`edge()` en un bucle da el mismo plan, y un
bucle no tiene árbol. **La expresión del DSL es el oráculo, no el origen.**

Cuatro casos, y el orden importa:

| caso | sale |
|---|---|
| ningún nodo | `Empty` |
| un nodo | `Execute` |
| el subgrafo se parte en componentes conexas | `Wave`, una rama por componente |
| hay un **corte serie** | `Sequence` de los dos lados |
| no hay corte | secuencia plana: no es serie-paralelo |

Un **corte serie** `(A, B)` es lo que hace un `>>`: las aristas que cruzan van
de **todos** los sinks de `A` a **todos** los sources de `B`, y de ningún otro
sitio. Las dos mitades de la comprobación hacen falta — sin la primera, una
arista que sale de un nodo interior pasa por buena.

Bastó con probar los **prefijos de un orden topológico**, y es demostrable: en
una composición serie todo nodo de `A` alcanza un sink de `A`, todo sink de `A`
tiene arista a todo source de `B`, y todo nodo de `B` es alcanzable desde un
source de `B`. Luego todo nodo de `A` precede a todo nodo de `B` en *cualquier*
orden topológico. No hay que enumerar subconjuntos.

### La regla que se descartó por el camino

Antes del corte serie se intentó cortar por un **nodo barrera** —uno tal que
`ancestros(x) ∪ {x} ∪ descendientes(x)` fuera todo—. Solo acierta cuando la
junta es un nodo único. El contraejemplo, que está como test:

```
(a >> a2 | b) >> (c | d)

con barrera →  Seq([ Wave([a, b]), a2, Wave([c, d]) ])   ← parte la rama
correcto    →  Seq([ Wave([Seq([a,a2]), b]), Wave([c,d]) ])
```

### Decisiones tomadas

1. **`Wave(Vec<Plan>)`, no `Vec<Execute>`.** Una rama es un plan entero. Lo
   restrictivo era más simple de ejecutar, pero no sabe expresar una rama de
   varios nodos, que es justo el caso.
2. **Una wave significa «se lanzan a la vez»**, no «son independientes». Esa
   fue la lección de CU4: una variante que solo describe estructura no compra
   nada.
3. **Las ramas son componentes conexas**, así que son disjuntas por
   construcción y ningún nodo puede aparecer en dos. El bug que mató a
   `Parallel` no puede volver: hay un test que lo comprueba sobre una batería
   de topologías.
4. **`std::thread::scope`, sin dependencias.** Presta `&Catalog` y `&Driver`
   sin envolverlos. Las cotas que lo permiten —`Node: Send + Sync`,
   `Driver: Send + Sync`— llevan puestas desde CU2 por otra razón: PyO3 exige
   `Send` en un pyclass. Rayon habría sido la respuesta obvia y la peor.
5. **Cada rama copia lo producido y devuelve lo suyo**; el padre funde al
   juntar. Copiar sale barato porque un `Value` se clona por `Arc`, y a cambio
   no hay ni un cerrojo.
6. **El error es el de la primera rama declarada**, no el de la que falló antes
   en el reloj. Si dos ramas se rompen a la vez, cuál llega primero es una
   carrera y el mensaje no puede depender de ella.
7. **Un panic dentro de una rama no se traga**: se propaga con
   `resume_unwind` después de que `scope` haya esperado a las demás.
8. **Una cadena lineal compila al plan de antes, idéntico.** Es la regresión
   que más importa: todo lo cerrado de CU2 a CU8 son cadenas.
9. **Lo que no es serie-paralelo se recorre en secuencia**, como antes. No es
   un fallo ni un aviso: es lo que había.

### La frontera afortunada

**La imagen del DSL son exactamente los grafos serie-paralelos.** `>>` compone
en serie conectando todos los terminales con todas las cabezas, `|` compone en
paralelo con unión disjunta, y no hay una tercera operación.

El patrón mínimo que no es serie-paralelo es la «N» —`a→c`, `a→d`, `b→d`—, y
**no se puede escribir con `>>` y `|`**. Para llegar a ella hay que usar
`node()`/`edge()`. Así que la línea se explica en una frase: *si lo escribiste
con el DSL, se paraleliza; si no, se paraleliza cuando se puede*. Que haya DAGs
sin árbol es un teorema, no un hueco del algoritmo — ver Valdes, Tarjan y
Lawler, «The recognition of series parallel digraphs», SIAM J. Comput. 11(2),
1982.

### El GIL, que es donde esto se cuelga

`Graph.forward` tenía el GIL cogido mientras corría el motor. En cuanto una
wave lanza hilos que llaman al `forward` de un objeto Python, esos hilos se
bloquean pidiéndolo y **el proceso entero se congela** — ni un
`join(timeout=…)` del hilo principal vuelve, porque también necesita el GIL.
La solución es una línea, `py.allow_threads`, y su test tiene que vivir **en
otro proceso**: un cuelgue así no lo puede cazar nada de dentro.

Lo que `allow_threads` no arregla, y conviene decirlo: dos nodos de Python
puros en la misma wave **se serializan entre ellos**. Se solapa lo que suelte
el GIL — torch en su dispatch, la espera de un driver de red, la E/S. Por eso
el test que mide concurrencia de verdad usa un driver, y no dos nodos.

### Cuestionario

**Descomposición** (`core/tests/unit/plan.rs`)
- [x] vacío, un nodo, y una cadena lineal idéntica a la de antes
- [x] abanico de salida, de entrada y diamante, cada uno con su wave
- [x] `a >> (b >> b2 >> b3 | c >> c2) >> d` da una wave de dos secuencias
- [x] `(a >> a2 | b) >> (c | d)` no parte la rama — el contraejemplo de la barrera
- [x] `(a | b) >> (c | d)` son dos waves, no una de cuatro
- [x] una wave dentro de la rama de otra wave
- [x] dos grafos sin relación, cada uno largo, son dos ramas
- [x] la N se recorre en secuencia, y no estropea el paralelismo de lo que tenga al lado

**Invariantes, sobre una batería de diez topologías**
- [x] ningún nodo se ejecuta dos veces ni se queda fuera
- [x] el orden que dicta el plan respeta todas las aristas
- [x] cada paso declara exactamente sus predecesores del grafo
- [x] las ramas de una wave no comparten ningún nodo
- [x] el mismo grafo compila siempre igual

**El oráculo** (`core/tests/unit/build.rs`)
- [x] siete expresiones del DSL, y su plan es el árbol que se escribió

**Ejecución** (`core/tests/unit/execution.rs`)
- [x] el orden **real** de ejecución respeta las aristas, con hilos de por medio
- [x] una rama entera corre en el mismo hilo, y dos ramas en hilos distintos
- [x] dos y tres ramas corren de verdad a la vez — sin dormir: quedan en verse,
      y si fueran secuenciales la primera agotaría el plazo
- [x] el resultado del diamante es el mismo repartido que en fila
- [x] lo que produce una rama por dentro llega a quien la lee
- [x] dos ramas que fallan dan siempre el error de la primera declarada
- [x] un panic dentro de una rama no se traga
- [x] dos ramas pueden tener al driver ocupado a la vez
- [x] una wave que es todo el plan devuelve el mapa de sus hojas

**Python** (`python/tests/test_waves.py`)
- [x] el motor suelta el GIL — en otro proceso, con plazo
- [x] dos nodos Python en la misma wave dan el resultado correcto aunque el
      GIL los serialice
- [x] el DSL con ramas da el mismo plan que `node()`/`edge()`
- [x] hilos, orden real, fallos y la N, como en Rust

### Lo que NO entró

**El dispositivo.** `Plan::Execute` sigue sin decir *dónde*. Esta rebanada es
la que lo habilita —una rama por hilo es lo que hace que fijar un dispositivo
por rama signifique algo—, pero `.on("cuda:1")` es el caso de uso siguiente.

**Micro-lotes.** Solapar dentro de una rama, no entre ramas, es otro problema y
otra variante.

**Repartir una wave entre procesos.** Necesita transporte, y `Opaque` no cruza
un cable. Ese es el otro caso de uso siguiente.

## CU10 — Dónde corre un nodo

```python
g = Graph.somatize(
    Tokenizar()
    >> (Encoder().on("cuda:0") | Otro().on("cuda:1"))
    >> Juntar()
)
g.devices()   # {"encoder": "cuda:0", "otro": "cuda:1"}
g.plan()      # el mismo de siempre: el plan dice cuándo, no dónde
```

Estado: **cerrado**. 110 tests en Rust, 119 en Python.

### La pregunta: ¿dónde vive el dispositivo?

El primer diseño lo metía en el plan, como `Plan::On { device, inner }`
envolviendo un subplan —la misma forma que va a tener `Plan::Remote`—. Se
descartó por una razón que lo tumba entero: **el plan determina el orden de
ejecución y la concurrencia, y colocar no cambia ni uno ni otra**. Son dos ejes
distintos y meterlos en el mismo tipo los ata sin necesidad.

Sacarlo del plan pagó de inmediato:

- `plan.rs` no se toca en todo el caso de uso.
- Desaparece la regla de colapsar tiradas contiguas del mismo dispositivo, que
  era lo más frágil del diseño: la única parte que tenía que ser canónica y
  podía dejar de serlo.
- «Colocar no cambia el plan» deja de ser algo que comprobar y pasa a ser
  cierto por construcción, porque `compile` no ve la colocación. El test sigue
  escrito, pero como aviso para el día que alguien intente meterla ahí.

Descartado también, y por qué:

| dónde | por qué no |
|---|---|
| en el `Node` (`fn device(&self)`) | mete una decisión de **orquestación** dentro del contrato de la implementación. El nodo no elige dónde corre; y además queda invisible: no se puede imprimir ni razonar sobre ella |
| en el `Graph` | un `Graph` es solo topología. Y el motor no lo mira —cada paso del plan es autónomo desde CU3—, así que habría obligado a pasarle el grafo al `Executor` |
| en el `Catalog` | era el finalista: el motor ya lo tiene en la mano. Pierde porque el catálogo es la mitad que **no** es dato, y una colocación sí lo es. Cuando un subgrafo viaje a otra máquina, la colocación viaja con él y las implementaciones no |
| una `Metadata` genérica | es el nombre genérico de `Placement`, y el genérico se paga caro: un saco `id → dict` no puede guardar un `Device` **tipado** |

### Decisiones tomadas

**1. `Placement` es un tipo propio, y se le da al motor como se le da el
driver.** Queda un cuarto hecho ortogonal, que es lo que la pregunta hizo
visible:

| pieza | contesta |
|---|---|
| `Graph` | **qué** hay y cómo se conecta |
| `Catalog` | **quién** lo ejecuta |
| `Placement` | **dónde** |
| `Plan` | **cuándo**, y con qué concurrencia |

Encaja con lo que el propio `Executor` tenía escrito de sí mismo: «ejecutar
necesita contexto —hoy el almacén y el driver— y mañana necesitará más».

**2. `Device` es un enum, no un `String` validado por forma.** El argumento que
lo decidió no es el de la exhaustividad sino éste: **con un enum, un typo es un
error al declarar**. `.on("cude:0")` falla donde se escribió; un
`Device(String)` que solo comprobara la forma lo daría por bueno y el fallo
saldría dentro de torch a mitad de un run.

El coste de que el vocabulario pase a ser nuestro se puede pagar porque **el
núcleo no hace `match` sobre un `Device` en ninguna otra parte**: no decide nada
según cuál sea, solo lo transporta. Añadir una variante son tres líneas —el
enum, un brazo de `FromStr` y otro de `Display`— y ningún otro sitio deja de
compilar.

**3. El índice de `cuda` es obligatorio.** En torch, `"cuda"` a secas significa
«la GPU actual», que es estado del hilo. Para quien coloca, «la actual» no es
una colocación: `.on("cuda")` se rechaza pidiendo `cuda:0`. Una declaración
ambigua no se puede escribir.

**4. `meta` entra como variante.** Es el único dispositivo que permite probar de
extremo a extremo que una colocación llega y se obedece **en cualquier
máquina**. La de desarrollo tiene una sola GPU, así que sin `meta` la mitad del
cuestionario dependería del hardware.

**5. Sin colocar ≠ colocado en `cpu`.** El primero es «donde ya esté», el
segundo es una orden de mover. Por eso `Placement::of` devuelve `Option` en vez
de un `Cpu` por defecto.

**6. El dispositivo llega por el `Ctx`, y el que obedece es el nodo.** Es la
consecuencia de que el núcleo no sepa qué es una GPU: su papel es transportar
la declaración hasta el punto de ejecución. `ctx.device` llega escrito como lo
escribe torch —`"cuda:0"`— para que se le pueda pasar a `.to()` sin traducir.

**7. `.on()` en el DSL, `place()` con el id, y una sola puerta.** `.on()` se
reparte a las hojas que no tengan sitio y **gana el de dentro**:
`(a.on("cuda:0") >> b).on("cuda:1")` deja `a` en la 0 y `b` en la 1. Pero
`.on()` necesita el objeto dentro de una expresión, y hay dos casos en que solo
queda el id: el grafo construido en un bucle, y —el que importa de verdad— la
colocación decidida **después**, con lo que haya en la máquina:

```python
for i, nid in enumerate(g.nodes()):
    g.place(nid, f"cuda:{i % torch.cuda.device_count()}")
```

No son dos caminos: `.on()` termina llamando a `place()`, así que la validación
se escribe una vez y el DSL la hereda. Ningún id huérfano es posible — `.on()`
solo nombra nodos de su propio `Wire`, y `place()` valida contra el grafo.

### Lo que `.on()` **no** es

`.on("cuda:1")` no es `torch.cuda.set_device(1)`. Para que un nodo compute en
una GPU tienen que pasar tres cosas, y el contexto ambiente solo afecta a la
tercera:

| qué | cómo | cuándo |
|---|---|---|
| los **parámetros** están allí | `modulo.to(dev)` | una vez |
| la **entrada** está allí | `x.to(dev)` | cada forward |
| lo **creado dentro** nace allí | `device=` explícito | cada forward |

El contraejemplo estaba ya en el repo: `test_pipeline_torch.py` crea el tensor
de índices con `torch.tensor(filas)` dentro del `forward`. Con el `Embedding`
movido a cuda, eso revienta con *«Expected all tensors to be on the same
device»*, y ningún `set_device` lo arregla.

De ahí que obedecer sea trabajo del nodo, y que el patrón se escriba a mano —
son cinco líneas, y hasta que se repitan tres veces no hay nada que sacar a una
clase base:

```python
def forward(self, x, ctx):
    if ctx.device:
        if self.colocada != ctx.device:
            self.lin.to(ctx.device)   # los parámetros, una vez
            self.colocada = ctx.device
        x = x.to(ctx.device)          # la entrada, cada vez
    return Done(Opaque(self.lin(x)))
```

### La postcondición, que es lo que evita el silencio

Un nodo que ignora su `ctx.device` correría en el sitio equivocado sin que se
notara, y eso es justo lo que este proyecto no tolera. Desde fuera solo se puede
mirar una cosa: **dónde acabó lo que devolvió**. Si no coincide, es un error con
nombre:

```
el nodo `encoder` falló: declaró `cuda:0` pero devolvió un valor en `cpu`
```

Se mira lo que tenga un `.device` que mirar —un tensor, suelto o dentro de un
`Opaque`—. Un nodo colocado que devuelve una lista de textos no se comprueba, y
tampoco tenía mucho sentido colocarlo.

**El caso que marca sin ser un error**: un nodo que corre en la GPU y termina a
propósito con un `.cpu()`. Se acepta a sabiendas —es el caso raro, el mensaje
dice exactamente lo que pasó, y la alternativa era el silencio—.

### Por qué esto venía después de las waves

Una rama de una wave corre entera en un hilo (decisión de CU9), así que un
dispositivo por rama significa algo. Al revés no funcionaba: agrupar por nivel
topológico habría hecho saltar de hilo a una rama, y el dispositivo de torch es
thread-local.

Y lo que hace barato todo el caso de uso: `.to()` entre dispositivos es
**diferenciable**, así que autograd atraviesa el salto y `Opaque` no ha tenido
que cambiar ni una línea. Hay test: dos capas, una en `cuda:0` y otra en `cpu`,
entrenando de extremo a extremo.

### Cuestionario

**Rust** (`core/tests/unit/device.rs`, `placement.rs`, `execution.rs`)
- [x] `cpu`, `cuda:N` y `meta` se leen, y la ida y la vuelta dan lo mismo
- [x] `cude:0` es un tipo desconocido; `cuda` pide índice; `cuda:`, `cuda:x`,
      `cuda:1:2`, `cpu:0` y `""` no tienen forma de dispositivo
- [x] `.on()` reparte por todo el trozo y gana el de dentro
- [x] cada rama de un `|` en su sitio, y lo no colocado sigue sin colocar
- [x] el nodo ve el suyo y solo el suyo — a nadie se le pega el del vecino
- [x] las ramas de una wave ven dispositivos distintos, cada una en su hilo
- [x] colocar no cambia el plan, ni el grafo, ni lo que produce

**Python** (`python/tests/test_dispositivo.py`)
- [x] `.on()` y `place()` dan el mismo grafo, y `.named` y `.on` conmutan
- [x] colocar después en un bucle, y recolocar pisa lo anterior
- [x] colocar un nodo que no existe falla, y cada nombre malo con su aviso
- [x] `ctx.device` llega, y sale en el `repr` del `Ctx`
- [x] la postcondición salta, y dice de qué nodo — sin torch, con un objeto
      cualquiera que sepa decir dónde está
- [x] con torch: `meta` de extremo a extremo sin hardware; un nodo que ignora
      su dispositivo se cuenta
- [x] con GPU: `cuda:0` → `cpu` en el mismo grafo, y el backward atraviesa el
      salto entrenando

### Lo que NO entró

**Elegir el dispositivo automáticamente.** Balanceo, «auto», mirar cuánta
memoria queda: eso es una política, y todavía no hay quien la pida.

**Partir un nodo entre dispositivos**, y `g.to("cuda")` para el grafo entero.

**`soma_next.torch`.** El patrón para obedecer una colocación se escribe a mano
en el test, que es donde se documenta hasta que se repita.

**Generalizar `Placement` a «un sitio, local o remoto»** para adelantar CU13.
Cuando llegue `Remote`, `Placement` estará ahí para crecer — o no; eso se decide
entonces.

### Medido, no afirmado

Con una sola GPU en la máquina de desarrollo, `cuda:1` no existe: el reparto
entre dos GPUs se puede **declarar** y no se puede **ejecutar** aquí. Los tests
lo dicen en su nombre en vez de dejarlo implícito.

Y sigue en pie el aviso de CU9: nada de justificar esto con un benchmark de dos
ramas en dos GPUs. CUDA lanza asíncrono y las dos ya se solapan ejecutándose en
secuencia; lo que las waves compran es tiempo de **host**.

## CU11 — El entrenamiento, fuera del grafo

```python
from soma_next.torch import Trainer, parameters

g = Graph.somatize(Encoder().on("cuda:0") >> Cabeza().on("cuda:0"))
t = Trainer(g, objetivo=cross_entropy,
            optimizador=torch.optim.Adam(parameters(g), lr=1e-3))

t.fit(datos, epocas=10)   # el azúcar
t.step(lote)              # la primitiva
```

Estado: **cerrado**. 110 tests en Rust, 136 en Python, y **cero líneas nuevas
en `core/`** — el primer caso de uso que no toca el núcleo.

### La pregunta: ¿el bucle de entrenamiento va dentro del grafo?

No, y hay dos razones independientes.

**La primera está en el contrato del nodo.** `forward(input, ctx) → Done |
Await` describe **un paso**: se ejecuta una vez por run, tiene un presupuesto
de 64 turnos, y `run()` no tiene recuperación parcial. Un entrenamiento dura
una tarde, muta su propio estado, emite métricas continuamente y falla de
maneras de las que uno quiere recuperarse. **El grafo opera a la escala de un
`forward`; un entrenamiento opera a la escala de una tarde.**

El original lo intentó: su trait de nodo lleva `fn fit(&self, x, y)`. La
factura se ve en sus propios tests — `soma-worker`, `soma-compiler`,
`soma-runtime` y `soma-agent` implementan todos un `fit` vacío solo para poder
existir. Es el mismo impuesto que CU6 quitó en el eje filtro/step.

**La segunda es que un grafo describe una red, y explorar es una familia de
redes.** Ese grafo es justo el artefacto que en CU13 se serializa y viaja; uno
que llevara cinco configuraciones dentro estaría mintiendo sobre la
arquitectura.

### Los tres niveles

| nivel | qué es | escala | qué reparte |
|---|---|---|---|
| el grafo | una red | un `forward` | trozos de un forward: waves, `Placement`, y `Remote` en su día |
| `Trainer` | un entrenamiento | una tarde | nada; repite forwards |
| un estudio | N entrenamientos | un experimento | runs enteros |

Y la regla que lo sostiene: **ningún nivel sabe que existe el de arriba.** El
grafo no sabe que lo entrenan; el trainer no sabe que hay otros trainers. La
composición entre niveles es composición de **funciones**, no de grafos.

### El nivel 3 no tiene tipo, y es a propósito

```python
estudio = {lr: Trainer(red(), ..., lr).fit(datos) for lr in (1e-4, 1e-2)}
mejor = min(estudio, key=lambda lr: estudio[lr].loss)
```

> Un grafo se gana el sueldo cuando hay **dependencias** que declarar. N
> entrenamientos independientes no tienen ninguna: son una lista. Modelar una
> lista como un grafo es pagar el precio de un DAG para no usarlo.

Se llegó a diseñar la alternativa —las N configuraciones como ramas de un `|`,
con un nodo que elige la mejor— y se descartó. Y también su variante astuta,
«un grafo, un plan, **N catálogos**», que se cae por algo concreto: `Catalog`
es `Clone`, pero clona `Arc`. Las N réplicas compartirían los objetos nodo, o
sea **los pesos**, y las cinco configuraciones entrenarían el mismo modelo
dando resultados que parecen buenos. Cada réplica hay que construirla — y en
cuanto la construyes, ya no tienes un grafo con N planes, tienes N grafos. Hay
test.

### Decisiones tomadas

**1. El Trainer recibe el grafo; nunca `g.fit(...)`.** Así el mismo grafo se
entrena de tres maneras sin tocarlo, y sigue siendo el artefacto que viaja.

**2. Vive en `soma_next.torch`.** Pérdida, `backward()` y optimizador son
torch; escribirlo neutral pediría un `Backend` con un solo implementor. El
núcleo no aprende qué es entrenar, y eso es la señal de que la separación
aguanta.

**3. `step(lote)` es la primitiva; `fit(datos, epocas)` es azúcar.** Es lo que
evita el camino del trainer-dios: parada temprana, *schedules* raros, rondas
federadas y PBT son un `while` que escribe el usuario sobre `step`, no una
lista creciente de opciones y *callbacks*. Una ronda federada es
`for _ in range(k): t.step(lote)`.

**4. Los parámetros se recogen por *duck typing*, y un grafo sin ellos falla al
construir el Trainer.** Se pregunta por `.parameters()` y se salta a quien no
lo tenga —un lematizador no entrena y no por eso deja de ser nodo—. Meterlo en
el contrato sería el `fit` del original otra vez. Y como el *duck typing* falla
callado —un grafo sin parámetros entrena la nada y muestra una pérdida plana—,
la lista vacía revienta al construir, igual que la postcondición de CU10.

**5. Van sin repetir, por identidad.** Dos nodos pueden compartir un módulo
—pesos atados entre embedding y salida— y entonces el mismo `Parameter` sale
dos veces.

**6. El optimizador lo construye quien llama.** Nada de `optimizador="adam"`,
que acabaría siendo un registro de nombres. Lo único que se comprueba es que
el optimizador y el grafo **compartan algún parámetro**: no compartir ninguno
no tiene lectura inocente. Cubrir solo una parte sí es legítimo y pasa —
congelar el encoder y entrenar la cabeza es exactamente eso.

**7. Los datos son un iterable de `(entrada, objetivo)`.** Un `DataLoader` lo
es. Descartado a propósito: **los datos como nodo fuente**, porque un nodo
produce *un* valor por ejecución; para ser un flujo tendría que recordar por
dónde va, y entonces dos ejecuciones del mismo grafo dejan de dar lo mismo.

**8. La pérdida es un invocable, no un nodo.** No es parte de la red: se cambia
sin tocar el modelo y en inferencia no existe. Como nodo, la salida del grafo
sería un escalar y el grafo solo serviría para entrenar.

### Lo que encontró el test con GPU

El objetivo no cruza el grafo. La **entrada** sí, y cada nodo la mueve a su
dispositivo porque eso es lo que hace un nodo colocado; el **objetivo** va
directo a la pérdida, así que no lo movía nadie. Con la última capa en
`cuda:0`, la salida sale de allí, el objetivo sigue en la cpu y torch para el
entrenamiento con *«expected all tensors to be on the same device»*.

Lo arregla el único que ve los dos lados. Y se mueve el objetivo, no la salida:
traer la salida a la cpu arrastraría el backward de vuelta por el cable en cada
paso.

### Cuestionario

**Python** (`python/tests/test_trainer.py`)
- [x] `parameters(g)` recoge lo de todos los nodos que tengan y salta a los que
      no; en orden de declaración; sin repetir un módulo compartido
- [x] un grafo sin parámetros falla **al construir el Trainer**
- [x] un optimizador de otro grafo falla; congelar una parte pasa
- [x] entrenar baja la pérdida, y `fit` da lo mismo que el bucle a mano
- [x] los pesos que actualiza el optimizador son los que usa el grafo
- [x] entrenar no cambia el grafo: `nodes()`, `edges()`, `plan()` y `devices()`
      idénticos antes y después
- [x] una entrada que no es un tensor cruza como siempre
- [x] **dos redes de la misma fábrica no comparten pesos**
- [x] la exploración de hiperparámetros, como comprensión de lista
- [x] con GPU: el optimizador sigue apuntando a los pesos después de que el
      nodo se mueva; el objetivo va a buscar a la salida; y las dos capas en
      dispositivos distintos entrenan

### Lo que NO entró

Checkpoints y reanudación · *callbacks* y parada temprana, que son un `while`
sobre `step` · métricas más allá de la pérdida · *schedulers* · acumulación de
gradiente · el estudio como tipo · y **exportar o cargar el estado de un
modelo**, que es la pregunta de CU14 y aquí no hacía falta: entrenar en local
no extrae ningún estado.

### Lo que esto le hizo al plan

- **La pregunta del estado deja de bloquear.** Ya no es «¿el estado de un nodo
  es un `Value`?» en el núcleo, sino «¿qué exporta un entrenamiento?» en el
  nivel 2, y se contesta en CU14 con el caso delante.
- **CU13 se parte en dos.** Repartir *un grafo* entre hosts es `Plan::Remote`,
  con dependencias a mitad del forward. Repartir *entrenamientos* —HPO,
  federado, data parallel— es «ejecuta esto entero allí», nivel 3, y no lo
  necesita. El original los tenía en el mismo enum: `ModelParallel` junto a
  `DataParallel`, `Federated` y `PopulationBased`.
- **CU14 cambia de forma.** «Una ronda federada es un grafo» se retira: es
  `map` y `reduce`. Un grafo solo se justificaría con topologías no planas
  —federación jerárquica, gossip—, y ese día no ha llegado.
- **`soma_next.torch` deja de estar pendiente** y se abre aquí.
