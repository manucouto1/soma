"""Entrenar un grafo desde fuera del grafo.

Las dos cosas que este fichero está defendiendo:

**El grafo no se entera.** Después de entrenar, sus nodos, sus aristas, su plan
y su colocación son idénticos. Lo que cambia son los pesos, que viven dentro de
los nodos y siempre vivieron ahí.

**Varios entrenamientos son una lista, no un grafo.** La exploración de
hiperparámetros está aquí abajo escrita como una comprensión de lista, sin un
solo tipo nuevo — y el test que importa de todos es el que comprueba que dos
runs de la misma fábrica **no comparten pesos**, porque compartirlos daría
resultados que parecen buenos y no lo son.
"""

import pytest

from soma_next import Done, Graph, Node, Opaque
from soma_next.torch import Trainer, parameters

torch = pytest.importorskip("torch")
nn = torch.nn

DENTRO, MEDIO, CLASES = 4, 3, 2
sin_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no hay CUDA")


class Capa(Node):
    """Un nodo con parámetros que obedece su colocación. El patrón de CU10."""

    def __init__(self, dentro, fuera):
        self.lin = nn.Linear(dentro, fuera)
        self.colocada = None

    def forward(self, x, ctx):
        if ctx.device:
            if self.colocada != ctx.device:
                self.lin.to(ctx.device)
                self.colocada = ctx.device
            x = x.to(ctx.device)
        return Done(Opaque(self.lin(x)))

    def parameters(self):
        return list(self.lin.parameters())


class Etiquetar(Node):
    """Sin parámetros: no todo nodo entrena, y no por eso deja de ser nodo."""

    def forward(self, x, ctx):
        return Done(x)


def red(gpu=None):
    """La fábrica: una configuración → una red entrenable, recién construida."""
    encoder, cabeza = Capa(DENTRO, MEDIO), Capa(MEDIO, CLASES)
    expresion = encoder.named("encoder") >> cabeza.named("cabeza")
    if gpu:
        expresion = expresion.on(gpu)
    return Graph.somatize(expresion)


def lotes(n=4):
    torch.manual_seed(0)
    return [(torch.randn(6, DENTRO), torch.randint(0, CLASES, (6,))) for _ in range(n)]


def entrenador(g, lr=0.05):
    return Trainer(
        g,
        objetivo=nn.functional.cross_entropy,
        optimizador=torch.optim.Adam(parameters(g), lr=lr),
    )


# ── Los parámetros de un grafo ──


def test_recoge_los_de_todos_los_nodos_que_tengan():
    g = red()
    assert len(parameters(g)) == 4  # peso y sesgo de cada capa


def test_se_salta_los_nodos_sin_parametros():
    g = Graph.somatize(Etiquetar().named("etiquetar") >> Capa(DENTRO, MEDIO).named("capa"))
    assert len(parameters(g)) == 2


def test_van_en_orden_de_declaracion_y_no_cambian_entre_llamadas():
    g = red()
    assert [id(p) for p in parameters(g)] == [id(p) for p in parameters(g)]


def test_un_modulo_compartido_no_sale_dos_veces():
    # Pesos atados: dos nodos, el mismo módulo. Un optimizador con duplicados
    # avisa o falla, según la versión de torch.
    compartida = Capa(DENTRO, DENTRO)
    g = Graph()
    g.node("a", compartida)
    g.node("b", compartida)
    g.edge("a", "b")
    assert len(parameters(g)) == 2


# ── Construir el Trainer: lo que se rechaza ──


def test_un_grafo_sin_parametros_falla_al_construir_el_trainer():
    # Y no con una pérdida plana veinte minutos después.
    g = Graph.somatize(Etiquetar().named("etiquetar"))
    with pytest.raises(ValueError, match="no tiene parámetros"):
        Trainer(g, objetivo=nn.functional.cross_entropy, optimizador=None)


def test_un_optimizador_de_otro_grafo_falla():
    g, otro = red(), red()
    with pytest.raises(ValueError, match="no actualiza ningún parámetro"):
        Trainer(
            g,
            objetivo=nn.functional.cross_entropy,
            optimizador=torch.optim.Adam(parameters(otro), lr=0.1),
        )


def test_congelar_una_parte_es_legitimo_y_pasa():
    # Entrenar solo la cabeza cubre una parte de los parámetros, no todos.
    g = red()
    cabeza = g.implementation("cabeza")
    entrenable = Trainer(
        g,
        objetivo=nn.functional.cross_entropy,
        optimizador=torch.optim.Adam(cabeza.parameters(), lr=0.1),
    )
    assert entrenable.step(lotes()[0]) > 0


# ── Entrenar ──


def test_entrenar_baja_la_perdida():
    t = entrenador(red())
    resultado = t.fit(lotes(), epocas=10)

    assert len(resultado.historia) == 40
    assert resultado.loss == resultado.historia[-1]
    assert resultado.loss < resultado.historia[0], f"no bajó: {resultado!r}"


def test_fit_da_lo_mismo_que_el_bucle_a_mano_sobre_step():
    torch.manual_seed(7)
    con_fit = entrenador(red()).fit(lotes(), epocas=2)

    torch.manual_seed(7)
    a_mano, historia = entrenador(red()), []
    for _ in range(2):
        for lote in lotes():
            historia.append(a_mano.step(lote))

    assert con_fit.historia == pytest.approx(historia)


def test_los_pesos_que_actualiza_el_optimizador_son_los_que_usa_el_grafo():
    g = red()
    antes = g.implementation("cabeza").lin.weight.detach().clone()
    entrenador(g, lr=0.1).step(lotes()[0])
    assert not torch.allclose(antes, g.implementation("cabeza").lin.weight)


def test_entrenar_no_cambia_el_grafo():
    g = red()
    foto = (g.nodes(), g.edges(), g.plan(), g.devices())
    entrenador(g).fit(lotes())
    assert (g.nodes(), g.edges(), g.plan(), g.devices()) == foto


def test_una_entrada_que_no_es_un_tensor_cruza_como_siempre():
    # `_cruzable` solo envuelve tensores; lo demás sigue el camino de siempre.
    class Cuenta(Node):
        def __init__(self):
            self.lin = nn.Linear(1, CLASES)

        def forward(self, textos, ctx):
            largos = torch.tensor([[float(len(t))] for t in textos])
            return Done(Opaque(self.lin(largos)))

        def parameters(self):
            return list(self.lin.parameters())

    g = Graph.somatize(Cuenta().named("cuenta"))
    lote = (["hola", "adiós"], torch.tensor([0, 1]))
    assert entrenador(g).step(lote) > 0


# ── Varios entrenamientos: una lista, no un grafo ──


def test_dos_redes_de_la_misma_fabrica_no_comparten_pesos():
    # El contraejemplo que descartó «un grafo, N catálogos»: clonar un catálogo
    # clona `Arc`, o sea que las réplicas compartirían los pesos y entrenarían
    # todas el mismo modelo. Hay que construir cada una.
    una, otra = red(), red()
    assert {id(p) for p in parameters(una)}.isdisjoint({id(p) for p in parameters(otra)})

    entrenador(una, lr=0.5).fit(lotes(), epocas=3)
    assert not torch.allclose(
        una.implementation("cabeza").lin.weight,
        otra.implementation("cabeza").lin.weight,
    ), "entrenar una movió los pesos de la otra"


def test_la_exploracion_de_hiperparametros_es_una_comprension_de_lista():
    # Sin un tipo nuevo, sin ramas en el grafo, sin waves. Tres redes, tres
    # entrenamientos, y el mejor sale de un `min`.
    datos = lotes()
    estudio = {lr: entrenador(red(), lr=lr).fit(datos, epocas=5) for lr in (1e-4, 1e-2)}

    mejor = min(estudio, key=lambda lr: estudio[lr].loss)
    assert mejor == 1e-2, {lr: r.loss for lr, r in estudio.items()}


# ── Con GPU: CU10 y CU11 a la vez ──


@sin_cuda
def test_el_optimizador_sigue_apuntando_a_los_pesos_tras_moverse_a_la_gpu():
    # El nodo se coloca **perezosamente**, en el primer forward, o sea después
    # de que el optimizador exista. `Module.to()` mueve los parámetros in-place
    # conservando los mismos objetos, así que debería seguir valiendo — y eso
    # es justo la clase de «debería» que hay que convertir en test.
    g = red(gpu="cuda:0")
    t = entrenador(g, lr=0.1)
    assert g.implementation("cabeza").lin.weight.device.type == "cpu", "aún sin mover"

    antes = g.implementation("cabeza").lin.weight.detach().clone()
    t.step(lotes()[0])

    peso = g.implementation("cabeza").lin.weight
    assert peso.device.type == "cuda", "el nodo se movió al ejecutarse"
    assert not torch.allclose(antes.cuda(), peso), "y el optimizador lo actualizó"


@sin_cuda
def test_entrenar_con_las_dos_capas_en_dispositivos_distintos():
    encoder, cabeza = Capa(DENTRO, MEDIO), Capa(MEDIO, CLASES)
    g = Graph.somatize(
        encoder.named("encoder").on("cuda:0") >> cabeza.named("cabeza").on("cpu")
    )
    resultado = entrenador(g, lr=0.1).fit(lotes(), epocas=5)
    assert resultado.loss < resultado.historia[0]


@sin_cuda
def test_el_objetivo_va_a_buscar_a_la_salida_a_su_dispositivo():
    # La asimetría que solo se ve en GPU: la entrada la mueve cada nodo, porque
    # cruza el grafo; el objetivo no entra en el grafo, así que no lo movía
    # nadie y la pérdida reventaba con «expected all tensors on the same
    # device». Lo arregla el único que ve los dos lados.
    g = red(gpu="cuda:0")
    entrada, objetivo = lotes()[0]
    assert objetivo.device.type == "cpu"

    entrenador(g).step((entrada, objetivo))  # la salida acaba en cuda:0

    assert objetivo.device.type == "cpu", "y el lote del usuario se queda como estaba"
