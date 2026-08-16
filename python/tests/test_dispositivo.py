"""Dónde corre un nodo: `.on("cuda:0")`, `place()` y lo que llega al `forward`.

Dos cosas que conviene tener claras antes de leer:

**Colocar no es ejecutar.** El núcleo no sabe mover nada a una GPU; lo que hace
es llevar la declaración hasta el `ctx` del nodo. Quien obedece es el nodo, con
`.to(ctx.device)`. Lo que impide que desobedecer salga gratis es la
postcondición: si un nodo colocado devuelve algo que está en otro sitio, es un
error con nombre.

**Colocar no cambia nada más.** Ni el plan, ni el orden, ni el resultado. Por
eso el dispositivo no vive en el `Plan`: el plan dice cuándo corre cada nodo, no
dónde.
"""

import pytest

from soma_next import Done, Graph, Node, Opaque

from conftest import Identidad, Sumar


class Mirar(Node):
    """Apunta el dispositivo que le llegó y devuelve su entrada."""

    def __init__(self):
        self.visto = "no ha corrido"

    def forward(self, x, ctx):
        self.visto = ctx.device
        return Done(x)


# ── Declarar dónde ──


def test_on_coloca_un_nodo():
    g = Graph.somatize(Sumar(1).named("a").on("cuda:0"))
    assert g.devices() == {"a": "cuda:0"}


def test_sin_on_no_hay_dispositivo():
    g = Graph.somatize(Sumar(1) >> Sumar(2))
    assert g.devices() == {}


def test_on_reparte_por_todo_el_trozo():
    g = Graph.somatize((Sumar(1).named("a") >> Sumar(2).named("b")).on("cpu"))
    assert g.devices() == {"a": "cpu", "b": "cpu"}


def test_gana_el_de_dentro():
    g = Graph.somatize(
        (Sumar(1).named("a").on("cuda:0") >> Sumar(2).named("b")).on("meta")
    )
    assert g.devices() == {"a": "cuda:0", "b": "meta"}


def test_cada_rama_en_su_sitio():
    g = Graph.somatize(
        Sumar(1).named("fuente")
        >> (Sumar(2).named("izq").on("cuda:0") | Sumar(3).named("der").on("cpu"))
    )
    assert g.devices() == {"izq": "cuda:0", "der": "cpu"}


def test_named_y_on_conmutan():
    uno = Graph.somatize(Sumar(1).named("a").on("cpu"))
    otro = Graph.somatize(Sumar(1).on("cpu").named("a"))
    assert uno.devices() == otro.devices() == {"a": "cpu"}


def test_place_hace_lo_mismo_que_on():
    # `.on()` necesita el objeto dentro de una expresión; `place()` solo el id,
    # que es lo único que queda cuando el grafo se construyó a mano o cuando la
    # colocación se decide después.
    dsl = Graph.somatize(Sumar(1).named("a").on("cuda:0"))

    a_mano = Graph()
    a_mano.node("a", Sumar(1))
    a_mano.place("a", "cuda:0")

    assert dsl.devices() == a_mano.devices()
    assert dsl.plan() == a_mano.plan()


def test_colocar_despues_en_un_bucle():
    # El caso que `.on()` no puede cubrir: la colocación sale de lo que haya en
    # la máquina, no de lo que se escribió en la expresión.
    g = Graph.somatize(Sumar(1).named("a") >> Sumar(2).named("b") >> Sumar(3).named("c"))
    for i, nid in enumerate(g.nodes()):
        g.place(nid, f"cuda:{i % 2}")

    assert g.devices() == {"a": "cuda:0", "b": "cuda:1", "c": "cuda:0"}


def test_recolocar_pisa_lo_anterior():
    g = Graph.somatize(Sumar(1).named("a").on("cpu"))
    g.place("a", "meta")
    assert g.devices() == {"a": "meta"}


# ── Lo que se rechaza, y dónde ──


@pytest.mark.parametrize(
    "malo, aviso",
    [
        ("cude:0", "no conozco el dispositivo"),
        ("gpu:0", "no conozco el dispositivo"),
        ("cuda", "no dice cuál"),
        ("cuda:", "no tiene forma de dispositivo"),
        ("cuda:x", "no tiene forma de dispositivo"),
        ("cpu:0", "no tiene forma de dispositivo"),
        ("", "no tiene forma de dispositivo"),
    ],
)
def test_un_nombre_que_no_nombra_un_sitio_falla_al_declarar(malo, aviso):
    # La razón de que `Device` sea un enum: el typo se cuenta aquí, no dentro
    # de torch a mitad de un run.
    with pytest.raises(ValueError, match=aviso):
        Graph.somatize(Sumar(1).on(malo))


def test_colocar_un_nodo_que_no_existe_falla():
    g = Graph.somatize(Sumar(1).named("a"))
    with pytest.raises(ValueError, match="encodr"):
        g.place("encodr", "cpu")


# ── Lo que le llega al nodo ──


def test_el_nodo_ve_donde_le_dijeron_que_corriera():
    mirar = Mirar()
    Graph.somatize(mirar.named("m").on("cuda:1")).forward(1.0)
    assert mirar.visto == "cuda:1"


def test_sin_colocar_no_ve_ninguno():
    mirar = Mirar()
    Graph.somatize(mirar.named("m")).forward(1.0)
    assert mirar.visto is None


def test_a_nadie_se_le_pega_el_del_vecino():
    primero, segundo = Mirar(), Mirar()
    Graph.somatize(primero.named("a").on("meta") >> segundo.named("b")).forward(1.0)
    assert primero.visto == "meta"
    assert segundo.visto is None


def test_cada_rama_de_una_wave_ve_el_suyo():
    izq, der = Mirar(), Mirar()
    g = Graph.somatize(
        Identidad().named("fuente")
        >> (izq.named("izq").on("cuda:0") | der.named("der").on("cuda:1"))
    )
    assert "Wave" in g.plan()
    g.forward(1.0)

    assert izq.visto == "cuda:0"
    assert der.visto == "cuda:1"


def test_el_dispositivo_sale_en_el_repr_del_ctx():
    class Repr(Node):
        def __init__(self):
            self.visto = None

        def forward(self, x, ctx):
            self.visto = repr(ctx)
            return Done(x)

    nodo = Repr()
    Graph.somatize(nodo.named("n").on("cpu")).forward(1.0)
    assert nodo.visto == "Ctx(turn=0, results=0, device=cpu)"


# ── Lo que colocar NO cambia ──


def test_colocar_no_cambia_el_plan():
    sin = Graph.somatize(Sumar(1).named("a") >> (Sumar(2).named("b") | Sumar(3).named("c")))
    con = Graph.somatize(
        (Sumar(1).named("a") >> (Sumar(2).named("b") | Sumar(3).named("c"))).on("meta")
    )
    assert sin.plan() == con.plan()
    assert "meta" not in con.plan(), "el plan dice cuándo, no dónde"


def test_colocar_no_cambia_el_resultado():
    sin = Graph.somatize(Sumar(1).named("a") >> Sumar(10).named("b"))
    con = Graph.somatize((Sumar(1).named("a") >> Sumar(10).named("b")).on("meta"))
    assert sin.forward(0.0) == con.forward(0.0) == 11.0


# ── La postcondición: desobedecer no sale gratis ──


class Falso:
    """Cualquier cosa que sepa decir dónde está. No hace falta torch."""

    def __init__(self, device):
        self.device = device


class Devuelve(Node):
    def __init__(self, valor):
        self.valor = valor

    def forward(self, x, ctx):
        return Done(Opaque(self.valor))


def test_un_nodo_que_devuelve_algo_de_otro_sitio_se_cuenta():
    g = Graph.somatize(Devuelve(Falso("cpu")).named("n").on("cuda:0"))
    with pytest.raises(ValueError, match="declaró `cuda:0` pero devolvió un valor en `cpu`"):
        g.forward(1.0)


def test_el_error_dice_de_que_nodo_es():
    g = Graph.somatize(Devuelve(Falso("cpu")).named("encoder").on("meta"))
    with pytest.raises(ValueError, match="el nodo `encoder` falló"):
        g.forward(1.0)


def test_obedecer_pasa_sin_ruido():
    g = Graph.somatize(Devuelve(Falso("cuda:0")).named("n").on("cuda:0"))
    assert g.forward(1.0).device == "cuda:0"


def test_lo_que_no_sabe_donde_esta_no_se_comprueba():
    # Un nodo colocado que devuelve texto no se puede comprobar desde fuera.
    # Tampoco tenía mucho sentido colocarlo.
    g = Graph.somatize(Identidad().named("n").on("cuda:0"))
    assert g.forward("hola") == "hola"


def test_sin_colocar_no_se_comprueba_nada():
    g = Graph.somatize(Devuelve(Falso("cpu")).named("n"))
    assert g.forward(1.0).device == "cpu"


# ── Con torch de verdad ──

torch = pytest.importorskip("torch")
nn = torch.nn


class Capa(Node):
    """El patrón: los parámetros una vez, la entrada cada vez.

    Es todo lo que hay que escribir para obedecer una colocación, y va aquí a
    mano a propósito: hasta que este mismo cuerpo se repita tres veces, no hay
    nada que sacar a una clase base.
    """

    def __init__(self, dentro, fuera):
        self.lin = nn.Linear(dentro, fuera)
        self.colocada = None

    def forward(self, x, ctx):
        if ctx.device:
            if self.colocada != ctx.device:
                self.lin.to(ctx.device)  # los parámetros, una vez
                self.colocada = ctx.device
            x = x.to(ctx.device)  # la entrada, cada vez
        return Done(Opaque(self.lin(x)))

    def parameters(self):
        return list(self.lin.parameters())


def test_meta_prueba_la_colocacion_sin_hardware():
    # `meta` es la razón de que la variante exista: comprueba de extremo a
    # extremo que la colocación llega y se obedece en cualquier máquina.
    g = Graph.somatize(Capa(4, 3).named("capa").on("meta"))
    salida = g.forward(Opaque(torch.zeros(2, 4)))

    assert str(salida.device) == "meta"
    assert salida.shape == (2, 3)


def test_un_nodo_que_ignora_su_dispositivo_se_cuenta():
    class Sorda(Node):
        def forward(self, x, ctx):
            return Done(Opaque(torch.zeros(2, 3)))  # nace en cpu, pase lo que pase

    g = Graph.somatize(Sorda().named("sorda").on("meta"))
    with pytest.raises(ValueError, match="declaró `meta` pero devolvió un valor en `cpu`"):
        g.forward(Opaque(torch.zeros(2, 4)))


# ── Y con GPU, si la hay ──

sin_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no hay CUDA")


@sin_cuda
def test_un_nodo_corre_en_la_gpu_y_el_siguiente_en_la_cpu():
    # Con una sola GPU en la máquina esto es lo que se puede probar de verdad:
    # el reparto entre dos GPUs queda declarado pero sin ejecutar aquí.
    g = Graph.somatize(Capa(4, 3).named("gpu").on("cuda:0") >> Capa(3, 2).named("cpu").on("cpu"))
    salida = g.forward(Opaque(torch.zeros(2, 4)))

    assert str(salida.device) == "cpu"


@sin_cuda
def test_el_backward_atraviesa_el_salto_de_dispositivo():
    # Es lo que hace barato todo CU10: `.to()` entre dispositivos es
    # diferenciable, así que `Opaque` no ha tenido que cambiar nada.
    torch.manual_seed(0)
    en_gpu, en_cpu = Capa(4, 3), Capa(3, 2)
    g = Graph.somatize(en_gpu.named("gpu").on("cuda:0") >> en_cpu.named("cpu").on("cpu"))
    objetivo = torch.tensor([0, 1])

    opt = torch.optim.Adam(en_gpu.parameters() + en_cpu.parameters(), lr=0.05)
    entrada = torch.randn(2, 4)
    primera = ultima = None
    for paso in range(20):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(g.forward(Opaque(entrada)), objetivo)
        loss.backward()
        opt.step()
        if paso == 0:
            primera = loss.item()
        ultima = loss.item()

    assert all(p.grad is not None for p in en_gpu.parameters()), (
        "los gradientes cruzaron de vuelta a la GPU"
    )
    assert ultima < primera, f"la pérdida bajó de {primera:.4f} a {ultima:.4f}"
