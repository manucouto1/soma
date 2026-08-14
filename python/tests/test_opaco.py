"""Valores que cruzan el grafo sin convertirse."""

import pytest

from soma_next import Done, Graph, Node, Opaque


class Envuelve(Node):
    """Devuelve lo que le llegue, opaco."""

    def forward(self, x, ctx):
        return Done(Opaque(x))


class Recuerda(Node):
    """Apunta el objeto que recibió, para poder comprobar identidad."""

    def __init__(self):
        self.visto = None

    def forward(self, x, ctx):
        self.visto = x
        return Done(Opaque(x))


# ── Cualquier objeto ──


def test_un_objeto_cualquiera_cruza_sin_convertirse(g):
    class Raro:
        pass

    original = Raro()
    recuerda = Recuerda()
    g.node("recuerda", recuerda)

    salida = g.forward(Opaque(original))
    assert salida is original, "tiene que salir el mismo objeto, no una copia"
    assert recuerda.visto is original


def test_el_nodo_lo_recibe_desenvuelto(g):
    recuerda = Recuerda()
    g.node("recuerda", recuerda)
    g.forward(Opaque({1, 2, 3}))  # un set, que sin envolver no cruzaría
    assert recuerda.visto == {1, 2, 3}


def test_atraviesa_varios_nodos_siendo_el_mismo(g):
    class Raro:
        pass

    original = Raro()
    for name in ("a", "b", "c"):
        g.node(name, Envuelve())
    g.edge("a", "b")
    g.edge("b", "c")
    assert g.forward(Opaque(original)) is original


def test_sin_envolver_sigue_dando_error(g):
    g.node("x", Envuelve())
    with pytest.raises(TypeError, match="Opaque"):
        g.forward({1, 2, 3})


def test_cabe_en_una_lista_y_en_un_mapa(g):
    class Raro:
        pass

    uno, otro = Raro(), Raro()

    class Lista(Node):
        def forward(self, x, ctx):
            return Done([Opaque(uno), Opaque(otro)])

    g.node("lista", Lista())
    salida = g.forward()
    assert salida[0] is uno and salida[1] is otro


def test_el_repr_dice_de_que_tipo_es():
    assert repr(Opaque({1: 2})) == "Opaque(dict)"


# ── El caso que motivó la variante ──


def test_el_autograd_de_torch_sobrevive_al_grafo(g):
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class Capa(Node, nn.Module):
        def __init__(self, m):
            nn.Module.__init__(self)
            self.m = m

        def forward(self, x, ctx):
            return Done(Opaque(self.m(x)))

    l1, l2 = nn.Linear(4, 3), nn.Linear(3, 2)
    g.node("l1", Capa(l1))
    g.node("relu", Capa(nn.ReLU()))
    g.node("l2", Capa(l2))
    g.edge("l1", "relu")
    g.edge("relu", "l2")

    x = torch.randn(5, 4, requires_grad=True)
    y = g.forward(Opaque(x))

    assert y.requires_grad, "la salida sigue enganchada a la gráfica"
    assert y.grad_fn is not None

    y.pow(2).sum().backward()
    assert x.grad is not None, "el backward atraviesa los tres nodos"
    assert l1.weight.grad is not None
    assert l2.weight.grad is not None


def test_convertir_a_numeros_rompe_el_autograd_que_es_por_lo_que_existe_opaque(g):
    torch = pytest.importorskip("torch")

    class Copia(Node):
        def forward(self, x, ctx):
            return Done(x.tolist())  # sin envolver: se convierte

    g.node("copia", Copia())
    x = torch.randn(3, requires_grad=True)
    salida = g.forward(Opaque(x))

    assert salida == pytest.approx(x.tolist())
    assert not torch.tensor(salida).requires_grad
