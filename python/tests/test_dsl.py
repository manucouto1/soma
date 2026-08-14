"""El DSL: el grafo como una expresión."""

import pytest

import soma_next
from soma_next import Filter, Step, build


class Sumar(Filter):
    def __init__(self, cuanto):
        self.cuanto = cuanto

    def forward(self, x):
        return x + self.cuanto


class Media(Filter):
    """Un agregador es un filtro que lee un mapa."""

    def forward(self, entradas):
        return sum(entradas.values()) / len(entradas)


class Eco(Step):
    def poll(self, ctx):
        return {"done": ctx["input"]}


# ── Encadenar ──


def test_un_solo_filtro_ya_es_un_grafo():
    g = build(Sumar(1))
    assert g.nodes() == ["sumar"]
    assert g.forward(41) == 42.0


def test_una_cadena():
    g = build(Sumar(1) >> Sumar(10) >> Sumar(100))
    assert g.nodes() == ["sumar", "sumar_2", "sumar_3"]
    assert g.forward(0) == 111.0


def test_named_pone_el_id():
    g = build(Sumar(1).named("primero") >> Sumar(10).named("segundo"))
    assert g.nodes() == ["primero", "segundo"]
    assert g.edges() == [("primero", "segundo")]


# ── Abrir y cerrar ramas ──


def test_un_diamante_se_lee_de_un_vistazo():
    g = build(Sumar(1) >> (Sumar(10).named("izq") | Sumar(100).named("der")) >> Media())
    assert g.edges() == [
        ("sumar", "izq"),
        ("sumar", "der"),
        ("izq", "media"),
        ("der", "media"),
    ]
    assert g.forward(0) == 56.0


def test_ramas_abiertas_salen_como_mapa():
    g = build(Sumar(1) >> (Sumar(10).named("izq") | Sumar(100).named("der")))
    assert g.forward(0) == {"izq": 11.0, "der": 101.0}


def test_una_rama_puede_ser_mas_larga_que_la_otra():
    g = build(
        Sumar(1).named("fuente")
        >> ((Sumar(1).named("izq") >> Sumar(1).named("izq2")) | Sumar(1).named("der"))
    )
    assert g.edges() == [
        ("fuente", "izq"),
        ("izq", "izq2"),
        ("fuente", "der"),
    ]


def test_tres_ramas():
    g = build(Sumar(0) >> (Sumar(1).named("a") | Sumar(2).named("b") | Sumar(3).named("c")))
    assert g.forward(0) == {"a": 1.0, "b": 2.0, "c": 3.0}


# ── Filtros y steps en la misma expresión ──


def test_un_step_encaja_donde_encajaria_un_filtro():
    g = build(Sumar(1) >> Eco())
    assert g.forward(41) == 42.0


# ── Heredar es opcional ──


def test_un_objeto_sin_heredar_vale_si_algo_a_su_lado_hereda():
    class Suelto:
        def forward(self, x):
            return x * 2

    g = build(Suelto() >> Sumar(1))
    assert g.forward(20) == 41.0


def test_lo_que_no_puede_ser_nodo_lo_dice():
    with pytest.raises(TypeError, match="le falta forward\\(\\) o poll\\(\\)"):
        build(Sumar(1) >> "esto no es un filtro")


# ── El DSL y las llamadas construyen lo mismo ──


def test_el_dsl_no_es_otra_cosa_que_node_y_edge():
    dsl = build(Sumar(1).named("a") >> Sumar(10).named("b"))

    a_mano = soma_next.Graph()
    a_mano.node("a", Sumar(1))
    a_mano.node("b", Sumar(10))
    a_mano.edge("a", "b")

    assert dsl.nodes() == a_mano.nodes()
    assert dsl.edges() == a_mano.edges()
    assert dsl.plan() == a_mano.plan()
