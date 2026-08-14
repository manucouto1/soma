"""El DSL: el grafo como una expresión."""

import pytest

from conftest import Media, Preguntar, Sumar
from soma_next import Done, Graph, Node


# ── Encadenar ──


def test_un_solo_nodo_ya_es_un_grafo():
    g = Graph.somatize(Sumar(1))
    assert g.nodes() == ["sumar"]
    assert g.forward(41) == 42.0


def test_una_cadena():
    g = Graph.somatize(Sumar(1) >> Sumar(10) >> Sumar(100))
    assert g.nodes() == ["sumar", "sumar_2", "sumar_3"]
    assert g.forward(0) == 111.0


def test_named_pone_el_id():
    g = Graph.somatize(Sumar(1).named("primero") >> Sumar(10).named("segundo"))
    assert g.nodes() == ["primero", "segundo"]
    assert g.edges() == [("primero", "segundo")]


# ── Abrir y cerrar ramas ──


def test_un_diamante_se_lee_de_un_vistazo():
    g = Graph.somatize(Sumar(1) >> (Sumar(10).named("izq") | Sumar(100).named("der")) >> Media())
    assert g.edges() == [
        ("sumar", "izq"),
        ("sumar", "der"),
        ("izq", "media"),
        ("der", "media"),
    ]
    assert g.forward(0) == 56.0


def test_ramas_abiertas_salen_como_mapa():
    g = Graph.somatize(Sumar(1) >> (Sumar(10).named("izq") | Sumar(100).named("der")))
    assert g.forward(0) == {"izq": 11.0, "der": 101.0}


def test_una_rama_puede_ser_mas_larga_que_la_otra():
    g = Graph.somatize(
        Sumar(1).named("fuente")
        >> ((Sumar(1).named("izq") >> Sumar(1).named("izq2")) | Sumar(1).named("der"))
    )
    assert g.edges() == [
        ("fuente", "izq"),
        ("izq", "izq2"),
        ("fuente", "der"),
    ]


def test_tres_ramas():
    g = Graph.somatize(Sumar(0) >> (Sumar(1).named("a") | Sumar(2).named("b") | Sumar(3).named("c")))
    assert g.forward(0) == {"a": 1.0, "b": 2.0, "c": 3.0}


def test_un_nodo_que_pide_turnos_encaja_donde_encajaria_cualquiera():
    from conftest import Gritar

    g = Graph.somatize(Sumar(1) >> Preguntar("x"))
    assert g.forward(41, driver=Gritar()) == "X"


# ── La clase obliga, y es la única puerta del DSL ──


def test_la_clase_obliga_a_implementar_forward():
    class SinForward(Node):
        pass

    with pytest.raises(TypeError, match="abstract method 'forward'"):
        SinForward()


def test_en_el_dsl_hay_que_heredar_de_node():
    class Suelto:
        def forward(self, x, ctx):
            return Done(x)

    with pytest.raises(TypeError, match="tiene que heredar de soma_next.Node"):
        Graph.somatize(Suelto() >> Sumar(1))


def test_lo_que_no_puede_ser_nodo_lo_dice():
    with pytest.raises(TypeError, match="tiene que heredar de soma_next.Node"):
        Graph.somatize(Sumar(1) >> "esto no es un nodo")


def test_un_objeto_de_fuera_sigue_entrando_por_la_puerta_de_abajo(g):
    class Ajeno:  # no hereda de nada nuestro
        def forward(self, x, ctx):
            return Done(x * 2)

    g.node("ajeno", Ajeno())
    assert g.forward(21) == 42.0


# ── El DSL y las llamadas construyen lo mismo ──


def test_el_dsl_no_es_otra_cosa_que_node_y_edge():
    dsl = Graph.somatize(Sumar(1).named("a") >> Sumar(10).named("b"))

    a_mano = Graph()
    a_mano.node("a", Sumar(1))
    a_mano.node("b", Sumar(10))
    a_mano.edge("a", "b")

    assert dsl.nodes() == a_mano.nodes()
    assert dsl.edges() == a_mano.edges()
    assert dsl.plan() == a_mano.plan()


def test_el_numero_de_argumentos_lo_sigue_contando_rust(g):
    with pytest.raises(ValueError, match="toma \\(objeto\\)"):
        g.node()
