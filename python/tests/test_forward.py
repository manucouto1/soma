"""Ejecutar: el motor está en Rust, esto solo aporta las implementaciones."""

import pytest

from conftest import Identidad, Media, Sumar
from soma_next import Done, Node


class Mayusculas(Node):
    def forward(self, x, ctx):
        return Done(x.upper())


class Romper(Node):
    def forward(self, x, ctx):
        raise RuntimeError("me rompí")


# ── El camino feliz ──


def test_un_grafo_vacio_devuelve_su_entrada(g):
    assert g.forward("intacto") == "intacto"


def test_un_solo_nodo(g):
    g.node("sumar", Sumar(1))
    assert g.forward(41) == 42.0


def test_una_cadena_encadena_las_salidas(g):
    g.node("a", Sumar(1))
    g.node("b", Sumar(10))
    g.node("c", Sumar(100))
    g.edge("a", "b")
    g.edge("b", "c")
    assert g.forward(0) == 111.0


def test_texto_cruza_la_frontera(g):
    g.node("gritar", Mayusculas())
    assert g.forward("hola") == "HOLA"


def test_una_lista_va_y_vuelve_igual(g):
    g.node("id", Identidad())
    assert g.forward([1, 2, 3]) == [1.0, 2.0, 3.0]


def test_una_lista_anidada_tambien(g):
    g.node("id", Identidad())
    assert g.forward([1, ["dos", None], 3]) == [1.0, ["dos", None], 3.0]


def test_un_dict_va_y_vuelve_igual(g):
    g.node("id", Identidad())
    assert g.forward({"b": 1, "a": ["dos", None]}) == {"b": 1.0, "a": ["dos", None]}


def test_sin_entrada_el_nodo_recibe_none(g):
    class Recibe(Node):
        def forward(self, x, ctx):
            assert x is None
            return Done("vale")

    g.node("recibe", Recibe())
    assert g.forward() == "vale"


# ── Los fallos ──


def test_un_objeto_sin_forward_falla_al_registrarlo(g):
    class NoLoEs:
        pass

    with pytest.raises(TypeError, match="le falta forward"):
        g.node("malo", NoLoEs())
    assert len(g) == 0


def test_la_excepcion_de_un_nodo_dice_cual_fue(g):
    g.node("bomba", Romper())
    with pytest.raises(ValueError, match="el nodo `bomba` falló"):
        g.forward(1)


def test_un_tipo_que_no_cruza_lo_dice(g):
    class Devuelve(Node):
        def forward(self, x, ctx):
            return Done({"no", "cruzo"})  # un set

    g.node("devuelve", Devuelve())
    with pytest.raises(ValueError, match="un `set` no cruza"):
        g.forward(1)


def test_un_bool_no_se_convierte_en_silencio(g):
    g.node("sumar", Sumar(1))
    with pytest.raises(TypeError, match="un bool no cruza"):
        g.forward(True)


def test_una_clave_que_no_es_texto_lo_dice(g):
    g.node("id", Identidad())
    with pytest.raises(TypeError, match="claves de un dict"):
        g.forward({1: "uno"})


def test_devolver_algo_que_no_es_una_transicion_lo_dice(g):
    class Confuso(Node):
        def forward(self, x, ctx):
            return "olvidé el Done"

    g.node("confuso", Confuso())
    with pytest.raises(ValueError, match="debe devolver Done"):
        g.forward(1)


# ── Abanicos, en las dos direcciones ──


def test_varias_hojas_salen_como_un_mapa_con_su_nombre(g):
    g.node("fuente", Sumar(1))
    g.node("izq", Sumar(10))
    g.node("der", Sumar(100))
    g.edge("fuente", "izq")
    g.edge("fuente", "der")

    assert g.forward(0) == {"izq": 11.0, "der": 101.0}


def test_a_un_nodo_con_dos_entradas_le_llega_un_mapa(g):
    g.node("izq", Sumar(10))
    g.node("der", Sumar(100))
    g.node("juntar", Media())
    g.edge("izq", "juntar")
    g.edge("der", "juntar")

    assert g.forward(0) == 55.0


def test_un_diamante_da_la_vuelta(g):
    g.node("fuente", Sumar(1))
    g.node("izq", Sumar(10))
    g.node("der", Sumar(100))
    g.node("juntar", Media())
    for a, b in (("fuente", "izq"), ("fuente", "der"), ("izq", "juntar"), ("der", "juntar")):
        g.edge(a, b)

    assert g.forward(0) == 56.0


def test_el_mapa_conserva_el_orden_en_que_se_declararon_las_aristas(g):
    class Claves(Node):
        def forward(self, entradas, ctx):
            return Done(list(entradas.keys()))

    g.node("segundo", Sumar(1))
    g.node("primero", Sumar(1))
    g.node("juntar", Claves())
    g.edge("segundo", "juntar")
    g.edge("primero", "juntar")

    assert g.forward(0) == ["segundo", "primero"]


def test_el_plan_se_puede_mirar(g):
    g.node("fuente", Sumar(1))
    g.node("izq", Sumar(10))
    g.edge("fuente", "izq")

    plan = g.plan()
    assert "Sequence" in plan
    assert "from" in plan
