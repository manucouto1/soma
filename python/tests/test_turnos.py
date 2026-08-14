"""Nodos que piden algo antes de terminar, y quien lo atiende.

No son un tipo aparte: lo único que los distingue es que a veces devuelven
`Await` en vez de `Done`.
"""

import pytest

from conftest import Gritar, Preguntar, Sumar
from soma_next import Await, Done, Node


class Insaciable(Node):
    def forward(self, x, ctx):
        return Await(["otra vez"])


class Tacano:
    """Un driver que devuelve menos resultados de los que le pidieron."""

    def perform(self, peticiones):
        return []


def test_quien_termina_a_la_primera_no_necesita_driver(g):
    g.node("sumar", Sumar(1))
    assert g.forward(41) == 42.0


def test_pide_algo_y_el_driver_se_lo_da(g):
    g.node("pregunta", Preguntar("hola"))
    assert g.forward(driver=Gritar()) == "HOLA"


def test_sin_driver_el_que_pide_lo_dice(g):
    g.node("pregunta", Preguntar("hola"))
    with pytest.raises(ValueError, match="no tiene driver"):
        g.forward()


def test_un_driver_sin_perform_falla_al_usarlo(g):
    g.node("pregunta", Preguntar("hola"))
    with pytest.raises(TypeError, match="le falta perform"):
        g.forward(driver=object())


def test_un_driver_que_devuelve_de_menos_lo_dice(g):
    g.node("pregunta", Preguntar("a", "b"))
    with pytest.raises(ValueError, match="devolvió 0"):
        g.forward(driver=Tacano())


def test_el_que_no_sabe_parar_gasta_sus_turnos(g):
    g.node("nunca", Insaciable())
    with pytest.raises(ValueError, match="no sabe parar"):
        g.forward(driver=Gritar())


def test_las_dos_clases_de_nodo_se_encadenan(g):
    g.node("sumar", Sumar(1))
    g.node("pregunta", Preguntar("x"))
    g.edge("sumar", "pregunta")
    assert g.forward(41, driver=Gritar()) == "X"


def test_el_plan_no_distingue_quien_pide_turnos(g):
    g.node("sumar", Sumar(1))
    g.node("pregunta", Preguntar("x"))
    g.edge("sumar", "pregunta")
    assert g.plan().count("Execute") == 2
    assert "Step {" not in g.plan()


def test_un_nodo_puede_evolucionar_sin_cambiar_de_tipo(g):
    """Empieza terminando siempre; una rama nueva le añade un turno."""

    class Evoluciona(Node):
        def forward(self, x, ctx):
            if ctx.turn > 0:
                return Done(ctx.results[0])
            return Await(["negativo"]) if x < 0 else Done(x)

    g.node("evoluciona", Evoluciona())
    assert g.forward(1) == 1.0                        # ni necesita driver
    assert g.forward(-1, driver=Gritar()) == "NEGATIVO"


def test_el_contexto_dice_el_turno_y_lo_que_trajo_el_driver(g):
    vistos = []

    class Mira(Node):
        def forward(self, x, ctx):
            vistos.append((ctx.turn, list(ctx.results)))
            if ctx.turn < 2:
                return Await([f"t{ctx.turn}"])
            return Done("fin")

    g.node("mira", Mira())
    assert g.forward("x", driver=Gritar()) == "fin"
    assert vistos == [(0, []), (1, ["T0"]), (2, ["T1"])]
