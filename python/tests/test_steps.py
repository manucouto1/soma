"""Nodos que pueden no terminar, y quien atiende lo que piden.

Debajo no hay un tipo aparte: lo que los distingue de un filtro es que a veces
devuelven `{"await": …}` en vez de `{"done": …}`.
"""

import pytest

import soma_next
from soma_next import Filter, Graph, Step


class Eco(Step):
    """Termina en el primer turno: un filtro con la firma larga."""

    def forward(self, x, ctx):
        return {"done": x}


class Preguntar(Step):
    """Pide cosas en el turno 0 y devuelve lo que le contesten."""

    def __init__(self, *peticiones):
        self.peticiones = list(peticiones)

    def forward(self, x, ctx):
        if ctx["turn"] == 0:
            return {"await": self.peticiones}
        return {"done": ctx["results"][0]}


class Insaciable(Step):
    def forward(self, x, ctx):
        return {"await": ["otra vez"]}


class Gritar:
    def perform(self, peticiones):
        return [p.upper() for p in peticiones]


class Tacano:
    """Devuelve menos resultados de los que le pidieron."""

    def perform(self, peticiones):
        return []


@pytest.fixture
def g():
    return Graph()


def test_uno_que_termina_a_la_primera_no_necesita_driver(g):
    g.step("eco", Eco())
    assert g.forward("hola") == "hola"


def test_pide_algo_y_el_driver_se_lo_da(g):
    g.step("pregunta", Preguntar("hola"))
    assert g.forward(driver=Gritar()) == "HOLA"


def test_sin_driver_el_que_pide_lo_dice(g):
    g.step("pregunta", Preguntar("hola"))
    with pytest.raises(ValueError, match="no tiene driver"):
        g.forward()


def test_sin_forward_falla_al_registrarlo(g):
    class NoLoEs:
        pass

    with pytest.raises(TypeError, match="le falta forward"):
        g.step("malo", NoLoEs())
    assert len(g) == 0


def test_un_driver_sin_perform_falla_al_usarlo(g):
    g.step("pregunta", Preguntar("hola"))
    with pytest.raises(TypeError, match="le falta perform"):
        g.forward(driver=object())


def test_un_driver_que_devuelve_de_menos_lo_dice(g):
    g.step("pregunta", Preguntar("a", "b"))
    with pytest.raises(ValueError, match="devolvió 0"):
        g.forward(driver=Tacano())


def test_el_que_no_sabe_parar_gasta_sus_turnos(g):
    g.step("nunca", Insaciable())
    with pytest.raises(ValueError, match="no sabe parar"):
        g.forward(driver=Gritar())


def test_devolver_cualquier_cosa_lo_dice(g):
    class Confuso(Step):
        def forward(self, x, ctx):
            return "no soy un dict"

    g.step("confuso", Confuso())
    with pytest.raises(ValueError, match="debe devolver"):
        g.forward()


def test_un_dict_sin_done_ni_await_lo_dice(g):
    class Confuso(Step):
        def forward(self, x, ctx):
            return {"quizas": 1}

    g.step("confuso", Confuso())
    with pytest.raises(ValueError, match='sin "done" ni "await"'):
        g.forward()


def test_las_dos_convenciones_se_encadenan(g):
    class Sumar(Filter):
        def forward(self, x):
            return x + 1

    g.node("sumar", Sumar())
    g.step("eco", Eco())
    g.edge("sumar", "eco")
    assert g.forward(41) == 42.0


def test_el_plan_no_los_distingue(g):
    class Sumar(Filter):
        def forward(self, x):
            return x + 1

    g.node("sumar", Sumar())
    g.step("eco", Eco())
    g.edge("sumar", "eco")
    # Los dos compilan al mismo paso: quién pide turnos se sabe al ejecutar.
    assert g.plan().count("Execute") == 2
    assert "Step {" not in g.plan()


def test_se_nombra_solo_como_un_filtro(g):
    assert g.step(Eco()) == "eco"
    assert g.step(Eco()) == "eco_2"
