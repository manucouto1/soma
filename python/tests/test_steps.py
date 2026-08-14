"""Steps: unidades que pueden no terminar, y quien atiende lo que piden."""

import pytest

import soma_next


class Eco:
    """Termina en el primer turno: un filtro disfrazado de step."""

    def poll(self, ctx):
        return {"done": ctx["input"]}


class Preguntar:
    """Pide una cosa en el turno 0 y devuelve lo que le contesten."""

    def __init__(self, *peticiones):
        self.peticiones = list(peticiones)

    def poll(self, ctx):
        if ctx["turn"] == 0:
            return {"await": self.peticiones}
        return {"done": ctx["results"][0]}


class Insaciable:
    def poll(self, ctx):
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
    return soma_next.Graph()


def test_un_step_que_termina_a_la_primera_no_necesita_driver(g):
    g.step("eco", Eco())
    assert g.forward("hola") == "hola"


def test_un_step_pide_algo_y_el_driver_se_lo_da(g):
    g.step("pregunta", Preguntar("hola"))
    assert g.forward(driver=Gritar()) == "HOLA"


def test_sin_driver_un_step_que_pide_lo_dice(g):
    g.step("pregunta", Preguntar("hola"))
    with pytest.raises(ValueError, match="no tiene driver"):
        g.forward()


def test_un_step_sin_poll_falla_al_registrarlo(g):
    class NoEsUnStep:
        pass

    with pytest.raises(TypeError, match="le falta poll"):
        g.step("malo", NoEsUnStep())
    assert len(g) == 0


def test_un_driver_sin_perform_falla_al_usarlo(g):
    g.step("pregunta", Preguntar("hola"))
    with pytest.raises(TypeError, match="le falta perform"):
        g.forward(driver=object())


def test_un_driver_que_devuelve_de_menos_lo_dice(g):
    g.step("pregunta", Preguntar("a", "b"))
    with pytest.raises(ValueError, match="devolvió 0"):
        g.forward(driver=Tacano())


def test_un_step_que_no_sabe_parar_gasta_sus_turnos(g):
    g.step("nunca", Insaciable())
    with pytest.raises(ValueError, match="no sabe parar"):
        g.forward(driver=Gritar())


def test_poll_que_devuelve_cualquier_cosa_lo_dice(g):
    class Confuso:
        def poll(self, ctx):
            return "no soy un dict"

    g.step("confuso", Confuso())
    with pytest.raises(ValueError, match="debe devolver"):
        g.forward()


def test_un_filtro_y_un_step_se_encadenan(g):
    class Sumar:
        def forward(self, x):
            return x + 1

    g.node("sumar", Sumar())
    g.step("eco", Eco())
    g.edge("sumar", "eco")
    assert g.forward(41) == 42.0


def test_el_step_se_nombra_solo_como_un_filtro(g):
    assert g.step(Eco()) == "eco"
    assert g.step(Eco()) == "eco_2"
