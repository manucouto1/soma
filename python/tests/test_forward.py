"""Ejecutar un grafo: el motor está en Rust, esto solo aporta los filtros."""

import pytest

import soma_next


class Sumar:
    def __init__(self, cuanto):
        self.cuanto = cuanto

    def forward(self, x):
        return x + self.cuanto


class Mayusculas:
    def forward(self, x):
        return x.upper()


class Romper:
    def forward(self, x):
        raise RuntimeError("me rompí")


class SinForward:
    pass


@pytest.fixture
def g():
    return soma_next.Graph()


# ── El camino feliz ──


def test_un_grafo_vacio_devuelve_su_entrada(g):
    assert g.forward("intacto") == "intacto"


def test_un_solo_filtro(g):
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


def test_una_lista_de_numeros_es_un_tensor(g):
    class Doblar:
        def forward(self, xs):
            return [x * 2 for x in xs]

    g.node("doblar", Doblar())
    assert g.forward([1, 2, 3]) == [2.0, 4.0, 6.0]


def test_sin_entrada_el_nodo_recibe_none(g):
    class Recibe:
        def forward(self, x):
            assert x is None
            return "vale"

    g.node("recibe", Recibe())
    assert g.forward() == "vale"


# ── Los fallos ──


def test_un_objeto_sin_forward_falla_al_registrarlo(g):
    with pytest.raises(TypeError, match="le falta forward"):
        g.node("malo", SinForward())
    assert len(g) == 0


def test_la_excepcion_de_un_filtro_dice_el_nodo(g):
    g.node("bomba", Romper())
    with pytest.raises(ValueError, match="el nodo `bomba` falló"):
        g.forward(1)


def test_un_tipo_que_no_cruza_lo_dice(g):
    class Devuelve:
        def forward(self, x):
            return {"no": "cruzo"}

    g.node("devuelve", Devuelve())
    with pytest.raises(ValueError, match="un `dict` no cruza"):
        g.forward(1)


def test_un_bool_no_se_convierte_en_silencio(g):
    g.node("sumar", Sumar(1))
    with pytest.raises(TypeError, match="un bool no cruza"):
        g.forward(True)


# ── Las decisiones pendientes, visibles desde Python ──


def test_juntar_dos_ramas_todavia_no_esta_decidido(g):
    for name in ("izq", "der", "juntar"):
        g.node(name, Sumar(1))
    g.edge("izq", "juntar")
    g.edge("der", "juntar")
    with pytest.raises(ValueError, match="cómo se combinan"):
        g.forward(0)


def test_dos_hojas_todavia_no_esta_decidido(g):
    g.node("a", Sumar(1))
    g.node("b", Sumar(1))
    with pytest.raises(ValueError, match="cuál es la salida"):
        g.forward(0)
