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


def test_una_lista_va_y_vuelve_igual(g):
    class Doblar:
        def forward(self, xs):
            return [x * 2 for x in xs]

    g.node("doblar", Doblar())
    assert g.forward([1, 2, 3]) == [2.0, 4.0, 6.0]


def test_una_lista_anidada_tambien(g):
    class Identidad:
        def forward(self, x):
            return x

    g.node("id", Identidad())
    assert g.forward([1, ["dos", None], 3]) == [1.0, ["dos", None], 3.0]


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
            return {"no", "cruzo"}  # un set, no un dict

    g.node("devuelve", Devuelve())
    with pytest.raises(ValueError, match="un `set` no cruza"):
        g.forward(1)


def test_un_bool_no_se_convierte_en_silencio(g):
    g.node("sumar", Sumar(1))
    with pytest.raises(TypeError, match="un bool no cruza"):
        g.forward(True)


# ── Las decisiones pendientes, visibles desde Python ──


# ── Abanicos, en las dos direcciones ──


class Media:
    """Un agregador es un filtro que lee un mapa. No hay ningún tipo detrás."""

    def forward(self, entradas):
        return sum(entradas.values()) / len(entradas)


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
    class Claves:
        def forward(self, entradas):
            return list(entradas.keys())

    g.node("segundo", Sumar(1))
    g.node("primero", Sumar(1))
    g.node("juntar", Claves())
    g.edge("segundo", "juntar")
    g.edge("primero", "juntar")

    assert g.forward(0) == ["segundo", "primero"]


def test_un_dict_va_y_vuelve_igual(g):
    class Identidad:
        def forward(self, x):
            return x

    g.node("id", Identidad())
    assert g.forward({"b": 1, "a": ["dos", None]}) == {"b": 1.0, "a": ["dos", None]}


def test_una_clave_que_no_es_texto_lo_dice(g):
    g.node("sumar", Sumar(1))
    with pytest.raises(TypeError, match="claves de un dict"):
        g.forward({1: "uno"})


def test_el_plan_se_puede_mirar(g):
    g.node("fuente", Sumar(1))
    g.node("izq", Sumar(10))
    g.edge("fuente", "izq")

    plan = g.plan()
    assert "Sequence" in plan
    assert "from" in plan
