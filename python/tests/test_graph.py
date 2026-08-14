"""El DSL de topología: node, edge y las consultas."""

import pytest

from conftest import Identidad, Sumar


def test_un_grafo_recien_creado_esta_vacio(g):
    assert len(g) == 0
    assert g.nodes() == []
    assert repr(g) == "Graph(0 nodos, 0 aristas)"


def test_node_con_id_explicito_devuelve_el_id(g):
    assert g.node("limpiar", Identidad()) == "limpiar"
    assert "limpiar" in g
    assert len(g) == 1


def test_node_sin_id_lo_deriva_de_la_clase(g):
    assert g.node(Identidad()) == "identidad"


def test_node_sin_id_desempata_sufijando(g):
    assert g.node(Identidad()) == "identidad"
    assert g.node(Identidad()) == "identidad_2"
    assert g.node(Identidad()) == "identidad_3"


def test_una_tuberia_de_dos_nodos(g):
    g.node("limpiar", Identidad())
    g.node("vectorizar", Identidad())
    g.edge("limpiar", "vectorizar")

    assert g.edges() == [("limpiar", "vectorizar")]
    assert g.roots() == ["limpiar"]
    assert g.leaves() == ["vectorizar"]
    assert g.topological_sort() == ["limpiar", "vectorizar"]
    assert repr(g) == "Graph(2 nodos, 1 aristas)"


def test_el_objeto_registrado_se_recupera(g):
    nodo = Identidad()
    g.node("limpiar", nodo)
    assert g.implementation("limpiar") is nodo
    assert g.implementation("no_existe") is None


def test_dos_nodos_no_pueden_llamarse_igual(g):
    g.node("limpiar", Identidad())
    with pytest.raises(ValueError, match="ya hay un nodo llamado `limpiar`"):
        g.node("limpiar", Sumar(1))
    assert len(g) == 1


def test_una_arista_a_un_nodo_que_no_existe(g):
    g.node("limpiar", Identidad())
    with pytest.raises(ValueError, match="no nombra ningún nodo"):
        g.edge("limpiar", "fantasma")
    assert g.edges() == []


def test_un_ciclo_se_rechaza_al_ponerlo(g):
    for name in ("a", "b", "c"):
        g.node(name, Identidad())
    g.edge("a", "b")
    g.edge("b", "c")
    with pytest.raises(ValueError, match="cerraría un ciclo"):
        g.edge("c", "a")
    assert len(g.edges()) == 2


def test_la_misma_arista_no_se_pone_dos_veces(g):
    g.node("a", Identidad())
    g.node("b", Identidad())
    g.edge("a", "b")
    with pytest.raises(ValueError, match="ya existe"):
        g.edge("a", "b")


def test_consultar_un_nodo_que_no_existe(g):
    with pytest.raises(ValueError, match="no nombra ningún nodo"):
        g.predecessors("fantasma")


def test_node_con_argumentos_absurdos(g):
    with pytest.raises(ValueError, match="toma \\(objeto\\)"):
        g.node()


def test_ramas_paralelas_que_se_juntan(g):
    for name in ("entrada", "izquierda", "derecha", "juntar"):
        g.node(name, Identidad())
    g.edge("entrada", "izquierda")
    g.edge("entrada", "derecha")
    g.edge("izquierda", "juntar")
    g.edge("derecha", "juntar")

    assert g.roots() == ["entrada"]
    assert g.leaves() == ["juntar"]
    assert g.predecessors("juntar") == ["izquierda", "derecha"]
    assert g.successors("entrada") == ["izquierda", "derecha"]

    orden = g.topological_sort()
    assert orden[0] == "entrada"
    assert orden[-1] == "juntar"
