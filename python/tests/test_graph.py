import soma_next


def test_un_grafo_recien_creado_esta_vacio():
    g = soma_next.Graph()
    assert len(g) == 0
    assert repr(g) == "Graph(0 nodos)"
