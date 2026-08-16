"""Los parámetros de un grafo: lo que un optimizador tiene que actualizar."""

from __future__ import annotations


def parameters(graph):
    """Los parámetros de todos los nodos del grafo que tengan.

    Se pregunta por `.parameters()` y se salta a quien no lo tenga — un
    tokenizador o un lematizador no tienen nada que entrenar y no por eso
    dejan de ser nodos. La alternativa, meter `parameters()` en el contrato
    del nodo, es el impuesto que el Soma original cobró con `fit`: sus dobles
    de test lo implementan vacío solo para poder existir.

    Van sin repetir, **por identidad**: dos nodos pueden compartir un módulo
    —pesos atados entre un embedding y la capa de salida es el caso clásico— y
    entonces el mismo `Parameter` sale dos veces. Un optimizador con
    duplicados avisa o falla, según la versión de torch.

    El orden es el de declaración de los nodos, así que dos llamadas dan lo
    mismo.
    """
    vistos, todos = set(), []
    for node_id in graph.nodes():
        implementation = graph.implementation(node_id)
        recoge = getattr(implementation, "parameters", None)
        if recoge is None:
            continue
        for parametro in recoge():
            if id(parametro) not in vistos:
                vistos.add(id(parametro))
                todos.append(parametro)
    return todos
