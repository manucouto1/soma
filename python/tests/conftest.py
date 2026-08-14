"""Nodos de mentira, compartidos por todos los tests.

Ninguno declara de qué "tipo" es: lo que los distingue es la transición que
devuelven.
"""

import pytest

from soma_next import Await, Done, Graph, Node


class Sumar(Node):
    def __init__(self, cuanto):
        self.cuanto = cuanto

    def forward(self, x, ctx):
        return Done(x + self.cuanto)


class Identidad(Node):
    def forward(self, x, ctx):
        return Done(x)


class Media(Node):
    """Un agregador es un nodo que lee un mapa. No hay ningún tipo detrás."""

    def forward(self, entradas, ctx):
        return Done(sum(entradas.values()) / len(entradas))


class Preguntar(Node):
    """Pide cosas en el turno 0 y devuelve lo que le contesten."""

    def __init__(self, *peticiones):
        self.peticiones = list(peticiones)

    def forward(self, x, ctx):
        if ctx.turn == 0:
            return Await(self.peticiones)
        return Done(ctx.results[0])


class Gritar:
    """Un driver."""

    def perform(self, peticiones):
        return [p.upper() for p in peticiones]


@pytest.fixture
def g():
    return Graph()
