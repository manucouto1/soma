"""Fake nodes, shared by every test.

None of them declares what "kind" it is: what tells them apart is the transition
they return.
"""

import pytest

from soma_next import Await, Done, Graph, Node


class Add(Node):
    def __init__(self, how_much):
        self.how_much = how_much

    def forward(self, x, ctx):
        return Done(x + self.how_much)


class Identity(Node):
    def forward(self, x, ctx):
        return Done(x)


class Mean(Node):
    """An aggregator is a node that reads a map. There is no type behind it."""

    def forward(self, inputs, ctx):
        return Done(sum(inputs.values()) / len(inputs))


class Ask(Node):
    """Asks for things on turn 0 and returns whatever it is told."""

    def __init__(self, *requests):
        self.requests = list(requests)

    def forward(self, x, ctx):
        if ctx.turn == 0:
            return Await(self.requests)
        return Done(ctx.results[0])


class Shout:
    """A driver."""

    def perform(self, requests):
        return [r.upper() for r in requests]


@pytest.fixture
def g():
    return Graph()
