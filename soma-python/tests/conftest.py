"""Fake nodes, shared by every test.

None of them declares what "kind" it is: what tells them apart is the transition
they return.
"""

import pytest

from somatize import Graph, Node


class Add(Node):
    def __init__(self, how_much):
        self.how_much = how_much

    def forward(self, x, ctx):
        return x + self.how_much


class Identity(Node):
    def forward(self, x, ctx):
        return x


class Mean(Node):
    """An aggregator is a node that reads a map. There is no type behind it."""

    def forward(self, inputs, ctx):
        return sum(inputs.values()) / len(inputs)


@pytest.fixture
def g():
    return Graph()
