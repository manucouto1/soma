"""The example's nodes. **Both sides** import this file.

That the catalog is built by a function — rather than travelling — is the
decision that separates soma from the original soma: an `Arc<dyn Node>`
does not cross a wire, so either it is there, or it arrives inside an artifact
someone knows how to open.
"""

import os

from somatize import Node


class Clean(Node):
    def forward(self, text, ctx):
        return text.strip().lower()


class Tokenize(Node):
    def forward(self, text, ctx):
        return text.split()


class Count(Node):
    def forward(self, words, ctx):
        return {"how_many": float(len(words)), "pid": float(os.getpid())}


class Oddities(Node):
    def forward(self, words, ctx):
        return {"long_ones": [w for w in words if len(w) > 5], "pid": float(os.getpid())}


class Join(Node):
    """A fan-in: receives a map keyed by each branch."""

    def forward(self, inputs, ctx):
        return {
            "how_many": inputs["count"]["how_many"],
            "long_ones": inputs["oddities"]["long_ones"],
            "pids": sorted({inputs["count"]["pid"], inputs["oddities"]["pid"]}),
        }


def nodes():
    return {
        "clean": Clean(),
        "tokenize": Tokenize(),
        "count": Count(),
        "oddities": Oddities(),
        "join": Join(),
    }


def graph(Graph, n, distributed=True):
    """The same network, distributed or whole here.

    That it is the same expression with and without `.at()` is not a
    convenience of the test: it is the guarantee. Distributing is a decision
    about **where**, not a change in what the graph computes.
    """
    away = (lambda node, host: node.at(host)) if distributed else (lambda node, _: node)
    return Graph.somatize(
        n["clean"].named("clean")
        >> away(n["tokenize"].named("tokenize"), "w1")
        >> (
            away(n["count"].named("count"), "w1")
            | away(n["oddities"].named("oddities"), "w2")
        )
        >> n["join"].named("join")
    )
