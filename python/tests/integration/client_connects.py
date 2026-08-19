"""The client against a worker that **was already running**.

It is the whole use case: the worker stood up on its own, in the background, and
this program only connects and sends it work. The address arrives as an
argument, as it would from an environment variable or a node file.

`sys.path` and not `PYTHONPATH`: the worker is someone else's process, started
earlier and from somewhere else, so `net` is **not** within its reach by any
means. Whatever it uses, it got over the wire.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from net import graph, nodes  # noqa: E402

from soma_next import Graph, Worker  # noqa: E402

addr = sys.argv[1]
n = nodes()
g = graph(Graph, n)

w1 = Worker.at(addr, mode="network", send=["net"])
w2 = Worker.at(addr, mode="network", send=["net"])

output = g.forward("  The Dog Runs Quickly  ", workers={"w1": w1, "w2": w2})
print("OUTPUT", output)
print("HERE", float(os.getpid()))
