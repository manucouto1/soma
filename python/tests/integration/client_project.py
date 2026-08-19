"""The client against a worker that **has the project**.

Not one line of code goes over the wire: what goes is the classes' names, the
version expected of each, and each node's state. The worker supplies the code
from its clone.

With `--lucky-expected` it does not fail even though the worker warns that its
version is another: it serves to check that it executes just the same and says so.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from net import graph, nodes  # noqa: E402

from soma_next import Graph, Worker  # noqa: E402

addr = sys.argv[1]
n = nodes()
g = graph(Graph, n)

w1 = Worker.at(addr)          # `project` is the default
w2 = Worker.at(addr)

output = g.forward("  The Dog Runs Quickly  ", workers={"w1": w1, "w2": w2})
print("OUTPUT", output)
print("HERE", float(os.getpid()))
from soma_next._remote import _pack  # noqa: E402

print("SIZE", len(_pack(n, None, "project", ())[1]))
