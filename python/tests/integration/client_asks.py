"""The client against a standing worker, with a node that asks the world.

`ask` returns `Await`, so it needs a driver **where it runs**, and where it runs
is someone else's process. The driver is not in the graph — nobody declared it —
so it gets there the only way anything gets there: packed in the artifact,
alongside the nodes, sent by `forward`.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from net import Shout, asking, nodes  # noqa: E402

from soma_next import Graph, Worker  # noqa: E402

addr = sys.argv[1]
g = asking(Graph, nodes())

w1 = Worker.at(addr, mode="network", send=["net"])

print("OUTPUT", g.forward("  The Dog Runs Quickly  ", driver=Shout(), workers={"w1": w1}))
print("HERE", float(os.getpid()))
