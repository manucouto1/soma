"""The client, starting the workers itself as child processes.

It is for testing and **not** for the use case: as long as the client starts the
process, there is no independent worker worth the name. That is what
`client_connects.py` is for.

What it does show well is the shape of the plan and that distributing does not
change the result, neither of which needs a standing worker to be checked.
"""

import os

from net import graph, nodes

from soma_next import Graph, Worker

n = nodes()
g = graph(Graph, n)

print("HOSTS", g.hosts())
print("PLAN", g.plan())

w1 = Worker.generic()
w2 = Worker.generic()

output = g.forward("  The Dog Runs Quickly  ", workers={"w1": w1, "w2": w2})
print("OUTPUT", output)
print("HERE", float(os.getpid()))
