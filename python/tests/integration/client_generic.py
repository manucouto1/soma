"""The client, without writing any worker.

On the other side runs `python -m soma_next.worker`, which comes with the
package and starts empty. With `--no-send` the package that has to travel is
deliberately left out, to see what it says.

`sys.path` and not `PYTHONPATH`: the first belongs to this process and the
second would be inherited by the worker, and then it could import `net` on its
own and `send=` would be doing nothing. The difference is exactly what this
client tests.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from net import graph, nodes  # noqa: E402

from soma_next import Graph, Worker  # noqa: E402

n = nodes()
g = graph(Graph, n)

send = [] if "--no-send" in sys.argv else ["net"]
w1 = Worker.generic(mode="network", send=send)
w2 = Worker.generic(mode="network", send=send)

output = g.forward("  The Dog Runs Quickly  ", workers={"w1": w1, "w2": w2})
print("OUTPUT", output)
print("HERE", float(os.getpid()))
