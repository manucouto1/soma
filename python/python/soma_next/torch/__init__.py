"""What only makes sense with torch in front of you: training.

The core does not know what a loss is, nor a gradient, nor an optimizer, and it
is not going to — writing this neutrally would ask for a `Backend` with a single
implementor. So it lives here, in Python, and `core/` does not change a line::

    from soma_next import Graph
    from soma_next.torch import Trainer, parameters

    g = Graph.somatize(Encoder().on("cuda:0") >> Head().on("cuda:0"))
    t = Trainer(g, objective=cross_entropy,
                optimizer=torch.optim.Adam(parameters(g), lr=1e-3))
    t.fit(data, epochs=10)

Training does not touch the graph: afterwards its nodes, its edges, its plan and
its placement are the same. What changes are the weights, which live inside the
nodes and always did.

That the package is called `torch` does not shadow the real one: in Python 3
imports are absolute, so `import torch` in here brings the usual one.
"""

from soma_next.torch._params import parameters
from soma_next.torch._trainer import Result, Trainer

__all__ = ["Result", "Trainer", "parameters"]
