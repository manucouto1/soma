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

Importing this also says how a tensor is written down, which is what lets a
graph keep what it produces: see `soma_next.torch._codec`.

Training does not touch the graph: afterwards its nodes, its edges, its plan and
its placement are the same. What changes are the weights, which live inside the
nodes and always did.

That the package is called `torch` does not shadow the real one: in Python 3
imports are absolute, so `import torch` in here brings the usual one.
"""

from soma_next.torch._codec import register as _register_the_tensor_codec
from soma_next.torch._freeze import freeze
from soma_next.torch._params import parameters
from soma_next.torch._trainer import NoGradient, Result, Trainer

# On being imported, and not on being asked: a graph that keeps what it produces
# needs a tensor to be writable **before** the first node runs, and by then
# nobody is going to remember to say so.
_register_the_tensor_codec()

__all__ = ["NoGradient", "Result", "Trainer", "freeze", "parameters"]
