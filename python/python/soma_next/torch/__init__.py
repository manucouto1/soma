"""Lo que solo tiene sentido con torch delante: entrenar.

El núcleo no sabe qué es una pérdida, ni un gradiente, ni un optimizador, y no
va a saberlo — escribir esto de forma neutral pediría un `Backend` con un solo
implementor. Así que vive aquí, en Python, y `core/` no cambia ni una línea::

    from soma_next import Graph
    from soma_next.torch import Trainer, parameters

    g = Graph.somatize(Encoder().on("cuda:0") >> Cabeza().on("cuda:0"))
    t = Trainer(g, objetivo=cross_entropy,
                optimizador=torch.optim.Adam(parameters(g), lr=1e-3))
    t.fit(datos, epocas=10)

Entrenar no toca el grafo: después de esto sus nodos, sus aristas, su plan y su
colocación son los mismos. Lo que cambia son los pesos, que viven dentro de los
nodos y siempre vivieron ahí.

Que el paquete se llame `torch` no pisa al de verdad: en Python 3 los imports
son absolutos, así que `import torch` aquí dentro trae el de siempre.
"""

from soma_next.torch._params import parameters
from soma_next.torch._trainer import Resultado, Trainer

__all__ = ["Resultado", "Trainer", "parameters"]
