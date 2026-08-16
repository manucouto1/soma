"""Entrenar **un** grafo. Ni el grafo sabe que existe esto, ni esto sabe que
hay otros entrenamientos.

Son tres niveles, y ninguno conoce al de arriba:

| nivel | qué es | escala |
|---|---|---|
| el grafo | una red | un `forward` |
| `Trainer` | un entrenamiento | una tarde |
| un estudio | N entrenamientos | un experimento |

El tercero **no tiene tipo**, y es a propósito: N entrenamientos independientes
son una lista de Python, y modelar una lista como un grafo es pagar el precio
de un DAG para no usarlo. Un grafo se gana el sueldo cuando hay dependencias
que declarar.

Y el entrenamiento no es un nodo por la misma razón: el contrato de un nodo
—`forward(input, ctx)`— describe **un paso**, con su presupuesto de turnos y
sin recuperación parcial. Un entrenamiento dura una tarde, muta su estado y
falla de maneras de las que uno quiere recuperarse. El Soma original metió
`fit` en el contrato del nodo, y la factura se ve en sus propios tests: cuatro
crates implementan un `fit` vacío solo para poder existir.
"""

from __future__ import annotations

import torch

from soma_next import Opaque
from soma_next.torch._params import parameters


class Resultado:
    """Lo que deja un entrenamiento: la pérdida, paso a paso.

    Es lo mínimo que hace comparables dos runs, que es lo único que alguien
    necesita hoy. Métricas, tiempos y checkpoints cuando haya quien los pida.
    """

    def __init__(self, historia):
        self.historia = historia

    @property
    def loss(self):
        """La última pérdida, o `None` si no se dio ni un paso."""
        return self.historia[-1] if self.historia else None

    def __repr__(self):
        if not self.historia:
            return "Resultado(sin pasos)"
        return (
            f"Resultado({len(self.historia)} pasos, "
            f"{self.historia[0]:.4f} → {self.historia[-1]:.4f})"
        )


class Trainer:
    """Entrena un grafo, sin que el grafo se entere.

    Recibe el grafo y no al revés —nada de `g.fit(...)`—: así el mismo grafo se
    entrena de tres maneras sin tocarlo, y sigue siendo el artefacto que se
    serializa y viaja.

    El optimizador lo construye quien llama::

        t = Trainer(g, objetivo=cross_entropy,
                    optimizador=torch.optim.Adam(parameters(g), lr=1e-3))

    Así se elige Adam o SGD sin que esto tenga que conocerlos, y sin un
    `optimizador="adam"` que acabe siendo un registro de nombres.
    """

    def __init__(self, graph, *, objetivo, optimizador):
        parametros = parameters(graph)
        if not parametros:
            raise ValueError(
                "este grafo no tiene parámetros: ningún nodo contesta a "
                "`.parameters()`, así que entrenarlo no cambiaría nada y la "
                "pérdida saldría plana"
            )
        _comprueba_que_se_hablan(parametros, optimizador)

        self.graph = graph
        self.objetivo = objetivo
        self.optimizador = optimizador

    def step(self, lote):
        """Un paso: adelante, pérdida, atrás, actualizar. Devuelve la pérdida.

        Es **la primitiva**, y `fit` es azúcar encima. Todo lo que no entra en
        un bucle de épocas —parada temprana, rondas federadas, PBT— se escribe
        como un `while` sobre esto, en vez de como opciones y *callbacks* de
        esta clase.
        """
        entrada, objetivo = lote
        self.optimizador.zero_grad()
        salida = self.graph.forward(_cruzable(entrada))
        perdida = self.objetivo(salida, _donde_la_salida(objetivo, salida))
        perdida.backward()
        self.optimizador.step()
        return perdida.item()

    def fit(self, datos, epocas=1):
        """Da un paso por lote, tantas épocas como digas.

        `datos` se recorre una vez por época, así que con más de una tiene que
        ser algo re-iterable —una lista, un `DataLoader`—: un generador se
        agota en la primera.
        """
        historia = []
        for _ in range(epocas):
            for lote in datos:
                historia.append(self.step(lote))
        return Resultado(historia)

    def __repr__(self):
        return f"Trainer({len(parameters(self.graph))} parámetros)"


def _cruzable(entrada):
    """Un tensor se envuelve para cruzar una arista; lo demás pasa tal cual.

    `Opaque` se pide a mano en todas partes para que nada cruce opaco por
    accidente. Aquí se pone solo, y es una excepción con motivo: esto es
    `soma_next.torch`, donde un tensor no es una sorpresa sino el caso. Lo que
    no es un tensor —una lista de textos, por ejemplo— sigue su camino y se
    convierte como siempre.
    """
    return Opaque(entrada) if isinstance(entrada, torch.Tensor) else entrada


def _donde_la_salida(objetivo, salida):
    """El objetivo va a buscar a la salida allá donde haya acabado.

    Hay una asimetría que solo se ve entrenando en GPU: la **entrada** cruza el
    grafo y cada nodo la mueve a su dispositivo, porque eso es lo que hace un
    nodo colocado. El **objetivo** no entra en el grafo — va directo a la
    pérdida—, así que nadie lo mueve nunca. Si el último nodo está en `cuda:0`,
    la salida sale de allí y el objetivo sigue en la cpu.

    Quien puede arreglarlo es el único que ve los dos: esto. Y mover el
    objetivo y no la salida no es indiferente — traer la salida a la cpu
    arrastraría el backward de vuelta por el cable en cada paso.
    """
    if torch.is_tensor(objetivo) and torch.is_tensor(salida):
        return objetivo.to(salida.device)
    return objetivo


def _comprueba_que_se_hablan(parametros, optimizador):
    """Que el optimizador y el grafo hablen de los mismos pesos.

    Solo se rechaza que **no compartan ninguno**, que no tiene lectura
    inocente: el optimizador se construyó sobre otro grafo, o sobre éste antes
    de que tuviera nodos.

    Cubrir solo una parte sí es legítimo y se deja pasar: entrenar únicamente
    la cabeza y congelar el encoder es exactamente eso.
    """
    del_grafo = {id(p) for p in parametros}
    del_optimizador = {
        id(p) for grupo in optimizador.param_groups for p in grupo["params"]
    }
    if not (del_grafo & del_optimizador):
        raise ValueError(
            f"el optimizador no actualiza ningún parámetro de este grafo: "
            f"tiene {len(del_optimizador)} y el grafo {len(del_grafo)}, y no "
            f"coincide ninguno. ¿Se construyó sobre otro grafo? "
            f"Se hace con `Adam(parameters(g), ...)`"
        )
