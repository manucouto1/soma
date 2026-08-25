"""Where the data comes from, and whether the model is learning what you meant.

Two halves that meet on the same word. **Where it comes from**::

    from somatize import Graph, Store
    from somatize.data import Parquet, settle, to_polars

    sms = Parquet(Store("/data"), "sms/train")
    g = Graph.somatize(sms.named("sms").frozen() >> Clean().named("clean").cached())
    settle(g)
    g.forward({"at": 0, "take": 64}, store="/data")

A source is a node, what it answers with is Arrow, and what the graph is handed
is a **coordinate** — which is what stops a cache weighing the batch on every
step. See `somatize.data._source`.

And **whether it is learning what you meant**, which is not about the network at
all::

    from somatize.data import contribution, leaning, shares

    said = contribution(g, batches, objective=mse, over=("symptoms", "text"))
    shares(said)     # {"symptoms": 0.01, "text": 0.99}
    leaning(said)    # {"symptoms": ["IGNORED_INPUT(symptoms)"], ...}

`somatize.health` asks whether a network is **learning**: gradients,
activations, channels, the update. This asks whether it is learning **what you
meant** — which no amount of looking at a gradient will ever say.

It exists because of a real project: symptom channels for detecting a
mental-health condition, where interpretability and performance could be had one
at a time and never together. Months went into the architecture. The signal was
in the self-disclosure and not in the symptoms, and one afternoon of taking
inputs away would have said so.
"""

from somatize.data._ablation import contribution, leaning, shares, shuffled
from somatize.data._figure import leaned
from somatize.data._frame import to_arrow, to_polars
from somatize.data._source import Parquet, settle

__all__ = [
    "Parquet",
    "contribution",
    "leaned",
    "leaning",
    "settle",
    "shares",
    "shuffled",
    "to_arrow",
    "to_polars",
]
