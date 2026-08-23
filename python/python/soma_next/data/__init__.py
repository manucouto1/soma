"""Whether the model is learning what you think it is learning.

The third question, and the one that is not about the network at all::

    from soma_next.data import contribution, leaning, shares

    said = contribution(g, batches, objective=mse, over=("symptoms", "text"))
    shares(said)     # {"symptoms": 0.01, "text": 0.99}
    leaning(said)    # {"symptoms": ["IGNORED_INPUT(symptoms)"], ...}

`soma_next.health` asks whether a network is **learning**: gradients,
activations, channels, the update. This asks whether it is learning **what you
meant** — which no amount of looking at a gradient will ever say.

It exists because of a real project: symptom channels for detecting a
mental-health condition, where interpretability and performance could be had one
at a time and never together. Months went into the architecture. The signal was
in the self-disclosure and not in the symptoms, and one afternoon of taking
inputs away would have said so.
"""

from soma_next.data._ablation import contribution, leaning, shares, shuffled
from soma_next.data._figure import leaned

__all__ = ["contribution", "leaned", "leaning", "shares", "shuffled"]
