"""One machine of a distributed study, as a script — because it has to be one.

Started `n` times against the same directory, this is the whole use case::

    python searching.py <store> <me> <trials> <port-a> <port-gpu> <messages> <epochs>

Nothing in here is told what the others are doing. A trial is a number, `ask` is
a function of that number, and `claim` settles who gets which — so a machine
works out its own configuration without replaying anybody else's.

It prints one line of JSON: which trials this machine took, and how they went.

# What a trial of a cut graph can be scored on

The curve is the **training** loss, and that is not laziness. `embed` is trained
where it runs, so its weights are over there and `export` refuses to hand back a
copy that never learnt anything — which is the right refusal. A held-out score
would therefore have to be produced on the machine that holds the weights, and
producing it is a forward through a graph whose session belongs to the trainer.

So: the number a study of a cut graph compares is the one the loop produces. The
day a held-out score is wanted, what has to travel is the **scoring**, not the
weights — the same way the trainer travelled in CU14.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import torch  # noqa: E402

from cluster import nodes, spam  # noqa: E402
from soma_next import Graph, Store, Worker  # noqa: E402
from soma_next.study import (  # noqa: E402
    DONE,
    PRUNED,
    Pruner,
    Sampler,
    Space,
    curves,
    finished,
    report,
    take,
)
from soma_next.torch import Split, Trainer, parameters  # noqa: E402

STUDY = "spam"

#: What is being searched over: three knobs, one of each kind there is. `lr` is
#: logarithmic because drawn linearly four fifths of this range sits above 0.06,
#: and the study would never see a small rate at all.
SPACE = (
    Space()
    .real("lr", 1e-3, 3e-1, log=True)
    .int("dim", 8, 48)
    .choice("opt", ["adam", "sgd"])
)

#: Even for **every prefix**, so two machines taking different numbers out of
#: the folder cannot land on neighbouring configurations. The seed is the only
#: thing they share.
SAMPLER = Sampler.sobol(seed=0)

#: Judged against the trials that finished, at the same epoch — and those were
#: run by other machines. It stops nothing: it answers, and the loop below stops
#: calling `step`.
PRUNER = Pruner.median(goal="min", warmup=1, startup=2)

OPTIMIZERS = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}


def machine(where, me, how_many, ports, messages, epochs):
    """Walk the trial numbers, take whatever nobody has, and run those."""
    store = Store(where)
    batches = spam.batches(*spam.messages(messages))

    mine = []
    for trial in range(how_many):
        point = SAMPLER.ask(SPACE, trial, finished(store, SPACE, study=STUDY))
        if not take(store, point, study=STUDY, trial=trial, me=me):
            continue
        mine.append(one(store, point, me, trial, batches, ports, epochs))
    return mine


def one(store, point, me, trial, batches, ports, epochs):
    """One trial: build it, train it, and give up on it if it is going badly."""
    _, trainer = built(point, ports)
    drawn, state, because = [], DONE, None

    for _ in range(epochs):
        drawn.append(
            sum(trainer.step(batch) for batch in batches) / len(batches)
        )
        report(store, point, drawn, study=STUDY, trial=trial, me=me)
        # Against what other machines finished, read off the same folder. It
        # answers; nothing is asked of the trainer and nothing interrupts it.
        because = PRUNER.verdict(drawn, curves(store, study=STUDY))
        if because:
            state = PRUNED
            break

    report(
        store, point, drawn, study=STUDY, trial=trial, me=me,
        state=state, because=because,
    )
    return {"trial": trial, "point": str(point), "state": state, "reports": drawn}


def built(point, ports):
    """The graph this configuration is, and the trainer that moves it.

    Three nodes on two machines. `Clean` goes where there is **no torch at all**,
    which is what a heterogeneous cluster is for: the cheap half of a pipeline
    does not need the expensive machine.
    """
    graph = Graph.somatize(
        nodes.Clean().named("clean").at("a")
        >> nodes.Embed(point["dim"]).named("embed").at("gpu")
        >> nodes.Classify(point["dim"], 2).named("head")
    )
    # The embedding runs on another machine, so it is trained **there**: the
    # trainer travels and stands beside it, activations go one way and `dL/da`
    # comes back the other. Nothing about the weights ever travels — until
    # somebody asks for them, which is what `export` is.
    trains = {"embed": Split(OPTIMIZERS[point["opt"]], lr=point["lr"])}
    trainer = Trainer(
        graph,
        objective=torch.nn.functional.cross_entropy,
        optimizer=OPTIMIZERS[point["opt"]](
            parameters(graph, without=trains), lr=point["lr"]
        ),
        trains=trains,
        workers={which: reachable(ports[which]) for which in ("a", "gpu")},
    )
    return graph, trainer


def reachable(port):
    return Worker.at(f"127.0.0.1:{port}", mode="network", send=["cluster.nodes"])


if __name__ == "__main__":
    where, me = sys.argv[1], sys.argv[2]
    how_many, port_a, port_gpu = (int(one) for one in sys.argv[3:6])
    messages, epochs = int(sys.argv[6]), int(sys.argv[7])
    print(
        json.dumps(
            machine(
                where, me, how_many, {"a": port_a, "gpu": port_gpu}, messages, epochs
            )
        )
    )
