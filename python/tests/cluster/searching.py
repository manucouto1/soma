"""The study, written the way somebody would write it.

This file is meant to be **read**. It is the whole use case and there is nothing
test-shaped in it: what is being searched over, how the pipeline is put together
for one configuration, and the loop. `test_searching.py` starts several of these
against one directory and checks what came out.

    python searching.py <store> <me> <a> <gpu> <messages>

Run it `n` times against the same `<store>` and that is a distributed study.
Nothing in here is told what the others are doing: a trial is a number, `ask` is
a function of that number, and `claim` settles who gets which — so a machine
works out its own configuration without replaying anybody else's.

# Two distributions at once, and they are not the same one

The **graph** is cut by `.at()`: tokenising goes to a worker with no torch in it
at all, the embedding to the one that has it. The **study** is cut by `claim`:
this script, running in several places, over one directory.

Both are here because both are real and confusing them is the easy mistake.

# What a trial of a cut graph can be scored on

The curve is the **training** loss, and that is not laziness. `embed` is trained
where it runs, so its weights are over there and `export` refuses to hand back a
copy that never learnt anything — the right refusal. A held-out score would have
to be produced on the machine that holds the weights, so what would have to
travel is the **scoring**, the same way the trainer travelled in CU14.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from cluster import spam  # noqa: E402
from soma_next import Graph, Store, Worker  # noqa: E402
from soma_next.study import (  # noqa: E402
    DONE,
    PRUNED,
    Pruner,
    Sampler,
    Space,
    curves,
    finished,
    in_flight,
    report,
    take,
)
from soma_next.torch import Split, Trainer, parameters  # noqa: E402

STUDY = "spam"
TRIALS = 8
EPOCHS = 3

#: What is being searched over: three knobs, one of each kind there is. `lr` is
#: logarithmic because drawn linearly four fifths of this range sits above 0.06,
#: and the study would never see a small learning rate at all.
space = (
    Space()
    .real("lr", 1e-3, 3e-1, log=True)
    .int("dim", 8, 48)
    .choice("opt", ["adam", "sgd"])
)

#: Even for **every prefix**, so two machines taking different numbers out of the
#: directory cannot land on neighbouring configurations. A uniform draw is even
#: only in expectation, and "they probably will not collide" is not what a study
#: spread over machines wants. The seed is the only thing they share.
sampler = Sampler.sobol(seed=0)

#: Judged against the trials that finished — which other machines ran. It stops
#: nothing: it answers, and the loop below stops calling `step`.
pruner = Pruner.median(goal="min", warmup=1, startup=2)

OPTIMIZERS = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}


def search(store, me, at, messages=600):
    """Take whichever trials nobody has taken, and run them.

    `n` of these against one `store` is the whole thing. There is no server, no
    port and no protocol: a directory, and `claim`.
    """
    batches = spam.batches(*spam.messages(messages))
    mine = []

    for trial in range(TRIALS):
        # What has finished **and** what the others are holding: a guided sampler
        # that is not told the second proposes where somebody already is, and two
        # machines spend two trials learning one thing.
        point = sampler.ask(
            space,
            trial,
            finished(store, space, study=STUDY) + in_flight(store, space, study=STUDY),
        )
        if not take(store, point, study=STUDY, trial=trial, me=me):
            continue  # somebody else got that number; on to the next

        # A `Point` is a mapping, so a configuration goes straight into whatever
        # builds the thing — no unpacking it knob by knob.
        trainer = training(**point, at=at)

        drawn, why = [], None
        for left in reversed(range(EPOCHS)):
            epoch = trainer.fit(batches)
            drawn.append(sum(epoch.history) / len(epoch.history))
            report(store, point, drawn, study=STUDY, trial=trial, me=me)
            # Only while there is another epoch to give up on. Asked after the
            # last one, a pruner would label a trial that ran the whole course as
            # pruned — and its score would stop counting as comparable for no
            # reason at all.
            if left and (why := pruner.verdict(drawn, curves(store, study=STUDY))):
                break

        state = PRUNED if why else DONE
        report(
            store, point, drawn, study=STUDY, trial=trial, me=me,
            state=state, because=why,
        )
        mine.append(
            {"trial": trial, "point": str(point), "state": state, "reports": drawn}
        )
    return mine


def training(lr, dim, opt, *, at):
    """One configuration, as a graph on two machines and a trainer over it.

    `at` is which port each host name reaches. The graph says `a` and `gpu`, and
    what those resolve to is this machine's business — which is the whole reason
    a host is a name.
    """
    graph = Graph.somatize(
        spam.Clean().named("clean").at("a")
        >> spam.Embed(dim).named("embed").at("gpu")
        >> spam.Classify(dim, 2).named("head")
    )
    # `embed` runs on another machine, so it is trained **there**: the trainer
    # travels and stands beside it, activations go one way and `dL/da` comes back
    # the other. Nothing about the weights ever travels.
    trains = {"embed": Split(OPTIMIZERS[opt], lr=lr)}
    return Trainer(
        graph,
        objective=torch.nn.functional.cross_entropy,
        optimizer=OPTIMIZERS[opt](parameters(graph, without=trains), lr=lr),
        trains=trains,
        # Fresh ones, per configuration and not per machine: a worker handle
        # carries the catalog it provisioned, so it belongs to **one graph**.
        # Holding one across two configurations is being told to reconnect.
        workers={
            name: Worker.at(
                f"127.0.0.1:{port}", mode="network", send=["cluster.spam"]
            )
            for name, port in at.items()
        },
    )


if __name__ == "__main__":
    store, me, cheap, with_torch, messages = sys.argv[1:6]
    print(
        json.dumps(
            search(
                Store(store), me, {"a": int(cheap), "gpu": int(with_torch)},
                int(messages),
            )
        )
    )
