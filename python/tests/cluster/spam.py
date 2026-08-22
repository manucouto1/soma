"""The SMS Spam Collection — a real dataset, and the containers never see it.

5,574 real text messages, 747 of them spam. It is here rather than generated
because what the study is being asked to find is a **learning rate that works on
this data**, and synthetic noise has no learning rate that works on it.

The imbalance is the useful part too: 13% of it is spam, so "always ham" already
gets 87% right and a loss that comes down is a loss that learnt something.

This module is the client's. It is not sent to any worker and it is in no image:
the workers get `cluster.nodes`, which has no `datasets` in it and does not need
one — data reaches a worker as the input of a graph, like everything else.
"""

from __future__ import annotations

NAME = "ucirvine/sms_spam"


def messages(how_many=1600):
    """`(texts, labels)` — the first `how_many`, shuffled, spam as 1.

    It raises rather than skipping. Whether an unreachable hub is a skip or a
    failure is the test's to decide, and this file is also imported by the
    processes the test starts, where `pytest.skip` would come out as a crash.
    """
    from datasets import load_dataset

    whole = load_dataset(NAME, split="train")

    # Seeded, so every machine in the study trains on the same messages in the
    # same order — otherwise two trials would differ by their data as well as by
    # their configuration, and the search would be comparing nothing.
    shuffled = whole.shuffle(seed=0).select(range(min(how_many, len(whole))))
    return list(shuffled["sms"]), list(shuffled["label"])


def batches(texts, labels, size=64):
    """The messages in batches, as a `Trainer` takes them."""
    import torch

    return [
        (texts[at : at + size], torch.tensor(labels[at : at + size], dtype=torch.long))
        for at in range(0, len(labels), size)
    ]
