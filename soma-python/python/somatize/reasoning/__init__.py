"""What somebody was trying to find out, read back and drawn.

`somatize.record` is what happened; this is what was thought about it. Neither
recovers the other: **if it can be recalculated it is record, and if somebody
thought it it is reasoning.**

It is written from the terminal, one move at a time, while somebody is
thinking — `somatize-tree ask`, `suppose`, `tried`, `found`, `decide` — and read
from here::

    from somatize import Store
    from somatize import reasoning

    store = Store("/scratch/inv")

    reasoning.moves(store, tree="inv")                  # every move, in the order they were made
    reasoning.standing(store, tree="inv")               # how each question and hypothesis stands
    reasoning.says(store, tree="inv")                   # what was said, and where it holds
    reasoning.folds(store, tree="inv")                  # what was abandoned, how many and why
    reasoning.cites(store, tree="inv")                  # which moves cite each commit
    reasoning.covered(store, tree="inv", by=["short"])  # what a scope reaches
    reasoning.figure(store, tree="inv")                 # the shape of it

Everything cross-references by **name**, because the store's slot stops
identifying a move the moment nobody is holding it in a variable — and reading
one back is exactly that moment.

`tree=` and never `soma-tree.toml`: a second reader of that file is how a
`--tree` said on the command line reaches the journal and not the walk, with
nothing saying so.
"""

from somatize.reasoning._figure import Card, cards, figure
from somatize.reasoning._read import (
    KINDS,
    STANDINGS,
    cites,
    covered,
    folds,
    moves,
    says,
    standing,
)

__all__ = [
    "KINDS",
    "STANDINGS",
    "Card",
    "cards",
    "cites",
    "covered",
    "figure",
    "folds",
    "moves",
    "says",
    "standing",
]
