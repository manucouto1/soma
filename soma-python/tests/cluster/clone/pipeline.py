"""The version of the code a worker with the project **already has**.

This file is mounted read-only into `worker-a` and `worker-b`, and the client
imports this very copy: both sides are at the same version, which is what
`mode="project"` needs in order to send forty bytes instead of a pickle.

`clone_old/pipeline.py` is the same module one version behind. Nothing about it
is staged: it is another file, in another image's mount, and the fingerprints
come out different because the code is different.
"""

from soma_next import Node

FACTOR = 2


class Scale(Node):
    """Multiplies. The version here is the one the graph is written against."""

    def forward(self, x, ctx):
        return x * FACTOR
