"""The same module, one version behind, mounted into `worker-old`.

Nothing here is staged: it is another file in another container's mount, and its
fingerprint comes out different because the code **is** different. What the
client sends says which version it was written against; this worker compares and
decides — stopping with `--strict`, and running its own with `--lucky`.
"""

from soma_next import Done, Node

FACTOR = 3


class Scale(Node):
    """Multiplies. An older version: it multiplies by something else."""

    def forward(self, x, ctx):
        return Done(x * FACTOR)
