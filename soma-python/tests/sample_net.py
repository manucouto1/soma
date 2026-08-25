"""A node in a separate module, to test what cloudpickle does not send by itself.

It lives here and not in `test_remote.py` on purpose: that one registers its own
module "by value" on import, so from there the case that matters could not be
checked — a node that comes from an **importable** module, which cloudpickle
serializes by reference and the worker cannot open.

This module is importable from the tests and **not** from the worker: the child
process inherits the working directory, which is `python/`, and `tests/` is not
on its `sys.path`.
"""

from soma_next import Node


class Greet(Node):
    def forward(self, x, ctx):
        return f"hello, {x}"
