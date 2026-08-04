"""`Graph` — the Rust runtime object plus the parts written in Python.

Some of what a `Graph` does cannot live in Rust. Training a
`DifferentiableFilter` has to keep torch tensors on the Python side,
because autograd does not survive the `Value` boundary; a checkpoint has
to import the user's filter classes; a notebook repr has to build HTML.
So `Graph` is a Python class over the Rust one.

It used to be the Rust class with those methods *assigned onto it* at
import time, from six different modules. Three consequences, and only the
first was obvious:

- nothing could see them. Not `help(Graph)`, not an IDE, not mypy, not a
  reader of the class.
- which methods existed depended on what had been imported, so the same
  object had a different surface in two programs.
- three of them shadowed a Rust method of the same name. `forward` was
  the dangerous one — two execution engines answering to one name — and
  is now dispatched from Rust; `compile` and `Study.run` are decorators
  and stay.

Assigning in the class body keeps every implementation where it is — the
training loop is still in `_orchestrator`, checkpointing still in
`_checkpoint` — while making the surface of `Graph` one list you can read.
"""

from __future__ import annotations

from soma._soma import Graph as _RustGraph

from soma import _audit, _checkpoint, _compile, _orchestrator, _study, _tracking
from soma.builder import somatize as _somatize


class Graph(_RustGraph):
    """A computational graph: filters and steps, wired and executed.

    Everything below is defined in Python. Everything else is inherited
    from the Rust class — `node`, `edge`, `fit`, `forward`, `run`, and
    the rest of the runtime surface.
    """

    # ── Training a differentiable graph (soma._orchestrator) ──
    materialize = _orchestrator.materialize
    train = _orchestrator.train
    eval = _orchestrator.eval
    to = _orchestrator.to
    parameters = _orchestrator.parameters
    set_optimizer = _orchestrator.set_optimizer
    make_optimizer = _orchestrator.make_optimizer
    optimizer = _orchestrator.optimizer
    context = _orchestrator.context
    backward = _orchestrator.backward
    step = _orchestrator.step
    zero_grad = _orchestrator.zero_grad
    freeze = _orchestrator.freeze

    # ── Checkpoints (soma._checkpoint) ──
    state = _checkpoint.state
    load_state = _checkpoint.load_state
    save = _checkpoint.save
    restore_optimizer = _checkpoint.restore_optimizer

    # ── Search and studies (soma._study) ──
    search_space = _study.graph_search_space
    apply_params = _study.apply_params
    study = _study.graph_study

    # ── Tracking and diagnostics ──
    track_run = _tracking.track_run
    gradient_audit = _audit.gradient_audit

    # ── Presentation ──
    compile = _compile.compile_with_repr

    @classmethod
    def load(cls, *args, **kwargs):
        return _checkpoint.load(cls, *args, **kwargs)

    @classmethod
    def somatize(cls, topology):
        """Build a graph from the fluent builder API.

        "You think it. Soma somatizes it."
        """
        return _somatize(topology)

