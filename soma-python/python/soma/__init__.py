"""Soma: A computational graph runtime for research pipelines."""

from soma._soma import Graph, Study, Worker, __version__
from soma.filter import Filter
from soma.search import search
from soma.lab import Lab
from soma.chain import Chain, Fork
from soma.builder import somatize as _somatize

try:
    from soma._composite import DifferentiableFilter
except ImportError:  # torch not installed — DifferentiableFilter is opt-in
    DifferentiableFilter = None   # type: ignore[assignment]

# Add Graph.somatize() classmethod — bridges the fluent builder API
# with the Rust Graph runtime.
#
# "You think it. Soma somatizes it."
Graph.somatize = classmethod(lambda cls, topology: _somatize(topology))

__all__ = [
    "Graph",
    "Study",
    "Worker",
    "Filter",
    "DifferentiableFilter",
    "Lab",
    "Chain",
    "Fork",
    "search",
    "__version__",
]
