"""Soma: A computational graph runtime for research pipelines."""

from soma._soma import Graph, Run, Study, Trial, Worker, __version__
from soma.filter import Filter
from soma._identity import CacheConfigError
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

# Install train/eval/forward/materialize/parameters on Graph for
# native gradient flow. Import for side-effects.
from soma import _orchestrator  # noqa: E402, F401

# Install state/load_state/save/load on Graph (depends on
# _orchestrator's freeze/py_state). Import for side-effects.
from soma import _checkpoint  # noqa: E402, F401

# Install search_space/apply_params/study on Graph. Import for
# side-effects.
from soma import _study  # noqa: E402, F401

# Install track_run on Graph. Import for side-effects.
from soma import _tracking  # noqa: E402, F401

# Wrap Graph.compile in CompileInfo (dict + notebook repr). Import for
# side-effects.
from soma import _compile  # noqa: E402, F401

from soma._experiments import experiments  # noqa: E402
from soma._runs import RunView, runs  # noqa: E402

# Install plot_*/dataframe methods on Study and RunView. The methods are
# always present; calling them without the `somatize[viz]` extra raises
# a helpful error.
from soma import viz as _viz  # noqa: E402

_viz._install()
from soma.viz import experiments_dataframe  # noqa: E402

# Install gradient_audit on Graph (depends on _orchestrator). Import
# for side-effects. Re-export the user-facing types.
try:
    from soma._audit import (
        Audit,
        AuditReport,
        AuditScope,
        ChannelConfig,
        FilterReport,
        GradientHealthError,
        StepRecord,
        Thresholds,
        audit_modules,
    )
except ImportError:  # torch not installed
    Audit = AuditReport = AuditScope = FilterReport = None  # type: ignore[assignment]
    ChannelConfig = StepRecord = None                # type: ignore[assignment]
    GradientHealthError = Thresholds = None          # type: ignore[assignment]
    audit_modules = None                             # type: ignore[assignment]

__all__ = [
    "Graph",
    "Run",
    "Study",
    "Trial",
    "Worker",
    "Filter",
    "DifferentiableFilter",
    "Lab",
    "Chain",
    "Fork",
    "search",
    "experiments",
    "runs",
    "RunView",
    "experiments_dataframe",
    "__version__",
    "Audit",
    "AuditScope",
    "AuditReport",
    "ChannelConfig",
    "FilterReport",
    "GradientHealthError",
    "StepRecord",
    "Thresholds",
    "audit_modules",
]
