"""Soma: A computational graph runtime for research pipelines."""

from soma._soma import (
    Agent,
    Judge,
    Run,
    SomaNodeNotFound,
    SomaPruned,
    SomaSchemaMismatch,
    SomaSuspended,
    Tool,
    Trial,
    Worker,
    __version__,
    models,
    providers,
    tool,
)
from soma._graph import Graph
from soma._soma import Pbt
from soma._study import Study
from soma.filter import Filter
from soma._identity import CacheConfigError
from soma.search import search
from soma.lab import Lab
from soma.chain import Chain, Fork
from soma import agentic
from soma import library
from soma.builder import somatize as _somatize

try:
    from soma._composite import DifferentiableFilter
except ImportError:  # torch not installed — DifferentiableFilter is opt-in
    DifferentiableFilter = None   # type: ignore[assignment,misc]







from soma._experiments import experiments  # noqa: E402
from soma._lineage import (  # noqa: E402
    checkout,
    detach,
    diff,
    find_similar,
    head,
    lineage,
    record_conclusion,
    reindex,
)
from soma._runs import RunView, runs  # noqa: E402

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
    Audit = AuditReport = AuditScope = FilterReport = None  # type: ignore[assignment,misc]
    ChannelConfig = StepRecord = None                # type: ignore[assignment,misc]
    GradientHealthError = Thresholds = None          # type: ignore[assignment,misc]
    audit_modules = None                             # type: ignore[assignment]

__all__ = [
    "Graph",
    # Agentic
    "Agent",
    "SomaSuspended",
    "SomaPruned",
    "SomaSchemaMismatch",
    "SomaNodeNotFound",
    "Judge",
    "agentic",
    "library",
    "Tool",
    "tool",
    "providers",
    "models",
    "Run",
    "Pbt",
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
    "checkout",
    "head",
    "detach",
    "reindex",
    "find_similar",
    "record_conclusion",
    "lineage",
    "diff",
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
