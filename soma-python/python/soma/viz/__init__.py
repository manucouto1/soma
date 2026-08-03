"""soma.viz — figures and tables over runs and studies.

Optuna-style plotting on ``Study`` and metric/efficiency views on
``RunView``, all returning interactive Plotly figures (rendered inline
in notebooks, serializable with ``fig.to_json()``)::

    study.plot_optimization_history()
    study.plot_parallel_coordinate()
    study.plot_timeline()
    run = soma.runs()[0]
    run.plot_metrics()
    run.plot_gantt()          # where the run's wall time went
    study.trials_dataframe()  # pandas projection

Requires the ``viz`` extra: ``pip install 'somatize[viz]'``. The
methods are always installed; only calling them needs plotly/pandas.
"""

from __future__ import annotations

from soma.viz._figures import (
    plot_gantt,
    plot_intermediate_values,
    plot_metrics,
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_pareto_front,
    plot_timeline,
)
from soma.viz._frames import experiments_dataframe, metrics_dataframe, trials_dataframe
from soma.viz._report import build_report, study_to_html, to_html
from soma.viz._health import (
    plot_audit,
    plot_channel_evolution,
    plot_channels,
    plot_health,
    plot_module_flow,
)

__all__ = [
    "plot_health",
    "plot_audit",
    "plot_module_flow",
    "plot_channels",
    "plot_channel_evolution",
    "plot_optimization_history",
    "plot_intermediate_values",
    "plot_parallel_coordinate",
    "plot_param_importances",
    "plot_timeline",
    "plot_pareto_front",
    "plot_metrics",
    "plot_gantt",
    "trials_dataframe",
    "metrics_dataframe",
    "experiments_dataframe",
    "build_report",
    "to_html",
    "study_to_html",
]
