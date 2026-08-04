"""Read the local experiments journal (``<root>/experiments.jsonl``).

Every completed tracked run and study appends one ``ExperimentRecord``
line; this is the Python-side reader for quick inspection::

    for exp in soma.experiments():
        print(exp["name"], exp["metrics"], exp["tags"])
"""

from __future__ import annotations

import json
import pathlib


def experiments(root: str = ".soma") -> list[dict]:
    """All recorded experiments under `root`, oldest first. A torn
    trailing line (crash mid-append) is skipped silently."""
    path = pathlib.Path(root) / "experiments.jsonl"
    if not path.exists():
        return []
    records: list[dict] = []
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    for i, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except ValueError:
            if i != len(lines) - 1:
                raise
    return records
