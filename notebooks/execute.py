#!/usr/bin/env python
"""Re-execute the tutorial notebooks in place, with their outputs.

    python notebooks/execute.py              # all of them
    python notebooks/execute.py 10 11 12     # just these

The notebooks ship executed so they can be read on GitHub without
running anything, which means the saved outputs are part of the source
and have to be regenerated deliberately.

Three things this does that a bare `jupyter nbconvert --execute` does
not, each because of a specific way the naive version goes wrong:

**A fresh temp working directory per notebook.** Notebooks write
`.soma/` run directories. Executed in place they would litter the repo,
and — worse — a notebook that demonstrates a cache *miss* would find a
hit left over from the previous run and silently teach the opposite of
what it says.

**A fresh `SOMA_CACHE_DIR` per notebook**, for the same reason one level
down: the miss→hit demonstrations in notebooks 02 and 04 are only
deterministic against an empty cache.

**A warning gate.** Any notebook whose stderr contains "Warning" is
treated as a failure and is *not* written back. Nearly every warning
soma can emit in a notebook means the notebook is wrong rather than
soma is: a filter without `_cache_version` (`getsource` is unavailable
under a headless kernel, so identity silently falls back to cloudpickle),
an un-materialized module, a mismatched channel sampling rate. Shipping
a tutorial whose output includes a warning teaches the warning.

Helper modules next to the notebooks (`campaign.py`) are copied into the
working directory, so `import campaign` resolves there exactly as it
does for a reader running `jupyter lab notebooks/`.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NB_DIR = Path(__file__).resolve().parent
TIMEOUT = 900


def _select(argv: list[str]) -> list[Path]:
    everything = sorted(NB_DIR.glob("*.ipynb"))
    if not argv:
        return everything
    wanted = tuple(argv)
    return [p for p in everything if p.name.startswith(wanted)]


def _has_warning(nb) -> str | None:
    for cell in nb.cells:
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream" and output.get("name") == "stderr":
                for line in output.get("text", "").splitlines():
                    if "Warning" in line:
                        return line.strip()
    return None


def _run(path: Path) -> bool:
    nb = nbformat.read(path, as_version=4)
    nbformat.validator.normalize(nb)  # older notebooks lack cell ids

    workdir = Path(tempfile.mkdtemp(prefix=f"nbexec-{path.stem[:4]}-"))
    for helper in NB_DIR.glob("*.py"):
        if helper.name != Path(__file__).name:
            shutil.copy(helper, workdir / helper.name)

    env_before = os.environ.get("SOMA_CACHE_DIR")
    os.environ["SOMA_CACHE_DIR"] = tempfile.mkdtemp(prefix="nbcache-")
    try:
        NotebookClient(
            nb,
            timeout=TIMEOUT,
            allow_errors=False,
            resources={"metadata": {"path": str(workdir)}},
        ).execute()
    except Exception as exc:  # noqa: BLE001 — reported, not raised
        print(f"  ✗ {path.name}: {type(exc).__name__}: {str(exc).splitlines()[0]}")
        return False
    finally:
        if env_before is None:
            os.environ.pop("SOMA_CACHE_DIR", None)
        else:
            os.environ["SOMA_CACHE_DIR"] = env_before

    warning = _has_warning(nb)
    if warning:
        print(f"  ✗ {path.name}: warning in output, not written back:\n      {warning}")
        return False

    nbformat.write(nb, path)
    outputs = sum(len(c.get("outputs", [])) for c in nb.cells)
    size = path.stat().st_size / 1024
    print(f"  ✓ {path.name}: {outputs} outputs, {size:.0f} KiB")
    return True


def main() -> int:
    selected = _select(sys.argv[1:])
    if not selected:
        print("no notebooks matched", file=sys.stderr)
        return 1
    print(f"executing {len(selected)} notebook(s) from {NB_DIR}")
    ok = [_run(p) for p in selected]
    failed = ok.count(False)
    print(f"{ok.count(True)} succeeded, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
