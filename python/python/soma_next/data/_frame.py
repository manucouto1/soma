"""A frame, in whichever dataframe you have installed.

What a source answers with is Arrow, and `polars`, `pandas` and `pyarrow` all
read Arrow. So the frame hands over IPC bytes and these turn them into one of
the three — rather than this library picking one and making it a dependency of
every worker that only counts rows.
"""

from __future__ import annotations

__all__ = ["to_arrow", "to_polars"]


def to_polars(frame):
    """That frame as a `polars.DataFrame`."""
    import polars

    return polars.read_ipc_stream(frame.ipc())


def to_arrow(frame):
    """That frame as a `pyarrow.Table`."""
    import pyarrow

    return pyarrow.ipc.open_stream(frame.ipc()).read_all()
