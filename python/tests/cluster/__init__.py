"""A package on purpose, and it is the only one under `tests/`.

Without it, `tests/cluster/conftest.py` is imported as the top-level module
`conftest` and **shadows** `tests/conftest.py`, which every other file in the
suite imports its doubles from. With it, this directory is `cluster` and the two
conftests stop being the same name.
"""
