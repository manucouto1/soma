"""Shared pytest configuration for the soma test suite."""

from __future__ import annotations

import warnings

import pytest


# PyTorch warns when a full backward hook fires while no input requires
# grad — benign in our fixtures (inputs are data, not parameters). This
# was previously silenced only in test_gradient_audit.py; the audit
# suites in test_diagnostics.py need it too.
@pytest.fixture(autouse=True)
def _silence_torch_hook_warning():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Full backward hook is firing when gradients are computed",
        )
        yield
