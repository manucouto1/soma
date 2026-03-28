"""Tests for the Soma Lab API."""
import pytest
import subprocess
import time
import sys
import os


class TestLabAPI:
    """Test Lab connection (requires no running server for unit tests)."""

    def test_lab_import(self):
        from soma import Lab
        assert Lab is not None

    def test_lab_connect_fails_on_bad_url(self):
        from soma import Lab
        with pytest.raises(ConnectionError):
            Lab.connect("http://127.0.0.1:19999")  # port that shouldn't be listening

    def test_lab_repr(self):
        from soma.lab import Lab
        # Can't connect but can test repr format
        lab = Lab.__new__(Lab)
        lab.url = "http://localhost:3000"
        assert repr(lab) == "Lab('http://localhost:3000')"
