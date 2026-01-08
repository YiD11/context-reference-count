"""Shared test fixtures."""

import uuid
from pathlib import Path

import pytest

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))


@pytest.fixture(autouse=True)
def reset_chromadb_state():
    """Reset ChromaDB state between tests to avoid collection conflicts."""
    yield
