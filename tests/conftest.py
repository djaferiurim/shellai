"""Shared pytest fixtures."""

import pytest


@pytest.fixture
def tmp_workspace(tmp_path):
    """A temporary directory to act as an agent workspace root."""
    return tmp_path
