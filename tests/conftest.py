"""Shared fixtures for logger tests."""

import logging
from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture
def reset_root_logger() -> Generator[logging.Logger]:
    """Restore root logger level and handlers after each test."""
    root = logging.getLogger()
    original_level = root.level
    original_handlers = list(root.handlers)
    yield root
    for handler in list(root.handlers):
        if handler not in original_handlers:
            handler.close()
            root.removeHandler(handler)
    root.setLevel(original_level)


@pytest.fixture
def logging_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated working directory for log file output.

    File handlers write into tmp_path/logs/ (CWD-relative), keeping the project
    root clean. The YAML config itself is loaded from my_logger's own dir.
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path
