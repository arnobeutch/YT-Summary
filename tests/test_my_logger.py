"""Tests for my_logger."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from types import TracebackType
from typing import TYPE_CHECKING

import colorama
import pytest

from scriber.logger import (
    ColorFormatter,
    MyJSONFormatter,
    NonErrorFilter,
    _excepthook,
    initialize_logger,
    install_excepthook,
    setup_logger,
)

if TYPE_CHECKING:
    from pathlib import Path


def _make_record(level: int, msg: str = "hello") -> logging.LogRecord:
    return logging.LogRecord(
        name="test",
        level=level,
        pathname=__file__,
        lineno=10,
        msg=msg,
        args=None,
        exc_info=None,
    )


class TestColorFormatter:
    def test_info_colored_blue(self) -> None:
        out = ColorFormatter().format(_make_record(logging.INFO))
        assert colorama.Fore.BLUE in out
        assert "INFO" in out
        assert "hello" in out

    @pytest.mark.parametrize(
        ("level", "color"),
        [
            (logging.DEBUG, colorama.Fore.GREEN),
            (logging.INFO, colorama.Fore.BLUE),
            (logging.WARNING, colorama.Fore.YELLOW),
            (logging.ERROR, colorama.Fore.RED),
            (logging.CRITICAL, colorama.Back.RED),
        ],
    )
    def test_each_level_colored(self, level: int, color: str) -> None:
        out = ColorFormatter().format(_make_record(level))
        assert color in out

    def test_unknown_level_no_color_wrap(self) -> None:
        # level between INFO and WARNING has no color mapping
        record = _make_record(logging.INFO + 5)
        out = ColorFormatter().format(record)
        # none of the known colors for known levels should appear
        for c in (
            colorama.Fore.GREEN,
            colorama.Fore.BLUE,
            colorama.Fore.YELLOW,
            colorama.Fore.RED,
            colorama.Back.RED,
        ):
            assert c not in out

    def test_does_not_mutate_original_record(self) -> None:
        record = _make_record(logging.INFO)
        original = record.levelname
        ColorFormatter().format(record)
        assert record.levelname == original


class TestMyJSONFormatter:
    def test_basic_json_with_fmt_keys(self) -> None:
        fmt = MyJSONFormatter(fmt_keys={"level": "levelname", "logger": "name"})
        out = fmt.format(_make_record(logging.INFO, "hello"))
        data = json.loads(out)
        assert data["level"] == "INFO"
        assert data["logger"] == "test"
        assert data["message"] == "hello"
        assert "timestamp" in data

    def test_default_no_fmt_keys(self) -> None:
        out = MyJSONFormatter().format(_make_record(logging.INFO))
        data = json.loads(out)
        assert data["message"] == "hello"
        assert "timestamp" in data

    def test_includes_exc_info(self) -> None:
        def _raise() -> None:
            raise ValueError("boom")

        exc_info: tuple[type[BaseException], BaseException, TracebackType | None] | None = None
        try:
            _raise()
        except ValueError as e:
            exc_info = (type(e), e, e.__traceback__)
        assert exc_info is not None
        record = logging.LogRecord(
            name="t",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="x",
            args=None,
            exc_info=exc_info,
        )
        data = json.loads(MyJSONFormatter().format(record))
        assert "exc_info" in data
        assert "ValueError" in data["exc_info"]

    def test_extra_attrs_merged(self) -> None:
        record = _make_record(logging.INFO)
        record.custom_field = "xyz"
        data = json.loads(MyJSONFormatter().format(record))
        assert data["custom_field"] == "xyz"

    def test_builtin_attrs_not_leaked(self) -> None:
        # without fmt_keys, keys like "filename", "funcName" should not appear
        data = json.loads(MyJSONFormatter().format(_make_record(logging.INFO)))
        assert "filename" not in data
        assert "funcName" not in data


class TestNonErrorFilter:
    def test_debug_passes(self) -> None:
        assert NonErrorFilter().filter(_make_record(logging.DEBUG)) is True

    def test_info_passes(self) -> None:
        assert NonErrorFilter().filter(_make_record(logging.INFO)) is True

    def test_warning_blocked(self) -> None:
        assert NonErrorFilter().filter(_make_record(logging.WARNING)) is False

    def test_error_blocked(self) -> None:
        assert NonErrorFilter().filter(_make_record(logging.ERROR)) is False

    def test_critical_blocked(self) -> None:
        assert NonErrorFilter().filter(_make_record(logging.CRITICAL)) is False


class TestSetupLogger:
    def test_creates_logs_directory(
        self,
        logging_env: Path,
        reset_root_logger: logging.Logger,
    ) -> None:
        _ = reset_root_logger
        setup_logger()
        assert (logging_env / "logs").is_dir()

    def test_attaches_handlers_to_root(
        self,
        logging_env: Path,
        reset_root_logger: logging.Logger,
    ) -> None:
        _ = logging_env
        setup_logger()
        assert len(reset_root_logger.handlers) >= 2


class TestExcepthook:
    def test_keyboard_interrupt_forwarded_to_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[tuple[object, ...]] = []

        def fake_default_hook(
            _exc_type: type[BaseException],
            _exc: BaseException,
            _tb: object,
        ) -> None:
            calls.append((_exc_type, _exc, _tb))

        monkeypatch.setattr(sys, "__excepthook__", fake_default_hook)
        _excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)
        assert len(calls) == 1

    def test_other_exception_logged_as_critical(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.CRITICAL, logger="scriber.logger"):
            _excepthook(ValueError, ValueError("x"), None)
        assert any("Unhandled exception" in r.message for r in caplog.records)


class TestInstallExcepthook:
    def test_replaces_sys_excepthook(self, monkeypatch: pytest.MonkeyPatch) -> None:
        original = sys.excepthook
        try:
            install_excepthook()
            assert sys.excepthook is _excepthook
        finally:
            monkeypatch.setattr(sys, "excepthook", original)


class TestInitializeLogger:
    def test_debug_flag_sets_debug(
        self,
        logging_env: Path,
        reset_root_logger: logging.Logger,
    ) -> None:
        _ = logging_env
        ns = argparse.Namespace(debug=True)
        initialize_logger(ns)
        assert reset_root_logger.level == logging.DEBUG

    def test_no_debug_keeps_yaml_level(
        self,
        logging_env: Path,
        reset_root_logger: logging.Logger,
    ) -> None:
        _ = logging_env
        ns = argparse.Namespace(debug=False)
        initialize_logger(ns)
        assert reset_root_logger.level == logging.INFO

    def test_missing_debug_attr_defaults_false(
        self,
        logging_env: Path,
        reset_root_logger: logging.Logger,
    ) -> None:
        _ = logging_env
        ns = argparse.Namespace()  # no `debug` attr
        initialize_logger(ns)
        assert reset_root_logger.level == logging.INFO
