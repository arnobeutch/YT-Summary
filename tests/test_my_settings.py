"""Tests for my_settings."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

import my_settings
from my_settings import Settings

if TYPE_CHECKING:
    from pathlib import Path


class TestLoadDotenv:
    def test_missing_file_is_noop(self, tmp_path: Path) -> None:
        # must not raise
        my_settings._load_dotenv(tmp_path / ".env")

    def test_basic_key_value(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        env = tmp_path / ".env"
        env.write_text("FOO=bar\nBAZ=qux\n")
        monkeypatch.delenv("FOO", raising=False)
        monkeypatch.delenv("BAZ", raising=False)
        my_settings._load_dotenv(env)
        assert os.environ["FOO"] == "bar"
        assert os.environ["BAZ"] == "qux"

    def test_comments_and_blank_lines_skipped(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env = tmp_path / ".env"
        env.write_text("# comment\n\n  \nKEY=value\n")
        monkeypatch.delenv("KEY", raising=False)
        my_settings._load_dotenv(env)
        assert os.environ["KEY"] == "value"

    def test_strip_quotes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        env = tmp_path / ".env"
        env.write_text("Q1=\"quoted\"\nQ2='single'\n")
        monkeypatch.delenv("Q1", raising=False)
        monkeypatch.delenv("Q2", raising=False)
        my_settings._load_dotenv(env)
        assert os.environ["Q1"] == "quoted"
        assert os.environ["Q2"] == "single"

    def test_line_without_equals_skipped(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env = tmp_path / ".env"
        env.write_text("NOEQUALSLINE\nGOOD=yes\n")
        monkeypatch.delenv("GOOD", raising=False)
        monkeypatch.delenv("NOEQUALSLINE", raising=False)
        my_settings._load_dotenv(env)
        assert os.environ["GOOD"] == "yes"
        assert "NOEQUALSLINE" not in os.environ

    def test_existing_env_wins(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        env = tmp_path / ".env"
        env.write_text("PREEXISTING=from_file\n")
        monkeypatch.setenv("PREEXISTING", "from_env")
        my_settings._load_dotenv(env)
        assert os.environ["PREEXISTING"] == "from_env"

    def test_value_with_internal_equals(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env = tmp_path / ".env"
        env.write_text("URL=https://x.com/?a=1&b=2\n")
        monkeypatch.delenv("URL", raising=False)
        my_settings._load_dotenv(env)
        assert os.environ["URL"] == "https://x.com/?a=1&b=2"


class TestSettingsFromEnv:
    def test_default_log_level(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)  # no .env in tmp_path
        monkeypatch.delenv("LOG_LEVEL", raising=False)
        s = Settings.from_env()
        assert s.log_level == "INFO"

    def test_respects_env_var(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        s = Settings.from_env()
        assert s.log_level == "DEBUG"

    def test_loads_from_dotenv(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("LOG_LEVEL", raising=False)
        (tmp_path / ".env").write_text("LOG_LEVEL=WARNING\n")
        s = Settings.from_env()
        assert s.log_level == "WARNING"
