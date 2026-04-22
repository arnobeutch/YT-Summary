"""Environment-based settings. Stdlib only — no pydantic dep.

Extend ``Settings`` with your own fields and map them in ``from_env``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Populate os.environ from a .env file (KEY=value per line). Existing vars win."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            continue
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    """Typed snapshot of process-wide settings."""

    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> Settings:
        """Load .env (if present) then read settings from os.environ."""
        _load_dotenv()
        return cls(log_level=os.environ.get("LOG_LEVEL", "INFO"))
