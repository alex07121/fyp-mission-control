"""Pytest fixtures with isolated temporary DB and ephemeral UDP ports."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app import create_app  # noqa: E402


@pytest.fixture()
def client(tmp_path: Path) -> Generator[TestClient, None, None]:
    db_file = tmp_path / "test_logs.sqlite3"
    app = create_app(
        robot_mode="mock",
        udp_port=0,  # OS assigns free UDP port cleanly
        db_path=db_file,
        default_agent_mode="mock",
    )
    with TestClient(app) as tc:
        yield tc
