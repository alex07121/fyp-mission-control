"""Cross-platform one-click launcher for FYP Mission Control (macOS & Ubuntu Linux)."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from types import FrameType
from typing import Optional

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app import GLOBAL_SHUTDOWN_EVENT  # noqa: E402
from backend.config import DEFAULT_HOST, DEFAULT_PORT, DEFAULT_ROBOT_MODE, DEFAULT_UDP_PORT  # noqa: E402


class CleanShutdownServer(uvicorn.Server):
    """Signals active streaming responses immediately upon SIGINT/SIGTERM for <0.3s clean exit (rc=0)."""

    def handle_exit(self, sig: int, frame: Optional[FrameType]) -> None:
        GLOBAL_SHUTDOWN_EVENT.set()
        super().handle_exit(sig, frame)
        # Clear captured signal list after setting should_exit so Uvicorn finishes lifespan & exits cleanly with code 0
        if hasattr(self, "_captured_signals"):
            self._captured_signals.clear()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    host = os.getenv("HOST", DEFAULT_HOST)
    port = int(os.getenv("PORT", str(DEFAULT_PORT)))
    udp_port = int(os.getenv("UDP_PORT", str(DEFAULT_UDP_PORT)))
    robot_mode = os.getenv("ROBOT_MODE", DEFAULT_ROBOT_MODE)

    logging.info(
        "Starting FYP Mission Control on http://%s:%d (UDP BCI=%d, ROBOT_MODE=%s)",
        host,
        port,
        udp_port,
        robot_mode,
    )
    config = uvicorn.Config(
        "backend.app:create_app",
        factory=True,
        host=host,
        port=port,
        log_level="info",
        timeout_graceful_shutdown=1,
    )
    server = CleanShutdownServer(config)
    server.run()


if __name__ == "__main__":
    main()
