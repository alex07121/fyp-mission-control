"""Red/Blue Stress Test 5: Real Process Lifecycle, Exporters (CSV/PNG/SVG), & Cross-Platform Audit."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _find_free_port(sock_type: int) -> int:
    with socket.socket(socket.AF_INET, sock_type) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_data_exporters_csv_png_svg_and_frontend_assets(client: TestClient) -> None:
    # Verify frontend HTML, CSS, JS, and live JPEG camera snapshot
    assert client.get("/").status_code == 200
    assert client.get("/static/styles.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    snap = client.get("/api/camera/snapshot.jpg")
    assert snap.status_code == 200 and snap.content[:2] == b"\xff\xd8"

    # Trigger quick E-stop / poison run so DB has records, then export CSV, PNG, SVG
    client.post("/api/faults", json={"depth_fault": "all_zero"})
    client.post("/api/bci/command", json={"option_id": 1})
    time.sleep(0.3)
    client.post("/api/faults", json={"depth_fault": "none"})

    csv_res = client.get("/api/logs/export/csv")
    assert csv_res.status_code == 200
    assert "created_at,command,source" in csv_res.text

    png_res = client.get("/api/logs/export/png")
    assert png_res.status_code == 200
    assert png_res.content[:8] == b"\x89PNG\r\n\x1a\n"

    svg_res = client.get("/api/logs/export/svg")
    assert svg_res.status_code == 200
    assert "<svg" in svg_res.text


def test_real_subprocess_startup_udp_and_clean_port_release(tmp_path: Path) -> None:
    http_port = _find_free_port(socket.SOCK_STREAM)
    udp_port = _find_free_port(socket.SOCK_DGRAM)
    env = os.environ.copy()
    env.update(
        {
            "HOST": "127.0.0.1",
            "PORT": str(http_port),
            "UDP_PORT": str(udp_port),
            "ROBOT_MODE": "mock",
            "AGENT_MODE": "mock",
            "FYP_DATA_DIR": str(tmp_path),
        }
    )

    proc = subprocess.Popen(
        [sys.executable, str(PROJECT_ROOT / "run.py")],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    try:
        base_url = f"http://127.0.0.1:{http_port}"
        ready = False
        for _ in range(40):
            try:
                r = httpx.get(f"{base_url}/api/health", timeout=0.4)
                if r.status_code == 200:
                    ready = True
                    break
            except Exception:
                time.sleep(0.1)
        assert ready, "run.py server failed to become ready in time"

        # Verify UDP packet on live subprocess
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as usock:
            usock.settimeout(1.0)
            usock.sendto(b'{"option_id": 5}', ("127.0.0.1", udp_port))
            ack, _ = usock.recvfrom(1024)
            assert b'"ok": true' in ack.lower()
    finally:
        t_stop = time.perf_counter()
        proc.send_signal(signal.SIGTERM)
        exit_code = proc.wait(timeout=3.0)
        stop_elapsed = time.perf_counter() - t_stop
        assert exit_code is not None
        assert stop_elapsed < 2.5

    # Verify both TCP & UDP ports are immediately released and re-bindable!
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as ts:
        ts.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        ts.bind(("127.0.0.1", http_port))
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as us:
        us.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        us.bind(("127.0.0.1", udp_port))


def test_cross_platform_no_hardcoded_user_paths_or_os_commands() -> None:
    this_file = Path(__file__).resolve()
    forbidden_tokens = [
        "/" + "Users" + "/",
        "si" + "ps ",
        "pb" + "copy",
        "osa" + "script",
    ]
    for path in PROJECT_ROOT.rglob("*"):
        if ".venv" in path.parts or "__pycache__" in path.parts or ".pytest_cache" in path.parts:
            continue
        if path.resolve() == this_file:
            continue
        if path.is_file() and path.suffix in {".py", ".html", ".js", ".css", ".md"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for tok in forbidden_tokens:
                assert tok not in text, f"Forbidden platform-specific token {tok!r} found in {path.name}"
