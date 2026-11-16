"""Red/Blue Stress Test 3: 320x240 JPEG Compression & <2.5s Cloud Agent Watchdog Brake."""

from __future__ import annotations

import time
from fastapi.testclient import TestClient


def test_jpeg_compression_size_and_mock_benchmark(client: TestClient) -> None:
    res = client.post("/api/agent/benchmark", json={"prompt": "Where is the red mug?", "mode": "mock"})
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    # Verify 320x240 compressed JPEG stays compact (~10-28 KB)
    assert 8.0 <= body["jpeg_kb"] <= 30.0
    assert body["direction"] in ("LEFT", "STRAIGHT", "RIGHT", "STOP")


def test_watchdog_stops_robot_under_2_5_seconds_on_5s_hang(client: TestClient) -> None:
    # Speed up look interval so cruise background thread fires promptly, then inject 5.0s hang
    client.post("/api/robot/reset", json={"scenario": "far_search"})
    client.post("/api/config", json={"agent_mode": "mock", "agent_interval_s": 0.8})

    # Inject 5.0s agent hang before mission starts
    client.post("/api/faults", json={"simulate_agent_delay_s": 5.0, "simulate_agent_error": False})

    t0 = time.perf_counter()
    r = client.post("/api/bci/command", json={"option_id": 1, "source": "watchdog_5s_test"})
    assert r.status_code == 200

    # Meanwhile verify main API thread is NEVER blocked while agent hangs!
    t_ping = time.perf_counter()
    ping_res = client.get("/api/status")
    ping_ms = (time.perf_counter() - t_ping) * 1000.0
    assert ping_res.status_code == 200
    assert ping_ms < 250.0, f"Main thread blocked for {ping_ms:.1f}ms!"

    # Wait until FSM aborts via watchdog — MUST complete strictly < 2.5s!
    while (time.perf_counter() - t0) < 2.48:
        st = client.get("/api/status").json()
        if st["fsm"]["state"] == "IDLE":
            break
        time.sleep(0.04)

    elapsed = time.perf_counter() - t0
    st = client.get("/api/status").json()
    assert elapsed < 2.50, f"Watchdog took {elapsed:.3f}s (>= 2.5s limit)!"
    assert st["fsm"]["state"] == "IDLE"
    assert st["agent"]["watchdog_trips"] >= 1
    assert "watchdog_abort" in st["fsm"]["last_abort_reason"]
    assert st["gatekeeper"]["velocity"]["vx"] == 0.0

    # Also test immediate cloud exception brake
    client.post("/api/faults", json={"simulate_agent_delay_s": 0.0, "simulate_agent_error": True})
    t1 = time.perf_counter()
    client.post("/api/bci/command", json={"option_id": 2, "source": "watchdog_err_test"})
    while (time.perf_counter() - t1) < 1.5:
        if client.get("/api/status").json()["fsm"]["state"] == "IDLE":
            break
        time.sleep(0.03)

    st2 = client.get("/api/status").json()
    assert st2["fsm"]["state"] == "IDLE"
    assert "watchdog_abort" in st2["fsm"]["last_abort_reason"]
    assert st2["gatekeeper"]["velocity"]["vx"] == 0.0

    client.post("/api/faults", json={"simulate_agent_delay_s": 0.0, "simulate_agent_error": False})
