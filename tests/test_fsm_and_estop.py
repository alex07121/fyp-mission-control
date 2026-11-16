"""Red/Blue Stress Test 4: Full FSM Pipeline, VLN Early-Stop Rejection, 12GB VRAM, & Zero-Race E-Stop."""

from __future__ import annotations

import math
import threading
import time
from fastapi.testclient import TestClient

from adapters.mock_robot import MockRobotAdapter


def test_full_closed_loop_80cm_and_20cm_with_vln_rejection_and_vram_guard(client: TestClient) -> None:
    # 1) Run 80cm mode closed-loop mission from default start (~2.0m)
    client.post("/api/robot/reset", json={"scenario": "default"})
    client.post("/api/config", json={"lens_mode": "80cm", "blind_walk_enabled": False, "cruise_speed_mps": 0.30})
    r1 = client.post("/api/bci/command", json={"option_id": 1, "source": "e2e_80cm"})
    assert r1.status_code == 200

    deadline = time.monotonic() + 18.0
    while time.monotonic() < deadline:
        st = client.get("/api/status").json()
        if st["fsm"]["state"] == "IDLE":
            break
        assert st["vram"]["allocated_gb"] < 12.0
        time.sleep(0.06)

    st1 = client.get("/api/status").json()
    assert st1["fsm"]["state"] == "IDLE"
    assert st1["vram"]["vram_violation"] is False
    assert st1["vram"]["peak_gb"] < 12.0
    runs = st1["logs_summary"]["recent_runs"]
    assert len(runs) >= 1
    run80 = runs[0]
    assert run80["success"] is True
    assert run80["vln_early_stop_rejections"] >= 1, "Expected Light-VLN premature arrived=True to be rejected!"
    assert 0.78 <= run80["final_stop_distance_m"] <= 0.86

    # 2) Run 20cm upgraded lens mode -> must stop cleanly around <= 0.45m!
    client.post("/api/robot/reset", json={"scenario": "default"})
    client.post("/api/config", json={"lens_mode": "20cm", "cruise_speed_mps": 0.32, "vla_model": "pi0.5"})
    r2 = client.post("/api/bci/command", json={"option_id": 2, "source": "e2e_20cm"})
    assert r2.status_code == 200

    deadline2 = time.monotonic() + 18.0
    while time.monotonic() < deadline2:
        st = client.get("/api/status").json()
        if st["fsm"]["state"] == "IDLE":
            break
        assert st["vram"]["allocated_gb"] < 12.0
        time.sleep(0.06)

    st2 = client.get("/api/status").json()
    run20 = st2["logs_summary"]["recent_runs"][0]
    assert run20["success"] is True
    assert 0.36 <= run20["final_stop_distance_m"] <= 0.46
    assert st2["vram"]["peak_gb"] < 12.0


def test_estop_zero_race_condition_against_rogue_background_threads() -> None:
    """
    Spawn 16 rogue background threads hammering adapter.move(0.4, 0.1, 1.0) continuously,
    fire trigger_estop(), and verify velocity stays strictly 0.0 and robot pose NEVER drifts!
    """
    sim = MockRobotAdapter()
    stop_rogues = threading.Event()

    def _rogue_mover() -> None:
        while not stop_rogues.is_set():
            sim.move(0.40, 0.10, 1.20)
            time.sleep(0.0005)

    threads = [threading.Thread(target=_rogue_mover, daemon=True) for _ in range(16)]
    for t in threads:
        t.start()

    try:
        time.sleep(0.06)  # Let robot move briefly before E-Stop
        sim.trigger_estop("Red-Team Race Condition Torture")
        pose_at_estop = sim.get_pose()

        # Keep all 16 rogue threads hammering move() for 250ms AFTER E-Stop is latched!
        for _ in range(25):
            vel = sim.get_velocity()
            assert vel.vx == 0.0 and vel.vy == 0.0 and vel.wz == 0.0
            time.sleep(0.01)

        pose_after = sim.get_pose()
        drift = math.hypot(pose_after.x - pose_at_estop.x, pose_after.y - pose_at_estop.y)
        assert drift == 0.0, f"Robot drifted {drift}m while E-Stop was latched!"
        assert sim.get_gatekeeper_status()["blocked_move_attempts"] > 50
    finally:
        stop_rogues.set()
        for t in threads:
            t.join(timeout=0.3)
        sim.shutdown()
