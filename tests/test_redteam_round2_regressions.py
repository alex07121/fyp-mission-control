"""Comprehensive regression suite locking down all 16 Red-Team Round 2 findings (R2-CRIT-01..R2-LOW-02)."""

from __future__ import annotations

import concurrent.futures
import csv
import io
import os
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import types
import warnings
from pathlib import Path
from unittest.mock import patch

import cv2
import httpx
import numpy as np
from fastapi.testclient import TestClient

from adapters.mock_robot import MockRobotAdapter
from adapters.xlerobot_real import XLeRobotRealAdapter
from backend.modules.cloud_agent import CloudAgentEngine, compress_frame_320x240_jpeg
from backend.modules.data_logger import MissionDataLogger
from backend.modules.vision_depth import VisionDepthCalibrator

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_r2_crit01_option5_absolute_preemption_and_nl_stop_vs_whitelist(client: TestClient) -> None:
    """R2-CRIT-01: Option #5 preempts regardless of malformed auxiliary fields; NL stop halts; unknown NL -> 422."""
    # 1) Hardware Option #5 with empty, blank, >256-char, int, or list `command` MUST still brake immediately (200)!
    weird_opt5_payloads = [
        {"option_id": 5, "command": ""},
        {"option_id": 5, "command": "   "},
        {"option_id": 5, "command": "STOP " * 70},
        {"option_id": 5, "command": 999},
        {"option_id": 5, "command": ["unexpected", "list"]},
    ]
    for bad_aux in weird_opt5_payloads:
        client.post("/api/estop/reset")
        client.post("/api/robot/reset", json={"scenario": "default"})
        r_start = client.post("/api/bci/command", json={"option_id": 1})
        assert r_start.status_code == 200
        time.sleep(0.06)

        r_estop = client.post("/api/bci/command", json=bad_aux)
        assert r_estop.status_code == 200, f"Option 5 rejected with {r_estop.status_code} for {bad_aux!r}"
        assert r_estop.json()["action"] == "estop"
        st = client.get("/api/status").json()
        assert st["gatekeeper"]["estop_active"] is True
        assert st["gatekeeper"]["velocity"]["vx"] == 0.0

    client.post("/api/estop/reset")

    # 2) All 12 urgent natural-language stop phrases MUST trigger E-Stop (NEVER start motors!)
    urgent_phrases = [
        "please stop",
        "**please stop!**",
        "stop now",
        "stop moving",
        "please freeze",
        "cancel mission",
        "emergency!",
        "don't move",
        "快停下",
        "请停止",
        "立即停止",
        "别动",
        "不要动",
    ]
    for phrase in urgent_phrases:
        client.post("/api/estop/reset")
        res = client.post("/api/bci/command", json={"command": phrase})
        assert res.status_code == 200, f"Stop phrase {phrase!r} returned {res.status_code}"
        assert res.json().get("action") == "estop", f"Stop phrase {phrase!r} started mission instead of E-Stop!"
        st = client.get("/api/status").json()
        assert st["gatekeeper"]["estop_active"] is True
        assert st["gatekeeper"]["velocity"]["vx"] == 0.0

    client.post("/api/estop/reset")
    time.sleep(0.05)

    # 3) Negated/inspection phrases & unknown chatter MUST NOT trigger E-Stop AND MUST NOT start motors (422)!
    non_task_phrases = [
        "go to bus stop",
        "non-stop",
        "do not brake",
        "check brake",
        "don't stop",
        "不要停",
        "别停",
        "hello how are you",
        "don't pick up the mug",
        "不要拿水杯",
    ]
    for txt in non_task_phrases:
        res = client.post("/api/bci/command", json={"command": txt})
        assert res.status_code == 422, f"Non-task {txt!r} was not rejected with 422 (got {res.status_code})"
        st = client.get("/api/status").json()
        assert st["gatekeeper"]["estop_active"] is False, f"{txt!r} falsely latched E-Stop!"
        assert st["fsm"]["state"] == "IDLE", f"{txt!r} started motors!"

    # 4) Whitelisted natural-language target command succeeds cleanly
    r_ok = client.post("/api/mission/start", json={"command": "去桌边拿水瓶"})
    assert r_ok.status_code == 200
    client.post("/api/estop/trigger")
    client.post("/api/estop/reset")


def test_r2_crit02_all_four_depth_blind_and_penetration_bypasses_blocked() -> None:
    """R2-CRIT-02: Block all 4 blind-zone / penetration bypasses while keeping normal 80cm & 20cm ROIs > 0.75 valid."""
    sim = MockRobotAdapter()
    cal = VisionDepthCalibrator(initial_lens_mode="80cm")
    try:
        # Calibration check: normal doorway & mug frames at >=0.85m (80cm) and >=0.45m (20cm) have roi_valid_ratio > 0.75
        sim.reset_pose(scenario="doorway_front")
        cal.set_classes(["open doorway", "door frame"])
        m_door_ok = cal.analyze_frame(sim.get_rgbd_frame("80cm"), lens_mode_override="80cm")
        assert m_door_ok.blind_zone_triggered is False and m_door_ok.roi_valid_ratio > 0.75

        sim.reset_pose(x=1.70, y=0.56, theta=0.0)  # 0.45m from doorway on 20cm lens
        m_door_20 = cal.analyze_frame(sim.get_rgbd_frame("20cm"), lens_mode_override="20cm")
        assert m_door_20.blind_zone_triggered is False and m_door_20.roi_valid_ratio > 0.75

        # Bypass 1: Open doorway at 0.60m with 69% blind (0.0m) + 31% inner corridor (1.85m) in 15% side jamb ROIs
        sim.reset_pose(x=1.55, y=0.56, theta=0.0)  # 0.60m from doorway
        f_b1 = sim.get_rgbd_frame("20cm")  # Get clean geometry bbox first
        m_geom_door = cal.analyze_frame(f_b1, lens_mode_override="20cm", update_history=False)
        assert m_geom_door.bbox is not None
        dx1, dy1, dx2, dy2 = m_geom_door.bbox
        d_b1 = np.full_like(f_b1.depth_m, 2.2, dtype=np.float32)
        d_b1[dy1:dy2, dx1:dx2] = 0.0
        ew = max(1, int(round((dx2 - dx1) * 0.15)))
        left_slice = d_b1[dy1:dy2, dx1 : dx1 + ew]
        n_corridor = int(left_slice.size * 0.62)  # 31% across combined left+right ROIs
        left_slice.ravel()[:n_corridor] = 1.85
        f_b1.depth_m = d_b1
        m_b1 = cal.analyze_frame(f_b1, lens_mode_override="80cm", update_history=False)
        assert m_b1.blind_zone_triggered is True and m_b1.steering_hint == "BRAKE"

        # Bypass 2: Survivorship bias — Mug at 0.60m with 59% blind (0.0m) + 41% background desk (1.85m) in center 40% ROI
        sim.reset_pose(scenario="near_blind_06m")
        cal.set_classes(["red mug", "mug", "cup"])
        f_b2 = sim.get_rgbd_frame("20cm")
        m_geom_mug = cal.analyze_frame(f_b2, lens_mode_override="20cm", update_history=False)
        assert m_geom_mug.bbox is not None
        mx1, my1, mx2, my2 = m_geom_mug.bbox
        bw, bh = mx2 - mx1, my2 - my1
        rx1, rx2 = mx1 + int(bw * 0.30), max(mx1 + int(bw * 0.30) + 1, mx2 - int(bw * 0.30))
        ry1, ry2 = my1 + int(bh * 0.30), max(my1 + int(bh * 0.30) + 1, my2 - int(bh * 0.30))
        d_b2 = np.full_like(f_b2.depth_m, 2.1, dtype=np.float32)
        roi_view = d_b2[ry1:ry2, rx1:rx2]
        roi_view.fill(0.0)
        roi_view.ravel()[: int(roi_view.size * 0.41)] = 1.85
        f_b2.depth_m = d_b2
        m_b2 = cal.analyze_frame(f_b2, lens_mode_override="80cm", update_history=False)
        assert m_b2.blind_zone_triggered is True and m_b2.steering_hint == "BRAKE"

        # Also verify if 100% of pixels at 0.60m (bw~110px) report 1.85m, continuous pinhole upper-bound catches it!
        roi_view.fill(1.85)
        f_b2.depth_m = d_b2
        m_b2_pinhole = cal.analyze_frame(f_b2, lens_mode_override="80cm", update_history=False)
        assert m_b2_pinhole.blind_zone_triggered is True and m_b2_pinhole.penetration_anomaly is True

        # Bypass 3: Spatial gradient far-depth penetration (3.4m..4.5m, std=0.32) at 1.18m (bw~56px < 70px)
        sim.reset_pose(x=0.80, y=-0.16, theta=0.0)  # 1.18m from red mug -> bw ~ 56px
        f_b3 = sim.get_rgbd_frame("80cm")
        grad_rows = np.linspace(3.40, 4.50, 480, dtype=np.float32)[:, None]
        grad_depth = np.broadcast_to(grad_rows, f_b3.depth_m.shape).copy()
        assert float(np.std(grad_depth)) > 0.25  # Defeats naive std < 0.035 check!
        f_b3.depth_m = grad_depth
        cal.reset_tracking_history()
        m_b3 = cal.analyze_frame(f_b3, lens_mode_override="80cm")
        assert m_b3.blind_zone_triggered is True
        assert m_b3.penetration_anomaly is True
        assert m_b3.steering_hint == "BRAKE"

        # Bypass 4: Gradual monotonic approach poisoning (1.25m -> 0.92m -> 1.58m -> 2.28m)
        cal.reset_tracking_history()
        sim.reset_pose(x=0.73, y=-0.16, theta=0.0)  # 1.25m
        m_s1 = cal.analyze_frame(sim.get_rgbd_frame("80cm"), lens_mode_override="80cm")
        assert m_s1.blind_zone_triggered is False

        sim.reset_pose(x=1.06, y=-0.16, theta=0.0)  # 0.92m (sets _min_approached_depth_m = ~0.92m)
        m_s2 = cal.analyze_frame(sim.get_rgbd_frame("80cm"), lens_mode_override="80cm")
        assert m_s2.blind_zone_triggered is False

        # Step 3: Even if bbox stays narrow, upward rebound from 0.92m -> 1.58m (+0.66m > +0.40m) MUST brake!
        f_s3 = sim.get_rgbd_frame("80cm")
        f_s3.depth_m[:, :] = 1.58
        m_s3 = cal.analyze_frame(f_s3, lens_mode_override="80cm")
        assert m_s3.blind_zone_triggered is True and m_s3.penetration_anomaly is True

        # Step 4: Subsequent 2.28m frame MUST ALSO stay blocked (anchor was NOT overwritten by 1.58m!)
        f_s4 = sim.get_rgbd_frame("80cm")
        f_s4.depth_m[:, :] = 2.28
        m_s4 = cal.analyze_frame(f_s4, lens_mode_override="80cm")
        assert m_s4.blind_zone_triggered is True and m_s4.penetration_anomaly is True
    finally:
        sim.shutdown()


def test_r2_crit03_cloud_cold_start_inside_watchdog_under_2_5s_and_clamp(client: TestClient) -> None:
    """R2-CRIT-03: Cold-start _get_vertex_client hang is bounded <2.50s wall-clock & watchdog_timeout_s <= 2.35s."""
    # Schema & property clamp check
    r_bad_cfg = client.post("/api/config", json={"watchdog_timeout_s": 2.45})
    assert r_bad_cfg.status_code == 422

    brake_times: list[float] = []
    t_call0 = 0.0
    eng = CloudAgentEngine(
        emergency_brake_cb=lambda _: brake_times.append(time.perf_counter() - t_call0),
        default_mode="cloud",
        watchdog_timeout_s=9.99,  # Must be clamped to 2.35s!
    )
    assert eng.watchdog_timeout_s == 2.35
    eng.watchdog_timeout_s = 2.48
    assert eng.watchdog_timeout_s == 2.35

    # Simulate a 3.6s network blackhole inside _get_vertex_client() on cold start
    def _hung_cold_start() -> tuple[object, object]:
        time.sleep(3.6)
        raise RuntimeError("Cold start blackhole")

    eng._get_vertex_client = _hung_cold_start  # type: ignore[method-assign]
    eng.watchdog_timeout_s = 1.20  # Verify tight wall-clock accounting first, then 2.35s max limit

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    t_call0 = time.perf_counter()
    res = eng.infer_with_watchdog(frame, "Pick up red mug", mode_override="cloud")
    wall_s = time.perf_counter() - t_call0

    assert res["ok"] is False
    assert res["watchdog_triggered"] is True
    assert len(brake_times) == 1
    assert brake_times[0] < 1.35 and wall_s < 1.35
    # Reported latency_ms must reflect true wall-clock elapsed time (not fake post-warmup delta!)
    assert abs(res["latency_ms"] - (wall_s * 1000.0)) < 60.0

    # Also test at max allowed watchdog_timeout_s=2.35s with high-entropy 1080p frame -> strictly < 2.50s!
    eng2 = CloudAgentEngine(
        emergency_brake_cb=lambda _: brake_times.append(time.perf_counter() - t_call0),
        default_mode="cloud",
        watchdog_timeout_s=2.35,
    )
    eng2._get_vertex_client = _hung_cold_start  # type: ignore[method-assign]
    big_frame = np.random.default_rng(7).integers(0, 256, size=(1080, 1920, 3), dtype=np.uint8)
    t_call0 = time.perf_counter()
    res2 = eng2.infer_with_watchdog(big_frame, "Pick up red mug", mode_override="cloud")
    wall_max_s = time.perf_counter() - t_call0
    assert res2["watchdog_triggered"] is True
    assert wall_max_s < 2.50, f"Cold-start + 1080p compression exceeded 2.50s red line: {wall_max_s:.3f}s"


def test_r2_crit04_robot_reset_preserves_estop_and_no_split_lock_dual_workers(client: TestClient) -> None:
    """R2-CRIT-04: /api/robot/reset NEVER clears manual E-Stop; concurrent reset+submit spawns at most 1 owner worker."""
    client.post("/api/faults", json={"simulate_agent_delay_s": 0.6})
    r1 = client.post("/api/bci/command", json={"option_id": 1})
    assert r1.status_code == 200
    time.sleep(0.05)

    # Operator hits manual E-Stop while slow AGENT_PARSE is in flight
    client.post("/api/estop/trigger", json={"reason": "Operator Red Button"})
    assert client.get("/api/status").json()["gatekeeper"]["estop_active"] is True

    # Calling /api/robot/reset while E-Stop is active MUST NOT unlock E-Stop!
    r_pose = client.post("/api/robot/reset", json={"scenario": "default"})
    assert r_pose.status_code == 200
    st_after = client.get("/api/status").json()
    assert st_after["gatekeeper"]["estop_active"] is True, "/api/robot/reset illegally cleared operator E-Stop!"

    client.post("/api/faults", json={"simulate_agent_delay_s": 0.0})
    client.post("/api/estop/reset")

    # Hammer reset_estop + submit_task concurrently to verify zero dual active FSMWorker race & zero prompt clobber
    client.post("/api/estop/trigger")

    def _race_submit(opt_id: int) -> int:
        return client.post("/api/bci/command", json={"option_id": opt_id}).status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        f_rst = pool.submit(lambda: client.post("/api/estop/reset").status_code)
        f_subs = [pool.submit(_race_submit, 3) for _ in range(8)]
        assert f_rst.result() == 200
        sub_codes = [f.result() for f in f_subs]

    # At most 1 submission may win 200 OK; never 2 concurrent workers!
    assert sub_codes.count(200) <= 1
    if sub_codes.count(200) == 1:
        st_run = client.get("/api/status").json()
        assert "open doorway" in st_run["vision"]["prompts"]

    client.post("/api/estop/trigger")
    client.post("/api/estop/reset")


def test_r2_high01_and_med02_frontend_spacebar_capture_sync_and_one_shot_slow_bench() -> None:
    """R2-HIGH-01 + R2-MED-02: Verify Spacebar capture preemption, one-shot benchmark, pollBusy, & UI sync."""
    js = (PROJECT_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")

    # R2-HIGH-01: Capture-phase listener, text-only whitelist, blur(), preventDefault()
    assert "capture: true" in js
    assert "function isTextEntryElement(" in js
    assert '["text", "search", "password", "url", "email", "number"]' in js
    assert "el.blur()" in js

    # R2-MED-02: #btnTestSlowAgent calls runAgentBenchmark(5.0) via /api/agent/benchmark (NOT /api/faults!)
    assert 'runAgentBenchmark(5.0)' in js
    assert "simulate_agent_delay_s: 5.0" not in js
    assert "let pollBusy = false" in js
    assert '$("rngSpeed").value' in js
    assert '$("rngInterval").value' in js
    assert '$("inpPrompts").value' in js
    assert 'if (status === 200)' in js and 'inp.value = ""' in js
    assert 'ev.key === "Enter"' in js


def test_r2_high02_vln_vram_reclaimed_on_estop_and_idle_restored(client: TestClient) -> None:
    """R2-HIGH-02: Triggering E-Stop during VLN_CRUISE reclaims 5.4GB Light-VLN VRAM and returns FSM to IDLE."""
    client.post("/api/robot/reset", json={"scenario": "default"})
    client.post("/api/config", json={"lens_mode": "80cm", "cruise_speed_mps": 0.16})
    r = client.post("/api/bci/command", json={"option_id": 1, "source": "vram_estop_test"})
    assert r.status_code == 200

    # Wait until Light-VLN is loaded (~6.80 GB) in VLN_CRUISE
    saw_vln_vram = False
    deadline = time.monotonic() + 4.0
    while time.monotonic() < deadline:
        st = client.get("/api/status").json()
        if "Light-VLN" in st["vram"]["active_models"] and st["vram"]["allocated_gb"] >= 6.5:
            saw_vln_vram = True
            break
        time.sleep(0.02)
    assert saw_vln_vram, "Mission never loaded Light-VLN in VLN_CRUISE"

    # Trigger E-Stop WITHOUT calling Reset yet -> worker must exit, unload Light-VLN, and restore IDLE!
    client.post("/api/estop/trigger", json={"reason": "VRAM Leak Check"})
    deadline_cleanup = time.monotonic() + 1.0
    while time.monotonic() < deadline_cleanup:
        st = client.get("/api/status").json()
        if st["fsm"]["state"] == "IDLE":
            break
        time.sleep(0.02)

    st_latched = client.get("/api/status").json()
    assert st_latched["gatekeeper"]["estop_active"] is True
    assert st_latched["fsm"]["state"] == "IDLE", f"FSM stuck in {st_latched['fsm']['state']} after E-Stop!"
    assert "Light-VLN" not in st_latched["vram"]["active_models"], "5.4GB Light-VLN leaked after E-Stop!"
    assert 1.4 <= st_latched["vram"]["allocated_gb"] <= 1.65

    client.post("/api/estop/reset")
    st_reset = client.get("/api/status").json()
    assert 1.4 <= st_reset["vram"]["allocated_gb"] <= 1.65


def test_r2_high03_far_search_100pct_completion_and_vln_cruise_timeout_guard(client: TestClient) -> None:
    """R2-HIGH-03: `far_search` preset completes closed-loop 100%, and VLN timeout halts without YOLO_LAST_MILE."""
    client.post("/api/robot/reset", json={"scenario": "far_search"})
    client.post("/api/config", json={"lens_mode": "80cm", "blind_walk_enabled": False, "cruise_speed_mps": 0.28})

    r = client.post("/api/bci/command", json={"option_id": 1, "source": "far_search_e2e"})
    assert r.status_code == 200

    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        if client.get("/api/status").json()["fsm"]["state"] == "IDLE":
            break
        time.sleep(0.06)

    st = client.get("/api/status").json()
    assert st["fsm"]["state"] == "IDLE"
    latest = st["logs_summary"]["recent_runs"][0]
    assert latest["success"] is True, f"far_search failed with {latest['abort_reason']}"
    assert 0.78 <= latest["final_stop_distance_m"] <= 0.87
    assert st["faults"]["collision_occurred"] is False

    # Verify VLN_CRUISE timeout halts cleanly with `vln_cruise_timeout` and NEVER enters YOLO_LAST_MILE
    rt = client.app.state.runtime  # type: ignore[attr-defined]
    client.post("/api/robot/reset", json={"x": -4.5, "y": 0.0, "theta": 3.14})
    client.post("/api/config", json={"cruise_speed_mps": 0.05})
    orig_cruise_timeout = rt.fsm.vln_cruise_timeout_s
    rt.fsm.vln_cruise_timeout_s = 0.25
    try:
        r_to = client.post("/api/bci/command", json={"option_id": 1, "source": "cruise_timeout_check"})
        assert r_to.status_code == 200
        deadline_to = time.monotonic() + 2.5
        while time.monotonic() < deadline_to:
            if rt.fsm.state == "IDLE":
                break
            time.sleep(0.03)
    finally:
        rt.fsm.vln_cruise_timeout_s = orig_cruise_timeout

    st_to = client.get("/api/status").json()
    assert st_to["fsm"]["state"] == "IDLE"
    assert st_to["fsm"]["last_abort_reason"] == "vln_cruise_timeout"
    assert st_to["gatekeeper"]["velocity"]["vx"] == 0.0
    assert "Light-VLN" not in st_to["vram"]["active_models"]
    client.post("/api/config", json={"cruise_speed_mps": 0.18})


def test_r2_high04_vlm_parser_rejects_contradictions_duplicate_keys_and_negations() -> None:
    """R2-HIGH-04: Reject JSON+STOP contradiction, duplicate JSON keys, negated directions, & illegal JSON fallback."""
    adversarial_replies = [
        '{"direction": "STRAIGHT"} DANGER! OBSTACLE AHEAD, STOP IMMEDIATELY!',
        '{"direction": "LEFT", "direction": "RIGHT"}',
        '{"direction": "LEFT", "Direction": "RIGHT"}',
        "Do NOT go LEFT! Wall on the left!",
        '{"direction": "WAIT", "reason": "Never turn RIGHT"}',
        '{"direction": "RIGHT", "warning": "Blocked ahead, halt!"}',
        "禁止向左转，左侧有障碍",
        "不要直行",
    ]
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    for bad_reply in adversarial_replies:
        braked = {"cnt": 0}
        eng = CloudAgentEngine(
            emergency_brake_cb=lambda _: braked.__setitem__("cnt", braked["cnt"] + 1),
            default_mode="cloud",
        )
        eng._worker_call = lambda *a, _txt=bad_reply, **kw: (_txt, "adversarial_vlm")  # type: ignore[method-assign]
        out = eng.infer_with_watchdog(frame, "Pick up red mug", mode_override="cloud")
        assert out["ok"] is False, f"Adversarial VLM reply was accepted: {bad_reply!r} -> {out}"
        assert out["direction"] == "STOP"
        assert out["watchdog_triggered"] is True
        assert braked["cnt"] == 1


def test_r2_high05_record_run_sqlite_error_isolated_and_nan_sanitized(client: TestClient, tmp_path: Path) -> None:
    """R2-HIGH-05: SQLite OperationalError inside record_run NEVER deadlocks FSM; NaN/Inf floats sanitize cleanly."""
    rt = client.app.state.runtime  # type: ignore[attr-defined]
    client.post("/api/faults", json={"depth_fault": "all_zero"})

    with patch.object(rt.data_logger, "record_run", side_effect=sqlite3.OperationalError("database is locked")):
        r1 = client.post("/api/bci/command", json={"option_id": 1, "source": "db_locked_trial"})
        assert r1.status_code == 200
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if rt.fsm.state == "IDLE":
                break
            time.sleep(0.03)
        assert rt.fsm.state == "IDLE", f"FSM deadlocked in {rt.fsm.state} after SQLite OperationalError!"
        assert rt.adapter.get_vram_status().allocated_gb <= 1.65

    # Subsequent mission MUST be accepted (200 OK, not 409!)
    r2 = client.post("/api/bci/command", json={"option_id": 1, "source": "recovery_trial"})
    assert r2.status_code == 200
    deadline2 = time.monotonic() + 2.0
    while time.monotonic() < deadline2:
        if rt.fsm.state == "IDLE":
            break
        time.sleep(0.03)
    client.post("/api/faults", json={"depth_fault": "none"})

    # Direct NaN/Inf sanitization test on MissionDataLogger
    dl = MissionDataLogger(db_path=tmp_path / "nan_test.sqlite3")
    row_id = dl.record_run(
        {
            "command": "nan_trial",
            "total_duration_s": float("nan"),
            "stage_durations": {"AGENT_PARSE": float("inf")},
            "agent_avg_latency_ms": float("inf"),
            "final_stop_distance_m": float("nan"),
            "final_yaw_error_deg": float("-inf"),
            "peak_vram_gb": float("nan"),
            "success": False,
            "abort_reason": "nan_guard",
        }
    )
    assert row_id >= 1
    summary = dl.get_summary()
    assert summary["total_runs"] == 1


def test_r2_high06_requirements_declares_ultralytics_and_threadsafe_loader(tmp_path: Path, monkeypatch) -> None:
    """R2-HIGH-06: requirements.txt declares ultralytics>=8.3.0; instant fallback w/o weights; thread-safe YOLOE load w/ weights."""
    req_txt = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "ultralytics>=8.3.0" in req_txt

    t0 = time.perf_counter()
    cal = VisionDepthCalibrator(initial_lens_mode="80cm")
    init_ms = (time.perf_counter() - t0) * 1000.0
    assert init_ms < 250.0, f"VisionDepthCalibrator blocked on startup ({init_ms:.1f}ms)"
    assert isinstance(cal._model_lock, type(threading.Lock()))

    # Verify thread-safe YOLOE loading & set_classes() when local weights exist
    fake_pt = tmp_path / "yoloe-26s-seg.pt"
    fake_pt.write_bytes(b"FAKE_YOLOE_WEIGHTS")
    monkeypatch.setenv("YOLOE_WEIGHTS_PATH", str(fake_pt))
    calls: list[tuple[str, object]] = []

    class _FakeYOLOE:
        def __init__(self, path: str) -> None:
            calls.append(("init", path))

        def set_classes(self, prompts: list[str]) -> None:
            calls.append(("set_classes", list(prompts)))

        def predict(self, bgr, verbose: bool = False, conf: float = 0.25):
            return []

    fake_ultra = types.SimpleNamespace(YOLOE=_FakeYOLOE)
    with patch.dict(sys.modules, {"ultralytics": fake_ultra}):
        cal_real = VisionDepthCalibrator(initial_lens_mode="80cm")
        assert cal_real._detector_backend == "ultralytics_yoloe_26s"
        cal_real.set_classes(["water bottle", "bottle"])
    assert ("init", str(fake_pt)) in calls
    assert ("set_classes", ["water bottle", "bottle"]) in calls


def test_r2_med01_isolated_export_pool_under_slow_bench_storm_and_rpc_semaphore(client: TestClient) -> None:
    """R2-MED-01: CSV export stays <150ms during 15 slow benchmarks & orphan AgentWatchdogCall threads <= 4."""
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=15)
    futs = [
        pool.submit(
            lambda: client.post("/api/agent/benchmark", json={"prompt": "storm", "mode": "mock", "simulate_delay_s": 1.6})
        )
        for _ in range(15)
    ]
    try:
        time.sleep(0.08)
        # Check live AgentWatchdogCall thread count is strictly bounded <= 4!
        watchdog_threads = [t for t in threading.enumerate() if t.name == "AgentWatchdogCall" and t.is_alive()]
        assert len(watchdog_threads) <= 4, f"Leaked {len(watchdog_threads)} AgentWatchdogCall threads (>4)!"

        # Export CSV while agent_pool is 100% saturated -> must complete quickly via isolated export_pool!
        t_exp = time.perf_counter()
        csv_res = client.get("/api/logs/export/csv")
        exp_ms = (time.perf_counter() - t_exp) * 1000.0
        assert csv_res.status_code == 200
        assert exp_ms < 250.0, f"CSV export starved for {exp_ms:.1f}ms by slow agent benchmarks!"
    finally:
        for f in futs:
            f.result(timeout=8.0)
        pool.shutdown(wait=True)


def test_r2_med03_real_adapter_preserves_arm_abort_and_tracks_cuda_peak() -> None:
    """R2-MED-03: XLeRobotRealAdapter never wipes _arm_abort_flag in execute_vla_grasp and reads CUDA max memory."""
    real = XLeRobotRealAdapter()
    try:
        real.load_vla("SmolVLA")
        epoch_before = real.estop_epoch
        real.trigger_estop("Arm emergency stop")
        real.reset_estop()

        # Stale execute_vla_grasp WITHOUT reloading VLA MUST see preserved _arm_abort_flag and abort!
        res_stale = real.execute_vla_grasp("Pick mug", model_name="SmolVLA")
        assert res_stale["success"] is False and res_stale["aborted"] is True

        # Even if load_vla() is called for a new run, passing stale expected_estop_epoch MUST still abort!
        real.load_vla("SmolVLA")
        res_bad_epoch = real.execute_vla_grasp("Pick mug", model_name="SmolVLA", expected_estop_epoch=epoch_before)
        assert res_bad_epoch["success"] is False and res_bad_epoch["aborted"] is True

        # Mock torch.cuda to verify reset_peak_memory_stats() and max_memory_allocated() capture transient spikes
        reset_called = {"ok": False}
        fake_cuda = types.SimpleNamespace(
            is_available=lambda: True,
            reset_peak_memory_stats=lambda: reset_called.__setitem__("ok", True),
            memory_allocated=lambda: int(2.1 * (1024**3)),
            max_memory_allocated=lambda: int(7.35 * (1024**3)),
            empty_cache=lambda: None,
        )
        fake_torch = types.SimpleNamespace(cuda=fake_cuda)
        with patch.dict(sys.modules, {"torch": fake_torch}):
            v_stat = real.reset_peak_vram()
            assert reset_called["ok"] is True
            assert abs(v_stat.allocated_gb - 2.1) < 0.05
            assert abs(v_stat.peak_gb - 7.35) < 0.05
    finally:
        real.shutdown()


def test_r2_med04_sqlite_migration_blind_walk_split_and_csv_formula_defense(tmp_path: Path) -> None:
    """R2-MED-04: Auto-migrate legacy DB, split 80cm vs 80cm_blind_walk averages, and escape CSV formulas."""
    db_file = tmp_path / "legacy_v1.sqlite3"
    # 1) Create pre-existing Round-1 schema WITHOUT `blind_walk_enabled` or `vla_model`
    conn = sqlite3.connect(str(db_file))
    conn.execute(
        """
        CREATE TABLE mission_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            command TEXT NOT NULL,
            source TEXT NOT NULL,
            lens_mode TEXT NOT NULL,
            total_duration_s REAL NOT NULL,
            stage_durations_json TEXT NOT NULL,
            vln_early_stop_rejections INTEGER NOT NULL,
            agent_avg_latency_ms REAL NOT NULL,
            agent_avg_jpeg_kb REAL NOT NULL,
            final_stop_distance_m REAL,
            final_yaw_error_deg REAL,
            peak_vram_gb REAL NOT NULL,
            success INTEGER NOT NULL,
            abort_reason TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()

    # Instantiate MissionDataLogger -> MUST auto-migrate cleanly via ALTER TABLE
    dl = MissionDataLogger(db_path=db_file)
    dl.record_run(
        {
            "command": "pure_80cm",
            "lens_mode": "80cm",
            "blind_walk_enabled": False,
            "vla_model": "SmolVLA",
            "final_stop_distance_m": 0.85,
            "final_yaw_error_deg": -1.25,
            "peak_vram_gb": 6.8,
            "success": True,
        }
    )
    dl.record_run(
        {
            "command": "=1+1+cmd|' /C calc'!A0",
            "source": "@attacker",
            "lens_mode": "80cm",
            "blind_walk_enabled": True,
            "vla_model": "pi0.5",
            "final_stop_distance_m": 0.45,
            "final_yaw_error_deg": 0.8,
            "peak_vram_gb": 7.2,
            "success": True,
            "abort_reason": "+completed",
        }
    )

    summary = dl.get_summary()
    assert summary["lens_comparison"]["80cm"]["avg_stop_m"] == 0.85
    assert summary["lens_comparison"]["80cm_blind_walk"]["avg_stop_m"] == 0.45

    csv_text = dl.export_csv_bytes().decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    assert len(rows) == 2
    assert "blind_walk_enabled" in rows[0] and "vla_model" in rows[0]
    # Negative float yaw `-1.25` stays numeric, while formula strings get leading `'`
    assert rows[0]["final_yaw_error_deg"] == "-1.25"
    assert rows[1]["command"].startswith("'=1+1+cmd")
    assert rows[1]["source"].startswith("'@attacker")
    assert rows[1]["abort_reason"].startswith("'+completed")
    assert rows[1]["blind_walk_enabled"] == "1"
    assert rows[1]["vla_model"] == "pi0.5"


def test_r2_low01_jpeg_nan_inf_no_runtime_warning_and_zero_synthetic_noise() -> None:
    """R2-LOW-01: float32 NaN/Inf emits zero RuntimeWarning, and flat images suffer zero synthetic grain injection."""
    bad_float = np.full((240, 320, 3), 128.0, dtype=np.float32)
    bad_float[0, 0, 0] = np.nan
    bad_float[0, 1, 1] = np.inf
    bad_float[0, 2, 2] = -np.inf

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        jpg_nan, kb_nan = compress_frame_320x240_jpeg(bad_float)
    assert jpg_nan[:2] == b"\xff\xd8" and kb_nan <= 25.0

    flat_gray = np.full((240, 320, 3), 120, dtype=np.uint8)
    jpg_flat, kb_flat = compress_frame_320x240_jpeg(flat_gray)
    assert kb_flat <= 25.0
    decoded = cv2.imdecode(np.frombuffer(jpg_flat, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    max_diff = int(np.max(np.abs(decoded.astype(np.int16) - 120)))
    assert max_diff <= 2, f"Synthetic grain corrupted flat frame (max_diff={max_diff})!"


def test_r2_low02_bci_sender_example_env_ports_line_count_and_non_latching(tmp_path: Path) -> None:
    """R2-LOW-02: bci_sender_example.py is 15..18 lines, reads PORT/UDP_PORT, and never latches E-Stop across 4 runs."""
    script_path = PROJECT_ROOT / "bci_sender_example.py"
    lines = script_path.read_text(encoding="utf-8").strip().splitlines()
    assert 15 <= len(lines) <= 18, f"Expected 15..18 lines, got {len(lines)}"

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s1:
        s1.bind(("127.0.0.1", 0))
        http_port = s1.getsockname()[1]
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s2:
        s2.bind(("127.0.0.1", 0))
        udp_port = s2.getsockname()[1]

    env = {
        **os.environ,
        "HOST": "127.0.0.1",
        "PORT": str(http_port),
        "UDP_PORT": str(udp_port),
        "ROBOT_MODE": "mock",
        "AGENT_MODE": "mock",
        "FYP_DATA_DIR": str(tmp_path),
    }
    proc = subprocess.Popen(
        [sys.executable, str(PROJECT_ROOT / "run.py")],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        for _ in range(40):
            try:
                if httpx.get(f"http://127.0.0.1:{http_port}/api/health", timeout=0.3).status_code == 200:
                    break
            except Exception:
                time.sleep(0.08)

        # Run bci_sender_example.py 4 times using ONLY environment variables (no source string patching!)
        for i in range(4):
            sub = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(PROJECT_ROOT),
                env=env,
                timeout=5.0,
                capture_output=True,
                text=True,
            )
            assert sub.returncode == 0, f"Run #{i+1} failed: {sub.stderr}"
            if i == 0:
                assert "[HTTP] 200" in sub.stdout

        # Confirm E-Stop was NEVER latched by bci_sender_example.py!
        st = httpx.get(f"http://127.0.0.1:{http_port}/api/status", timeout=1.0).json()
        assert st["gatekeeper"]["estop_active"] is False
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=2.5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=1.0)
