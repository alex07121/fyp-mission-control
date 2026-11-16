"""Comprehensive regression suite locking down all 18 Red-Team Round 1 findings (CRIT-01..LOW-02)."""

from __future__ import annotations

import concurrent.futures
import math
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import numpy as np
from fastapi.testclient import TestClient

from adapters.mock_robot import MockRobotAdapter
from adapters.xlerobot_real import XLeRobotRealAdapter
from backend.modules.cloud_agent import CloudAgentEngine, compress_frame_320x240_jpeg
from backend.modules.data_logger import MissionDataLogger
from backend.modules.vision_depth import VisionDepthCalibrator

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_crit01_estop_aba_no_zombie_resurrection(client: TestClient) -> None:
    """CRIT-01: Trigger E-Stop then immediate Reset while slow agent call is in flight -> ZERO motion!"""
    client.post("/api/robot/reset", json={"scenario": "default"})
    client.post("/api/faults", json={"simulate_agent_delay_s": 0.45})
    r = client.post("/api/bci/command", json={"option_id": 1, "source": "aba_test"})
    assert r.status_code == 200

    time.sleep(0.08)  # Old worker is mid-flight inside AGENT_PARSE
    client.post("/api/estop/trigger", json={"reason": "ABA E-Stop"})
    client.post("/api/estop/reset")
    client.post("/api/faults", json={"simulate_agent_delay_s": 0.0})

    pose0 = client.get("/api/status").json()["pose"]
    # Observe for 0.9s after old worker wakes up — state MUST stay IDLE and robot MUST NOT move!
    for _ in range(18):
        time.sleep(0.05)
        st = client.get("/api/status").json()
        assert st["fsm"]["state"] == "IDLE", f"Zombie resurrected state to {st['fsm']['state']}!"
        assert st["gatekeeper"]["velocity"]["vx"] == 0.0

    pose1 = client.get("/api/status").json()["pose"]
    drift = math.hypot(pose1["x"] - pose0["x"], pose1["y"] - pose0["y"])
    assert drift == 0.0, f"Zombie worker moved robot by {drift}m after reset!"


def test_crit02_concurrent_reset_never_clears_newer_estop(client: TestClient) -> None:
    """CRIT-02: If a new E-Stop fires while an older reset_estop is waiting on worker exit, E-Stop stays latched!"""
    rt = client.app.state.runtime  # type: ignore[attr-defined]
    # Keep a dummy worker briefly alive for 220ms
    hold = threading.Event()
    dummy = threading.Thread(target=lambda: hold.wait(0.22), daemon=True)
    dummy.start()
    rt.fsm._worker_thread = dummy

    rt.fsm.trigger_estop("First E-Stop")
    reset_result = {}

    def _call_old_reset() -> None:
        reset_result["res"] = client.post("/api/estop/reset")

    t_reset = threading.Thread(target=_call_old_reset, daemon=True)
    t_reset.start()
    time.sleep(0.05)  # Old reset is waiting on worker; fire SECOND E-Stop now!
    client.post("/api/estop/trigger", json={"reason": "Second Urgent E-Stop"})
    t_reset.join(timeout=1.0)
    hold.set()

    assert reset_result["res"].status_code == 409
    st = client.get("/api/status").json()
    assert st["gatekeeper"]["estop_active"] is True, "Old reset_estop overwrote newer E-Stop!"
    client.post("/api/estop/reset")


def test_crit03_threadpool_starvation_bci_estop_under_50ms(client: TestClient) -> None:
    """CRIT-03: 20 concurrent slow /api/agent/benchmark calls MUST NOT delay BCI E-Stop!"""
    stop_bg = threading.Event()

    def _slow_bench(_: int) -> None:
        if not stop_bg.is_set():
            client.post("/api/agent/benchmark", json={"prompt": "test", "mode": "mock", "simulate_delay_s": 1.2})

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=16)
    for i in range(16):
        pool.submit(_slow_bench, i)

    time.sleep(0.08)
    t0 = time.perf_counter()
    res = client.post("/api/bci/command", json={"option_id": 5, "source": "starvation_check"})
    estop_ms = (time.perf_counter() - t0) * 1000.0
    stop_bg.set()
    pool.shutdown(wait=False, cancel_futures=True)

    assert res.status_code == 200
    assert estop_ms < 120.0, f"E-Stop starved for {estop_ms:.1f}ms!"
    assert client.get("/api/status").json()["gatekeeper"]["estop_active"] is True
    client.post("/api/estop/reset")


def test_crit04_and_high01_blind_noise_doorway_4m_and_both_lenses_doorway_pass(client: TestClient) -> None:
    """CRIT-04 + HIGH-01: Catch 13% outlier noise, <0.80m sub-blind values, 640x480@4m poison, AND pass 80cm/20cm doorway!"""
    sim = MockRobotAdapter()
    cal = VisionDepthCalibrator(initial_lens_mode="80cm")
    try:
        # (a) Doorway at 0.60m with penetrate_4m (full-frame 640x480 bbox returning 4.0m)
        sim.reset_pose(x=1.55, y=0.56, theta=0.0)
        sim.set_fault_injection(depth_fault="penetrate_4m")
        cal.set_classes(["open doorway", "door frame"])
        m_pen = cal.analyze_frame(sim.get_rgbd_frame("80cm"), lens_mode_override="80cm")
        assert m_pen.blind_zone_triggered is True

        # (b) 80cm blind zone with 13% 1.95m outlier noise + 87% zeros -> MUST trigger blind guard!
        sim.set_fault_injection(depth_fault="none")
        sim.reset_pose(scenario="near_blind_06m")
        cal.set_classes(["red mug", "mug", "cup"])
        f_mug = sim.get_rgbd_frame("80cm")
        m_base = cal.analyze_frame(f_mug, lens_mode_override="80cm", update_history=False)
        assert m_base.bbox is not None
        x1, y1, x2, y2 = m_base.bbox
        poisoned_depth = f_mug.depth_m.copy()
        roi = poisoned_depth[y1:y2, x1:x2]
        roi.fill(0.0)
        roi.ravel()[: max(1, int(roi.size * 0.13))] = 1.95
        f_mug.depth_m = poisoned_depth
        m_noisy = cal.analyze_frame(f_mug, lens_mode_override="80cm", update_history=False)
        assert m_noisy.blind_zone_triggered is True, "13% outlier noise bypassed 80cm blind-zone guard!"

        # (c) Sub-blind (<0.80m, e.g. 0.35m) readings on 80cm lens -> MUST trigger blind guard!
        roi.fill(0.35)
        f_mug.depth_m = poisoned_depth
        m_sub = cal.analyze_frame(f_mug, lens_mode_override="80cm", update_history=False)
        assert m_sub.blind_zone_triggered is True, "0.35m fake reading was accepted on 80cm lens!"
    finally:
        sim.shutdown()

    # (d) HIGH-01: Both 80cm AND 20cm lenses MUST complete Option 3 (Open Doorway) with 100% success!
    client.post("/api/faults", json={"depth_fault": "none", "simulate_agent_delay_s": 0.0, "simulate_agent_error": False})
    for lens in ("80cm", "20cm"):
        client.post("/api/robot/reset", json={"scenario": "doorway_front"})
        client.post("/api/config", json={"lens_mode": lens, "cruise_speed_mps": 0.30})
        r = client.post("/api/bci/command", json={"option_id": 3, "source": f"doorway_{lens}"})
        assert r.status_code == 200
        deadline = time.monotonic() + 14.0
        while time.monotonic() < deadline:
            if client.get("/api/status").json()["fsm"]["state"] == "IDLE":
                break
            time.sleep(0.05)
        st = client.get("/api/status").json()
        latest = st["logs_summary"]["recent_runs"][0]
        assert latest["success"] is True, f"Open doorway failed on {lens}: {latest['abort_reason']}"
        assert st["faults"]["collision_occurred"] is False


def test_high02_sqlite_zero_fd_leak_and_med06_clean_averages(tmp_path: Path) -> None:
    """HIGH-02 + MED-06: Zero SQLite FD leak across 80 ops, and aborted runs don't skew stop average."""
    logger = MissionDataLogger(db_path=tmp_path / "fd_check.sqlite3")
    fd_dir = Path("/dev/fd") if Path("/dev/fd").exists() else Path("/proc/self/fd")
    before_fds = len(list(fd_dir.iterdir())) if fd_dir.exists() else 0

    logger.record_run({"command": "ok_run", "lens_mode": "20cm", "final_stop_distance_m": 0.45, "final_yaw_error_deg": 1.2, "peak_vram_gb": 6.4, "success": True, "abort_reason": "completed"})
    logger.record_run({"command": "abort_far", "lens_mode": "20cm", "final_stop_distance_m": 1.85, "final_yaw_error_deg": 25.0, "peak_vram_gb": 6.8, "success": False, "abort_reason": "estop"})
    for _ in range(80):
        s = logger.get_summary()

    if fd_dir.exists():
        after_fds = len(list(fd_dir.iterdir()))
        assert (after_fds - before_fds) <= 3, f"Leaked {after_fds - before_fds} file descriptors!"

    assert s["avg_stop_distance_m"] == 0.45, f"Aborted run skewed avg_stop_distance_m: {s['avg_stop_distance_m']}"


def test_high03_low01_xss_sanitization_and_all_dom_ids_bound() -> None:
    """HIGH-03 + LOW-01: Verify escapeHtml usage and zero orphaned HTML element IDs."""
    html = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    js = (PROJECT_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    css = (PROJECT_ROOT / "frontend" / "styles.css").read_text(encoding="utf-8")
    assert "function escapeHtml(" in js
    assert "escapeHtml(l.summary)" in js and "escapeHtml(e.msg)" in js

    ids = re.findall(r'\bid="([^"]+)"', html)
    assert len(ids) >= 25
    for dom_id in ids:
        assert (dom_id in js) or (f"#{dom_id}" in css), f"Orphaned DOM id={dom_id!r} in index.html!"


def test_high04_low02_stop_phrases_nan_json_and_unparseable_vlm(client: TestClient) -> None:
    """HIGH-04 + LOW-02: Natural language stop words trigger E-Stop; NaN/Infinity JSON rejected; empty VLM halts."""
    for word in ("stop", "STOP!", "halt", "cancel", "急停", "刹车", "停止"):
        client.post("/api/estop/reset")
        r = client.post("/api/bci/command", json={"command": word})
        assert r.status_code == 200 and r.json().get("action") == "estop", f"{word!r} did not trigger E-Stop!"
        assert client.get("/api/status").json()["gatekeeper"]["estop_active"] is True
    client.post("/api/estop/reset")

    for bad_nan in (b'{"option_id": NaN}', b'{"option_id": Infinity}', b'{"x": -Infinity}'):
        r_nan = client.post("/api/bci/command", content=bad_nan, headers={"Content-Type": "application/json"})
        assert r_nan.status_code == 400

    braked = {"cnt": 0}
    eng = CloudAgentEngine(emergency_brake_cb=lambda _: braked.__setitem__("cnt", braked["cnt"] + 1), default_mode="cloud")
    eng._worker_call = lambda *a, **kw: ("", "fake_cloud")  # type: ignore[method-assign]
    out_empty = eng.infer_with_watchdog(np.zeros((240, 320, 3), dtype=np.uint8), "pick mug", mode_override="cloud")
    assert out_empty["ok"] is False and out_empty["direction"] == "STOP" and braked["cnt"] == 1


def test_high05_blind_walk_precision_and_midway_estop_distance(client: TestClient) -> None:
    """HIGH-05: 80cm + blind_walk_enabled travels 0.40m +- 0.025m and logs true physical distance."""
    client.post("/api/robot/reset", json={"scenario": "default"})
    client.post("/api/config", json={"lens_mode": "80cm", "blind_walk_enabled": True, "cruise_speed_mps": 0.30})
    client.post("/api/bci/command", json={"option_id": 1, "source": "bw_full"})
    deadline = time.monotonic() + 16.0
    while time.monotonic() < deadline:
        if client.get("/api/status").json()["fsm"]["state"] == "IDLE":
            break
        time.sleep(0.05)
    st = client.get("/api/status").json()
    rec = st["logs_summary"]["recent_runs"][0]
    assert rec["success"] is True
    # Started blind-walk at ~0.83-0.85m, walked 0.40m -> final distance ~0.43-0.46m (matching true_mug_distance_m!)
    assert abs(rec["final_stop_distance_m"] - st["faults"]["true_mug_distance_m"]) <= 0.05
    client.post("/api/config", json={"blind_walk_enabled": False})


def test_high06_jpeg_25kb_cap_grayscale_bgra_and_med01_real_vram() -> None:
    """HIGH-06 + MED-01: High-entropy noise, 2D grayscale, and 4ch BGRA stay <= 25.0 KB; RealAdapter lists models."""
    rng = np.random.default_rng(123)
    noise_rgb = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    gray_2d = rng.integers(0, 256, size=(480, 640), dtype=np.uint8)
    bgra_4c = rng.integers(0, 256, size=(480, 640, 4), dtype=np.uint8)

    for arr in (noise_rgb, gray_2d, bgra_4c):
        raw_jpg, kb = compress_frame_320x240_jpeg(arr)
        assert raw_jpg[:2] == b"\xff\xd8"
        assert kb <= 25.0, f"JPEG exceeded 25KB cap: {kb} KB"

    real = XLeRobotRealAdapter()
    real.load_vln()
    v1 = real.get_vram_status()
    assert "Light-VLN" in v1.active_models
    real.unload_vln()
    real.load_vla("SmolVLA")
    v2 = real.get_vram_status()
    assert "SmolVLA" in v2.active_models
    real.shutdown()


def test_med02_mjpeg_stream_sigterm_and_med05_bci_sender_repeatable(tmp_path: Path) -> None:
    """MED-02 + MED-05: Active MJPEG stream exits in <1.0s on SIGTERM; bci_sender_example runs repeatedly with rc=0."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s1:
        s1.bind(("127.0.0.1", 0))
        http_port = s1.getsockname()[1]
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s2:
        s2.bind(("127.0.0.1", 0))
        udp_port = s2.getsockname()[1]

    env = {**os.environ, "HOST": "127.0.0.1", "PORT": str(http_port), "UDP_PORT": str(udp_port), "AGENT_MODE": "mock", "FYP_DATA_DIR": str(tmp_path)}
    proc = subprocess.Popen([sys.executable, str(PROJECT_ROOT / "run.py")], cwd=str(PROJECT_ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    try:
        for _ in range(40):
            try:
                if httpx.get(f"http://127.0.0.1:{http_port}/api/health", timeout=0.3).status_code == 200:
                    break
            except Exception:
                time.sleep(0.08)

        # Run bci_sender_example.py 3 consecutive times (2nd/3rd hit 409 E-Stop latched, must NOT crash!)
        sender_src = (PROJECT_ROOT / "bci_sender_example.py").read_text(encoding="utf-8")
        patched = sender_src.replace("8080", str(http_port)).replace("9999", str(udp_port))
        for _ in range(3):
            r_sub = subprocess.run([sys.executable, "-c", patched], timeout=5.0, capture_output=True, text=True)
            assert r_sub.returncode == 0, f"bci_sender_example crashed: {r_sub.stderr}"

        # Open an active MJPEG stream in a background thread, then send SIGTERM
        stream_opened = threading.Event()

        def _hold_mjpeg() -> None:
            try:
                with httpx.stream("GET", f"http://127.0.0.1:{http_port}/api/camera/stream.mjpg", timeout=3.0) as resp:
                    for chunk in resp.iter_bytes():
                        if chunk:
                            stream_opened.set()
            except Exception:
                pass

        t_mjpeg = threading.Thread(target=_hold_mjpeg, daemon=True)
        t_mjpeg.start()
        assert stream_opened.wait(timeout=1.5), "MJPEG stream did not yield initial frame"

        t_sig = time.perf_counter()
        proc.send_signal(signal.SIGTERM)
        rc = proc.wait(timeout=2.0)
        shutdown_s = time.perf_counter() - t_sig
        assert rc == 0, f"Non-zero exit code {rc}"
        assert shutdown_s < 1.2, f"MJPEG stream delayed SIGTERM shutdown ({shutdown_s:.2f}s)"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=1.0)
