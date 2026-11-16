"""Comprehensive regression suite locking down all 4 Red-Team Round 4 findings (R4-HIGH-01..R4-LOW-01)."""

from __future__ import annotations

import json
import socket
import threading
import time
import types
from fastapi.testclient import TestClient

from adapters.mock_robot import MockRobotAdapter
from backend.modules.vision_depth import VisionDepthCalibrator


def test_r4_high01_yoloe_zero_boxes_returns_none_without_hsv_fallback_and_exception_fallback() -> None:
    """R4-HIGH-01: Loaded YOLOE returning 0 boxes MUST return None (NO_TARGET_IN_VIEW), never falling through to HSV."""
    sim = MockRobotAdapter()
    try:
        sim.reset_pose(x=0.0, y=0.0, theta=0.0)  # Red mug, blue bottle, table, doorway all visible in HSV
        frame = sim.get_rgbd_frame("80cm")

        cal = VisionDepthCalibrator(initial_lens_mode="80cm")
        cal.set_classes(["red mug", "mug", "cup"])

        # Sanity check: without YOLOE weights, builtin HSV detects the red mug
        m_builtin = cal.analyze_frame(frame, update_history=False)
        assert m_builtin.detected is True and m_builtin.label == "red mug"

        # 1) Mount YOLOE stub returning 0 boxes (both empty list and Results object with empty boxes)
        for empty_return in (
            [],
            [types.SimpleNamespace(boxes=[], names={0: "red mug"})],
            [types.SimpleNamespace(boxes=None, names={0: "red mug"})],
        ):
            cal._yoloe_model = types.SimpleNamespace(
                predict=lambda bgr, verbose=False, conf=0.25, _ret=empty_return: _ret,
                set_classes=lambda prompts: None,
            )
            m_zero = cal.analyze_frame(frame, update_history=False)
            assert m_zero.detected is False, f"0-box YOLOE fell through to HSV for {empty_return!r}: {m_zero}"
            assert m_zero.label == "none"
            assert m_zero.confidence == 0.0
            assert m_zero.bbox is None
            assert m_zero.steering_hint == "SEARCH"
            assert m_zero.depth_status == "NO_TARGET_IN_VIEW"

        # 2) Mount YOLOE stub that raises an inference exception -> MUST gracefully fall back to builtin detector
        def _broken_predict(bgr, verbose=False, conf=0.25):
            raise RuntimeError("Simulated TensorRT/CUDA execution error")

        cal._yoloe_model = types.SimpleNamespace(predict=_broken_predict, set_classes=lambda prompts: None)
        m_err_fb = cal.analyze_frame(frame, update_history=False)
        assert m_err_fb.detected is True and m_err_fb.label == "red mug"
    finally:
        sim.shutdown()


def test_r4_med01_bci_hybrid_option_and_command_negation_and_split_brain_blocked(client: TestClient) -> None:
    """R4-MED-01: Hybrid (`option_id` + `command`) packets enforce negation filter & System-1/System-2 target consistency."""
    client.post("/api/estop/reset")
    client.post("/api/robot/reset", json={"scenario": "default"})

    # 1) Negated or unrecognized commands smuggled alongside valid option_id=1..4 MUST be rejected with 422!
    negated_or_bad_hybrids = [
        {"option_id": 1, "command": "Do not pick up the red mug"},
        {"option_id": 1, "command": "don't grab the cup"},
        {"option_id": 1, "command": "不要拿红色水杯"},
        {"option_id": 2, "command": "Do not pick up the water bottle"},
        {"option_id": 2, "command": "别拿水瓶"},
        {"option_id": 3, "command": "Do not go to the doorway"},
        {"option_id": 3, "command": "不去门口了"},
        {"option_id": 4, "command": "pick up the apple on keyboard"},
        {"option_id": 1, "command": "Pick up red mug and water bottle"},
    ]
    for bad in negated_or_bad_hybrids:
        res = client.post("/api/bci/command", json=bad)
        assert res.status_code == 422, f"Hybrid negated/invalid payload accepted ({res.status_code}): {bad!r}"
        st = client.get("/api/status").json()
        assert st["fsm"]["state"] == "IDLE"
        assert st["gatekeeper"]["estop_active"] is False

    # 2) Conflicting System-1/System-2 split-brain payloads (mismatched target or task_type) MUST return 422!
    conflicting_hybrids = [
        {"option_id": 1, "command": "前往敞开的门口 (Go to open doorway)"},
        {"option_id": 1, "command": "Pick up water bottle"},
        {"option_id": 2, "command": "去桌边拿红色水杯"},
        {"option_id": 2, "command": "Go to open doorway"},
        {"option_id": 3, "command": "Pick up the red mug"},
        {"option_id": 3, "command": "先看门口方向再去拿红色水杯"},
        {"option_id": 4, "command": "前往敞开的门口"},
        {"option_id": 4, "command": "拿蓝色水瓶"},
    ]
    for conflict in conflicting_hybrids:
        res = client.post("/api/bci/command", json=conflict)
        assert res.status_code == 422, f"Split-brain hybrid payload accepted ({res.status_code}): {conflict!r}"
        assert "Conflicting" in res.json().get("error", "")
        assert client.get("/api/status").json()["fsm"]["state"] == "IDLE"

    # Also verify over LAN UDP socket
    udp_port = client.get("/api/status").json()["bci"]["udp_port"]
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(1.0)
        sock.sendto(
            json.dumps({"option_id": 1, "command": "前往敞开的门口"}).encode("utf-8"),
            ("127.0.0.1", udp_port),
        )
        ack = json.loads(sock.recvfrom(2048)[0].decode("utf-8"))
        assert ack["status"] == 422 and ack["ok"] is False

    # 3) Consistent hybrid payloads (official label_zh, official command_text, or matching NL command) succeed with 200
    for valid_hybrid in (
        {"option_id": 1, "command": "抓红色水杯"},
        {"option_id": 1, "command": "Pick up the red mug"},
        {"option_id": 3, "command": "前往敞开的门口 (Go to open doorway)"},
    ):
        r_ok = client.post("/api/bci/command", json=valid_hybrid)
        assert r_ok.status_code == 200, f"Valid hybrid rejected: {valid_hybrid!r} -> {r_ok.text}"
        client.post("/api/estop/trigger")
        client.post("/api/estop/reset")

    # 4) Emergency stop preemption still wins immediately on Option #5 or E-Stop text in hybrid packet
    for estop_hybrid in (
        {"option_id": 5, "command": "Do not pick up the red mug"},
        {"option_id": 1, "command": "急停！"},
    ):
        client.post("/api/estop/reset")
        r_es = client.post("/api/bci/command", json=estop_hybrid)
        assert r_es.status_code == 200 and r_es.json().get("action") == "estop"
        assert client.get("/api/status").json()["gatekeeper"]["estop_active"] is True
    client.post("/api/estop/reset")


def test_r4_med02_collision_guard_never_records_success_and_resets_per_mission(client: TestClient) -> None:
    """R4-MED-02: Any physical collision halts immediately with success=False & obstacle_collision_guard; resets cleanly next run."""
    rt = client.app.state.runtime  # type: ignore[attr-defined]
    client.post("/api/estop/reset")

    # 1) Verify step_vln enforces vx == 0.0 (in-place rotation) whenever |yaw_error| > 14.0 deg
    rt.adapter.reset_pose(x=0.5, y=0.0, theta=1.0)  # Large yaw error (~60 deg) relative to red mug
    step = rt.adapter.step_vln("Pick up red mug", agent_direction_hint="RIGHT")
    assert abs(step["yaw_err_deg"]) > 14.0
    assert step["vx"] == 0.0, f"Expected vx=0.0 for large yaw error {step['yaw_err_deg']}deg, got {step['vx']}"

    # 2) Inject mid-cruise collision flag -> mission MUST halt immediately and record success=False, obstacle_collision_guard
    client.post("/api/robot/reset", json={"scenario": "default"})
    client.post("/api/config", json={"lens_mode": "80cm", "blind_walk_enabled": False, "cruise_speed_mps": 0.22})
    r1 = client.post("/api/bci/command", json={"option_id": 1, "source": "injected_collision_run"})
    assert r1.status_code == 200
    time.sleep(0.08)
    rt.adapter.collision_occurred = True

    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if client.get("/api/status").json()["fsm"]["state"] == "IDLE":
            break
        time.sleep(0.03)

    st_col = client.get("/api/status").json()
    assert st_col["fsm"]["state"] == "IDLE"
    assert st_col["gatekeeper"]["velocity"]["vx"] == 0.0
    col_run = st_col["logs_summary"]["recent_runs"][0]
    assert col_run["source"] == "bci_http:injected_collision_run"
    assert col_run["success"] is False, "Collided mission illegally recorded success=True!"
    assert col_run["abort_reason"] == "obstacle_collision_guard"

    # 3) Next mission from safe pose WITHOUT calling /api/robot/reset must reset per-mission collision flag and succeed!
    client.post("/api/config", json={"cruise_speed_mps": 0.34})
    r2 = client.post("/api/bci/command", json={"option_id": 1, "source": "clean_recovery_run"})
    assert r2.status_code == 200
    deadline2 = time.monotonic() + 16.0
    while time.monotonic() < deadline2:
        if client.get("/api/status").json()["fsm"]["state"] == "IDLE":
            break
        time.sleep(0.05)

    st_ok = client.get("/api/status").json()
    assert st_ok["faults"]["collision_occurred"] is False
    ok_run = st_ok["logs_summary"]["recent_runs"][0]
    assert ok_run["source"] == "bci_http:clean_recovery_run"
    assert ok_run["success"] is True and ok_run["abort_reason"] == "completed"

    # 4) Exact Red-Team R4-MED-02 close-table pose `(1.6422, -0.1729, -0.623)` -> Option #3 doorway MUST have 0 collisions!
    client.post("/api/robot/reset", json={"x": 1.6422, "y": -0.1729, "theta": -0.623})
    client.post("/api/config", json={"lens_mode": "20cm", "blind_walk_enabled": False, "cruise_speed_mps": 0.32})
    r3 = client.post("/api/bci/command", json={"option_id": 3, "source": "exact_r4_repro_pose"})
    assert r3.status_code == 200
    deadline3 = time.monotonic() + 18.0
    while time.monotonic() < deadline3:
        if client.get("/api/status").json()["fsm"]["state"] == "IDLE":
            break
        time.sleep(0.04)
    st_r4 = client.get("/api/status").json()
    assert st_r4["faults"]["collision_occurred"] is False
    assert st_r4["logs_summary"]["recent_runs"][0]["success"] is True

    # 5) Cold-spawned `near_blind_06m` on 80cm lens with depth_fault="none" MUST still immediately blind-brake (<0.80m)!
    client.post("/api/robot/reset", json={"scenario": "near_blind_06m"})
    client.post("/api/config", json={"lens_mode": "80cm", "blind_walk_enabled": False})
    r_nb = client.post("/api/bci/command", json={"option_id": 1, "source": "cold_near_blind_80cm"})
    assert r_nb.status_code == 200
    deadline_nb = time.monotonic() + 2.0
    while time.monotonic() < deadline_nb:
        if client.get("/api/status").json()["fsm"]["state"] == "IDLE":
            break
        time.sleep(0.03)
    st_nb = client.get("/api/status").json()
    assert "blind_zone_brake" in st_nb["fsm"]["last_abort_reason"]
    assert st_nb["faults"]["collision_occurred"] is False


def test_r4_med02_chained_5_missions_without_reset_zero_collision_all_lens_modes(client: TestClient) -> None:
    """
    R4-MED-02: Execute continuous un-reset 5-mission chain `#1 -> #2 -> #3 -> #4 -> #1` across:
      - 80cm (blind_walk=False)
      - 80cm (blind_walk=True)
      - 20cm (blind_walk=False)
    Verify 0 transient collisions and 100% success=True on all 15 runs!
    """
    rt = client.app.state.runtime  # type: ignore[attr-defined]
    client.post("/api/estop/reset")
    client.post("/api/faults", json={"depth_fault": "none", "simulate_agent_delay_s": 0.0, "simulate_agent_error": False})

    chain_options = [1, 2, 3, 4, 1]
    mode_matrix = [
        ("80cm", False),
        ("80cm", True),
        ("20cm", False),
    ]

    for lens_mode, bw_flag in mode_matrix:
        # Reset ONLY ONCE at the very start of the 5-task sequence!
        client.post("/api/robot/reset", json={"scenario": "default"})
        client.post(
            "/api/config",
            json={
                "lens_mode": lens_mode,
                "blind_walk_enabled": bw_flag,
                "cruise_speed_mps": 0.35,
                "vla_model": "SmolVLA",
            },
        )

        saw_any_collision = {"hit": False}
        stop_watch = threading.Event()

        def _collision_watcher() -> None:
            while not stop_watch.is_set():
                if getattr(rt.adapter, "collision_occurred", False):
                    saw_any_collision["hit"] = True
                time.sleep(0.015)

        watcher = threading.Thread(target=_collision_watcher, daemon=True)
        watcher.start()
        try:
            for step_idx, opt_id in enumerate(chain_options):
                tag = f"chain_{lens_mode}_bw{int(bw_flag)}_s{step_idx}_opt{opt_id}"
                res = client.post("/api/bci/command", json={"option_id": opt_id, "source": tag})
                assert res.status_code == 200, f"Step {tag} dispatch failed: {res.text}"

                deadline = time.monotonic() + 22.0
                while time.monotonic() < deadline:
                    if rt.fsm.state == "IDLE":
                        break
                    time.sleep(0.04)

                st = client.get("/api/status").json()
                assert st["fsm"]["state"] == "IDLE", f"Step {tag} timed out in state {st['fsm']['state']}"
                assert st["faults"]["collision_occurred"] is False, f"Collision occurred on {tag}! Pose={st['pose']}"
                assert saw_any_collision["hit"] is False, f"Transient collision detected during {tag}!"

                latest = st["logs_summary"]["recent_runs"][0]
                assert latest["source"] == f"bci_http:{tag}"
                assert latest["success"] is True, f"Chained task {tag} failed: {latest}"
                assert latest["abort_reason"] == "completed"
                assert st["vram"]["vram_violation"] is False
        finally:
            stop_watch.set()
            watcher.join(timeout=0.3)

    client.post("/api/config", json={"lens_mode": "80cm", "blind_walk_enabled": False, "cruise_speed_mps": 0.18})


def test_r4_low01_builtin_open_vocab_no_red_mug_hallucination_and_table_support(client: TestClient) -> None:
    """R4-LOW-01: Unmatched open-vocabulary prompts return detected=False (never hallucinating 'red mug'); 'table' works."""
    sim = MockRobotAdapter()
    cal = VisionDepthCalibrator(initial_lens_mode="80cm")
    cal._yoloe_model = None  # Explicitly test builtin open-vocabulary detector
    try:
        sim.reset_pose(x=0.0, y=0.0, theta=0.0)
        frame = sim.get_rgbd_frame("80cm")

        for unmatched_prompts in (
            ["apple", "keyboard"],
            ["banana", "laptop", "backpack"],
            ["infrared sensor", "credit card"],
        ):
            cal.set_classes(unmatched_prompts)
            meas = cal.analyze_frame(frame, update_history=False)
            assert meas.detected is False, f"Hallucinated {meas.label!r} for unmatched prompts {unmatched_prompts!r}!"
            assert meas.label == "none"
            assert meas.bbox is None
            assert meas.depth_status == "NO_TARGET_IN_VIEW"

        # Verify wood brown laboratory table/desk IS cleanly supported without false scale_poison
        cal.set_classes(["table", "desk"])
        m_tbl = cal.analyze_frame(frame, update_history=False)
        assert m_tbl.detected is True
        assert m_tbl.label == "table"
        assert m_tbl.blind_zone_triggered is False
        assert m_tbl.penetration_anomaly is False
        assert m_tbl.depth_status == "OK"
    finally:
        sim.shutdown()
