"""Comprehensive regression suite locking down all 8 Red-Team Round 3 findings (R3-HIGH-01..R3-LOW-01)."""

from __future__ import annotations

import csv
import io
import threading
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from backend.modules.cloud_agent import CloudAgentEngine

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_r3_high01_compound_estop_clauses_no_stop_and_negated_task_rejection(client: TestClient) -> None:
    """R3-HIGH-01: Compound ZH/EN stop clauses preempt as E-Stop; negated/hesitation commands return 422."""
    compound_estop_phrases = [
        "去门口拿水杯，不对，停！",
        "去门口拿水杯，算了停吧！",
        "去拿水杯，停住！",
        "去拿水杯，快点停！",
        "先暂停，去门口拿水杯算了",
        "去门口拿水杯，不对，取消！",
        "给我停！",
        "中止！",
        "终止！",
        "算了！",
        "go to the doorway no stop",
        "no robot stop",
        "wait no stop right now",
    ]
    for phrase in compound_estop_phrases:
        client.post("/api/estop/reset")
        res = client.post("/api/bci/command", json={"command": phrase})
        assert res.status_code == 200, f"Compound E-Stop {phrase!r} failed with {res.status_code}"
        body = res.json()
        assert body.get("action") == "estop", f"Compound E-Stop {phrase!r} started motors instead of E-Stop: {body}"
        st = client.get("/api/status").json()
        assert st["gatekeeper"]["estop_active"] is True
        assert st["gatekeeper"]["velocity"]["vx"] == 0.0
        assert st["fsm"]["state"] == "IDLE"

    client.post("/api/estop/reset")

    # Also verify mid-cruise preemption with `"去门口拿水杯，不对，停！"`
    client.post("/api/robot/reset", json={"scenario": "default"})
    r_go = client.post("/api/bci/command", json={"option_id": 1})
    assert r_go.status_code == 200
    time.sleep(0.08)
    r_mid_stop = client.post("/api/bci/command", json={"command": "去门口拿水杯，不对，停！"})
    assert r_mid_stop.status_code == 200 and r_mid_stop.json().get("action") == "estop"
    assert client.get("/api/status").json()["gatekeeper"]["estop_active"] is True
    client.post("/api/estop/reset")

    # Negated / retracted / hesitation commands MUST return 422 (neither E-Stop nor motor start!)
    negated_or_retracted = [
        "不去门口了",
        "不拿水杯了",
        "不是拿水杯",
        "慢着，先等一下",
        "等一下再去拿水杯",
        "no mug",
        "not the mug",
        "wrong cup",
        "wait grab the mug",
        "hold on",
        "do not stop",
        "no stopping",
    ]
    for txt in negated_or_retracted:
        res = client.post("/api/bci/command", json={"command": txt})
        assert res.status_code == 422, f"Negated phrase {txt!r} was not rejected with 422 (got {res.status_code})"
        st = client.get("/api/status").json()
        assert st["gatekeeper"]["estop_active"] is False, f"{txt!r} falsely latched E-Stop!"
        assert st["fsm"]["state"] == "IDLE", f"{txt!r} started motors!"

    # Benign `"不"` compound `"去拿不锈钢水杯"` still dispatches cleanly
    r_benign = client.post("/api/bci/command", json={"command": "去拿不锈钢水杯"})
    assert r_benign.status_code == 200 and r_benign.json().get("ok") is True
    client.post("/api/estop/trigger")
    client.post("/api/estop/reset")


def test_r3_high02_vlm_parser_rejects_all_unsafe_error_negated_and_verbose_prose() -> None:
    """R3-HIGH-02: Reject `safe:false`, non-empty `error`, `Unsafe to go LEFT`, Chinese wall/crash warnings, & >3-word prose."""
    adversarial_vlm_outputs = [
        '{"direction": "LEFT", "safe": false}',
        '{"direction": "LEFT", "ok": "false"}',
        '{"direction": "LEFT", "valid": 0}',
        '{"direction": "STRAIGHT", "error": "camera_blind"}',
        '{"direction": "RIGHT", "blocked": true}',
        "Unsafe to go LEFT",
        "Unable to move STRAIGHT",
        "Going RIGHT is wrong",
        "前方有墙，如果选 LEFT 会撞墙",
        "左边走不通，LEFT 被堵死了",
        "Sorry I cannot see clearly so maybe try LEFT later",
    ]
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    for bad in adversarial_vlm_outputs:
        braked = {"cnt": 0}
        eng = CloudAgentEngine(
            emergency_brake_cb=lambda _: braked.__setitem__("cnt", braked["cnt"] + 1),
            default_mode="cloud",
        )
        eng._worker_call = lambda *a, _txt=bad, **kw: (_txt, "adversarial_r3")  # type: ignore[method-assign]
        out = eng.infer_with_watchdog(frame, "Pick up red mug", mode_override="cloud")
        assert out["ok"] is False, f"Hazardous VLM output accepted: {bad!r} -> {out}"
        assert out["direction"] == "STOP"
        assert out["watchdog_triggered"] is True
        assert braked["cnt"] == 1

    # Confirm valid concise replies & benign `blocked: false` / `safe: true` JSON still pass
    eng_ok = CloudAgentEngine(emergency_brake_cb=lambda _: None, default_mode="cloud")
    for good, expected_dir in (
        ("LEFT", "LEFT"),
        ("Direction: RIGHT", "RIGHT"),
        ('{"direction": "STRAIGHT", "safe": true, "blocked": false}', "STRAIGHT"),
    ):
        eng_ok._worker_call = lambda *a, _txt=good, **kw: (_txt, "clean_vlm")  # type: ignore[method-assign]
        res_ok = eng_ok.infer_with_watchdog(frame, "Pick up red mug", mode_override="cloud")
        assert res_ok["ok"] is True and res_ok["direction"] == expected_dir


def test_r3_high03_mixed_vocabulary_no_mislabeling_or_false_4m_penetration_deadlock(client: TestClient) -> None:
    """R3-HIGH-03: Mixed prompts `['cup','mug','bottle','door','doorway']` never mislabel doorway as 'cup' or false-lock."""
    mixed_prompts = ["cup", "mug", "bottle", "door", "doorway"]
    r_cfg = client.post("/api/config", json={"lens_mode": "80cm", "detection_prompts": mixed_prompts})
    assert r_cfg.status_code == 200

    # 1) `default` scenario: must lock onto `"cup"` (red mug at ~1.98m) with `center_40pct` and `depth_status == "OK"`
    client.post("/api/robot/reset", json={"scenario": "default"})
    rt = client.app.state.runtime  # type: ignore[attr-defined]
    m_def = rt.vision.analyze_frame(rt.adapter.get_rgbd_frame("80cm"), lens_mode_override="80cm")
    assert m_def.detected is True
    assert m_def.label == "cup"
    assert m_def.is_doorway is False
    assert m_def.roi_mode == "center_40pct"
    assert m_def.blind_zone_triggered is False and m_def.penetration_anomaly is False
    assert m_def.depth_status == "OK"

    # 2) `doorway_front` scenario AND closer doorway-only pose `(1.20, 0.56, 0.0)`: zero false 4.0m penetration brake!
    for pose_payload in ({"scenario": "doorway_front"}, {"x": 1.20, "y": 0.56, "theta": 0.0}):
        client.post("/api/robot/reset", json=pose_payload)
        m_dw = rt.vision.analyze_frame(rt.adapter.get_rgbd_frame("80cm"), lens_mode_override="80cm")
        assert m_dw.detected is True
        assert m_dw.blind_zone_triggered is False, f"False brake on {pose_payload}: {m_dw.to_dict()}"
        assert m_dw.penetration_anomaly is False, f"False penetration anomaly on {pose_payload}: {m_dw.to_dict()}"
        assert m_dw.depth_status == "OK"
        if "door" in m_dw.label:
            assert m_dw.is_doorway is True and m_dw.roi_mode == "edges_15pct"

    # 3) Putting `"door"` first in mixed prompts prioritizes doorway with `edges_15pct`
    client.post("/api/config", json={"detection_prompts": ["door", "doorway", "cup", "mug"]})
    client.post("/api/robot/reset", json={"scenario": "doorway_front"})
    m_prio = rt.vision.analyze_frame(rt.adapter.get_rgbd_frame("80cm"), lens_mode_override="80cm")
    assert m_prio.label == "door" and m_prio.is_doorway is True and m_prio.roi_mode == "edges_15pct"
    assert m_prio.blind_zone_triggered is False and m_prio.depth_status == "OK"


def test_r3_high04_already_at_stop_distance_zero_creep_across_8_repeats(client: TestClient) -> None:
    """R3-HIGH-04: Repeating `#1 抓红色水杯` 8 times after reaching <=0.45m causes 0.000m forward creep!"""
    client.post("/api/robot/reset", json={"scenario": "default"})
    client.post("/api/config", json={"lens_mode": "20cm", "blind_walk_enabled": False, "cruise_speed_mps": 0.32})

    # Initial approach from 2.0m -> ~0.446..0.450m
    assert client.post("/api/bci/command", json={"option_id": 1, "source": "initial_20cm"}).status_code == 200
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if client.get("/api/status").json()["fsm"]["state"] == "IDLE":
            break
        time.sleep(0.04)

    st0 = client.get("/api/status").json()
    d_parked = st0["faults"]["true_mug_distance_m"]
    assert 0.42 <= d_parked <= 0.46

    # Fire 8 consecutive repeats WITHOUT resetting pose — robot MUST NOT creep forward into blind/collision zone!
    for rep in range(8):
        r_rep = client.post("/api/bci/command", json={"option_id": 1, "source": f"repeat_{rep}"})
        assert r_rep.status_code == 200
        dl_rep = time.monotonic() + 6.0
        while time.monotonic() < dl_rep:
            if client.get("/api/status").json()["fsm"]["state"] == "IDLE":
                break
            time.sleep(0.03)
        st_rep = client.get("/api/status").json()
        latest = st_rep["logs_summary"]["recent_runs"][0]
        assert latest["success"] is True, f"Repeat #{rep+1} failed: {latest}"
        assert latest["stage_durations"].get("VLN_CRUISE", 0.0) == 0.0
        assert st_rep["faults"]["collision_occurred"] is False

    d_final = client.get("/api/status").json()["faults"]["true_mug_distance_m"]
    assert abs(d_final - d_parked) <= 0.002, f"Robot crept forward from {d_parked:.3f}m to {d_final:.3f}m!"


def test_r3_med01_mock_immune_to_hung_cloud_sem_http_timeout_auto_degrade_and_safe_bench(client: TestClient) -> None:
    """R3-MED-01: Local mock bypasses saturated `_rpc_sem`; HttpOptions(timeout=2200); auto degrades on cold timeout; safe bench."""
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    release_hung = threading.Event()

    eng = CloudAgentEngine(emergency_brake_cb=lambda _: None, default_mode="cloud", watchdog_timeout_s=0.55)
    eng._worker_call = lambda *a, **kw: (release_hung.wait(timeout=3.0) and ("STRAIGHT", "hung"))  # type: ignore[method-assign]

    # 1) Occupy all 4 `_rpc_sem` slots with hung background calls
    bg_futs = [
        threading.Thread(target=lambda: eng.infer_with_watchdog(frame, "Pick mug", mode_override="cloud"), daemon=True)
        for _ in range(4)
    ]
    for t in bg_futs:
        t.start()
    time.sleep(0.08)

    try:
        # Pure local `mock` mode MUST succeed in <50ms even while all 4 cloud slots are 100% hung!
        t_m0 = time.perf_counter()
        res_mock = eng.infer_with_watchdog(frame, "Pick up red mug", mode_override="mock")
        mock_ms = (time.perf_counter() - t_m0) * 1000.0
        assert res_mock["ok"] is True
        assert res_mock["provider"] == "mock_local"
        assert mock_ms < 60.0

        # Stationary `auto` mode (`allow_auto_fallback_on_timeout=True`) MUST also degrade cleanly to `mock_fallback`!
        res_auto_sat = eng.infer_with_watchdog(
            frame,
            "Pick up red mug",
            mode_override="auto",
            allow_auto_fallback_on_timeout=True,
        )
        assert res_auto_sat["ok"] is True and res_auto_sat["provider"] == "mock_fallback"
    finally:
        release_hung.set()
        for t in bg_futs:
            t.join(timeout=1.0)

    # 2) Verify stationary `auto` mode degrades on >watchdog cold-start OAuth hang, while moving cruise (`False`) brakes!
    brakes = {"n": 0}
    eng_auto = CloudAgentEngine(
        emergency_brake_cb=lambda _: brakes.__setitem__("n", brakes["n"] + 1),
        default_mode="auto",
        watchdog_timeout_s=0.55,
    )

    def _slow_cold_oauth(*a, **kw):
        time.sleep(1.1)
        return "STRAIGHT", "vertex"

    eng_auto._worker_call = _slow_cold_oauth  # type: ignore[method-assign]
    r_stat = eng_auto.infer_with_watchdog(frame, "Pick up red mug", allow_auto_fallback_on_timeout=True)
    assert r_stat["ok"] is True and r_stat["provider"] == "mock_fallback" and brakes["n"] == 0

    r_cruise = eng_auto.infer_with_watchdog(frame, "Pick up red mug", allow_auto_fallback_on_timeout=False)
    assert r_cruise["ok"] is False and r_cruise["watchdog_triggered"] is True and brakes["n"] == 1

    # 3) Verify `_get_vertex_client` passes `HttpOptions(timeout=2200)`
    eng_client_chk = CloudAgentEngine(emergency_brake_cb=lambda _: None, default_mode="cloud")
    with patch("google.genai.Client") as mock_genai_cls:
        eng_client_chk._get_vertex_client()
        assert mock_genai_cls.called
        http_opts = mock_genai_cls.call_args.kwargs.get("http_options")
        assert http_opts is not None and http_opts.timeout == 2200

    # 4) Verify `/api/agent/benchmark` does NOT slam chassis brake while FSM is in `YOLO_LAST_MILE`
    rt = client.app.state.runtime  # type: ignore[attr-defined]
    client.post("/api/estop/reset")
    with patch.object(rt.adapter, "stop", wraps=rt.adapter.stop) as spy_stop:
        with rt.fsm._lock:
            orig_state = rt.fsm._state
            rt.fsm._state = "YOLO_LAST_MILE"
        try:
            b_res = client.post("/api/agent/benchmark", json={"prompt": "test", "mode": "mock", "simulate_error": True})
            assert b_res.status_code == 200 and b_res.json()["ok"] is False
            assert spy_stop.call_count == 0, "Benchmark erroneously braked robot during YOLO_LAST_MILE!"
        finally:
            with rt.fsm._lock:
                rt.fsm._state = orig_state


def test_r3_med02_zombie_worker_ownership_guard_pre_vla_check_and_pose_reset_reason(client: TestClient) -> None:
    """R3-MED-02: Zombie worker never clobbers new mission's `last_abort_reason` or brakes it; pose reset logs `pose_reset_aborted`."""
    rt = client.app.state.runtime  # type: ignore[attr-defined]
    client.post("/api/estop/reset")
    client.post("/api/robot/reset", json={"scenario": "default"})

    # 1) Non-E-Stop `/api/robot/reset` during active mission MUST log `pose_reset_aborted` (NOT `estop_interrupted`!)
    client.post("/api/config", json={"lens_mode": "80cm", "cruise_speed_mps": 0.12})
    assert client.post("/api/bci/command", json={"option_id": 1, "source": "pose_rst_test"}).status_code == 200
    time.sleep(0.10)
    r_rst = client.post("/api/robot/reset", json={"scenario": "default"})
    assert r_rst.status_code == 200 and r_rst.json()["estop_active"] is False

    st_rst = client.get("/api/status").json()
    assert st_rst["fsm"]["state"] == "IDLE"
    assert st_rst["logs_summary"]["recent_runs"][0]["abort_reason"] == "pose_reset_aborted"

    # 2) Zombie Mission #1 delayed 0.65s in AGENT_PARSE must NEVER clobber Mission #2's `last_abort_reason` or brake it!
    orig_infer = rt.agent.infer_with_watchdog
    call_idx = {"n": 0}

    def _delayed_first_infer(*args, **kwargs):
        call_idx["n"] += 1
        if call_idx["n"] == 1:
            time.sleep(0.65)
        return orig_infer(*args, **kwargs)

    with patch.object(rt.agent, "infer_with_watchdog", side_effect=_delayed_first_infer):
        assert client.post("/api/bci/command", json={"option_id": 1, "source": "zombie_m1"}).status_code == 200
        time.sleep(0.05)
        # Preempt Mission #1 via E-Stop -> Reset (waits 0.25s; Mission #1 still asleep until 0.65s!)
        client.post("/api/estop/trigger")
        assert client.post("/api/estop/reset").status_code == 200

        # Launch Mission #2 at ~0.32s while Mission #1 is STILL asleep
        client.post("/api/config", json={"cruise_speed_mps": 0.30})
        assert client.post("/api/bci/command", json={"option_id": 3, "source": "active_m2"}).status_code == 200
        time.sleep(0.45)  # Mission #1 wakes up & exits around 0.65s while Mission #2 is actively cruising!

        st_mid = client.get("/api/status").json()
        if st_mid["fsm"]["state"] != "IDLE":
            assert st_mid["fsm"]["last_abort_reason"] == "running", (
                f"Zombie Mission #1 clobbered active Mission #2 last_abort_reason: {st_mid['fsm']['last_abort_reason']!r}"
            )

        deadline = time.monotonic() + 12.0
        while time.monotonic() < deadline:
            if client.get("/api/status").json()["fsm"]["state"] == "IDLE":
                break
            time.sleep(0.05)

    st_end = client.get("/api/status").json()
    assert st_end["fsm"]["last_abort_reason"] == "completed"
    runs_by_src = {r["source"]: r for r in st_end["logs_summary"]["recent_runs"]}
    assert runs_by_src["bci_http:active_m2"]["success"] is True
    assert runs_by_src["bci_http:zombie_m1"]["abort_reason"] == "estop_interrupted"


def test_r3_med03_blind_walk_logged_only_when_actually_executed_and_ui_20cm_guard(client: TestClient) -> None:
    """R3-MED-03: Doorway `navigate_only` & `20cm` runs record `blind_walk_enabled=False`; `80cm_blind_walk` stays pure ~0.45m."""
    js = (PROJECT_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    assert "bwEl.disabled = isLens20" in js
    assert 'r.lens_mode === "80cm" && r.blind_walk_enabled' in js

    # Turn `blind_walk_enabled=True` globally and run all 3 cases
    client.post("/api/config", json={"lens_mode": "80cm", "blind_walk_enabled": True, "cruise_speed_mps": 0.32})

    # Case A: 80cm + Option #1 (mug pick) -> executes blind walk -> `blind_walk_enabled == True` (~0.45m)
    client.post("/api/robot/reset", json={"scenario": "default"})
    assert client.post("/api/bci/command", json={"option_id": 1, "source": "bw_mug_80"}).status_code == 200
    while client.get("/api/status").json()["fsm"]["state"] != "IDLE":
        time.sleep(0.05)

    # Case B: 80cm + Option #3 (doorway navigate_only) -> NO blind walk -> MUST record `blind_walk_enabled == False` (~0.85m)
    client.post("/api/robot/reset", json={"scenario": "doorway_front"})
    assert client.post("/api/bci/command", json={"option_id": 3, "source": "bw_door_80"}).status_code == 200
    while client.get("/api/status").json()["fsm"]["state"] != "IDLE":
        time.sleep(0.05)

    # Case C: 20cm + Option #1 (mug pick) -> NO blind walk -> MUST record `blind_walk_enabled == False` (~0.45m)
    client.post("/api/robot/reset", json={"scenario": "default"})
    client.post("/api/config", json={"lens_mode": "20cm", "blind_walk_enabled": True})
    assert client.post("/api/bci/command", json={"option_id": 1, "source": "bw_mug_20"}).status_code == 200
    while client.get("/api/status").json()["fsm"]["state"] != "IDLE":
        time.sleep(0.05)

    summary = client.get("/api/logs/summary").json()
    by_src = {r["source"]: r for r in summary["recent_runs"]}
    assert by_src["bci_http:bw_mug_80"]["blind_walk_enabled"] is True
    assert by_src["bci_http:bw_door_80"]["blind_walk_enabled"] is False
    assert by_src["bci_http:bw_mug_20"]["blind_walk_enabled"] is False
    assert 0.42 <= summary["lens_comparison"]["80cm_blind_walk"]["avg_stop_m"] <= 0.48
    client.post("/api/config", json={"lens_mode": "80cm", "blind_walk_enabled": False})


def test_r3_low01_prompt_length_422_cap_and_ansi_crlf_sanitization(client: TestClient) -> None:
    """R3-LOW-01: Reject >48-char / >12-item prompts with 422; strip ANSI escapes & CR/LF from `source` and `reason`."""
    assert client.post("/api/config", json={"detection_prompts": ["A" * 50000]}).status_code == 422
    assert client.post("/api/config", json={"detection_prompts": [""]}).status_code == 422
    assert client.post("/api/config", json={"detection_prompts": [f"cls_{i}" for i in range(13)]}).status_code == 422

    dirty_source = "evil_src\r\n=CMD|' /C calc'!A0\x1b[31m"
    r_bci = client.post("/api/bci/command", json={"option_id": 5, "source": dirty_source})
    assert r_bci.status_code == 200

    st = client.get("/api/status").json()
    logged_src = st["bci"]["recent_logs"][0]["source"]
    estop_reason = st["gatekeeper"]["estop_reason"]
    for s_val in (logged_src, estop_reason):
        assert "\r" not in s_val and "\n" not in s_val and "\x1b" not in s_val
        assert "[31m" not in s_val

    r_estop = client.post("/api/estop/trigger", json={"reason": "manual\r\n\x1b[32malert\t!"})
    assert r_estop.status_code == 200
    clean_reason = r_estop.json()["gatekeeper"]["estop_reason"]
    assert "\r" not in clean_reason and "\n" not in clean_reason and "\x1b" not in clean_reason and "[32m" not in clean_reason
    client.post("/api/estop/reset")

    # Verify SQLite & CSV export sanitize embedded newlines + formula leads
    rt = client.app.state.runtime  # type: ignore[attr-defined]
    rt.data_logger.record_run(
        {
            "command": "line1\r\n=2+2",
            "source": dirty_source,
            "lens_mode": "80cm",
            "success": False,
            "abort_reason": "\x1b[31m+aborted\r\nnow",
        }
    )
    csv_text = client.get("/api/logs/export/csv").content.decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    last = rows[-1]
    assert "\r" not in last["source"] and "\n" not in last["source"] and "\x1b" not in last["source"]
    assert last["abort_reason"].startswith("'+aborted")
