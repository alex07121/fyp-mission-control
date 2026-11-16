"""Red/Blue Stress Test 2: 80cm Blind-Zone Poisoning, Open-Doorway 4.0m Penetration, & 20cm Lens."""

from __future__ import annotations

import time
import numpy as np
from fastapi.testclient import TestClient

from adapters.mock_robot import MockRobotAdapter
from backend.modules.vision_depth import VisionDepthCalibrator


def test_open_doorway_edges_vs_center_penetration() -> None:
    sim = MockRobotAdapter()
    try:
        # Place robot 1.50m directly in front of open doorway
        sim.reset_pose(scenario="doorway_front")
        frame = sim.get_rgbd_frame(lens_mode="80cm")

        cal = VisionDepthCalibrator(initial_lens_mode="80cm")
        cal.set_classes(["open doorway", "door frame", "doorway"])
        meas = cal.analyze_frame(frame)

        assert meas.detected is True
        assert meas.is_doorway is True
        assert meas.roi_mode == "edges_15pct"
        assert meas.blind_zone_triggered is False
        assert meas.median_depth_m is not None

        # Center of open doorway in depth map penetrates to 4.0m corridor!
        x1, y1, x2, y2 = meas.bbox  # type: ignore[misc]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        center_depth = float(np.nanmedian(frame.depth_m[cy - 10 : cy + 10, cx - 10 : cx + 10]))
        assert center_depth > 3.5, f"Expected ~4.0m corridor center depth, got {center_depth}"

        # Yet our 15% left/right edge extractor measures true ~1.50m frame distance!
        assert 1.35 <= meas.median_depth_m <= 1.65, f"Doorway edge depth should be ~1.50m, got {meas.median_depth_m}"
    finally:
        sim.shutdown()


def test_80cm_blind_zone_at_06m_vs_20cm_new_lens() -> None:
    sim = MockRobotAdapter()
    try:
        # Robot at 0.60m from red mug (<0.80m blind zone on 80cm lens!)
        sim.reset_pose(scenario="near_blind_06m")
        cal = VisionDepthCalibrator(initial_lens_mode="80cm")
        cal.set_classes(["red mug", "mug", "cup"])

        f80 = sim.get_rgbd_frame(lens_mode="80cm")
        m80 = cal.analyze_frame(f80, lens_mode_override="80cm")
        assert m80.blind_zone_triggered is True
        assert m80.steering_hint == "BRAKE"

        # Switching to 20cm upgraded lens at exact same 0.60m pose gives valid ~0.60m depth!
        f20 = sim.get_rgbd_frame(lens_mode="20cm")
        m20 = cal.analyze_frame(f20, lens_mode_override="20cm")
        assert m20.blind_zone_triggered is False
        assert m20.median_depth_m is not None
        assert 0.52 <= m20.median_depth_m <= 0.68
    finally:
        sim.shutdown()


def test_poison_attacks_all_zero_nan_and_penetrate_4m_halt_safely(client: TestClient) -> None:
    for fault_mode in ("all_zero", "all_nan", "penetrate_4m"):
        client.post("/api/robot/reset", json={"scenario": "near_blind_06m"})
        client.post("/api/faults", json={"depth_fault": fault_mode})

        r = client.post("/api/bci/command", json={"option_id": 1, "source": f"poison_{fault_mode}"})
        assert r.status_code == 200

        # Must safely abort & brake within < 1.5s without infinite loop or table collision!
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            st = client.get("/api/status").json()
            if st["fsm"]["state"] == "IDLE":
                break
            time.sleep(0.05)

        st = client.get("/api/status").json()
        assert st["fsm"]["state"] == "IDLE", f"Stuck in loop under fault={fault_mode}"
        assert st["faults"]["collision_occurred"] is False, f"Robot collided under fault={fault_mode}!"
        vel = st["gatekeeper"]["velocity"]
        assert vel["vx"] == 0.0 and vel["wz"] == 0.0
        assert "blind_zone_brake" in st["fsm"]["last_abort_reason"]

    client.post("/api/faults", json={"depth_fault": "none"})
