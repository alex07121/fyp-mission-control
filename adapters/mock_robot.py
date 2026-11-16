"""Closed-loop 3D-projected RGB-D simulator + 12GB VRAM tracker for XLeRobot."""

from __future__ import annotations

import gc
import math
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from adapters.base import Pose2D, RGBDFrame, RobotAdapter, VRAMStatus
from backend.config import (
    LENS_PROFILES,
    TOTAL_VRAM_GB,
    VRAM_IDLE_BASE_GB,
    VRAM_POST_UNLOAD_GB,
    VRAM_VLA_MODELS_GB,
    VRAM_VLN_DELTA_GB,
)
from backend.modules.remote_vln_client import RemoteVLNClient


class MockRobotAdapter(RobotAdapter):
    """Hardware-free RGB-D & odometry simulator with blind-zone and fault injection."""

    # World coordinates (x forward, y left, in meters)
    OBJECTS: Dict[str, Dict[str, Any]] = {
        "doorway": {"x": 2.15, "y": 0.56, "w": 0.88, "h": 1.65, "z_c": 0.72},
        "table": {"x": 2.08, "y": -0.30, "w": 0.84, "h": 0.42, "z_c": 0.21},
        "mug": {"x": 1.98, "y": -0.16, "w": 0.13, "h": 0.15, "z_c": 0.49},
        "bottle": {"x": 2.00, "y": -0.44, "w": 0.10, "h": 0.23, "z_c": 0.53},
    }

    def __init__(self) -> None:
        super().__init__()
        self._state_lock = threading.RLock()
        self._pose = Pose2D(0.0, 0.0, 0.0)
        self._raw_vel = (0.0, 0.0, 0.0)
        self._arm_abort = False

        # Fault injection & VLN early-stop simulation controls
        self.depth_fault: str = "none"  # "none" | "all_zero" | "all_nan" | "penetrate_4m"
        self.force_vln_early_stop: bool = True
        self.collision_occurred: bool = False

        # 12GB VRAM state
        self._vln_loaded: bool = False
        self._vla_loaded: Optional[str] = None
        self._allocated_vram: float = VRAM_IDLE_BASE_GB
        self._peak_vram: float = VRAM_IDLE_BASE_GB
        self._vram_violation: bool = False
        self._remote_vln = RemoteVLNClient()

        # Precompute neutral gray/slate laboratory wall + terrazzo floor backdrop (zero HSV color interference)
        self._rng = np.random.default_rng(42)
        bg = np.zeros((480, 640, 3), dtype=np.int16)
        bg[:240, :] = (46, 44, 42)
        bg[240:, :] = (66, 68, 70)
        tex_rng = np.random.default_rng(101)
        speckle = tex_rng.integers(-10, 11, size=(480, 640, 1), dtype=np.int16)
        lab_bg = np.clip(bg + speckle, 0, 255).astype(np.uint8)
        for gx in range(0, 640, 64):
            cv2.line(lab_bg, (gx, 0), (gx, 240), (56, 54, 52), 1)
            cv2.line(lab_bg, (gx, 240), (max(0, min(639, 320 + int((gx - 320) * 1.5))), 479), (54, 56, 58), 1)
        for gy in range(260, 480, 36):
            cv2.line(lab_bg, (0, gy), (640, gy), (56, 58, 60), 1)
        cv2.line(lab_bg, (0, 240), (640, 240), (95, 95, 95), 1)
        self._lab_bg_rgb = lab_bg

        # Background 50Hz kinematics thread
        self._stop_evt = threading.Event()
        self._sim_thread = threading.Thread(target=self._kinematics_loop, name="MockSimLoop", daemon=True)
        self._sim_thread.start()

    def _kinematics_loop(self) -> None:
        last_t = time.monotonic()
        while not self._stop_evt.wait(0.02):
            now = time.monotonic()
            dt = min(0.08, max(0.0, now - last_t))
            last_t = now
            # Query gatekeeper BEFORE acquiring _state_lock to prevent ABBA lock inversion
            is_estop = self.estop_active
            with self._state_lock:
                vx, vy, wz = self._raw_vel
                if is_estop:
                    vx, vy, wz = 0.0, 0.0, 0.0
                    self._raw_vel = (0.0, 0.0, 0.0)
                if abs(vx) > 1e-6 or abs(vy) > 1e-6 or abs(wz) > 1e-6:
                    c, s = math.cos(self._pose.theta), math.sin(self._pose.theta)
                    nx = self._pose.x + (vx * c - vy * s) * dt
                    ny = self._pose.y + (vx * s + vy * c) * dt
                    nt = math.atan2(math.sin(self._pose.theta + wz * dt), math.cos(self._pose.theta + wz * dt))
                    self._pose = Pose2D(nx, ny, nt)
                    # Check physical table/pillar/doorframe collision (< 0.16m to solid object)
                    if self._is_pose_colliding(nx, ny):
                        self.collision_occurred = True

    def _is_pose_colliding(self, nx: float, ny: float) -> bool:
        for key in ("mug", "bottle"):
            obj = self.OBJECTS[key]
            if math.hypot(obj["x"] - nx, obj["y"] - ny) < 0.16:
                return True
        door = self.OBJECTS["doorway"]
        if abs(door["x"] - nx) < 0.14 and abs(door["y"] - ny) < (door["w"] * 0.5):
            return True
        return False

    def reset_collision_flag(self) -> None:
        """R4-MED-02: Reset per-mission collision flag at mission start (remains True if currently overlapping obstacle)."""
        with self._state_lock:
            self.collision_occurred = self._is_pose_colliding(self._pose.x, self._pose.y)

    @staticmethod
    def _resolve_target_key(target_kind: str) -> str:
        low = target_kind.lower()
        has_mug = any(w in low for w in ("mug", "cup", "teacup", "杯"))
        has_bottle = (
            ("bottle" in low)
            or ("瓶" in low)
            or (
                not has_mug
                and (
                    bool(re.search(r"\bwater\b", low))
                    or any(w in low for w in ("拿水", "取水", "抓水", "喝水", "端水"))
                )
            )
        )
        if ("door" in low or "门" in low) and not (has_mug or has_bottle):
            return "doorway"
        if has_bottle and not has_mug:
            return "bottle"
        return "mug"

    def _do_move(self, vx: float, vy: float, wz: float) -> None:
        with self._state_lock:
            self._raw_vel = (vx, vy, wz)

    def _do_stop(self) -> None:
        with self._state_lock:
            self._raw_vel = (0.0, 0.0, 0.0)

    def _do_abort_arms(self) -> None:
        with self._state_lock:
            self._arm_abort = True

    def set_fault_injection(
        self,
        depth_fault: Optional[str] = None,
        force_vln_early_stop: Optional[bool] = None,
    ) -> Dict[str, Any]:
        with self._state_lock:
            if depth_fault is not None:
                self.depth_fault = depth_fault
            if force_vln_early_stop is not None:
                self.force_vln_early_stop = bool(force_vln_early_stop)
            return {"depth_fault": self.depth_fault, "force_vln_early_stop": self.force_vln_early_stop}

    def get_true_target_distance(self, target_kind: str = "mug") -> float:
        pose = self.get_pose()
        key = self._resolve_target_key(target_kind)
        obj = self.OBJECTS[key]
        return math.hypot(obj["x"] - pose.x, obj["y"] - pose.y)

    def get_pose(self) -> Pose2D:
        with self._state_lock:
            return Pose2D(self._pose.x, self._pose.y, self._pose.theta)

    def reset_pose(
        self,
        x: float = 0.0,
        y: float = 0.0,
        theta: float = 0.0,
        scenario: Optional[str] = None,
    ) -> Pose2D:
        self.stop()
        with self._state_lock:
            self.collision_occurred = False
            if scenario == "near_blind_06m":
                self._pose = Pose2D(1.38, -0.16, 0.0)  # 0.60m from red mug (<0.80m blind zone!)
            elif scenario == "doorway_front":
                self._pose = Pose2D(0.65, 0.56, 0.0)   # 1.50m facing open doorway
            elif scenario == "far_search":
                self._pose = Pose2D(-0.35, 0.0, 0.65)  # Rotated away, needs VLN + Agent
            else:
                self._pose = Pose2D(x, y, theta)
            return Pose2D(self._pose.x, self._pose.y, self._pose.theta)

    @staticmethod
    def _project_box(
        pose: Pose2D, wx: float, wy: float, wz_c: float, real_w: float, real_h: float
    ) -> Optional[Tuple[int, int, int, int, float]]:
        dx, dy = wx - pose.x, wy - pose.y
        c, s = math.cos(pose.theta), math.sin(pose.theta)
        z_cam = dx * c + dy * s
        y_left = -dx * s + dy * c
        if z_cam <= 0.08:
            return None
        fx, fy, cx, cy = 510.0, 510.0, 320.0, 240.0
        cam_z = 0.55
        u_c = cx - fx * (y_left / z_cam)
        v_c = cy - fy * ((wz_c - cam_z) / z_cam)
        pw = max(6.0, fx * (real_w / z_cam))
        ph = max(6.0, fy * (real_h / z_cam))
        x1, y1 = int(round(u_c - pw * 0.5)), int(round(v_c - ph * 0.5))
        x2, y2 = int(round(u_c + pw * 0.5)), int(round(v_c + ph * 0.5))
        if x2 <= 0 or x1 >= 640 or y2 <= 0 or y1 >= 480:
            return None
        return x1, y1, x2, y2, float(z_cam)

    def get_rgbd_frame(self, lens_mode: str = "80cm") -> RGBDFrame:
        with self._state_lock:
            pose = Pose2D(self._pose.x, self._pose.y, self._pose.theta)
            fault = self.depth_fault

        H, W = 480, 640
        rgb = self._lab_bg_rgb.copy()

        # Default wall depth ~3.4m, floor gradient 1.0m..3.4m
        depth = np.full((H, W), max(1.2, 3.3 - pose.x * 0.4), dtype=np.float32)
        floor_rows = np.linspace(3.2, 0.95, H - 240, dtype=np.float32)[:, None]
        depth[240:, :] = np.minimum(depth[240:, :], floor_rows)

        # 1. Open Doorway: solid left/right frame at true z_cam, center cavity penetrates to 4.0m!
        d_box = self._project_box(pose, **self._box_args("doorway"))
        if d_box is not None:
            x1, y1, x2, y2, z_d = d_box
            cx1, cy1 = max(0, x1), max(0, y1)
            cx2, cy2 = min(W, x2), min(H, y2)
            bw = max(1, x2 - x1)
            vis_bw = max(1, cx2 - cx1)
            pillar_w = max(6, int(round(bw * 0.25)))
            min_vis_jamb = max(6, int(round(vis_bw * 0.19)))
            top_h = max(4, int(round((y2 - y1) * 0.14)))

            # Calibrate visible jamb bounds so 15% side ROIs stay >80% on solid jambs even at 0.45m close-up
            lx2 = min(cx2, max(min(W, cx1 + min_vis_jamb), min(W, max(0, x1 + pillar_w))))
            rx1 = max(cx1, min(max(0, cx2 - min_vis_jamb), max(0, min(W, x2 - pillar_w))))
            ty2 = min(H, max(0, y1 + top_h))

            # Draw far corridor inside open doorway (4.0m penetration!)
            in_x1, in_x2 = lx2, rx1
            in_y1 = max(cy1, ty2)
            if in_x2 > in_x1 and cy2 > in_y1:
                rgb[in_y1:cy2, in_x1:in_x2] = (28, 30, 34)
                cv2.line(rgb, (in_x1, (in_y1 + cy2) // 2), (in_x2, (in_y1 + cy2) // 2), (55, 65, 75), 1)
                depth[in_y1:cy2, in_x1:in_x2] = 4.00  # Open corridor far background

            # Left pillar, right pillar, and lintel at true doorway distance z_d
            frame_col = (40, 185, 235)  # Distinct amber/cyan-edged doorway frame
            if lx2 > cx1:
                rgb[cy1:cy2, cx1:lx2] = frame_col
                depth[cy1:cy2, cx1:lx2] = z_d
            if cx2 > rx1:
                rgb[cy1:cy2, rx1:cx2] = frame_col
                depth[cy1:cy2, rx1:cx2] = z_d
            if ty2 > cy1 and cx2 > cx1:
                rgb[cy1:ty2, cx1:cx2] = frame_col
                depth[cy1:ty2, cx1:cx2] = z_d
            cv2.rectangle(rgb, (cx1, cy1), (cx2 - 1, cy2 - 1), (0, 220, 255), 2)

        # 2. Lab Table
        t_box = self._project_box(pose, **self._box_args("table"))
        if t_box is not None:
            x1, y1, x2, y2, z_t = t_box
            cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
            if cx2 > cx1 and cy2 > cy1:
                rgb[cy1:cy2, cx1:cx2] = (55, 92, 138)  # Wood brown BGR
                cv2.rectangle(rgb, (cx1, cy1), (cx2 - 1, cy2 - 1), (40, 70, 110), 2)
                depth[cy1:cy2, cx1:cx2] = z_t

        # 3. Red Mug & Blue Water Bottle (sorted back-to-front)
        items = []
        for name, color, cap_col in (
            ("bottle", (230, 145, 20), (245, 245, 245)),
            ("mug", (28, 28, 232), (70, 70, 250)),
        ):
            proj = self._project_box(pose, **self._box_args(name))
            if proj is not None:
                items.append((proj[4], name, color, cap_col, proj))
        items.sort(key=lambda it: it[0], reverse=True)

        for z_o, name, color, cap_col, (x1, y1, x2, y2, _) in items:
            cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
            if cx2 > cx1 and cy2 > cy1:
                rgb[cy1:cy2, cx1:cx2] = color
                top_band = max(2, (cy2 - cy1) // 6)
                rgb[cy1 : min(cy2, cy1 + top_band), cx1:cx2] = cap_col
                depth[cy1:cy2, cx1:cx2] = z_o
                if fault == "penetrate_4m":
                    depth[cy1:cy2, cx1:cx2] = 4.00

        # Add subtle realistic sensor noise (+-3mm)
        with self._state_lock:
            noise = self._rng.normal(0.0, 0.003, size=depth.shape).astype(np.float32)
            nan_rand = self._rng.random(size=depth.shape)
        depth = depth + noise

        # Apply physical depth camera near blind-zone (<0.80m on 80cm lens, <0.20m on 20cm lens)
        blind_thresh = LENS_PROFILES.get(lens_mode, LENS_PROFILES["80cm"])["blind_zone_m"]
        blind_mask = depth < blind_thresh
        if np.any(blind_mask):
            nan_selector = nan_rand < 0.25
            depth[blind_mask & (~nan_selector)] = 0.0
            depth[blind_mask & nan_selector] = np.nan

        # Apply explicit red-team fault injection overrides
        if fault == "all_zero":
            depth.fill(0.0)
        elif fault == "all_nan":
            depth.fill(np.nan)
        elif fault == "penetrate_4m":
            # Simulate severe depth penetration returning 4.0m across central view
            depth[:, :] = 4.00

        return RGBDFrame(rgb=rgb, depth_m=depth, timestamp=time.time())

    def _box_args(self, key: str) -> Dict[str, float]:
        o = self.OBJECTS[key]
        return {"wx": o["x"], "wy": o["y"], "wz_c": o["z_c"], "real_w": o["w"], "real_h": o["h"]}

    def load_vln(self) -> VRAMStatus:
        if self._remote_vln.enabled:
            self._remote_vln.load()
        with self._state_lock:
            if not self._vln_loaded:
                self._vln_loaded = True
            self._recalc_vram()
            return self.get_vram_status()

    def step_vln(self, instruction: str, agent_direction_hint: str = "STRAIGHT") -> Dict[str, Any]:
        if self._remote_vln.enabled:
            frame = self.get_rgbd_frame()
            res = self._remote_vln.step(frame.rgb, instruction, agent_direction_hint)
            if res and res.get("ok"):
                return {
                    "vx": float(res.get("vx", 0.0)),
                    "vy": float(res.get("vy", 0.0)),
                    "wz": float(res.get("wz", 0.0)),
                    "arrived": bool(res.get("arrived", False)),
                }
        target_key = self._resolve_target_key(instruction)
        pose = self.get_pose()
        obj = self.OBJECTS[target_key]
        att_x = obj["x"] - pose.x
        att_y = obj["y"] - pose.y
        dist = math.hypot(att_x, att_y)

        hint = agent_direction_hint.upper()
        if hint == "STOP":
            return {"vx": 0.0, "vy": 0.0, "wz": 0.0, "arrived": False, "vln_est_dist_m": round(dist, 3)}

        inv_d = 1.0 / max(1e-6, dist)
        ux, uy = att_x * inv_d, att_y * inv_d

        # R4-MED-02: Local obstacle repulsive field preventing corner-clipping against table/mug/bottle/doorway
        hazards: List[Tuple[float, float, float]] = []
        if target_key == "doorway":
            hazards.append((self.OBJECTS["mug"]["x"], self.OBJECTS["mug"]["y"], 0.50))
            hazards.append((self.OBJECTS["bottle"]["x"], self.OBJECTS["bottle"]["y"], 0.50))
            hazards.append((self.OBJECTS["table"]["x"], self.OBJECTS["table"]["y"], 0.46))
        elif target_key == "mug":
            hazards.append((self.OBJECTS["bottle"]["x"], self.OBJECTS["bottle"]["y"], 0.34))
            hazards.append((self.OBJECTS["doorway"]["x"], self.OBJECTS["doorway"]["y"], 0.38))
        else:
            hazards.append((self.OBJECTS["mug"]["x"], self.OBJECTS["mug"]["y"], 0.34))
            hazards.append((self.OBJECTS["doorway"]["x"], self.OBJECTS["doorway"]["y"], 0.38))

        for hx, hy, r_inf in hazards:
            rx, ry = pose.x - hx, pose.y - hy
            dh = math.hypot(rx, ry)
            if 1e-4 < dh < r_inf:
                gain = 0.85 * ((r_inf - dh) / r_inf) ** 2 / max(0.14, dh)
                ux += (rx / dh) * gain
                uy += (ry / dh) * gain

        desired_yaw = math.atan2(uy, ux)
        yaw_err = math.atan2(math.sin(desired_yaw - pose.theta), math.cos(desired_yaw - pose.theta))
        yaw_err_deg = math.degrees(yaw_err)

        # R2-HIGH-03: Preserve closed-loop heading convergence — NEVER overwrite large negative yaw_err with +0.18!
        wz_nav = 1.35 * yaw_err
        if abs(yaw_err) <= 0.22:
            if hint == "LEFT":
                wz_nav += 0.10
            elif hint == "RIGHT":
                wz_nav -= 0.10
        wz = max(-0.75, min(0.75, wz_nav))
        # R4-MED-02: Strictly rotate in place (vx=0.0) whenever |yaw_error| > 14.0 deg before advancing!
        vx = 0.0 if abs(yaw_err_deg) > 14.0 else 0.18

        # Realistic Light-VLN pain point: falsely claims arrived=True prematurely around 1.2m~1.75m!
        premature_arrived = bool(self.force_vln_early_stop and (1.15 <= dist <= 1.75))
        return {
            "vx": vx,
            "vy": 0.0,
            "wz": wz,
            "arrived": premature_arrived,
            "vln_est_dist_m": round(dist, 3),
            "yaw_err_deg": round(yaw_err_deg, 2),
        }

    def unload_vln(self) -> VRAMStatus:
        if self._remote_vln.enabled:
            self._remote_vln.unload()
        with self._state_lock:
            self._vln_loaded = False
            self._recalc_vram(post_unload=True)
        gc.collect()
        return self.get_vram_status()

    def load_vla(self, model_name: str = "SmolVLA") -> VRAMStatus:
        is_estop = self.estop_active
        with self._state_lock:
            if not is_estop:
                self._arm_abort = False
            self._vla_loaded = model_name if model_name in VRAM_VLA_MODELS_GB else "SmolVLA"
            self._recalc_vram()
        return self.get_vram_status()

    def execute_vla_grasp(
        self,
        instruction: str,
        model_name: str = "SmolVLA",
        abort_check: Optional[Callable[[], bool]] = None,
        expected_estop_epoch: Optional[int] = None,
    ) -> Dict[str, Any]:
        for _ in range(12):
            stale_epoch = expected_estop_epoch is not None and self.estop_epoch != expected_estop_epoch
            with self._state_lock:
                arm_aborted = self._arm_abort
            if self.estop_active or stale_epoch or arm_aborted or (abort_check and abort_check()):
                return {"success": False, "aborted": True, "model": model_name}
            time.sleep(0.05)
        return {"success": True, "aborted": False, "model": model_name}

    def unload_vla(self) -> VRAMStatus:
        with self._state_lock:
            self._vla_loaded = None
            self._recalc_vram()
        gc.collect()
        return self.get_vram_status()

    def reset_peak_vram(self) -> VRAMStatus:
        with self._state_lock:
            self._vram_violation = False
            self._recalc_vram()
            self._peak_vram = self._allocated_vram
        return self.get_vram_status()

    def _recalc_vram(self, post_unload: bool = False) -> None:
        base = VRAM_POST_UNLOAD_GB if (post_unload or self._vla_loaded) else VRAM_IDLE_BASE_GB
        total = base
        if self._vln_loaded:
            total += VRAM_VLN_DELTA_GB
        if self._vla_loaded:
            total += VRAM_VLA_MODELS_GB.get(self._vla_loaded, 4.8)
        self._allocated_vram = round(total, 2)
        self._peak_vram = max(self._peak_vram, self._allocated_vram)
        if self._allocated_vram >= TOTAL_VRAM_GB:
            self._vram_violation = True

    def get_vram_status(self) -> VRAMStatus:
        with self._state_lock:
            models: List[str] = ["YOLOE-26s-seg"]
            if self._vln_loaded:
                models.append("Light-VLN")
            if self._vla_loaded:
                models.append(self._vla_loaded)
            return VRAMStatus(
                allocated_gb=self._allocated_vram,
                total_gb=TOTAL_VRAM_GB,
                peak_gb=self._peak_vram,
                active_models=models,
                vram_violation=self._vram_violation,
            )

    def shutdown(self) -> None:
        self._stop_evt.set()
        self.stop()
        if self._sim_thread.is_alive():
            self._sim_thread.join(timeout=0.5)
