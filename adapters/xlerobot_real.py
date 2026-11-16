"""
Real Hardware Adapter Template for XLeRobot + Dual SO-ARM101 (Ubuntu + RTX 5070 12GB).

To switch from simulation to real robot on the Ubuntu laptop:
1. Fill in the minimal LeRobot / RealSense / Orbbec hooks marked with `TODO(REAL_ROBOT)` below.
2. Start Mission Control with `ROBOT_MODE=real python run.py`.
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from adapters.base import Pose2D, RGBDFrame, RobotAdapter, VRAMStatus
from backend.modules.remote_vln_client import RemoteVLNClient

logger = logging.getLogger("fyp.adapter.real")


class XLeRobotRealAdapter(RobotAdapter):
    """Minimal thread-safe plug-and-play hardware wrapper for XLeRobot + SO-ARM101."""

    def __init__(self) -> None:
        super().__init__()
        self._state_lock = threading.RLock()
        self._pose = Pose2D(0.0, 0.0, 0.0)
        self._vln_model: Any = None
        self._vla_model: Optional[str] = None
        self._arm_abort_flag: bool = False
        self._peak_vram_gb: float = 1.4
        self._remote_vln = RemoteVLNClient()
        # TODO(REAL_ROBOT): Connect LeRobot XLeRobot base & RGB-D camera here
        logger.info("XLeRobotRealAdapter initialized (awaiting hardware driver hooks).")

    def _do_move(self, vx: float, vy: float, wz: float) -> None:
        # TODO(REAL_ROBOT): self.robot.send_base_velocity(vx=vx, vy=vy, wz=wz)
        logger.debug("Real base move command: vx=%.3f, vy=%.3f, wz=%.3f", vx, vy, wz)

    def _do_stop(self) -> None:
        # TODO(REAL_ROBOT): self.robot.stop_base()
        logger.debug("Real base STOP command issued.")

    def _do_abort_arms(self) -> None:
        with self._state_lock:
            self._arm_abort_flag = True
        # TODO(REAL_ROBOT): self.robot.hold_arms_position()

    def get_rgbd_frame(self, lens_mode: str = "80cm") -> RGBDFrame:
        # TODO(REAL_ROBOT): Read aligned BGR (uint8, 480x640x3) + Depth (float32, meters)
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        depth_m = np.full((480, 640), 1.5, dtype=np.float32)
        return RGBDFrame(rgb=rgb, depth_m=depth_m, timestamp=time.time())

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
            self._pose = Pose2D(x, y, theta)
            return Pose2D(self._pose.x, self._pose.y, self._pose.theta)

    def load_vln(self) -> VRAMStatus:
        if self._remote_vln.enabled:
            self._remote_vln.load()
        with self._state_lock:
            # TODO(REAL_ROBOT): Load Light-VLN checkpoint onto CUDA device (if not using VLN_REMOTE_URL)
            self._vln_model = True
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
        # TODO(REAL_ROBOT): Run single Light-VLN step and return (vx, wz, arrived)
        wz_map = {"LEFT": 0.25, "RIGHT": -0.25, "STRAIGHT": 0.0, "STOP": 0.0}
        vx = 0.0 if agent_direction_hint.upper() == "STOP" else 0.16
        return {"vx": vx, "vy": 0.0, "wz": wz_map.get(agent_direction_hint.upper(), 0.0), "arrived": False}

    def unload_vln(self) -> VRAMStatus:
        if self._remote_vln.enabled:
            self._remote_vln.unload()
        with self._state_lock:
            self._vln_model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        return self.get_vram_status()

    def load_vla(self, model_name: str = "SmolVLA") -> VRAMStatus:
        is_estop = self.estop_active
        with self._state_lock:
            if not is_estop:
                self._arm_abort_flag = False
            # TODO(REAL_ROBOT): Load SmolVLA or pi0.5 policy via LeRobot
            self._vla_model = model_name
        return self.get_vram_status()

    def execute_vla_grasp(
        self,
        instruction: str,
        model_name: str = "SmolVLA",
        abort_check: Optional[Callable[[], bool]] = None,
        expected_estop_epoch: Optional[int] = None,
    ) -> Dict[str, Any]:
        # R2-MED-03: NEVER clear `_arm_abort_flag` here — preserve any prior E-Stop arm abort latch
        # and strictly enforce `expected_estop_epoch` gatekeeper consistency!
        for _ in range(10):
            stale_epoch = expected_estop_epoch is not None and self.estop_epoch != expected_estop_epoch
            with self._state_lock:
                aborted = self._arm_abort_flag
            if self.estop_active or stale_epoch or aborted or (abort_check and abort_check()):
                return {"success": False, "aborted": True, "model": model_name}
            time.sleep(0.05)
        return {"success": True, "aborted": False, "model": model_name}

    def unload_vla(self) -> VRAMStatus:
        with self._state_lock:
            self._vla_model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        return self.get_vram_status()

    def reset_peak_vram(self) -> VRAMStatus:
        with self._state_lock:
            self._peak_vram_gb = 0.0
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        return self.get_vram_status()

    def get_vram_status(self) -> VRAMStatus:
        with self._state_lock:
            models: List[str] = ["YOLOE-26s-seg"]
            sim_alloc = 1.4
            if self._vln_model is not None:
                models.append("Light-VLN")
                sim_alloc += 5.4
            if self._vla_model is not None:
                models.append(str(self._vla_model))
                sim_alloc += 4.8 if self._vla_model == "SmolVLA" else 5.6

            allocated = sim_alloc
            cuda_peak_gb = 0.0
            try:
                import torch

                if torch.cuda.is_available():
                    cuda_gb = float(torch.cuda.memory_allocated() / (1024**3))
                    cuda_peak_gb = float(torch.cuda.max_memory_allocated() / (1024**3))
                    if cuda_gb > 0.05:
                        allocated = cuda_gb
            except Exception:
                pass

            self._peak_vram_gb = max(self._peak_vram_gb, allocated, cuda_peak_gb)
            return VRAMStatus(
                allocated_gb=allocated,
                total_gb=12.0,
                peak_gb=self._peak_vram_gb,
                active_models=models,
                vram_violation=(allocated >= 12.0 or self._peak_vram_gb >= 12.0),
            )

    def shutdown(self) -> None:
        self.stop()
        self.unload_vln()
        self.unload_vla()
