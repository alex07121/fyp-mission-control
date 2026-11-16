"""Abstract base RobotAdapter with epoch-protected atomic Motor Gatekeeper E-Stop latch."""

from __future__ import annotations

import abc
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("fyp.adapter")


@dataclass
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0  # radians

    def to_dict(self) -> Dict[str, float]:
        return {
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "theta_rad": round(self.theta, 4),
            "theta_deg": round(math.degrees(self.theta), 2),
        }


@dataclass
class VelocityCommand:
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "vx": round(self.vx, 4),
            "vy": round(self.vy, 4),
            "wz": round(self.wz, 4),
        }


@dataclass
class RGBDFrame:
    rgb: np.ndarray  # uint8 BGR image (H, W, 3)
    depth_m: np.ndarray  # float32 aligned depth map in meters (H, W)
    timestamp: float
    fx: float = 510.0
    fy: float = 510.0
    cx: float = 320.0
    cy: float = 240.0


@dataclass
class VRAMStatus:
    allocated_gb: float = 1.4
    total_gb: float = 12.0
    peak_gb: float = 1.4
    active_models: List[str] = field(default_factory=lambda: ["YOLOE-26s-seg"])
    vram_violation: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allocated_gb": round(self.allocated_gb, 2),
            "total_gb": round(self.total_gb, 2),
            "peak_gb": round(self.peak_gb, 2),
            "utilization_pct": round((self.allocated_gb / max(0.1, self.total_gb)) * 100.0, 1),
            "active_models": list(self.active_models),
            "vram_violation": self.vram_violation,
        }


class RobotAdapter(abc.ABC):
    """
    Base interface for XLeRobot + SO-ARM101 dual-arm mobile manipulator.

    Enforces a monotonic epoch-guarded Motor Gatekeeper Latch (`_estop_active`, `_estop_epoch`):
    1. When `_estop_active` is True, ALL non-zero `move()` calls are blocked & forced to zero.
    2. When `expected_estop_epoch` is passed by a mission worker, ANY mismatch with current
       `_estop_epoch` permanently blocks that stale zombie worker even AFTER `reset_estop()`!
    3. `reset_estop(expected_epoch)` refuses to clear E-Stop if a newer E-Stop arrived concurrently.
    """

    def __init__(self) -> None:
        self._gate_lock = threading.RLock()
        self._estop_active: bool = False
        self._estop_epoch: int = 0
        self._estop_reason: str = ""
        self._estop_timestamp: Optional[float] = None
        self._blocked_move_attempts: int = 0
        self._current_vel = VelocityCommand(0.0, 0.0, 0.0)

    @property
    def estop_active(self) -> bool:
        with self._gate_lock:
            return self._estop_active

    @property
    def estop_epoch(self) -> int:
        with self._gate_lock:
            return self._estop_epoch

    def get_gatekeeper_status(self) -> Dict[str, Any]:
        with self._gate_lock:
            return {
                "estop_active": self._estop_active,
                "estop_epoch": self._estop_epoch,
                "estop_reason": self._estop_reason,
                "estop_timestamp": self._estop_timestamp,
                "blocked_move_attempts": self._blocked_move_attempts,
                "velocity": self._current_vel.to_dict(),
            }

    def move(
        self,
        vx: float,
        vy: float = 0.0,
        wz: float = 0.0,
        expected_estop_epoch: Optional[int] = None,
    ) -> bool:
        """
        Thread-safe velocity command gated by both `_estop_active` and `_estop_epoch`.
        Returns True if command was applied, False if blocked/clamped to stop.
        """
        if not (math.isfinite(vx) and math.isfinite(vy) and math.isfinite(wz)):
            logger.warning("Rejected non-finite velocity command: (%s, %s, %s)", vx, vy, wz)
            self.stop()
            return False

        with self._gate_lock:
            stale_epoch = (expected_estop_epoch is not None) and (expected_estop_epoch != self._estop_epoch)
            if self._estop_active or stale_epoch:
                if abs(vx) > 1e-9 or abs(vy) > 1e-9 or abs(wz) > 1e-9:
                    self._blocked_move_attempts += 1
                    if self._blocked_move_attempts == 1 or self._blocked_move_attempts % 500 == 0:
                        logger.warning(
                            "[GATEKEEPER] Blocked move(%.3f, %.3f, %.3f) [estop=%s, stale_epoch=%s] (#%d)",
                            vx,
                            vy,
                            wz,
                            self._estop_active,
                            stale_epoch,
                            self._blocked_move_attempts,
                        )
                self._current_vel = VelocityCommand(0.0, 0.0, 0.0)
                self._do_stop()
                return False

            clamped_vx = max(-0.45, min(0.45, float(vx)))
            clamped_vy = max(-0.35, min(0.35, float(vy)))
            clamped_wz = max(-1.80, min(1.80, float(wz)))
            self._current_vel = VelocityCommand(clamped_vx, clamped_vy, clamped_wz)
            self._do_move(clamped_vx, clamped_vy, clamped_wz)
            return True

    def stop(self) -> None:
        """Immediately command zero velocity under gatekeeper lock."""
        with self._gate_lock:
            self._current_vel = VelocityCommand(0.0, 0.0, 0.0)
            self._do_stop()

    def trigger_estop(self, reason: str = "Manual E-STOP") -> Dict[str, Any]:
        """Atomically increment E-Stop epoch, latch E-Stop, halt base, and abort arms."""
        with self._gate_lock:
            self._estop_epoch += 1
            self._estop_active = True
            self._estop_reason = reason
            self._estop_timestamp = time.time()
            self._current_vel = VelocityCommand(0.0, 0.0, 0.0)
            self._do_stop()
            self._do_abort_arms()
            logger.critical("[E-STOP LATCHED epoch=%d] Reason: %s", self._estop_epoch, reason)
            return self.get_gatekeeper_status()

    def reset_estop(self, expected_epoch: Optional[int] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Release E-Stop latch ONLY if no newer E-Stop occurred since `expected_epoch`.
        Returns `(unlocked_ok, gatekeeper_status)`.
        """
        with self._gate_lock:
            self._current_vel = VelocityCommand(0.0, 0.0, 0.0)
            self._do_stop()
            if expected_epoch is not None and expected_epoch != self._estop_epoch:
                logger.warning(
                    "[E-STOP RESET ABORTED] Newer E-Stop epoch %d superseded reset target %d!",
                    self._estop_epoch,
                    expected_epoch,
                )
                return False, self.get_gatekeeper_status()
            self._estop_active = False
            self._estop_reason = ""
            logger.info("[E-STOP RESET epoch=%d] Motor gatekeeper unlocked.", self._estop_epoch)
            return True, self.get_gatekeeper_status()

    def get_velocity(self) -> VelocityCommand:
        with self._gate_lock:
            return VelocityCommand(
                self._current_vel.vx,
                self._current_vel.vy,
                self._current_vel.wz,
            )

    @abc.abstractmethod
    def _do_move(self, vx: float, vy: float, wz: float) -> None:
        """Low-level hardware velocity write."""

    @abc.abstractmethod
    def _do_stop(self) -> None:
        """Low-level hardware immediate zero-velocity brake."""

    @abc.abstractmethod
    def _do_abort_arms(self) -> None:
        """Abort dual SO-ARM101 follower arm execution safely."""

    @abc.abstractmethod
    def get_rgbd_frame(self, lens_mode: str = "80cm") -> RGBDFrame:
        """Acquire aligned 640x480 BGR image and float32 depth map in meters."""

    @abc.abstractmethod
    def get_pose(self) -> Pose2D:
        """Return estimated 2D robot odometry pose."""

    @abc.abstractmethod
    def reset_pose(
        self,
        x: float = 0.0,
        y: float = 0.0,
        theta: float = 0.0,
        scenario: Optional[str] = None,
    ) -> Pose2D:
        """Reset robot pose or simulation scenario."""

    @abc.abstractmethod
    def load_vln(self) -> VRAMStatus:
        """Load Light-VLN navigation model onto GPU."""

    @abc.abstractmethod
    def step_vln(self, instruction: str, agent_direction_hint: str = "STRAIGHT") -> Dict[str, Any]:
        """Execute one Light-VLN navigation inference step."""

    @abc.abstractmethod
    def unload_vln(self) -> VRAMStatus:
        """Unload Light-VLN model and reclaim GPU VRAM before VLA execution."""

    @abc.abstractmethod
    def load_vla(self, model_name: str = "SmolVLA") -> VRAMStatus:
        """Load SmolVLA or pi0.5 dual-arm manipulation policy onto GPU."""

    @abc.abstractmethod
    def execute_vla_grasp(
        self,
        instruction: str,
        model_name: str = "SmolVLA",
        abort_check: Optional[Callable[[], bool]] = None,
        expected_estop_epoch: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute closed-loop dual SO-ARM101 grasping policy."""

    @abc.abstractmethod
    def unload_vla(self) -> VRAMStatus:
        """Unload VLA policy and return to idle baseline VRAM."""

    @abc.abstractmethod
    def reset_peak_vram(self) -> VRAMStatus:
        """Reset per-mission peak VRAM watermark to current allocated VRAM."""

    @abc.abstractmethod
    def get_vram_status(self) -> VRAMStatus:
        """Query current and peak GPU VRAM usage."""

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Cleanly stop motors, release camera/GPU handles, and terminate threads."""
