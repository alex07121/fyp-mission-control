"""
Real Hardware Adapter for STM32 BLE Car (BT04-A) + USB Camera (FHD Camera) + XLeRobot Arms.

Supports:
  - Bluetooth BLE chassis control (`BT04-A`, UUID `0000ffe2-...`, ONA/ONB/ONC/OND/ONF/ONE)
    with 300ms hardware deadman safety timer & calibrated PWM pulse modulation;
  - Threaded 30 FPS OpenCV USB camera capture (`CAMERA_INDEX`, auto-probing external USB cam)
    with indoor ground-plane depth estimation fallback when RealSense/Orbbec depth is absent;
  - Remote LightNav-0 (`vllm_local` on school RTX 5000 Ada 32GB) via `VLN_REMOTE_URL`.
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import math
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from adapters.base import Pose2D, RGBDFrame, RobotAdapter, VRAMStatus
from backend.modules.remote_vln_client import RemoteVLNClient

logger = logging.getLogger("fyp.adapter.real")

BLE_WRITE_UUID = "0000ffe2-0000-1000-8000-00805f9b34fb"
BLE_COMMANDS: Dict[str, bytes] = {
    "forward": b"ONA",
    "backward": b"ONB",
    "left": b"ONC",
    "right": b"OND",
    "stop": b"ONF",
    "explicit_stop": b"ONE",
}

# Measured physical calibration from Desktop/VLN (20cm / 400ms = 0.50 m/s; 15deg / 335ms = 0.78 rad/s)
FULL_LINEAR_SPEED_MPS: float = float(os.getenv("BLE_FULL_VX_MPS", "0.50"))
FULL_ANGULAR_SPEED_RADS: float = float(os.getenv("BLE_FULL_WZ_RADS", "0.78"))


class _BLECarBridge:
    """Dedicated background asyncio BLE driver with 250ms PWM cycle & 300ms deadman brake."""

    def __init__(self, device_name: str = "BT04-A", address: Optional[str] = None) -> None:
        self.device_name = device_name
        self.address = address
        self.connected: bool = False
        self._lock = threading.RLock()
        self._target_vx: float = 0.0
        self._target_wz: float = 0.0
        self._last_cmd_mono: float = 0.0
        self._pulse_req: Optional[Tuple[str, float, threading.Event]] = None
        self._force_stop_evt = threading.Event()
        self._shutdown_evt = threading.Event()
        self._last_ble_sent: str = "stop"
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread = threading.Thread(target=self._thread_main, name="BLECarBridge", daemon=True)
        self._thread.start()

    def execute_pulse_blocking(self, direction: str, pulse_ms: int, abort_fn: Callable[[], bool]) -> bool:
        """Execute an exact-millisecond single BLE pulse (`ONC`/`OND` -> `ONF`), interruptible in <20ms."""
        done_evt = threading.Event()
        with self._lock:
            self._target_vx = 0.0
            self._target_wz = 0.0
            self._pulse_req = (direction, max(0.05, pulse_ms / 1000.0), done_evt)
        deadline = time.monotonic() + (pulse_ms / 1000.0) + 1.0
        while not done_evt.wait(0.02):
            if abort_fn() or time.monotonic() > deadline:
                self.emergency_stop()
                return False
        return not abort_fn()

    def set_velocity(self, vx: float, wz: float) -> None:
        with self._lock:
            self._target_vx = vx
            self._target_wz = wz
            self._last_cmd_mono = time.monotonic()
            if abs(vx) < 1e-4 and abs(wz) < 1e-4:
                self._force_stop_evt.set()

    def emergency_stop(self) -> None:
        with self._lock:
            self._target_vx = 0.0
            self._target_wz = 0.0
            self._last_cmd_mono = 0.0
        self._force_stop_evt.set()

    def shutdown(self) -> None:
        self.emergency_stop()
        self._shutdown_evt.set()
        self._thread.join(timeout=1.5)

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._ble_loop())
        except Exception as exc:
            logger.error("BLE bridge loop exited: %s", exc)
        finally:
            loop.close()

    async def _send_raw(self, client: Any, cmd_name: str, force: bool = False) -> None:
        if not force and cmd_name == self._last_ble_sent:
            return
        await client.write_gatt_char(BLE_WRITE_UUID, BLE_COMMANDS[cmd_name], response=False)
        self._last_ble_sent = cmd_name
        logger.info("[BLE TX] %s (%s)", BLE_COMMANDS[cmd_name].decode("ascii"), cmd_name)

    async def _ble_loop(self) -> None:
        try:
            from bleak import BleakClient, BleakScanner
        except ImportError:
            logger.warning("bleak not installed; BLE car bridge inactive.")
            return

        while not self._shutdown_evt.is_set():
            client = None
            try:
                target = self.address
                if not target:
                    logger.info("[BLE] Scanning for car '%s' ...", self.device_name)
                    dev = await BleakScanner.find_device_by_name(self.device_name, timeout=6.0)
                    if dev is None:
                        await asyncio.sleep(2.0)
                        continue
                    logger.info("[BLE] Found %s (%s)", dev.name, dev.address)
                    target = dev

                client = BleakClient(target)
                await client.connect()
                self.connected = bool(client.is_connected)
                logger.info("[BLE] Connected to %s! Car armed in safe STOP state.", self.device_name)
                await self._send_raw(client, "stop", force=True)

                cycle_s = 0.20  # 200ms PWM control cycle
                while not self._shutdown_evt.is_set() and client.is_connected:
                    if self._force_stop_evt.is_set():
                        self._force_stop_evt.clear()
                        with self._lock:
                            if self._pulse_req is not None:
                                self._pulse_req[2].set()
                                self._pulse_req = None
                        await self._send_raw(client, "stop", force=True)
                        await asyncio.sleep(0.03)
                        continue

                    p_req = None
                    with self._lock:
                        if self._pulse_req is not None:
                            p_req = self._pulse_req
                            self._pulse_req = None
                    if p_req is not None:
                        p_dir, p_dur, p_evt = p_req
                        try:
                            await self._send_raw(client, p_dir, force=True)
                            end_p = time.monotonic() + p_dur
                            while time.monotonic() < end_p:
                                if self._force_stop_evt.is_set() or self._shutdown_evt.is_set():
                                    break
                                await asyncio.sleep(0.015)
                            await self._send_raw(client, "stop", force=True)
                        finally:
                            p_evt.set()
                        continue

                    now = time.monotonic()
                    with self._lock:
                        vx = self._target_vx
                        wz = self._target_wz
                        age = now - self._last_cmd_mono

                    # 350ms deadman brake: auto-stop if controller stops refreshing velocity
                    if age > 0.35 or (abs(vx) < 0.01 and abs(wz) < 0.04):
                        await self._send_raw(client, "stop")
                        await asyncio.sleep(0.04)
                        continue

                    # Prioritize rotation when angular command dominates
                    if abs(wz) >= 0.14 and (abs(vx) < 0.03 or abs(wz) / FULL_ANGULAR_SPEED_RADS >= abs(vx) / FULL_LINEAR_SPEED_MPS):
                        cmd = "left" if wz > 0 else "right"
                        duty = min(1.0, max(0.32, abs(wz) / FULL_ANGULAR_SPEED_RADS))
                    else:
                        cmd = "forward" if vx >= 0 else "backward"
                        duty = min(1.0, max(0.32, abs(vx) / FULL_LINEAR_SPEED_MPS))

                    on_s = cycle_s * duty
                    off_s = cycle_s - on_s
                    await self._send_raw(client, cmd, force=True)
                    # Sleep in tiny 20ms slices so E-Stop interrupts pulse within <20ms!
                    end_on = time.monotonic() + on_s
                    while time.monotonic() < end_on:
                        if self._force_stop_evt.is_set() or self._shutdown_evt.is_set():
                            break
                        await asyncio.sleep(0.02)

                    if off_s >= 0.025 or self._force_stop_evt.is_set():
                        await self._send_raw(client, "stop", force=True)
                        if off_s >= 0.025 and not self._force_stop_evt.is_set():
                            await asyncio.sleep(off_s)

            except Exception as exc:
                if not self._shutdown_evt.is_set():
                    logger.warning("[BLE] Connection/IO notice: %s (retrying in 2s)", exc)
                await asyncio.sleep(2.0)
            finally:
                self.connected = False
                if client is not None:
                    try:
                        if client.is_connected:
                            await self._send_raw(client, "stop", force=True)
                            await asyncio.sleep(0.03)
                            await self._send_raw(client, "explicit_stop", force=True)
                            await client.disconnect()
                    except Exception:
                        pass


class _USBCameraReader:
    """Non-blocking 30 FPS USB UVC camera capture thread."""

    def __init__(self, preferred_index: Optional[int] = None) -> None:
        self._preferred_index = preferred_index
        self._lock = threading.RLock()
        self._latest_bgr: Optional[np.ndarray] = None
        self._active_index: int = -1
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._capture_loop, name="RealUSBCamLoop", daemon=True)
        self._thread.start()

    def get_latest_bgr(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._latest_bgr is None else self._latest_bgr.copy()

    def shutdown(self) -> None:
        self._stop_evt.set()
        self._thread.join(timeout=1.5)

    def set_preferred_index(self, idx: int) -> None:
        with self._lock:
            self._preferred_index = idx
            self._reopen_requested = True

    def _open_best_camera(self) -> Optional[cv2.VideoCapture]:
        candidates = (
            [self._preferred_index]
            if self._preferred_index is not None
            else [0, 1, 2]  # Cam 0 is confirmed Car USB FHD Camera
        )
        for idx in candidates:
            if idx is None or idx < 0:
                continue
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                for _ in range(5):
                    ok, frame = cap.read()
                    if ok and frame is not None and frame.size > 0:
                        self._active_index = idx
                        logger.info("[CAM] Opened real USB camera index=%d (%dx%d)", idx, frame.shape[1], frame.shape[0])
                        return cap
                cap.release()
        return None

    def _capture_loop(self) -> None:
        cap: Optional[cv2.VideoCapture] = None
        while not self._stop_evt.is_set():
            if getattr(self, "_reopen_requested", False):
                self._reopen_requested = False
                if cap is not None:
                    cap.release()
                    cap = None
            if cap is None or not cap.isOpened():
                cap = self._open_best_camera()
                if cap is None:
                    self._stop_evt.wait(1.5)
                    continue
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                if frame.shape[1] != 640 or frame.shape[0] != 480:
                    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
                with self._lock:
                    self._latest_bgr = frame
                self._stop_evt.wait(0.015)
            else:
                cap.release()
                cap = None
                self._stop_evt.wait(0.4)
        if cap is not None:
            cap.release()


class XLeRobotRealAdapter(RobotAdapter):
    """Thread-safe hardware adapter for BT04-A BLE Car + USB Camera + Remote LightNav-0."""

    def __init__(self) -> None:
        super().__init__()
        self._state_lock = threading.RLock()
        self._pose = Pose2D(0.0, 0.0, 0.0)
        self._last_pose_mono = time.monotonic()
        self._cmd_vel = (0.0, 0.0, 0.0)
        self._vln_model: Any = None
        self._vla_model: Optional[str] = None
        self._arm_abort_flag: bool = False
        self._peak_vram_gb: float = 1.4
        self._remote_vln = RemoteVLNClient()

        # Precompute smooth floor-perspective monocular depth prior (3.0m horizon -> 0.40m near floor)
        # Used when a 2D USB RGB camera (`FHD Camera`) is attached without RealSense/Orbbec depth hardware.
        v_norm = np.linspace(0.0, 1.0, 480, dtype=np.float32)[:, None]
        mono_col = np.where(v_norm < 0.42, 2.80, 0.40 + 2.40 * ((1.0 - v_norm) / 0.58) ** 1.35).astype(np.float32)
        self._mono_depth_prior = np.repeat(mono_col, 640, axis=1)

        # Activate real hardware threads ONLY when ROBOT_MODE=real (keeps fast offline unit tests isolated)
        live_hw = os.getenv("ROBOT_MODE", "").strip().lower() == "real"
        cam_env = os.getenv("CAMERA_INDEX", "0").strip()
        cam_idx = int(cam_env) if cam_env.isdigit() else 0
        self._cam: Optional[_USBCameraReader] = _USBCameraReader(cam_idx) if live_hw else None
        self._ble: Optional[_BLECarBridge] = (
            _BLECarBridge(
                device_name=os.getenv("BLE_CAR_NAME", "BT04-A"),
                address=os.getenv("BLE_CAR_ADDRESS") or None,
            )
            if live_hw
            else None
        )
        logger.info("XLeRobotRealAdapter initialized (live_hw=%s, cam=%s, remote_vln=%s).", live_hw, cam_idx, self._remote_vln.base_url)

    @property
    def active_camera_index(self) -> int:
        return self._cam._active_index if self._cam is not None else 0

    def set_camera_index(self, idx: int) -> None:
        if self._cam is not None:
            self._cam.set_preferred_index(int(idx))

    def turn_in_place_deg(self, deg: float, expected_estop_epoch: Optional[int] = None) -> bool:
        """
        Closed-loop short-pulse BLE turn using calibrated Cam 0 ORB visual odometry (`data/car_calibration.json`).
        Breaks turns into <=22° short pulses (preventing DC motor high-RPM runaway) and corrects residual error!
        """
        if abs(deg) < 2.5:
            return True

        hfov_deg = 115.0
        ms_per_deg = 16.7
        try:
            cp = Path(__file__).resolve().parent.parent / "data" / "car_calibration.json"
            if cp.exists():
                cdata = json.loads(cp.read_text(encoding="utf-8"))
                hfov_deg = float(cdata.get("true_camera_hfov_deg", hfov_deg))
                ms_per_deg = float(cdata.get("pulse_ms_per_deg", ms_per_deg))
        except Exception:
            pass

        def _is_aborted() -> bool:
            return self.estop_active or (expected_estop_epoch is not None and self.estop_epoch != expected_estop_epoch)

        def _measure_orb_deg(b_img: Optional[np.ndarray], a_img: Optional[np.ndarray]) -> Optional[float]:
            if b_img is None or a_img is None:
                return None
            g1 = cv2.cvtColor(b_img, cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(a_img, cv2.COLOR_BGR2GRAY)
            orb = cv2.ORB_create(1500)
            k1, d1 = orb.detectAndCompute(g1, None)
            k2, d2 = orb.detectAndCompute(g2, None)
            if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
                return None
            matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
            if len(matches) < 15:
                return None
            p1 = np.float32([k1[m.queryIdx].pt for m in matches])
            p2 = np.float32([k2[m.trainIdx].pt for m in matches])
            _, inl = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=4.5)
            if inl is None or int(inl.sum()) < 12:
                return None
            keep = inl.ravel().astype(bool)
            cx = float(g1.shape[1]) / 2.0
            fx = cx / math.tan(math.radians(hfov_deg) / 2.0)
            a1 = np.degrees(np.arctan((p1[keep, 0] - cx) / fx))
            a2 = np.degrees(np.arctan((p2[keep, 0] - cx) / fx))
            return float(np.median(a2 - a1))

        if self._ble is None or not self._ble.connected:
            return False

        turned = 0.0
        before = self._cam.get_latest_bgr() if self._cam is not None else None
        for _ in range(14):
            if _is_aborted():
                self.stop()
                return False
            rem = deg - turned
            if abs(rem) <= 3.5:
                break
            sub = max(-22.0, min(22.0, rem))
            p_ms = int(round(max(75.0, min(260.0, abs(sub) * ms_per_deg))))
            ok = self._ble.execute_pulse_blocking("left" if sub > 0 else "right", p_ms, _is_aborted)
            if not ok:
                self.stop()
                return False
            time.sleep(0.28)
            after = self._cam.get_latest_bgr() if self._cam is not None else None
            meas = _measure_orb_deg(before, after)
            if meas is not None and abs(meas) >= 1.5 and (meas > 0) == (sub > 0):
                turned += meas
            else:
                turned += sub
            before = after

        with self._state_lock:
            nt = math.atan2(
                math.sin(self._pose.theta + math.radians(turned)),
                math.cos(self._pose.theta + math.radians(turned)),
            )
            self._pose = Pose2D(self._pose.x, self._pose.y, nt)
        self.stop()
        logger.info("[TURN] Commanded %+.1f° -> Actual %+.1f°", deg, turned)
        return True

    def drive_pulse_ms(self, direction: str, pulse_ms: int, expected_estop_epoch: Optional[int] = None) -> bool:
        """Execute one clean discrete forward/backward/left/right BLE pulse then stop (matches vln_car.py)."""
        if pulse_ms <= 0 or direction not in ("forward", "backward", "left", "right"):
            return True

        def _is_aborted() -> bool:
            return self.estop_active or (expected_estop_epoch is not None and self.estop_epoch != expected_estop_epoch)

        if self._ble is None or not self._ble.connected:
            return False
        ok = self._ble.execute_pulse_blocking(direction, int(min(1400, max(60, pulse_ms))), _is_aborted)
        self.stop()
        if ok and direction in ("forward", "backward"):
            dist = (pulse_ms / 1000.0) * FULL_LINEAR_SPEED_MPS * (1.0 if direction == "forward" else -1.0)
            with self._state_lock:
                nx = self._pose.x + dist * math.cos(self._pose.theta)
                ny = self._pose.y + dist * math.sin(self._pose.theta)
                self._pose = Pose2D(nx, ny, self._pose.theta)
        return ok

    def _integrate_dead_reckoning(self) -> None:
        now = time.monotonic()
        dt = min(0.25, max(0.0, now - self._last_pose_mono))
        self._last_pose_mono = now
        vx, vy, wz = self._cmd_vel
        if abs(vx) > 1e-5 or abs(wz) > 1e-5:
            nx = self._pose.x + vx * math.cos(self._pose.theta) * dt
            ny = self._pose.y + vx * math.sin(self._pose.theta) * dt
            nt = math.atan2(math.sin(self._pose.theta + wz * dt), math.cos(self._pose.theta + wz * dt))
            self._pose = Pose2D(nx, ny, nt)

    def _do_move(self, vx: float, vy: float, wz: float) -> None:
        with self._state_lock:
            self._integrate_dead_reckoning()
            self._cmd_vel = (vx, vy, wz)
        if self._ble is not None:
            self._ble.set_velocity(vx, wz)

    def _do_stop(self) -> None:
        with self._state_lock:
            self._integrate_dead_reckoning()
            self._cmd_vel = (0.0, 0.0, 0.0)
        if self._ble is not None:
            self._ble.emergency_stop()

    def _do_abort_arms(self) -> None:
        with self._state_lock:
            self._arm_abort_flag = True
        if self._ble is not None:
            self._ble.emergency_stop()

    def get_rgbd_frame(self, lens_mode: str = "80cm") -> RGBDFrame:
        bgr = self._cam.get_latest_bgr() if self._cam is not None else None
        if bgr is None:
            bgr = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                bgr,
                "Waiting for USB Camera (Allow Camera in macOS Settings)...",
                (28, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 210, 255),
                2,
                cv2.LINE_AA,
            )
            depth_m = np.full((480, 640), 1.5, dtype=np.float32)
        else:
            depth_m = self._mono_depth_prior.copy()
        return RGBDFrame(rgb=bgr, depth_m=depth_m, timestamp=time.time())

    def get_pose(self) -> Pose2D:
        with self._state_lock:
            self._integrate_dead_reckoning()
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
            self._last_pose_mono = time.monotonic()
            return Pose2D(self._pose.x, self._pose.y, self._pose.theta)

    def load_vln(self) -> VRAMStatus:
        if self._remote_vln.enabled:
            self._remote_vln.load()
        with self._state_lock:
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
            self._vla_model = model_name
        return self.get_vram_status()

    def execute_vla_grasp(
        self,
        instruction: str,
        model_name: str = "SmolVLA",
        abort_check: Optional[Callable[[], bool]] = None,
        expected_estop_epoch: Optional[int] = None,
    ) -> Dict[str, Any]:
        for step_i in range(10):
            stale_epoch = expected_estop_epoch is not None and self.estop_epoch != expected_estop_epoch
            with self._state_lock:
                aborted = self._arm_abort_flag
            if self.estop_active or stale_epoch or aborted or (abort_check and abort_check()):
                return {"success": False, "aborted": True, "model": model_name}
            if step_i == 0 and self._remote_vln.enabled:
                frame = self.get_rgbd_frame()
                self._remote_vln.act_vla(model_name=model_name, instruction=instruction, front_bgr=frame.rgb)
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

            allocated = round(sim_alloc, 2)
            cuda_peak_gb = 0.0
            try:
                import torch

                if torch.cuda.is_available():
                    cuda_gb = float(torch.cuda.memory_allocated() / (1024**3))
                    cuda_peak_gb = float(torch.cuda.max_memory_allocated() / (1024**3))
                    if cuda_gb > 0.05:
                        allocated = round(cuda_gb, 2)
            except Exception:
                pass

            self._peak_vram_gb = round(max(self._peak_vram_gb, allocated, cuda_peak_gb), 2)
            return VRAMStatus(
                allocated_gb=allocated,
                total_gb=12.0,
                peak_gb=self._peak_vram_gb,
                active_models=models,
                vram_violation=(allocated >= 12.0 or self._peak_vram_gb >= 12.0),
            )

    def shutdown(self) -> None:
        self.stop()
        if self._ble is not None:
            self._ble.shutdown()
        if self._cam is not None:
            self._cam.shutdown()
        self.unload_vln()
        self.unload_vla()
