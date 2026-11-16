"""Module 4: Epoch-Guarded FSM Orchestrator + 12GB VRAM Time-Sharing + Zero-Race E-Stop."""

from __future__ import annotations

import asyncio
import collections
import gc
import logging
import math
import threading
import time
from typing import Any, Deque, Dict, List, Optional, Tuple

from adapters.base import RobotAdapter
from backend.config import LENS_PROFILES
from backend.modules.bci_gateway import sanitize_bci_field
from backend.modules.cloud_agent import CloudAgentEngine
from backend.modules.data_logger import MissionDataLogger
from backend.modules.vision_depth import VisionDepthCalibrator

logger = logging.getLogger("fyp.fsm")

FSM_STATES = ("IDLE", "AGENT_PARSE", "VLN_CRUISE", "YOLO_LAST_MILE", "VRAM_SWAP_AND_VLA")


class FSMController:
    """
    Coordinates BCI tasks with strict lock ordering (`self._lock` -> `adapter._gate_lock`),
    monotonic cancellation epochs (`_mission_epoch`, `estop_epoch`), and hardware ownership (`_owner_mission_id`).
    """

    def __init__(
        self,
        adapter: RobotAdapter,
        vision: VisionDepthCalibrator,
        agent: CloudAgentEngine,
        data_logger: MissionDataLogger,
    ) -> None:
        self.adapter = adapter
        self.vision = vision
        self.agent = agent
        self.data_logger = data_logger

        self._lock = threading.RLock()
        self._state: str = "IDLE"
        self._mission_epoch: int = 0
        self._owner_mission_id: int = 0
        self._cancel_cause: str = ""
        self._active_task: Optional[Dict[str, Any]] = None
        self._abort_evt = threading.Event()
        self._shutdown_evt = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        self.cruise_speed_mps: float = 0.18
        self.agent_interval_s: float = 1.50
        self.blind_walk_enabled: bool = False
        self.vla_model: str = "SmolVLA"
        self.vln_cruise_timeout_s: float = 18.0

        self._events: Deque[Dict[str, Any]] = collections.deque(maxlen=100)
        self._vln_rejections_current: int = 0
        self._last_abort_reason: str = "none"
        self._latest_jpeg: bytes = b""
        self._last_completed_pose: Optional[Tuple[float, float]] = None
        self._last_completed_blind_walk: bool = False

        self._cam_thread = threading.Thread(target=self._camera_hud_loop, name="CamHUDLoop", daemon=True)
        self._cam_thread.start()

    @staticmethod
    def _release_gpu_cache() -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _log_event(self, level: str, msg: str) -> None:
        entry = {"ts": round(time.time(), 3), "time_str": time.strftime("%H:%M:%S"), "level": level, "msg": msg}
        with self._lock:
            self._events.appendleft(entry)
        logger.info("[%s] %s", level, msg)

    def _camera_hud_loop(self) -> None:
        while not self._shutdown_evt.wait(0.065):
            try:
                frame = self.adapter.get_rgbd_frame(self.vision.lens_mode)
                meas = self.vision.analyze_frame(frame, update_history=False)
                jpg = self.vision.render_annotated_jpeg(frame, meas, fsm_state=self.state)
                with self._lock:
                    self._latest_jpeg = jpg
            except Exception as exc:  # pragma: no cover
                logger.debug("HUD loop error: %s", exc)

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def get_latest_jpeg(self) -> bytes:
        with self._lock:
            if self._latest_jpeg:
                return self._latest_jpeg
        frame = self.adapter.get_rgbd_frame(self.vision.lens_mode)
        meas = self.vision.analyze_frame(frame, update_history=False)
        return self.vision.render_annotated_jpeg(frame, meas, fsm_state=self.state)

    def trigger_estop(self, reason: str = "Manual E-STOP") -> Dict[str, Any]:
        """
        R2-CRIT-04 + R2-HIGH-02: Single outer `self._lock` wraps motor gatekeeper latch + epoch bump.
        If no worker is alive, immediately reclaim any VLN/VLA VRAM and ensure IDLE state.
        """
        clean_reason = sanitize_bci_field(reason, default="Manual E-STOP", max_len=100)
        with self._lock:
            gate = self.adapter.trigger_estop(clean_reason)
            self._mission_epoch += 1
            self._cancel_cause = "estop"
            self._last_completed_pose = None
            self._last_completed_blind_walk = False
            self._abort_evt.set()
            if self._worker_thread is None or not self._worker_thread.is_alive():
                self.adapter.unload_vln()
                self.adapter.unload_vla()
                self._release_gpu_cache()
                self._state = "IDLE"
                self._active_task = None
        self._log_event("CRITICAL", f"E-STOP LATCHED (epoch={gate['estop_epoch']}): {clean_reason}")
        return gate

    def reset_estop(self) -> Dict[str, Any]:
        """Synchronous atomic epoch-checked E-Stop reset with guaranteed VLN/VLA VRAM cleanup."""
        with self._lock:
            target_e_epoch = self.adapter.estop_epoch
            pre_owner_id = self._owner_mission_id
            self._mission_epoch += 1
            self._cancel_cause = "estop"
            self._abort_evt.set()
            worker = self._worker_thread

        if worker and worker.is_alive():
            deadline = time.monotonic() + 0.25
            while worker.is_alive() and time.monotonic() < deadline:
                if self.adapter.estop_epoch != target_e_epoch:
                    break
                time.sleep(0.01)

        with self._lock:
            unlocked, gate = self.adapter.reset_estop(expected_epoch=target_e_epoch)
            if not unlocked:
                return {**gate, "reset_aborted_due_to_new_estop": True}
            if self.adapter.estop_epoch == target_e_epoch and not self.adapter.estop_active:
                if self._owner_mission_id == pre_owner_id:
                    self.adapter.unload_vln()
                    self.adapter.unload_vla()
                    self._release_gpu_cache()
                    self._state = "IDLE"
                    self._active_task = None
                    self._abort_evt.clear()
        self._log_event("INFO", f"E-STOP reset cleanly (epoch={target_e_epoch}).")
        return gate

    async def reset_estop_async(self) -> Dict[str, Any]:
        """Non-blocking async E-Stop reset immune to threadpool starvation and split-lock races."""
        with self._lock:
            target_e_epoch = self.adapter.estop_epoch
            pre_owner_id = self._owner_mission_id
            self._mission_epoch += 1
            self._cancel_cause = "estop"
            self._abort_evt.set()
            worker = self._worker_thread

        if worker and worker.is_alive():
            deadline = time.monotonic() + 0.25
            while worker.is_alive() and time.monotonic() < deadline:
                if self.adapter.estop_epoch != target_e_epoch:
                    break
                await asyncio.sleep(0.01)

        with self._lock:
            unlocked, gate = self.adapter.reset_estop(expected_epoch=target_e_epoch)
            if not unlocked:
                return {**gate, "reset_aborted_due_to_new_estop": True}
            if self.adapter.estop_epoch == target_e_epoch and not self.adapter.estop_active:
                if self._owner_mission_id == pre_owner_id:
                    self.adapter.unload_vln()
                    self.adapter.unload_vla()
                    self._release_gpu_cache()
                    self._state = "IDLE"
                    self._active_task = None
                    self._abort_evt.clear()
        self._log_event("INFO", f"E-STOP reset cleanly (epoch={target_e_epoch}).")
        return gate

    async def abort_active_mission_for_pose_reset(self) -> None:
        """
        R2-CRIT-04 + R3-MED-02: Safely cancel any active mission & unload VRAM for `/api/robot/reset`
        WITHOUT ever unlocking a latched E-Stop, and tag cancellation as `pose_reset` (not `estop_interrupted`).
        """
        with self._lock:
            pre_owner_id = self._owner_mission_id
            self._last_completed_pose = None
            self._last_completed_blind_walk = False
            had_active = self._state != "IDLE" or (self._worker_thread is not None and self._worker_thread.is_alive())
            if had_active:
                self._mission_epoch += 1
                if not self.adapter.estop_active:
                    self._cancel_cause = "pose_reset"
                self._abort_evt.set()
            self.adapter.stop()
            worker = self._worker_thread

        if worker and worker.is_alive():
            deadline = time.monotonic() + 0.30
            while worker.is_alive() and time.monotonic() < deadline:
                await asyncio.sleep(0.01)

        with self._lock:
            if self._owner_mission_id == pre_owner_id:
                self.adapter.stop()
                self.adapter.unload_vln()
                self.adapter.unload_vla()
                self._release_gpu_cache()
                self._state = "IDLE"
                self._active_task = None

    def submit_task(self, task_spec: Dict[str, Any]) -> Tuple[int, str]:
        """Atomically admit mission, increment `_owner_mission_id`, and start worker under `self._lock`."""
        with self._lock:
            if self.adapter.estop_active:
                return 409, "Rejected: E-Stop is latched. Reset E-Stop first."
            if self._state != "IDLE":
                return 409, f"Rejected: Mission already active in state {self._state}."

            self._owner_mission_id += 1
            self._mission_epoch += 1
            my_owner_id = self._owner_mission_id
            my_m_epoch = self._mission_epoch
            my_e_epoch = self.adapter.estop_epoch
            self._cancel_cause = ""
            self._abort_evt.clear()
            self._state = "AGENT_PARSE"
            self._active_task = dict(task_spec)
            self._vln_rejections_current = 0
            self._last_abort_reason = "running"

            self.adapter.reset_peak_vram()
            reset_col = getattr(self.adapter, "reset_collision_flag", None)
            if callable(reset_col):
                reset_col()
            t = threading.Thread(
                target=self._run_mission_pipeline,
                args=(dict(task_spec), my_owner_id, my_m_epoch, my_e_epoch),
                name=f"FSMWorker-{my_owner_id}",
                daemon=True,
            )
            self._worker_thread = t
            t.start()
        return 200, f"Accepted mission: {task_spec.get('command')}"

    def _is_owner(self, my_owner_id: int) -> bool:
        with self._lock:
            return self._owner_mission_id == my_owner_id

    def _stop_if_owner(self, my_owner_id: int) -> None:
        """R3-MED-02: Prevent a superseded zombie worker from braking a newly admitted active mission."""
        if self._is_owner(my_owner_id):
            self.adapter.stop()

    def _has_collision(self) -> bool:
        return bool(getattr(self.adapter, "collision_occurred", False))

    def _is_at_last_completed_pose(self, pose: Any) -> bool:
        with self._lock:
            lp = self._last_completed_pose
        if lp is None:
            return False
        return math.hypot(pose.x - lp[0], pose.y - lp[1]) <= 0.05

    def _execute_standoff_retreat(
        self,
        target_x: float,
        my_owner_id: int,
        my_m_epoch: int,
        my_e_epoch: int,
        deadline: float,
    ) -> Tuple[bool, str]:
        """
        R4-MED-02: Smoothly clear close-workstation obstacles before large-yaw turns in chained tasks.
        1) If nose is turned sideways (|theta| > 0.55 rad), rotate in place (vx=0) until |theta| <= 0.55 rad;
        2) Reverse cleanly along the obstacle-free -x corridor until pose.x <= target_x, then reset depth history.
        """
        self._log_event("INFO", f"Executing close-workstation clearance retreat to x <= {target_x:.2f}m...")
        while time.monotonic() < deadline:
            if not self._is_epoch_valid(my_m_epoch, my_e_epoch) or not self._is_owner(my_owner_id):
                self._stop_if_owner(my_owner_id)
                return False, self._resolve_interrupt_reason(my_e_epoch)
            if self._has_collision():
                self._stop_if_owner(my_owner_id)
                return False, "obstacle_collision_guard"

            p = self.adapter.get_pose()
            if p.x <= target_x:
                break

            if abs(p.theta) > 0.55:
                wz_align = max(-0.95, min(0.95, -1.6 * p.theta))
                if abs(wz_align) < 0.25:
                    wz_align = -0.25 if p.theta >= 0.0 else 0.25
                if not self.adapter.move(0.0, 0.0, wz_align, expected_estop_epoch=my_e_epoch):
                    return False, self._resolve_interrupt_reason(my_e_epoch)
            else:
                rem_x = p.x - target_x
                rev_vx = -max(0.08, min(0.26, rem_x * 1.1 + 0.06))
                wz_hold = max(-0.50, min(0.50, -1.2 * p.theta))
                if not self.adapter.move(rev_vx, 0.0, wz_hold, expected_estop_epoch=my_e_epoch):
                    return False, self._resolve_interrupt_reason(my_e_epoch)
            time.sleep(0.025)
        else:
            self._stop_if_owner(my_owner_id)
            return False, "vln_cruise_timeout"

        self._stop_if_owner(my_owner_id)
        self.vision.reset_tracking_history()
        return True, ""

    def _resolve_interrupt_reason(self, my_e_epoch: int, stage_suffix: str = "") -> str:
        """R3-MED-02: Distinguish non-estop pose reset (`pose_reset_aborted`) from genuine E-Stop preemption."""
        if self._shutdown_evt.is_set():
            return "server_shutdown"
        if self.adapter.estop_active or self.adapter.estop_epoch != my_e_epoch:
            return f"estop_{stage_suffix}" if stage_suffix else "estop_interrupted"
        with self._lock:
            if self._cancel_cause == "pose_reset":
                return "pose_reset_aborted"
            if self._cancel_cause == "shutdown":
                return "server_shutdown"
        return f"estop_{stage_suffix}" if stage_suffix else "estop_interrupted"

    def _is_epoch_valid(self, my_m_epoch: int, my_e_epoch: int) -> bool:
        if self._shutdown_evt.is_set() or self.adapter.estop_active:
            return False
        if self.adapter.estop_epoch != my_e_epoch:
            return False
        with self._lock:
            return (not self._abort_evt.is_set()) and (self._mission_epoch == my_m_epoch)

    def _set_state_if_valid(self, new_state: str, my_m_epoch: int, my_e_epoch: int) -> bool:
        with self._lock:
            if self._shutdown_evt.is_set() or self.adapter.estop_active or self.adapter.estop_epoch != my_e_epoch:
                return False
            if self._abort_evt.is_set() or self._mission_epoch != my_m_epoch:
                return False
            self._state = new_state
        self._log_event("STATE", f"Transition -> {new_state}")
        return True

    def _run_mission_pipeline(
        self,
        task: Dict[str, Any],
        my_owner_id: int,
        my_m_epoch: int,
        my_e_epoch: int,
    ) -> None:
        t_mission_start = time.perf_counter()
        stage_durations: Dict[str, float] = {}
        telemetry_lock = threading.Lock()
        agent_latencies: List[float] = []
        agent_jpegs: List[float] = []
        final_dist: Optional[float] = None
        final_yaw: Optional[float] = None
        success = False
        abort_reason = "completed"
        lens_used = self.vision.lens_mode
        blind_walk_configured = bool(self.blind_walk_enabled)
        blind_walk_executed = False
        vla_model_used = str(self.vla_model)
        command = str(task.get("command", "Pick up red mug"))
        source = str(task.get("source", "manual"))
        task_type = str(task.get("task_type", "navigate_and_pick"))

        try:
            # R2-CRIT-04: Validate epoch on Line 1 BEFORE touching vision classes or hardware!
            prompts = task.get("target_prompts") or ["red mug", "mug", "cup"]
            with self._lock:
                if not self._is_epoch_valid(my_m_epoch, my_e_epoch) or not self._is_owner(my_owner_id):
                    abort_reason = self._resolve_interrupt_reason(my_e_epoch)
                    return
                self.vision.set_classes(prompts)
                self.vision.reset_tracking_history()

            if self._has_collision():
                self._stop_if_owner(my_owner_id)
                abort_reason = "obstacle_collision_guard"
                self._log_event("ERROR", "Obstacle collision guard triggered at mission start!")
                return

            # ---------------- STAGE 1: AGENT_PARSE ----------------
            t_stage = time.perf_counter()
            stop_thresh = LENS_PROFILES[lens_used]["stop_distance_m"]
            retreat_target_x = 1.02 if lens_used == "80cm" else 1.18
            pose0 = self.adapter.get_pose()
            chained_from_completed = (
                self._is_at_last_completed_pose(pose0)
                and getattr(self.adapter, "depth_fault", "none") == "none"
            )

            frame0 = self.adapter.get_rgbd_frame(lens_used)
            meas0 = self.vision.analyze_frame(
                frame0,
                lens_mode_override=lens_used,
                update_history=not chained_from_completed,
            )

            # Defer local <80cm ROI blind-zone abort ONLY when chained directly from our own completed close stop
            # AND global frame depth is healthy (never defer global wipeout or 4m penetration poisoning!).
            defer_local_blind_for_retreat = (
                chained_from_completed
                and pose0.x > retreat_target_x
                and meas0.blind_zone_triggered
                and not meas0.penetration_anomaly
                and meas0.global_valid_ratio >= 0.20
                and meas0.depth_status.startswith("BLIND_ZONE_GUARD")
            )

            if meas0.blind_zone_triggered and not defer_local_blind_for_retreat:
                self._stop_if_owner(my_owner_id)
                abort_reason = f"blind_zone_brake:{meas0.depth_status}"
                self._log_event("WARN", f"Immediate blind-zone/poison brake: {meas0.depth_status}")
                stage_durations["AGENT_PARSE"] = round(time.perf_counter() - t_stage, 3)
                return

            # R3-HIGH-04: Short-circuit VLN_CRUISE when target is ALREADY within precision stop threshold & aligned!
            # Prevents 2-frame blind forward creep (`0.446m -> 0.411m -> 0.251m`) on repeated triggers.
            already_within_stop = (
                not defer_local_blind_for_retreat
                and meas0.detected
                and meas0.median_depth_m is not None
                and meas0.median_depth_m <= stop_thresh
                and abs(meas0.yaw_error_deg) <= 12.0
                and self.agent.simulate_delay_s <= 0.0
                and not self.agent.simulate_error
            )

            if already_within_stop:
                final_dist = meas0.median_depth_m
                final_yaw = meas0.yaw_error_deg
                stage_durations["AGENT_PARSE"] = round(time.perf_counter() - t_stage, 3)
                stage_durations["VLN_CRUISE"] = 0.0
                self._log_event(
                    "INFO",
                    f"Target '{meas0.label}' already at {meas0.median_depth_m:.3f}m <= {stop_thresh:.2f}m -> Skipping VLN_CRUISE!",
                )
            else:
                ag_res = self.agent.infer_with_watchdog(
                    frame0.rgb,
                    command,
                    enforce_brake_on_fail=True,
                    should_suppress_brake=lambda: (not self._is_owner(my_owner_id))
                    or (not self._is_epoch_valid(my_m_epoch, my_e_epoch)),
                    allow_auto_fallback_on_timeout=True,
                )
                with telemetry_lock:
                    agent_latencies.append(ag_res["latency_ms"])
                    agent_jpegs.append(ag_res["jpeg_kb"])
                stage_durations["AGENT_PARSE"] = round(time.perf_counter() - t_stage, 3)

                if not self._is_epoch_valid(my_m_epoch, my_e_epoch) or not self._is_owner(my_owner_id):
                    self._stop_if_owner(my_owner_id)
                    abort_reason = self._resolve_interrupt_reason(my_e_epoch)
                    return

                if not ag_res["ok"] or ag_res.get("watchdog_triggered"):
                    self._stop_if_owner(my_owner_id)
                    abort_reason = f"watchdog_abort:{ag_res.get('error', 'agent_fail')}"
                    self._log_event("ERROR", abort_reason)
                    return

                if ag_res.get("direction") == "STOP":
                    self._stop_if_owner(my_owner_id)
                    abort_reason = "agent_directed_stop"
                    self._log_event("WARN", "Cloud Agent returned STOP direction; halting mission safely.")
                    return

                # ---------------- STAGE 2: VLN_CRUISE ----------------
                if not self._set_state_if_valid("VLN_CRUISE", my_m_epoch, my_e_epoch):
                    abort_reason = self._resolve_interrupt_reason(my_e_epoch)
                    return

                t_stage = time.perf_counter()
                self.adapter.load_vln()
                cruise_deadline = time.monotonic() + max(0.05, float(self.vln_cruise_timeout_s))

                cur_p0 = self.adapter.get_pose()
                need_standoff_retreat = (cur_p0.x > retreat_target_x) and (
                    defer_local_blind_for_retreat
                    or (not meas0.detected)
                    or (meas0.median_depth_m is None)
                    or (abs(meas0.yaw_error_deg) > 12.0)
                )
                if need_standoff_retreat:
                    with self._lock:
                        self._last_completed_pose = None
                        self._last_completed_blind_walk = False
                    ok_ret, ret_err = self._execute_standoff_retreat(
                        retreat_target_x,
                        my_owner_id,
                        my_m_epoch,
                        my_e_epoch,
                        cruise_deadline,
                    )
                    if not ok_ret:
                        abort_reason = ret_err
                        stage_durations["VLN_CRUISE"] = round(time.perf_counter() - t_stage, 3)
                        return

                shared_dir = {"hint": ag_res.get("direction", "STRAIGHT"), "watchdog_err": ""}
                cruise_done = threading.Event()

                def _async_agent_look_loop() -> None:
                    while not cruise_done.is_set():
                        if cruise_done.wait(timeout=max(0.4, self.agent_interval_s)):
                            break
                        if not self._is_epoch_valid(my_m_epoch, my_e_epoch) or cruise_done.is_set():
                            break
                        f_bg = self.adapter.get_rgbd_frame(lens_used)
                        res_bg = self.agent.infer_with_watchdog(
                            f_bg.rgb,
                            command,
                            enforce_brake_on_fail=True,
                            should_suppress_brake=lambda: cruise_done.is_set()
                            or (not self._is_owner(my_owner_id))
                            or (not self._is_epoch_valid(my_m_epoch, my_e_epoch)),
                            allow_auto_fallback_on_timeout=False,
                        )
                        if cruise_done.is_set() or not self._is_epoch_valid(my_m_epoch, my_e_epoch):
                            break
                        with telemetry_lock:
                            agent_latencies.append(res_bg["latency_ms"])
                            agent_jpegs.append(res_bg["jpeg_kb"])
                        if not res_bg["ok"] or res_bg.get("watchdog_triggered"):
                            shared_dir["watchdog_err"] = str(res_bg.get("error", "watchdog_timeout"))
                            self._stop_if_owner(my_owner_id)
                            cruise_done.set()
                            break
                        shared_dir["hint"] = res_bg.get("direction", "STRAIGHT")

                bg_agent_thread = threading.Thread(target=_async_agent_look_loop, name="AsyncCruiseAgent", daemon=True)
                bg_agent_thread.start()

                stable_frames = 0
                locked_for_handoff = False

                try:
                    while time.monotonic() < cruise_deadline:
                        if not self._is_epoch_valid(my_m_epoch, my_e_epoch):
                            self._stop_if_owner(my_owner_id)
                            abort_reason = self._resolve_interrupt_reason(my_e_epoch)
                            return
                        if self._has_collision():
                            self._stop_if_owner(my_owner_id)
                            abort_reason = "obstacle_collision_guard"
                            self._log_event("ERROR", "Cruise halted by obstacle_collision_guard!")
                            return
                        if shared_dir["watchdog_err"]:
                            self._stop_if_owner(my_owner_id)
                            abort_reason = f"watchdog_abort:{shared_dir['watchdog_err']}"
                            self._log_event("ERROR", f"Cruise halted by Watchdog: {abort_reason}")
                            return
                        if shared_dir["hint"] == "STOP":
                            self._stop_if_owner(my_owner_id)
                            abort_reason = "agent_directed_stop"
                            self._log_event("WARN", "Cruise Agent advised STOP; halting base.")
                            return

                        frame = self.adapter.get_rgbd_frame(lens_used)
                        meas = self.vision.analyze_frame(frame, lens_mode_override=lens_used)
                        if meas.blind_zone_triggered:
                            self._stop_if_owner(my_owner_id)
                            abort_reason = f"blind_zone_brake:{meas.depth_status}"
                            self._log_event("WARN", f"Cruise halted by depth guard: {meas.depth_status}")
                            return

                        vln_step = self.adapter.step_vln(command, agent_direction_hint=shared_dir["hint"])
                        if vln_step.get("arrived"):
                            if meas.median_depth_m is None or meas.median_depth_m > (stop_thresh + 0.08):
                                with self._lock:
                                    self._vln_rejections_current += 1
                                d_str = f"{meas.median_depth_m:.2f}m" if meas.median_depth_m else "unknown"
                                self._log_event(
                                    "GUARD",
                                    f"Rejected premature Light-VLN arrived=True at depth={d_str} (target <= {stop_thresh:.2f}m)",
                                )

                        if meas.detected and meas.median_depth_m is not None:
                            final_dist = meas.median_depth_m
                            final_yaw = meas.yaw_error_deg
                            stable_frames += 1
                            if stable_frames >= 3 and meas.median_depth_m <= 1.48:
                                locked_for_handoff = True
                                self._stop_if_owner(my_owner_id)
                                self._log_event(
                                    "INFO",
                                    f"YOLOE locked '{meas.label}' at {meas.median_depth_m:.2f}m -> Preempting Light-VLN!",
                                )
                                break
                        else:
                            stable_frames = 0

                        step_wz = float(vln_step.get("wz", 0.0))
                        step_vx_raw = float(vln_step.get("vx", self.cruise_speed_mps))
                        # R3-HIGH-04 + R4-MED-02: Clamp forward linear velocity to 0.0 when within `stop_thresh + 0.05`
                        # OR whenever heading error exceeds 14.0 deg (rotate in place before translating forward)!
                        if (
                            step_vx_raw <= 1e-4
                            or (meas.detected and abs(meas.yaw_error_deg) > 14.0)
                            or (meas.detected and meas.median_depth_m is not None and meas.median_depth_m <= (stop_thresh + 0.05))
                        ):
                            eff_vx = 0.0
                        else:
                            eff_vx = self.cruise_speed_mps * (0.45 if abs(step_wz) > 0.45 else 1.0)

                        moved = self.adapter.move(
                            eff_vx,
                            0.0,
                            step_wz,
                            expected_estop_epoch=my_e_epoch,
                        )
                        if not moved:
                            abort_reason = self._resolve_interrupt_reason(my_e_epoch)
                            return
                        time.sleep(0.04)
                finally:
                    # R2-HIGH-03 + R3-MED-02: Halt chassis (if still owner) BEFORE joining background thread!
                    self._stop_if_owner(my_owner_id)
                    cruise_done.set()
                    bg_agent_thread.join(timeout=0.12)
                    stage_durations["VLN_CRUISE"] = round(time.perf_counter() - t_stage, 3)

                if not self._is_epoch_valid(my_m_epoch, my_e_epoch):
                    self._stop_if_owner(my_owner_id)
                    abort_reason = self._resolve_interrupt_reason(my_e_epoch)
                    return

                # R2-HIGH-03: NEVER blindly transition into YOLO_LAST_MILE if VLN_CRUISE timed out without target lock!
                if not locked_for_handoff:
                    self._stop_if_owner(my_owner_id)
                    abort_reason = "vln_cruise_timeout"
                    self._log_event("WARN", "VLN_CRUISE timed out without target lock; halting safely.")
                    return

            # ---------------- STAGE 3: YOLO_LAST_MILE ----------------
            if not self._set_state_if_valid("YOLO_LAST_MILE", my_m_epoch, my_e_epoch):
                abort_reason = self._resolve_interrupt_reason(my_e_epoch)
                return

            t_stage = time.perf_counter()
            prof = LENS_PROFILES[lens_used]
            stop_dist = prof["stop_distance_m"]
            slow_zone = prof["slow_zone_m"]
            lost_count = 0
            servo_deadline = time.monotonic() + 18.0
            reached_stop = False

            while time.monotonic() < servo_deadline:
                if not self._is_epoch_valid(my_m_epoch, my_e_epoch):
                    self._stop_if_owner(my_owner_id)
                    abort_reason = self._resolve_interrupt_reason(my_e_epoch)
                    return
                if self._has_collision():
                    self._stop_if_owner(my_owner_id)
                    abort_reason = "obstacle_collision_guard"
                    self._log_event("ERROR", "Last-mile halted by obstacle_collision_guard!")
                    return

                frame = self.adapter.get_rgbd_frame(lens_used)
                meas = self.vision.analyze_frame(frame, lens_mode_override=lens_used)

                if meas.blind_zone_triggered:
                    self._stop_if_owner(my_owner_id)
                    abort_reason = f"blind_zone_brake:{meas.depth_status}"
                    self._log_event("WARN", f"Last-mile emergency depth brake: {meas.depth_status}")
                    return

                if not meas.detected or meas.median_depth_m is None:
                    lost_count += 1
                    if lost_count > 20:
                        self._stop_if_owner(my_owner_id)
                        abort_reason = "target_lost_in_last_mile"
                        return
                    search_wz = -0.25 if (final_yaw is not None and final_yaw > 0.0) else 0.25
                    if not self.adapter.move(0.0, 0.0, search_wz, expected_estop_epoch=my_e_epoch):
                        abort_reason = self._resolve_interrupt_reason(my_e_epoch)
                        return
                    time.sleep(0.04)
                    continue

                lost_count = 0
                d_now = meas.median_depth_m
                yaw_deg = meas.yaw_error_deg
                final_dist = d_now
                final_yaw = yaw_deg

                wz = max(-0.65, min(0.65, -0.038 * yaw_deg))

                if d_now <= stop_dist:
                    if abs(yaw_deg) > 5.0:
                        if not self.adapter.move(0.0, 0.0, wz, expected_estop_epoch=my_e_epoch):
                            abort_reason = self._resolve_interrupt_reason(my_e_epoch)
                            return
                        time.sleep(0.04)
                        continue
                    self._stop_if_owner(my_owner_id)
                    reached_stop = True
                    self._log_event(
                        "INFO",
                        f"Precision depth brake locked at d={d_now:.3f}m (target {stop_dist:.2f}m), yaw_err={yaw_deg:+.2f}deg",
                    )
                    break

                # R4-MED-02: Rotate in place (vx=0.0) whenever |yaw_error| > 14.0 deg before translating forward!
                if abs(yaw_deg) > 14.0:
                    vx = 0.0
                else:
                    speed = self.cruise_speed_mps * (0.55 if d_now <= slow_zone else 1.0)
                    vx = max(0.06, min(speed, 0.65 * (d_now - stop_dist) + 0.05))
                if not self.adapter.move(vx, 0.0, wz, expected_estop_epoch=my_e_epoch):
                    abort_reason = self._resolve_interrupt_reason(my_e_epoch)
                    return
                time.sleep(0.04)

            if not reached_stop:
                self._stop_if_owner(my_owner_id)
                abort_reason = "last_mile_timeout"
                return

            if self._has_collision():
                self._stop_if_owner(my_owner_id)
                abort_reason = "obstacle_collision_guard"
                self._log_event("ERROR", "Obstacle collision guard triggered after last-mile approach!")
                return

            # Precision odometry/monotonic closed-loop 40cm blind-walk (`R3-MED-03`: record true `blind_walk_executed`!)
            if (
                lens_used == "80cm"
                and blind_walk_configured
                and task_type == "navigate_and_pick"
                and (final_dist is None or final_dist >= 0.80)
            ):
                blind_walk_executed = True
                target_bw_m = prof["blind_walk_distance_m"]
                bw_speed = prof["blind_walk_speed_mps"]
                pre_bw_dist = final_dist if final_dist is not None else stop_dist
                bw_start_pose = self.adapter.get_pose()
                t_bw0 = time.monotonic()
                bw_hard_limit = t_bw0 + (target_bw_m / max(0.05, bw_speed)) + 0.40
                self._log_event("INFO", f"Executing precision 40cm blind-walk (@ {bw_speed:.2f}m/s)...")

                while time.monotonic() < bw_hard_limit:
                    cur_p = self.adapter.get_pose()
                    traveled = math.hypot(cur_p.x - bw_start_pose.x, cur_p.y - bw_start_pose.y)
                    final_dist = round(max(0.05, pre_bw_dist - traveled), 3)

                    if not self._is_epoch_valid(my_m_epoch, my_e_epoch):
                        self._stop_if_owner(my_owner_id)
                        cur_p = self.adapter.get_pose()
                        traveled = math.hypot(cur_p.x - bw_start_pose.x, cur_p.y - bw_start_pose.y)
                        final_dist = round(max(0.05, pre_bw_dist - traveled), 3)
                        abort_reason = self._resolve_interrupt_reason(my_e_epoch, stage_suffix="during_blind_walk")
                        return

                    if self._has_collision():
                        self._stop_if_owner(my_owner_id)
                        abort_reason = "obstacle_collision_guard"
                        self._log_event("ERROR", "Blind-walk halted by obstacle_collision_guard!")
                        return

                    rem = target_bw_m - traveled
                    if rem <= 0.004:
                        break
                    # Taper speed in final 4cm so 50Hz kinematics step never overshoots 0.40m
                    v_step = max(0.03, min(bw_speed, rem * 4.5))
                    if not self.adapter.move(v_step, 0.0, 0.0, expected_estop_epoch=my_e_epoch):
                        cur_p = self.adapter.get_pose()
                        traveled = math.hypot(cur_p.x - bw_start_pose.x, cur_p.y - bw_start_pose.y)
                        final_dist = round(max(0.05, pre_bw_dist - traveled), 3)
                        abort_reason = self._resolve_interrupt_reason(my_e_epoch, stage_suffix="during_blind_walk")
                        return
                    time.sleep(0.005)

                self._stop_if_owner(my_owner_id)
                time.sleep(0.025)
                end_p = self.adapter.get_pose()
                actual_traveled = math.hypot(end_p.x - bw_start_pose.x, end_p.y - bw_start_pose.y)
                final_dist = round(max(0.05, pre_bw_dist - actual_traveled), 3)
                if self._has_collision():
                    abort_reason = "obstacle_collision_guard"
                    self._log_event("ERROR", "Obstacle collision guard triggered at blind-walk finish!")
                    return
                self._log_event("INFO", f"Blind-walk finished: traveled={actual_traveled:.3f}m, final_dist={final_dist:.3f}m")

            stage_durations["YOLO_LAST_MILE"] = round(time.perf_counter() - t_stage, 3)

            # ---------------- STAGE 4: VRAM_SWAP_AND_VLA ----------------
            if not self._set_state_if_valid("VRAM_SWAP_AND_VLA", my_m_epoch, my_e_epoch):
                abort_reason = self._resolve_interrupt_reason(my_e_epoch)
                return

            t_stage = time.perf_counter()
            self._stop_if_owner(my_owner_id)

            vram_after_unload = self.adapter.unload_vln()
            self._release_gpu_cache()
            self._log_event("VRAM", f"Unloaded Light-VLN -> VRAM dropped to {vram_after_unload.allocated_gb:.2f} GB")

            # R3-MED-02: Re-verify cancellation/E-Stop epoch BEFORE loading 4.2GB/5.8GB VLA weights!
            if not self._is_epoch_valid(my_m_epoch, my_e_epoch) or not self._is_owner(my_owner_id):
                self._stop_if_owner(my_owner_id)
                abort_reason = self._resolve_interrupt_reason(my_e_epoch)
                return

            if task_type == "navigate_and_pick":
                vram_vla = self.adapter.load_vla(vla_model_used)
                self._log_event(
                    "VRAM",
                    f"Loaded {vla_model_used} -> VRAM={vram_vla.allocated_gb:.2f} GB (Peak={vram_vla.peak_gb:.2f} GB / 12.0 GB)",
                )
                grasp_res = self.adapter.execute_vla_grasp(
                    command,
                    model_name=vla_model_used,
                    abort_check=lambda: (not self._is_epoch_valid(my_m_epoch, my_e_epoch)) or self._has_collision(),
                    expected_estop_epoch=my_e_epoch,
                )
                self.adapter.unload_vla()
                self._release_gpu_cache()
                if self._has_collision():
                    abort_reason = "obstacle_collision_guard"
                    self._log_event("ERROR", "Obstacle collision guard triggered during VLA execution!")
                    return
                if grasp_res.get("aborted") or not grasp_res.get("success") or (not self._is_epoch_valid(my_m_epoch, my_e_epoch)):
                    abort_reason = self._resolve_interrupt_reason(my_e_epoch, stage_suffix="during_vla")
                    return

            if self._has_collision():
                self._stop_if_owner(my_owner_id)
                success = False
                abort_reason = "obstacle_collision_guard"
                self._log_event("ERROR", "Final completion check blocked by obstacle_collision_guard!")
                return

            stage_durations["VRAM_SWAP_AND_VLA"] = round(time.perf_counter() - t_stage, 3)
            success = True
            abort_reason = "completed"
            done_pose = self.adapter.get_pose()
            with self._lock:
                if self._owner_mission_id == my_owner_id:
                    self._last_completed_pose = (done_pose.x, done_pose.y)
                    self._last_completed_blind_walk = blind_walk_executed
            self._log_event("SUCCESS", f"Mission completed! stop_d={final_dist}m, yaw_err={final_yaw}deg")

        except Exception as exc:
            self._stop_if_owner(my_owner_id)
            abort_reason = f"exception:{exc}"
            logger.exception("Mission pipeline error: %s", exc)
        finally:
            peak_vram_gb = 1.4
            rej_cnt = 0
            try:
                peak_vram_gb = self.adapter.get_vram_status().peak_gb
                with self._lock:
                    still_owner = self._owner_mission_id == my_owner_id
                    rej_cnt = self._vln_rejections_current

                # R2-HIGH-02: Reclaim 5.4GB Light-VLN / VLA VRAM whenever this mission was NOT superseded by a new task!
                if still_owner:
                    self.adapter.stop()
                    self.adapter.unload_vln()
                    self.adapter.unload_vla()
                    self._release_gpu_cache()
                peak_vram_gb = max(peak_vram_gb, self.adapter.get_vram_status().peak_gb)
            except Exception as cleanup_exc:  # pragma: no cover
                logger.error("Hardware/VRAM cleanup error in FSM finally: %s", cleanup_exc)

            # R2-HIGH-05: Isolate SQLite `record_run` so DB lock/IO errors NEVER deadlock FSM state!
            try:
                total_dur = round(time.perf_counter() - t_mission_start, 3)
                with telemetry_lock:
                    avg_lat = round(sum(agent_latencies) / len(agent_latencies), 1) if agent_latencies else 0.0
                    avg_kb = round(sum(agent_jpegs) / len(agent_jpegs), 2) if agent_jpegs else 0.0

                self.data_logger.record_run(
                    {
                        "command": command,
                        "source": source,
                        "lens_mode": lens_used,
                        "blind_walk_enabled": blind_walk_executed,
                        "vla_model": vla_model_used,
                        "total_duration_s": total_dur,
                        "stage_durations": stage_durations,
                        "vln_early_stop_rejections": rej_cnt,
                        "agent_avg_latency_ms": avg_lat,
                        "agent_avg_jpeg_kb": avg_kb,
                        "final_stop_distance_m": final_dist,
                        "final_yaw_error_deg": final_yaw,
                        "peak_vram_gb": peak_vram_gb,
                        "success": success,
                        "abort_reason": abort_reason,
                    }
                )
            except Exception as db_exc:
                logger.error("Isolated database error in record_run (FSM recovery unaffected): %s", db_exc)
            finally:
                with self._lock:
                    # R3-MED-02: Only update `_last_abort_reason` & `_state` if this worker STILL owns the FSM!
                    if self._owner_mission_id == my_owner_id:
                        if not success:
                            self._last_completed_pose = None
                            self._last_completed_blind_walk = False
                        self._last_abort_reason = abort_reason
                        self._state = "IDLE"
                        self._active_task = None

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "state": self._state,
                "mission_epoch": self._mission_epoch,
                "owner_mission_id": self._owner_mission_id,
                "states_order": list(FSM_STATES),
                "active_task": self._active_task,
                "cruise_speed_mps": self.cruise_speed_mps,
                "agent_interval_s": self.agent_interval_s,
                "blind_walk_enabled": self.blind_walk_enabled,
                "vla_model": self.vla_model,
                "vln_early_stop_rejections": self._vln_rejections_current,
                "last_abort_reason": self._last_abort_reason,
                "recent_events": list(self._events)[:35],
            }

    def shutdown(self) -> None:
        self._shutdown_evt.set()
        with self._lock:
            self._mission_epoch += 1
            self._cancel_cause = "shutdown"
            self._abort_evt.set()
        self.adapter.stop()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=0.5)
        self.adapter.unload_vln()
        self.adapter.unload_vla()
        if self._cam_thread.is_alive():
            self._cam_thread.join(timeout=0.4)
