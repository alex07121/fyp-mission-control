"""
High-Speed Whole-Room Object Analyzer & Fast Async Object Navigator (Gemini Flash-Lite + LightNav-0 + BLE Car).

Speed Architecture (from START_HERE_PLAN.md Section 2 — Farewell to '1 meter in 10 minutes'):
  1. Target in view: Kick out slow step-by-step waiting! Align once, then drive in big smooth strides
     (Far = 800ms / ~40cm, Mid = 520ms / ~26cm, Near = 220ms / ~10cm) while a background thread checks
     frames asynchronously without stopping the car!
  2. Target out of view: Use school RTX 5000 Ada LightNav-0 (`vllm_local`) + remembered bearing to find it.
  3. Immediate E-Stop preemption (<15ms slice check) at every instant.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from backend.config import DATA_DIR, GCP_LOCATION, GCP_PROJECT_ID, GEMINI_MODEL_ID
from backend.modules.cloud_agent import _normalize_to_bgr
from backend.modules.remote_vln_client import RemoteVLNClient

logger = logging.getLogger("fyp.room_explorer")

CAMERA_HFOV_DEG: float = 115.0
ROOM_MEMORY_PATH: Path = DATA_DIR / "room_memory_latest.json"

_VIEW_SCAN_PROMPT = """You are analyzing an indoor room view from a small robot car's front camera (~10cm above ground).
List up to 8 distinct physical objects or landmarks visible in THIS image (e.g., orange chair, black backpack, black tote bag, white box, red mug, water bottle, door, cabinet, table, fire extinguisher).
Skip plain walls, ceiling, and empty floor.
For each object return:
- name_en: concise English name with color/type (e.g. "orange chair", "black bag")
- name_zh: concise Chinese name (e.g. "橙色椅子", "黑色手提包")
- x: horizontal center in image from 0.0 (far left) to 1.0 (far right)
- distance: "near" (<0.8m), "mid" (0.8-2.5m), or "far" (>2.5m)
Reply ONLY with valid JSON:
{"scene_zh": "<15字内中文场景简述>", "objects": [{"name_en": "...", "name_zh": "...", "x": 0.5, "distance": "mid"}]}"""

# NOTE: Use .replace("__TARGET__", target_en) instead of .format() so JSON braces never raise KeyError!
_ARRIVAL_CHECK_TEMPLATE = """You watch a small robot car's front camera (wide-angle, 10cm above floor) driving toward target: "__TARGET__".
Return JSON only:
- visible: true if "__TARGET__" is visible in the image, else false
- x: horizontal center of "__TARGET__" from 0.0 (left edge) to 1.0 (right edge), 0.5 if not visible
- distance: "arrived" (very close <=35cm, fills large lower portion of image), "near" (35-90cm), "mid" (0.9-2.2m), or "far" (>2.2m)
Reply ONLY with JSON: {"visible": true, "x": 0.5, "distance": "mid"}"""


def _wrap_deg(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def _x_to_offset_deg(x: float, hfov_deg: float = CAMERA_HFOV_DEG) -> float:
    xc = min(1.0, max(0.0, float(x)))
    tan_half = math.tan(math.radians(hfov_deg) / 2.0)
    # Positive = LEFT, Negative = RIGHT
    return math.degrees(math.atan((0.5 - xc) * 2.0 * tan_half))


def _encode_scan_jpeg(raw_bgr: np.ndarray, width: int = 320, quality: int = 75) -> bytes:
    """Fast low-payload JPEG (~14-18 KB) for sub-second Vertex AI inference."""
    bgr = _normalize_to_bgr(raw_bgr)
    h, w = bgr.shape[:2]
    new_h = max(120, int(round(h * width / max(1, w))))
    resized = cv2.resize(bgr, (width, new_h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else b""


class RoomExplorer:
    """Fast whole-room object scanner & asynchronous big-stride navigator."""

    def __init__(self, adapter: Any, data_logger: Optional[Any] = None) -> None:
        self.adapter = adapter
        self.data_logger = data_logger
        self._lock = threading.RLock()
        self._school_hub = RemoteVLNClient()
        self._vertex_client: Any = None
        self._status: str = "IDLE"
        self._progress_msg: str = "就绪：点击「分析眼前画面物体」或「360° 防缠线扫全屋」"
        self._scene_summary: str = ""
        self._objects: List[Dict[str, Any]] = []
        self._worker: Optional[threading.Thread] = None
        self._abort_evt = threading.Event()
        self._load_cached_memory()

    def _load_cached_memory(self) -> None:
        if ROOM_MEMORY_PATH.exists():
            try:
                data = json.loads(ROOM_MEMORY_PATH.read_text(encoding="utf-8"))
                self._objects = data.get("objects", [])
                self._scene_summary = data.get("scene_summary", "")
            except Exception:
                pass

    def _get_client(self) -> Any:
        if self._vertex_client is None:
            from google import genai
            from google.genai import types

            self._vertex_client = genai.Client(
                vertexai=True,
                project=GCP_PROJECT_ID,
                location=GCP_LOCATION,
                http_options=types.HttpOptions(timeout=6000),
            )
        return self._vertex_client

    def _ask_vlm_json(self, jpeg_bytes: bytes, prompt: str, max_tokens: int = 350) -> Dict[str, Any]:
        # Primary Compute Path: 100% School RTX 5000 Ada 32GB (`qwen3-vl:4b-instruct` via Ollama on ubuntu@324)
        if self._school_hub.enabled:
            res_school = self._school_hub.ask_agent_vlm(
                jpeg_bytes=jpeg_bytes,
                prompt=prompt,
                expect_json=True,
                max_tokens=max_tokens,
                timeout_s=14.0,
            )
            if res_school.get("ok"):
                raw_s = str(res_school.get("text", "")).strip()
                m_s = re.search(r"\{.*\}", raw_s, re.S)
                return json.loads(m_s.group(0)) if m_s else {}
            if os.getenv("ALLOW_CLOUD_FALLBACK", "0").strip() != "1":
                raise RuntimeError(f"School GPU VLM failed: {res_school.get('error')}")

        from google.genai import types

        client = self._get_client()
        img_part = types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")
        cfg = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
            response_mime_type="application/json",
            temperature=0.0,
            max_output_tokens=max_tokens,
        )
        res = client.models.generate_content(
            model=GEMINI_MODEL_ID,
            contents=[img_part, prompt],
            config=cfg,
        )
        raw = (res.text or "").strip()
        m = re.search(r"\{.*\}", raw, re.S)
        return json.loads(m.group(0)) if m else {}

    def abort(self) -> None:
        self._abort_evt.set()
        self.adapter.stop()
        with self._lock:
            if self._status != "IDLE":
                self._progress_msg = "⛔ 已急停中断"
                self._status = "IDLE"

    def get_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "status": self._status,
                "progress_msg": self._progress_msg,
                "scene_summary": self._scene_summary,
                "objects": list(self._objects),
            }

    def start_scan(self, full_360: bool = False) -> Dict[str, Any]:
        if self.adapter.estop_active:
            return {"ok": False, "error": "E-Stop 已锁死，请先点击顶部『解除急停 Reset』"}
        with self._lock:
            if self._status != "IDLE":
                return {"ok": False, "error": f"当前忙碌中: {self._status}"}
            self._abort_evt.clear()
            self._status = "SCANNING_360" if full_360 else "SCANNING_VIEW"
            self._progress_msg = "正在启动 360° 快速扫全屋..." if full_360 else "正在极速识别眼前画面物件..."
            t = threading.Thread(target=self._run_scan_worker, args=(full_360,), name="RoomScanWorker", daemon=True)
            self._worker = t
            t.start()
        return {"ok": True, "mode": "360" if full_360 else "view"}

    def _run_scan_worker(self, full_360: bool) -> None:
        my_e_epoch = self.adapter.estop_epoch
        try:
            # Parallel VLM inference WHILE turning to the next angle -> cuts 360° scan time in HALF!
            schedule = [(0.0, 90.0), (90.0, 90.0), (180.0, -270.0), (-90.0, 90.0)] if full_360 else [(0.0, 0.0)]
            steps = len(schedule)
            captured_views: List[tuple[float, bytes]] = []

            for i, (heading_deg, next_turn_deg) in enumerate(schedule):
                if self._abort_evt.is_set() or self.adapter.estop_active or self.adapter.estop_epoch != my_e_epoch:
                    return
                with self._lock:
                    self._progress_msg = (
                        f"📸 抓拍视角 {i + 1}/{steps}（朝向 {_wrap_deg(heading_deg):+.0f}°）..."
                        if full_360
                        else "📸 正在分析眼前画面..."
                    )
                time.sleep(0.15 if i == 0 else 0.22)
                frame = self.adapter.get_rgbd_frame()
                captured_views.append((heading_deg, _encode_scan_jpeg(frame.rgb, width=416, quality=78)))

                if full_360 and abs(next_turn_deg) > 1.0 and hasattr(self.adapter, "turn_in_place_deg"):
                    if next_turn_deg == -270.0:
                        for _ in range(3):
                            if self._abort_evt.is_set() or self.adapter.estop_active:
                                return
                            self.adapter.turn_in_place_deg(-90.0, expected_estop_epoch=my_e_epoch)
                    else:
                        self.adapter.turn_in_place_deg(next_turn_deg, expected_estop_epoch=my_e_epoch)

            # Analyze all 4 views concurrently in parallel (~1.2s total instead of 4 x 1.2s!)
            with self._lock:
                self._progress_msg = "⚡ 正在并发识别全部视角物件 (~1.2s)..."

            import concurrent.futures

            raw_found: List[Dict[str, Any]] = []
            scenes: List[str] = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futs = [
                    (h_deg, pool.submit(self._ask_vlm_json, jb, _VIEW_SCAN_PROMPT, 450))
                    for h_deg, jb in captured_views
                ]
                for h_deg, fut in futs:
                    parsed = fut.result(timeout=8.0)
                    sc = str(parsed.get("scene_zh", "")).strip()
                    if sc:
                        scenes.append(sc)
                    for obj in parsed.get("objects", []):
                        name_en = str(obj.get("name_en", "")).strip().lower()
                        name_zh = str(obj.get("name_zh", "")).strip() or name_en
                        if not name_en:
                            continue
                        x_val = float(obj.get("x", 0.5))
                        if full_360 and (x_val < 0.05 or x_val > 0.95):
                            continue
                        bearing = round(_wrap_deg(h_deg + _x_to_offset_deg(x_val)), 1)
                        raw_found.append(
                            {
                                "name_en": name_en,
                                "name_zh": name_zh,
                                "bearing_deg": bearing,
                                "distance": str(obj.get("distance", "mid")),
                            }
                        )

            merged = self._merge_objects(raw_found)
            summary = "；".join(dict.fromkeys(scenes[:4])) or f"共发现 {len(merged)} 个物件"
            ROOM_MEMORY_PATH.write_text(
                json.dumps({"updated_at": time.time(), "scene_summary": summary, "objects": merged}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            with self._lock:
                self._objects = merged
                self._scene_summary = summary
                self._progress_msg = f"✅ 扫描完成！共识别 {len(merged)} 个物件，点击任一按钮即刻快速开过去"
        except Exception as exc:
            logger.error("Room scan failed: %s", exc)
            with self._lock:
                self._progress_msg = f"扫描出错: {exc}"
        finally:
            self.adapter.stop()
            with self._lock:
                self._status = "IDLE"

    @staticmethod
    def _merge_objects(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        for it in items:
            for m in merged:
                if m["name_en"] == it["name_en"] and abs(_wrap_deg(m["bearing_deg"] - it["bearing_deg"])) <= 28.0:
                    if it["distance"] == "near":
                        m["distance"] = "near"
                    break
            else:
                en = it["name_en"]
                merged.append({**it, "lightnav_prompt": f"Go to the {en} in front and stop."})
        return merged

    def start_navigate_to(self, target_en: str, target_zh: str = "", bearing_deg: float = 0.0) -> Dict[str, Any]:
        if self.adapter.estop_active:
            return {"ok": False, "error": "E-Stop 已锁死，请先解除急停"}
        clean_en = re.sub(r"[^a-zA-Z0-9 _\-]", "", target_en).strip() or "target object"
        label = target_zh.strip() or clean_en
        with self._lock:
            if self._status != "IDLE":
                return {"ok": False, "error": f"当前忙碌中: {self._status}"}
            self._abort_evt.clear()
            self._status = "NAVIGATING"
            self._progress_msg = f"🚀 出发前往：{label} ({clean_en})"
            t = threading.Thread(
                target=self._run_fast_nav_worker,
                args=(clean_en, label, float(bearing_deg)),
                name="FastObjectNavWorker",
                daemon=True,
            )
            self._worker = t
            t.start()
        return {"ok": True, "target_en": clean_en, "target_zh": label, "bearing_deg": bearing_deg}

    def _check_target_once(self, bgr: np.ndarray, target_en: str) -> Dict[str, Any]:
        prompt = _ARRIVAL_CHECK_TEMPLATE.replace("__TARGET__", target_en)
        return self._ask_vlm_json(_encode_scan_jpeg(bgr, width=320, quality=72), prompt, max_tokens=120)

    def _run_fast_nav_worker(self, target_en: str, label: str, bearing_deg: float) -> None:
        """
        High-speed navigation loop:
          - Phase 1: Turn cleanly to remembered `bearing_deg` if off-axis.
          - Phase 2: Async background Gemini visual tracker (~0.9s on 320px thumbnail) +
                     Big-stride driving (`850ms` far / `550ms` mid / `240ms` near) with ZERO stop-and-wait
                     when target is ahead; falls back to LightNav-0 only when target is out of sight!
        """
        t0 = time.perf_counter()
        my_e_epoch = self.adapter.estop_epoch
        success = False
        abort_reason = "aborted"
        nav_done = threading.Event()

        try:
            # 1. Fast initial turn if object is off-center
            if abs(bearing_deg) >= 14.0 and hasattr(self.adapter, "turn_in_place_deg"):
                with self._lock:
                    self._progress_msg = f"🔄 正在转向 {bearing_deg:+.0f}° 对准『{label}』..."
                self.adapter.turn_in_place_deg(bearing_deg, expected_estop_epoch=my_e_epoch)
                time.sleep(0.15)

            if self._abort_evt.is_set() or self.adapter.estop_active or self.adapter.estop_epoch != my_e_epoch:
                abort_reason = "estop_interrupted"
                return

            lightnav_inst = f"Go to the {target_en} in front and stop."
            self.adapter.load_vln()
            remote_vln = getattr(self.adapter, "_remote_vln", None)
            if remote_vln is not None and remote_vln.enabled:
                remote_vln.load(instruction=lightnav_inst)

            # Do ONE fast initial look (~0.9s) so we immediately lock onto the target before moving!
            f0 = self.adapter.get_rgbd_frame()
            init_chk = {"visible": True, "x": 0.5, "distance": "mid"}
            try:
                init_chk = self._check_target_once(f0.rgb, target_en)
            except Exception as exc:
                logger.warning("Initial target check notice: %s", exc)

            obs_lock = threading.RLock()
            shared_obs: Dict[str, Any] = {
                "visible": bool(init_chk.get("visible", False)),
                "offset_deg": _x_to_offset_deg(float(init_chk.get("x", 0.5))),
                "distance": str(init_chk.get("distance", "mid")).lower(),
                "arrived_streak": 1 if (init_chk.get("visible") and str(init_chk.get("distance")).lower() == "arrived") else 0,
                "updated_mono": time.monotonic(),
            }

            # Background async visual tracker — runs continuously while car drives without blocking wheels!
            def _bg_visual_tracker() -> None:
                while not nav_done.is_set() and not self._abort_evt.is_set():
                    try:
                        fr = self.adapter.get_rgbd_frame()
                        chk = self._check_target_once(fr.rgb, target_en)
                        vis = bool(chk.get("visible", False))
                        dist_s = str(chk.get("distance", "mid")).lower()
                        off_d = _x_to_offset_deg(float(chk.get("x", 0.5)))
                        with obs_lock:
                            prev_st = int(shared_obs["arrived_streak"])
                            new_st = (prev_st + 1) if (vis and dist_s == "arrived") else 0
                            shared_obs.update(
                                visible=vis,
                                offset_deg=off_deg if (off_deg := off_d) is not None else 0.0,
                                distance=dist_s,
                                arrived_streak=new_st,
                                updated_mono=time.monotonic(),
                            )
                            if new_st >= 1 and dist_s == "arrived":
                                # Brake immediately from background thread the instant we arrive!
                                self.adapter.stop()
                    except Exception as bg_exc:
                        logger.warning("BG tracker check warning: %s", bg_exc)
                    nav_done.wait(0.15)

            bg_thread = threading.Thread(target=_bg_visual_tracker, name="AsyncRoomNavVision", daemon=True)
            bg_thread.start()

            has_pulse = hasattr(self.adapter, "drive_pulse_ms")
            deadline = time.monotonic() + 60.0
            step_idx = 0

            while time.monotonic() < deadline:
                if self._abort_evt.is_set() or self.adapter.estop_active or self.adapter.estop_epoch != my_e_epoch:
                    abort_reason = "estop_interrupted"
                    return

                step_idx += 1
                with obs_lock:
                    vis = bool(shared_obs["visible"])
                    off_deg = float(shared_obs["offset_deg"])
                    dist_cls = str(shared_obs["distance"])
                    arr_st = int(shared_obs["arrived_streak"])

                # Arrived check: stop immediately when target fills lower close view!
                if vis and dist_cls == "arrived" and arr_st >= 1:
                    self.adapter.stop()
                    # Quick stationary double-check so motion blur didn't fool it
                    time.sleep(0.15)
                    re_chk = self._check_target_once(self.adapter.get_rgbd_frame().rgb, target_en)
                    if re_chk.get("visible") and str(re_chk.get("distance", "")).lower() in ("arrived", "near"):
                        self.adapter.stop()
                        success = True
                        abort_reason = "completed"
                        with self._lock:
                            self._progress_msg = f"✅ 已快速抵达『{label}』面前并停稳！（共 {step_idx} 大步，耗时 {time.perf_counter() - t0:.1f}s）"
                        return

                if vis:
                    # FAST DIRECT VISUAL SERVO — NO LIGHTNAV ZIG-ZAG, BIG STRIDES!
                    if abs(off_deg) > 16.0 and hasattr(self.adapter, "turn_in_place_deg"):
                        turn_cmd = max(-35.0, min(35.0, off_deg * 0.75))
                        with self._lock:
                            self._progress_msg = f"🎯 锁定『{label}』({dist_cls})，微调对准 {turn_cmd:+.0f}°..."
                        if not self.adapter.turn_in_place_deg(turn_cmd, expected_estop_epoch=my_e_epoch):
                            abort_reason = "estop_interrupted"
                            return
                        with obs_lock:
                            shared_obs["offset_deg"] = 0.0
                        time.sleep(0.08)
                        continue

                    # Target is right ahead! Drive in BIG fast strides!
                    stride_ms = 850 if dist_cls == "far" else (560 if dist_cls == "mid" else 240)
                    with self._lock:
                        self._progress_msg = f"🚀 正对『{label}』({dist_cls}) 大步直冲 ({stride_ms}ms)..."
                    if has_pulse:
                        if not self.adapter.drive_pulse_ms("forward", stride_ms, expected_estop_epoch=my_e_epoch):
                            abort_reason = "estop_interrupted"
                            return
                    else:
                        self.adapter.move(0.22, 0.0, 0.0, expected_estop_epoch=my_e_epoch)
                        time.sleep(stride_ms / 1000.0)
                        self.adapter.stop()
                    # Tiny 60ms pause between big strides (near-continuous smooth motion!)
                    time.sleep(0.06 if dist_cls != "near" else 0.20)
                else:
                    # Target temporarily out of view -> ask LightNav-0 for big search step!
                    fr = self.adapter.get_rgbd_frame()
                    if remote_vln is not None and remote_vln.enabled:
                        with remote_vln._lock:
                            remote_vln._last_rpc_mono = 0.0
                        vln_out = remote_vln.step(fr.rgb, lightnav_inst, "STRAIGHT") or {}
                    else:
                        vln_out = self.adapter.step_vln(lightnav_inst, "STRAIGHT")

                    wp0 = vln_out.get("wp0") or [0.15, 0.0, 0.0]
                    fwd_m, lat_m, yaw_rad = float(wp0[0]), float(wp0[1]), float(wp0[2])
                    hdg = math.degrees(math.atan2(lat_m, fwd_m)) if (fwd_m > 0.01 or abs(lat_m) > 0.01) else math.degrees(yaw_rad)
                    with self._lock:
                        self._progress_msg = f"🧭 视野未见『{label}』，LightNav-0 寻路推进中..."
                    if abs(hdg) >= 16.0 and hasattr(self.adapter, "turn_in_place_deg"):
                        if not self.adapter.turn_in_place_deg(max(-30.0, min(30.0, hdg)), expected_estop_epoch=my_e_epoch):
                            abort_reason = "estop_interrupted"
                            return
                    elif has_pulse:
                        if not self.adapter.drive_pulse_ms("forward", 450, expected_estop_epoch=my_e_epoch):
                            abort_reason = "estop_interrupted"
                            return
                    time.sleep(0.10)

            abort_reason = "navigation_timeout"
            with self._lock:
                self._progress_msg = f"⚠️ 前往『{label}』超时已自动停车"
        finally:
            nav_done.set()
            self.adapter.stop()
            self.adapter.unload_vln()
            with self._lock:
                self._status = "IDLE"
