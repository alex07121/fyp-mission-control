"""Module 2: Vertex AI Cloud Agent (gemini-3.5-flash-lite MINIMAL) + <2.5s Watchdog."""

from __future__ import annotations

import collections
import json
import logging
import math
import os
import re
import threading
import time
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.config import (
    AGENT_JPEG_HEIGHT,
    AGENT_JPEG_WIDTH,
    AGENT_WATCHDOG_HARD_LIMIT_S,
    DEFAULT_WATCHDOG_TIMEOUT_S,
    GCP_LOCATION,
    GCP_PROJECT_ID,
    GEMINI_MODEL_ID,
)

logger = logging.getLogger("fyp.agent")

_DIR_TOKEN_RE = re.compile(r"\b(LEFT|RIGHT|STRAIGHT|FORWARD|STOP)\b", re.IGNORECASE)
_VLM_HAZARD_OR_STOP_RE = re.compile(
    r"\b(STOP|ESTOP|E-STOP|HALT|ABORT|BRAKE|FREEZE|WAIT|HOLD"
    r"|DANGER\w*|UNSAFE|HAZARD\w*|OBSTACLE\w*|BLOCK\w*|WALL\w*|CRASH\w*|COLLID\w*|COLLISION\w*"
    r"|WARNING\w*|CAUTION|ERROR\w*|FAIL\w*|FAULT\w*|INVALID|BLIND"
    r"|UNABLE|IMPOSSIBLE|WRONG|BAD|DEAD|SORRY|CANNOT|CAN['’`]?T|CANT|REFUS\w*)\b",
    re.IGNORECASE,
)
_VLM_NEGATION_EN_RE = re.compile(
    r"\b(not|never|don['’`]?t|dont|no|none|neither|without|avoid\w*|prohibit\w*|forbid\w*"
    r"|cannot|can['’`]?t|cant|won['’`]?t|wont|shouldn['’`]?t|shouldnt|mustn['’`]?t"
    r"|unsafe|unable|impossible|wrong|bad|error\w*|fail\w*)\b",
    re.IGNORECASE,
)
_VLM_HAZARD_OR_NEG_ZH = (
    "急停",
    "停止",
    "停下",
    "暂停",
    "危险",
    "不安全",
    "障碍",
    "受阻",
    "撞",
    "墙",
    "堵",
    "不通",
    "死胡同",
    "盲区",
    "错",
    "错误",
    "失败",
    "异常",
    "故障",
    "抱歉",
    "无法",
    "不能",
    "不可",
    "不要",
    "禁止",
    "切勿",
    "别",
    "勿",
    "没法",
    "不行",
)
_JSON_POSITIVE_SAFETY_KEYS = {"safe", "ok", "valid", "clear", "passable", "reachable", "allowed", "success"}
_JSON_NEGATIVE_HAZARD_KEYS = {
    "error",
    "err",
    "hazard",
    "warning",
    "blocked",
    "obstacle",
    "obstacles",
    "collision",
    "danger",
    "fault",
    "abort",
    "crash",
}


def _strict_unique_json_pairs(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    """Reject duplicate JSON keys (including case-insensitive collisions) in VLM output."""
    out: Dict[str, Any] = {}
    seen_lower = set()
    for k, v in pairs:
        lk = k.strip().lower()
        if k in out or lk in seen_lower:
            raise ValueError(f"Duplicate JSON key rejected in VLM reply: {k!r}")
        seen_lower.add(lk)
        out[k] = v
    return out


def _normalize_to_bgr(img: np.ndarray) -> np.ndarray:
    if not isinstance(img, np.ndarray) or img.size == 0:
        return np.zeros((AGENT_JPEG_HEIGHT, AGENT_JPEG_WIDTH, 3), dtype=np.uint8)
    if img.dtype == np.uint8:
        arr = img
    else:
        # R2-LOW-01: Sanitize NaN/+Inf/-Inf BEFORE uint8 cast to eliminate RuntimeWarning
        clean = np.nan_to_num(img, nan=0.0, posinf=255.0, neginf=0.0)
        arr = np.clip(clean, 0.0, 255.0).astype(np.uint8)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr
    return np.zeros((AGENT_JPEG_HEIGHT, AGENT_JPEG_WIDTH, 3), dtype=np.uint8)


def compress_frame_320x240_jpeg(raw_frame: np.ndarray) -> Tuple[bytes, float]:
    """
    Safely convert 2D/3D/4D image to 320x240 BGR and encode natural JPEG strictly <= 25.0 KB
    without injecting synthetic pixel noise (`R2-LOW-01`).
    """
    bgr = _normalize_to_bgr(raw_frame)
    resized = cv2.resize(bgr, (AGENT_JPEG_WIDTH, AGENT_JPEG_HEIGHT), interpolation=cv2.INTER_AREA)

    best_bytes = b""
    # Strictly descending quality ladder on clean natural pixels — zero synthetic grain corruption!
    for quality in (92, 84, 74, 62, 50, 38, 26, 16):
        ok, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            best_bytes = buf.tobytes()
            if len(best_bytes) <= 25000:
                break

    # Extreme high-entropy white-noise guard: gentle denoise guarantees <= 25.0 KB
    if len(best_bytes) > 25000:
        smoothed = cv2.GaussianBlur(resized, (5, 5), 1.6)
        for quality in (45, 32, 20, 10):
            ok, buf = cv2.imencode(".jpg", smoothed, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if ok:
                best_bytes = buf.tobytes()
                if len(best_bytes) <= 25000:
                    break

    return best_bytes, round(len(best_bytes) / 1024.0, 2)


class CloudAgentEngine:
    """
    Fast VLM planner & asynchronous look-while-moving direction advisor.
    Enforces a strict <2.5s wall-clock timeout watchdog (including cold-start) and max 4 inflight RPCs.
    """

    MAX_CONCURRENT_RPC_THREADS: int = 4
    MAX_WATCHDOG_TIMEOUT_S: float = 2.35

    def __init__(
        self,
        emergency_brake_cb: Callable[[str], None],
        default_mode: str = "cloud",
        watchdog_timeout_s: float = DEFAULT_WATCHDOG_TIMEOUT_S,
    ) -> None:
        self._brake_cb = emergency_brake_cb
        self._lock = threading.RLock()
        self.mode: str = default_mode if default_mode in {"cloud", "mock", "auto", "school"} else "cloud"
        self._school_hub_cache: Dict[str, Any] = {}
        self._watchdog_timeout_s: float = self.MAX_WATCHDOG_TIMEOUT_S
        self.watchdog_timeout_s = watchdog_timeout_s

        self.simulate_delay_s: float = 0.0
        self.simulate_error: bool = False

        self._history: Deque[Dict[str, Any]] = collections.deque(maxlen=60)
        self._watchdog_trips: int = 0
        self._inflight_since: Optional[float] = None
        self._client_lock = threading.Lock()
        self._client: Any = None
        self._genai_types: Any = None
        # R2-MED-01: Hard bound on concurrent/orphaned background RPC threads
        self._rpc_sem = threading.BoundedSemaphore(self.MAX_CONCURRENT_RPC_THREADS)

    @property
    def watchdog_timeout_s(self) -> float:
        with self._lock:
            return self._watchdog_timeout_s

    @watchdog_timeout_s.setter
    def watchdog_timeout_s(self, val: float) -> None:
        fval = float(val)
        if not math.isfinite(fval):
            fval = DEFAULT_WATCHDOG_TIMEOUT_S
        with self._lock:
            self._watchdog_timeout_s = min(self.MAX_WATCHDOG_TIMEOUT_S, max(0.50, fval))

    def _get_vertex_client(self) -> Tuple[Any, Any]:
        """Instantiate Vertex AI SDK client lazily with strict 2200ms HTTP socket timeout."""
        with self._client_lock:
            if self._client is None:
                from google import genai
                from google.genai import types

                self._client = genai.Client(
                    vertexai=True,
                    project=GCP_PROJECT_ID,
                    location=GCP_LOCATION,
                    http_options=types.HttpOptions(timeout=2200),
                )
                self._genai_types = types
            return self._client, self._genai_types

    def warmup_in_background(self) -> None:
        """R3-MED-01: Non-blocking daemon pre-initialization of `genai.Client` on server startup."""
        if self.mode not in {"auto", "cloud"}:
            return

        def _bg_init() -> None:
            try:
                self._get_vertex_client()
            except Exception as exc:
                logger.debug("Background Vertex client init skipped: %s", exc)

        threading.Thread(target=_bg_init, name="VertexClientWarmup", daemon=True).start()

    def set_fault_injection(
        self,
        simulate_delay_s: Optional[float] = None,
        simulate_error: Optional[bool] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            if simulate_delay_s is not None:
                self.simulate_delay_s = max(0.0, float(simulate_delay_s))
            if simulate_error is not None:
                self.simulate_error = bool(simulate_error)
            return {
                "simulate_delay_s": self.simulate_delay_s,
                "simulate_error": self.simulate_error,
            }

    @staticmethod
    def _parse_direction_and_labels(raw_text: str, command: str) -> Dict[str, Any]:
        """
        R2-HIGH-04: Hardened VLM output parser rejecting duplicate JSON keys, invalid JSON fallback,
        JSON + hazard/STOP contradictions, and negated direction warnings ("Do NOT go LEFT").
        """
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise ValueError("Empty VLM response")

        cleaned = raw_text.strip()
        direction: Optional[str] = None

        # 1. Strict JSON branch — if braces exist, MUST parse cleanly with valid direction; NEVER fall back to prose!
        if "{" in cleaned or "}" in cleaned:
            m_json = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if not m_json:
                raise ValueError(f"Malformed JSON braces in VLM reply: {cleaned[:60]!r}")
            try:
                obj = json.loads(m_json.group(0), object_pairs_hook=_strict_unique_json_pairs)
            except Exception as exc:
                raise ValueError(f"Invalid or duplicate-key VLM JSON: {exc}") from exc

            if not isinstance(obj, dict) or "direction" not in obj or not isinstance(obj["direction"], str):
                raise ValueError(f"Missing string 'direction' in VLM JSON: {cleaned[:60]!r}")

            cand = obj["direction"].strip().upper()
            if cand == "FORWARD":
                cand = "STRAIGHT"
            if cand not in {"LEFT", "RIGHT", "STRAIGHT", "STOP"}:
                raise ValueError(f"Illegal VLM JSON direction {cand!r} (prose fallback forbidden)")

            if cand != "STOP":
                # R3-HIGH-02: Inspect structured boolean/error/hazard JSON fields before trusting `direction`
                aux_parts: List[str] = []
                for k, v in obj.items():
                    lk = k.strip().lower()
                    if lk == "direction":
                        continue
                    if lk in _JSON_POSITIVE_SAFETY_KEYS or lk.startswith(("is_safe", "is_clear", "is_valid", "can_")):
                        if (
                            v is False
                            or (isinstance(v, (int, float)) and v == 0)
                            or (isinstance(v, str) and v.strip().lower() in {"false", "no", "0", "unsafe", "invalid", "blocked", "error"})
                        ):
                            raise ValueError(f"VLM JSON safety flag rejected ({k}={v!r}) for direction={cand}")
                    if lk in _JSON_NEGATIVE_HAZARD_KEYS:
                        if (
                            v is True
                            or (isinstance(v, (int, float)) and not isinstance(v, bool) and v != 0)
                            or (isinstance(v, (list, dict)) and len(v) > 0)
                            or (isinstance(v, str) and v.strip().lower() not in {"", "none", "null", "false", "0", "no", "ok"})
                        ):
                            raise ValueError(f"VLM JSON hazard/error field rejected ({k}={v!r}) for direction={cand}")
                    if isinstance(v, bool) and v is False and lk not in (_JSON_NEGATIVE_HAZARD_KEYS | {"stop", "estop"}):
                        raise ValueError(f"VLM JSON negative boolean ({k}=False) rejected for direction={cand}")
                    if v is not None and not isinstance(v, bool):
                        aux_parts.append(json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else str(v))

                outside_text = (cleaned[: m_json.start()] + " " + cleaned[m_json.end() :]).strip()
                aux_fields_text = " ".join(aux_parts)
                combined_context = f"{outside_text} {aux_fields_text}".strip()
                if combined_context:
                    if (
                        _VLM_HAZARD_OR_STOP_RE.search(combined_context)
                        or _VLM_NEGATION_EN_RE.search(combined_context)
                        or any(zh in combined_context for zh in _VLM_HAZARD_OR_NEG_ZH)
                    ):
                        raise ValueError(
                            f"Contradictory hazard/stop/negation alongside JSON direction={cand}: {combined_context[:60]!r}"
                        )
                    extra_dirs = {
                        ("STRAIGHT" if t.upper() == "FORWARD" else t.upper())
                        for t in _DIR_TOKEN_RE.findall(outside_text)
                    }
                    if extra_dirs and extra_dirs != {cand}:
                        raise ValueError(f"Conflicting prose direction {extra_dirs} vs JSON {cand}")
            direction = cand
        else:
            # 2. Plain-text branch — reject any negation, hazard/error term, or >3-word explanatory prose!
            tokens = [t.upper() for t in _DIR_TOKEN_RE.findall(cleaned)]
            norm_tokens = ["STRAIGHT" if t == "FORWARD" else t for t in tokens]
            unique_dirs = set(norm_tokens)

            if not unique_dirs:
                raise ValueError(f"Unparseable VLM direction reply: {cleaned[:60]!r}")

            has_negation = bool(_VLM_NEGATION_EN_RE.search(cleaned)) or any(
                zh in cleaned for zh in _VLM_HAZARD_OR_NEG_ZH
            )
            if has_negation and not (unique_dirs == {"STOP"} and cleaned.strip().upper() == "STOP"):
                raise ValueError(f"Negated or hazardous VLM direction warning rejected: {cleaned[:60]!r}")

            if unique_dirs != {"STOP"}:
                if _VLM_HAZARD_OR_STOP_RE.search(cleaned) or any(zh in cleaned for zh in _VLM_HAZARD_OR_NEG_ZH):
                    raise ValueError(f"Contradictory movement + hazard/stop in VLM reply: {cleaned[:60]!r}")

                word_tokens = re.sub(r"[^\w\u4e00-\u9fff]+", " ", cleaned).split()
                cjk_count = sum(1 for ch in cleaned if "\u4e00" <= ch <= "\u9fff")
                if len(word_tokens) > 3 or cjk_count > 4:
                    raise ValueError(f"Verbose prose VLM reply rejected (>3 words): {cleaned[:60]!r}")

            if len(unique_dirs) != 1:
                raise ValueError(f"Ambiguous multi-direction VLM reply {unique_dirs}: {cleaned[:60]!r}")
            direction = next(iter(unique_dirs))

        low_cmd = command.lower()
        if ("door" in low_cmd or "门" in low_cmd) and not any(
            w in low_cmd for w in ("mug", "cup", "bottle", "杯", "瓶")
        ):
            labels = ["open doorway", "door frame", "doorway", "door"]
        elif "bottle" in low_cmd or "瓶" in low_cmd:
            labels = ["water bottle", "bottle"]
        else:
            labels = ["red mug", "mug", "cup"]

        return {"direction": direction, "target_prompts": labels, "raw_reply": cleaned[:180]}

    def _mock_infer(self, bgr_frame: np.ndarray, command: str) -> str:
        bgr = _normalize_to_bgr(bgr_frame)
        W = bgr.shape[1]
        mid_x = W * 0.5
        deadband = W * 0.07
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        low = command.lower()
        if "bottle" in low or "瓶" in low:
            mask = cv2.inRange(hsv, np.array([95, 90, 80]), np.array([130, 255, 255]))
        elif ("door" in low or "门" in low) and not any(w in low for w in ("mug", "cup", "bottle", "杯", "瓶")):
            mask = cv2.inRange(hsv, np.array([14, 110, 120]), np.array([36, 255, 255]))
        else:
            m1 = cv2.inRange(hsv, np.array([0, 120, 100]), np.array([10, 255, 255]))
            m2 = cv2.inRange(hsv, np.array([170, 120, 100]), np.array([180, 255, 255]))
            mask = cv2.bitwise_or(m1, m2)

        _, xs = np.where(mask > 0)
        if len(xs) >= 20:
            cx = float(np.mean(xs))
            if cx < (mid_x - deadband):
                return "LEFT"
            if cx > (mid_x + deadband):
                return "RIGHT"
            return "STRAIGHT"

        # R2-HIGH-03: When target is out of FOV (e.g. `far_search`), inspect visible room landmarks
        # (doorway frame or brown lab desk) to steer toward the workspace, or sweep consistently RIGHT!
        m_door = cv2.inRange(hsv, np.array([14, 110, 120]), np.array([36, 255, 255]))
        m_table = cv2.inRange(hsv, np.array([10, 60, 70]), np.array([26, 210, 190]))
        _, lx = np.where(cv2.bitwise_or(m_door, m_table) > 0)
        if len(lx) >= 25:
            return "RIGHT" if float(np.mean(lx)) >= mid_x else "LEFT"
        return "RIGHT"

    def _worker_call(
        self,
        jpeg_bytes: bytes,
        bgr_frame: np.ndarray,
        prompt: str,
        effective_mode: str,
        delay_s: float,
        force_err: bool,
    ) -> Tuple[str, str]:
        if delay_s > 0.0:
            time.sleep(delay_s)
        if force_err:
            raise RuntimeError("Injected Cloud Agent API network/service failure")

        if effective_mode == "mock":
            return self._mock_infer(bgr_frame, prompt), "mock_local"

        # Route Agent VLM inference to School 32GB GPU (`qwen3-vl:4b-instruct`) whenever SCHOOL_GPU_URL / VLN_REMOTE_URL is set!
        school_url = (os.getenv("SCHOOL_GPU_URL") or os.getenv("VLN_REMOTE_URL") or "").strip().rstrip("/")
        if school_url and effective_mode in ("auto", "school"):
            from backend.modules.remote_vln_client import RemoteVLNClient

            with self._client_lock:
                hub = self._school_hub_cache.get(school_url)
                if hub is None:
                    hub = RemoteVLNClient(base_url=school_url, timeout_s=2.2)
                    self._school_hub_cache[school_url] = hub
            q_school = (
                f"Task: {prompt}. Look at this robot camera view. "
                "Should the robot steer LEFT, STRAIGHT, or RIGHT toward the target? "
                'Reply ONLY JSON: {"direction": "LEFT"} or {"direction": "STRAIGHT"} or {"direction": "RIGHT"}.'
            )
            s_out = hub.ask_agent_vlm(jpeg_bytes, q_school, expect_json=True, max_tokens=32, timeout_s=2.2)
            if s_out.get("ok") and s_out.get("text"):
                return str(s_out["text"]).strip(), str(s_out.get("model", "school_gpu_vlm"))
            if effective_mode == "school":
                raise RuntimeError(s_out.get("error", "School GPU VLM failed"))
        elif effective_mode == "school" and not school_url:
            raise RuntimeError("SCHOOL_GPU_URL is not set for school agent mode")

        try:
            client, types = self._get_vertex_client()
            img_part = types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")
            cfg = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
                max_output_tokens=16,
                temperature=0.0,
            )
            query = (
                f"Task: {prompt}. Look at this 320x240 robot view. "
                "Should the robot steer LEFT, STRAIGHT, or RIGHT toward the target? Reply ONE word: LEFT, STRAIGHT, or RIGHT."
            )
            res = client.models.generate_content(
                model=GEMINI_MODEL_ID,
                contents=[img_part, query],
                config=cfg,
            )
            txt = res.text if res.text is not None else ""
            return txt.strip(), f"vertex:{GEMINI_MODEL_ID}"
        except Exception as exc:
            if effective_mode == "auto":
                logger.warning("Vertex AI unavailable in auto mode (%s); falling back to mock.", exc)
                return self._mock_infer(bgr_frame, prompt), "mock_fallback"
            raise

    def infer_with_watchdog(
        self,
        bgr_frame: np.ndarray,
        command: str,
        mode_override: Optional[str] = None,
        extra_delay_s: float = 0.0,
        force_error: bool = False,
        enforce_brake_on_fail: bool = True,
        should_suppress_brake: Optional[Callable[[], bool]] = None,
        allow_auto_fallback_on_timeout: bool = False,
    ) -> Dict[str, Any]:
        """
        R2-CRIT-03 + R3-MED-01:
        - Pure local `mock` mode (without fault delay/error) executes synchronously in-memory, immune to `_rpc_sem` starvation;
        - Stationary `auto` mode (`allow_auto_fallback_on_timeout=True` with zero fault injection) gracefully falls back to
          `_mock_infer` if cold-start OAuth or WAN blackhole hits the watchdog limit;
        - Moving `VLN_CRUISE`, explicit `cloud` mode, and any fault injection strictly trip `<2.50s` watchdog brake!
        """
        t0 = time.perf_counter()

        with self._lock:
            eff_mode = mode_override or self.mode
            sim_delay = max(self.simulate_delay_s, float(extra_delay_s))
            sim_err = bool(self.simulate_error or force_error)
            timeout_budget = min(AGENT_WATCHDOG_HARD_LIMIT_S - 0.15, self._watchdog_timeout_s)

        jpeg_bytes, jpeg_kb = compress_frame_320x240_jpeg(bgr_frame)
        can_auto_degrade = bool(
            eff_mode == "auto" and allow_auto_fallback_on_timeout and sim_delay <= 0.0 and not sim_err
        )

        # R3-MED-01: Pure local mock mode bypasses `_rpc_sem` & thread spawning -> 100% immune to hung cloud threads!
        if eff_mode == "mock" and sim_delay <= 0.0 and not sim_err:
            suppressed = bool(should_suppress_brake and should_suppress_brake())
            txt = self._mock_infer(bgr_frame, command)
            parsed = self._parse_direction_and_labels(txt, command)
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            rec_mock = {
                "ok": True,
                "watchdog_triggered": False,
                "direction": parsed["direction"],
                "target_prompts": parsed["target_prompts"],
                "raw_reply": parsed["raw_reply"],
                "jpeg_kb": jpeg_kb,
                "latency_ms": latency_ms,
                "provider": "mock_local",
                "ts": round(time.time(), 3),
            }
            if not suppressed:
                with self._lock:
                    self._history.appendleft(rec_mock)
            return rec_mock

        # R2-MED-01: Prevent unbounded orphan `AgentWatchdogCall` thread accumulation on repeated hangs
        if not self._rpc_sem.acquire(blocking=False):
            suppressed = bool(should_suppress_brake and should_suppress_brake())
            if can_auto_degrade:
                txt_fb = self._mock_infer(bgr_frame, command)
                parsed_fb = self._parse_direction_and_labels(txt_fb, command)
                latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                rec_fb = {
                    "ok": True,
                    "watchdog_triggered": False,
                    "direction": parsed_fb["direction"],
                    "target_prompts": parsed_fb["target_prompts"],
                    "raw_reply": parsed_fb["raw_reply"],
                    "jpeg_kb": jpeg_kb,
                    "latency_ms": latency_ms,
                    "provider": "mock_fallback",
                    "ts": round(time.time(), 3),
                }
                if not suppressed:
                    with self._lock:
                        self._history.appendleft(rec_fb)
                return rec_fb

            reason = f"Agent RPC concurrency cap ({self.MAX_CONCURRENT_RPC_THREADS}) saturated by hung calls"
            if not suppressed:
                with self._lock:
                    self._watchdog_trips += 1
                logger.error("[WATCHDOG BRAKE] %s -> Fast-failing & forcing STOP!", reason)
                if enforce_brake_on_fail:
                    self._brake_cb(reason)
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            rec_cap = {
                "ok": False,
                "watchdog_triggered": not suppressed,
                "suppressed": suppressed,
                "error": reason,
                "direction": "STOP",
                "target_prompts": [],
                "jpeg_kb": jpeg_kb,
                "latency_ms": latency_ms,
                "provider": eff_mode,
                "ts": round(time.time(), 3),
            }
            if not suppressed:
                with self._lock:
                    self._history.appendleft(rec_cap)
            return rec_cap

        with self._lock:
            self._inflight_since = time.monotonic()

        result_box: Dict[str, Any] = {}
        done_evt = threading.Event()

        def _runner() -> None:
            try:
                txt, provider = self._worker_call(jpeg_bytes, bgr_frame, command, eff_mode, sim_delay, sim_err)
                parsed = self._parse_direction_and_labels(txt, command)
                result_box["ok"] = True
                result_box["parsed"] = parsed
                result_box["provider"] = provider
            except Exception as exc:
                result_box["ok"] = False
                result_box["error"] = str(exc)
            finally:
                self._rpc_sem.release()
                done_evt.set()

        worker = threading.Thread(target=_runner, name="AgentWatchdogCall", daemon=True)
        try:
            worker.start()
        except Exception:
            self._rpc_sem.release()
            raise

        elapsed_pre = time.perf_counter() - t0
        wait_budget = max(0.01, timeout_budget - elapsed_pre)
        finished = done_evt.wait(timeout=wait_budget)

        with self._lock:
            self._inflight_since = None

        suppressed = bool(should_suppress_brake and should_suppress_brake())

        if not finished or not result_box.get("ok", False):
            # R3-MED-01: If default `auto` mode timed out on cold-start OAuth while base is stationary, degrade to mock!
            if (not finished) and can_auto_degrade:
                logger.warning("Vertex AI cold-start/RPC exceeded %.2fs in stationary auto mode; degrading to mock_fallback.", timeout_budget)
                txt_fb = self._mock_infer(bgr_frame, command)
                parsed_fb = self._parse_direction_and_labels(txt_fb, command)
                latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                rec_fb = {
                    "ok": True,
                    "watchdog_triggered": False,
                    "direction": parsed_fb["direction"],
                    "target_prompts": parsed_fb["target_prompts"],
                    "raw_reply": parsed_fb["raw_reply"],
                    "jpeg_kb": jpeg_kb,
                    "latency_ms": latency_ms,
                    "provider": "mock_fallback",
                    "ts": round(time.time(), 3),
                }
                if not suppressed:
                    with self._lock:
                        self._history.appendleft(rec_fb)
                return rec_fb

            pre_brake_ms = (time.perf_counter() - t0) * 1000.0
            if not finished:
                reason = f"Agent Watchdog Timeout ({pre_brake_ms:.0f}ms > {timeout_budget*1000:.0f}ms limit)"
            else:
                err_str = result_box.get("error", "Invalid/Empty Agent reply")
                reason = f"Agent Failure ({err_str[:90]})"

            if not suppressed:
                with self._lock:
                    self._watchdog_trips += 1
                logger.error("[WATCHDOG BRAKE] %s -> Forcing immediate base STOP!", reason)
                if enforce_brake_on_fail:
                    self._brake_cb(reason)

            latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            rec = {
                "ok": False,
                "watchdog_triggered": not suppressed,
                "suppressed": suppressed,
                "error": reason,
                "direction": "STOP",
                "target_prompts": [],
                "jpeg_kb": jpeg_kb,
                "latency_ms": latency_ms,
                "provider": eff_mode,
                "ts": round(time.time(), 3),
            }
            if not suppressed:
                with self._lock:
                    self._history.appendleft(rec)
            return rec

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        parsed = result_box["parsed"]
        rec = {
            "ok": True,
            "watchdog_triggered": False,
            "direction": parsed["direction"],
            "target_prompts": parsed["target_prompts"],
            "raw_reply": parsed["raw_reply"],
            "jpeg_kb": jpeg_kb,
            "latency_ms": latency_ms,
            "provider": result_box["provider"],
            "ts": round(time.time(), 3),
        }
        if not suppressed:
            with self._lock:
                self._history.appendleft(rec)
        return rec

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            ok_latencies = [h["latency_ms"] for h in self._history if h.get("ok")]
            avg_ms = round(sum(ok_latencies) / len(ok_latencies), 1) if ok_latencies else 0.0
            return {
                "mode": self.mode,
                "model": GEMINI_MODEL_ID,
                "project": GCP_PROJECT_ID,
                "watchdog_timeout_s": self._watchdog_timeout_s,
                "watchdog_trips": self._watchdog_trips,
                "simulate_delay_s": self.simulate_delay_s,
                "simulate_error": self.simulate_error,
                "avg_latency_ms": avg_ms,
                "last_call": self._history[0] if self._history else None,
                "recent_calls": list(self._history)[:15],
            }
