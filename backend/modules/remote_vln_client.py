"""Persistent HTTP/HTTPS client for All-in-One School 32GB GPU Hub (Agent VLM + LightNav-0 + YOLOE + VLA)."""

from __future__ import annotations

import base64
import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from urllib.parse import urlparse

import cv2
import numpy as np

from backend.modules.cloud_agent import _normalize_to_bgr

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger("fyp.school_gpu.client")

LIGHTNAV_FRAME_W: int = 448
LIGHTNAV_FRAME_H: int = 256
LIGHTNAV_STEP_INTERVAL_S: float = 0.25


def resolve_school_gpu_url(explicit_url: Optional[str] = None) -> str:
    """Resolve school 32GB GPU Hub URL from explicit arg or `SCHOOL_GPU_URL` / `VLN_REMOTE_URL`."""
    if explicit_url is not None:
        return explicit_url.strip().rstrip("/")
    return (os.getenv("SCHOOL_GPU_URL") or os.getenv("VLN_REMOTE_URL") or "").strip().rstrip("/")


def _should_bypass_proxy(url: str) -> bool:
    """Bypass Clash/system HTTP proxy for localhost AND Tailscale `100.x.y.z` addresses (`HANDOFF.md` Sec 2)."""
    host = (urlparse(url).hostname or "").lower()
    return host in ("127.0.0.1", "localhost", "::1") or host.startswith("100.") or host.startswith("192.168.")


def encode_lightnav_jpeg(raw_bgr: np.ndarray, quality: int = 80) -> bytes:
    bgr = _normalize_to_bgr(raw_bgr)
    resized = cv2.resize(bgr, (LIGHTNAV_FRAME_W, LIGHTNAV_FRAME_H), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else b""


class RemoteVLNClient:
    """Unified thin client routing VLM Agent, LightNav-0, YOLOE, and VLA to the School 32GB GPU Hub."""

    def __init__(self, base_url: Optional[str] = None, timeout_s: float = 8.0) -> None:
        self.base_url: str = resolve_school_gpu_url(base_url)
        self.timeout_s: float = timeout_s
        self._client: Optional["httpx.Client"] = None
        self._lock = threading.RLock()
        self._last_rpc_mono: float = 0.0
        self._cached_step: Optional[Dict[str, Any]] = None
        self._force_next_reset: bool = True
        self._last_gpu_telemetry: Optional[Dict[str, Any]] = None

    @property
    def last_gpu_telemetry(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return dict(self._last_gpu_telemetry) if self._last_gpu_telemetry is not None else None

    def _record_telemetry(self, data: Any) -> None:
        if isinstance(data, dict) and ("vram_total_gb" in data or "vln_slot_state" in data):
            with self._lock:
                self._last_gpu_telemetry = data

    def _get_http(self) -> "httpx.Client":
        if self._client is None:
            import httpx

            bypass = _should_bypass_proxy(self.base_url)
            self._client = httpx.Client(timeout=self.timeout_s, trust_env=not bypass)
        return self._client

    @property
    def enabled(self) -> bool:
        return bool(self.base_url)

    def _post_json(self, path: str, payload: Dict[str, Any], timeout_override: Optional[float] = None) -> Dict[str, Any]:
        resp = self._get_http().post(
            f"{self.base_url}{path}",
            json=payload,
            timeout=timeout_override if timeout_override is not None else self.timeout_s,
        )
        resp.raise_for_status()
        out = resp.json()
        self._record_telemetry(out)
        return out

    def health(self) -> Dict[str, Any]:
        resp = self._get_http().get(f"{self.base_url}/health")
        resp.raise_for_status()
        out = resp.json()
        self._record_telemetry(out)
        return out

    # ---------------- 1. School Ollama VLM (`qwen3-vl:4b-instruct`) ----------------
    def ask_agent_vlm(
        self,
        jpeg_bytes: Optional[bytes],
        prompt: str,
        system_prompt: str = "",
        expect_json: bool = True,
        max_tokens: int = 400,
        timeout_s: float = 12.0,
    ) -> Dict[str, Any]:
        if not self.enabled:
            return {"ok": False, "error": "SCHOOL_GPU_URL not configured"}
        img_b64 = base64.b64encode(jpeg_bytes).decode("ascii") if jpeg_bytes else None
        return self._post_json(
            "/agent/vlm",
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "image_jpeg_b64": img_b64,
                "expect_json": expect_json,
                "max_tokens": max_tokens,
            },
            timeout_override=timeout_s,
        )

    # ---------------- 2. School LightNav-0 (`vllm_local`) ----------------
    def load(self, instruction: str = "") -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        with self._lock:
            self._last_rpc_mono = 0.0
            self._cached_step = None
            self._force_next_reset = True
        try:
            return self._post_json("/vln/load", {"instruction": instruction})
        except Exception as exc:
            logger.warning("School GPU /vln/load failed (%s): %s", self.base_url, exc)
            return None

    def step(
        self,
        bgr_frame: Optional[np.ndarray],
        instruction: str,
        agent_direction_hint: str = "STRAIGHT",
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        if agent_direction_hint.strip().upper() == "STOP":
            return {"ok": True, "vx": 0.0, "vy": 0.0, "wz": 0.0, "arrived": False}

        now = time.monotonic()
        with self._lock:
            if (
                not self._force_next_reset
                and self._cached_step is not None
                and (now - self._last_rpc_mono) < (LIGHTNAV_STEP_INTERVAL_S - 0.01)
            ):
                return dict(self._cached_step)
            force_reset = self._force_next_reset

        img_b64 = base64.b64encode(encode_lightnav_jpeg(bgr_frame)).decode("ascii") if bgr_frame is not None else None
        try:
            res = self._post_json(
                "/vln/step",
                {
                    "instruction": instruction,
                    "agent_direction_hint": agent_direction_hint,
                    "image_jpeg_b64": img_b64,
                    "force_reset": force_reset,
                },
                timeout_override=3.5,
            )
            with self._lock:
                self._last_rpc_mono = time.monotonic()
                self._cached_step = res
                self._force_next_reset = False
            return res
        except Exception as exc:
            logger.warning("School GPU /vln/step failed (%s): %s", self.base_url, exc)
            return None

    def unload(self, hard_unload: bool = False) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        with self._lock:
            self._last_rpc_mono = 0.0
            self._cached_step = None
            self._force_next_reset = True
        try:
            return self._post_json("/vln/unload", {"hard_unload": hard_unload})
        except Exception as exc:
            logger.warning("School GPU /vln/unload failed (%s): %s", self.base_url, exc)
            return None

    # ---------------- 3. School YOLOE Open-Vocab Detection ----------------
    def detect_yoloe(self, bgr_frame: np.ndarray, prompts: List[str], conf: float = 0.25) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        bgr = _normalize_to_bgr(bgr_frame)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            return None
        try:
            return self._post_json(
                "/vision/detect",
                {
                    "image_jpeg_b64": base64.b64encode(buf.tobytes()).decode("ascii"),
                    "prompts": prompts,
                    "conf": conf,
                },
                timeout_override=1.2,
            )
        except Exception:
            return None

    # ---------------- 4. School VLA Manipulation Policy (Dynamic Slot B) ----------------
    def load_vla(self, model_name: str = "SmolVLA") -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        try:
            return self._post_json("/vla/load", {"model_name": model_name}, timeout_override=6.0)
        except Exception as exc:
            logger.warning("School GPU /vla/load failed (%s): %s", self.base_url, exc)
            return None

    def act_vla(
        self,
        model_name: str,
        instruction: str,
        front_bgr: Optional[np.ndarray] = None,
        joint_state: Optional[List[float]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        fb64 = None
        if front_bgr is not None:
            ok, buf = cv2.imencode(".jpg", _normalize_to_bgr(front_bgr), [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if ok:
                fb64 = base64.b64encode(buf.tobytes()).decode("ascii")
        try:
            return self._post_json(
                "/vla/act",
                {
                    "model_name": model_name,
                    "instruction": instruction,
                    "front_jpeg_b64": fb64,
                    "joint_state": joint_state,
                },
                timeout_override=3.0,
            )
        except Exception as exc:
            logger.warning("School GPU /vla/act failed (%s): %s", self.base_url, exc)
            return None

    def unload_vla(self, hard_unload: bool = False) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        try:
            return self._post_json("/vla/unload", {"hard_unload": hard_unload}, timeout_override=4.0)
        except Exception as exc:
            logger.warning("School GPU /vla/unload failed (%s): %s", self.base_url, exc)
            return None
