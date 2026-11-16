"""Persistent HTTP/HTTPS client for remote LightNav-0 inference (4 FPS ZOH pacing + 448x256 RGB JPEG)."""

from __future__ import annotations

import base64
import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, Optional

import cv2
import numpy as np

if TYPE_CHECKING:
    import httpx

from backend.modules.cloud_agent import _normalize_to_bgr

logger = logging.getLogger("fyp.vln.remote")

# LightNav-0 native training resolution (H=256, W=448) & 4 FPS (0.25s) control cadence
LIGHTNAV_FRAME_W: int = 448
LIGHTNAV_FRAME_H: int = 256
LIGHTNAV_STEP_INTERVAL_S: float = 0.25


def encode_lightnav_jpeg(raw_bgr: np.ndarray, quality: int = 80) -> bytes:
    """Resize BGR camera frame to LightNav-0 native 448x256 and encode ~18KB JPEG."""
    bgr = _normalize_to_bgr(raw_bgr)
    resized = cv2.resize(bgr, (LIGHTNAV_FRAME_W, LIGHTNAV_FRAME_H), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else b""


class RemoteVLNClient:
    """Sends 4 FPS 448x256 frames over QUIC/TCP tunnel to the school 32GB LightNav-0 server."""

    def __init__(self, base_url: Optional[str] = None, timeout_s: float = 3.0) -> None:
        raw_url = (base_url if base_url is not None else os.getenv("VLN_REMOTE_URL", "")).strip()
        self.base_url: str = raw_url.rstrip("/")
        self.timeout_s: float = timeout_s
        self._client: Optional[httpx.Client] = None
        self._lock = threading.RLock()
        self._last_rpc_mono: float = 0.0
        self._cached_step: Optional[Dict[str, Any]] = None
        self._force_next_reset: bool = True

    def _get_http(self) -> "httpx.Client":
        if self._client is None:
            import httpx

            is_local = ("127.0.0.1" in self.base_url) or ("localhost" in self.base_url)
            self._client = httpx.Client(timeout=self.timeout_s, trust_env=not is_local)
        return self._client

    @property
    def enabled(self) -> bool:
        return bool(self.base_url)

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        resp = self._get_http().post(f"{self.base_url}{path}", json=payload)
        resp.raise_for_status()
        return resp.json()

    def health(self) -> Dict[str, Any]:
        resp = self._get_http().get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

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
            logger.warning("Remote VLN /vln/load failed (%s): %s", self.base_url, exc)
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
            # Zero-Order Hold at 4 FPS (DT = 0.25s) matching LightNav-0 training video rate
            if (
                not self._force_next_reset
                and self._cached_step is not None
                and (now - self._last_rpc_mono) < (LIGHTNAV_STEP_INTERVAL_S - 0.01)
            ):
                return dict(self._cached_step)
            force_reset = self._force_next_reset

        img_b64: Optional[str] = None
        if bgr_frame is not None:
            jpeg_bytes = encode_lightnav_jpeg(bgr_frame)
            img_b64 = base64.b64encode(jpeg_bytes).decode("ascii")

        try:
            res = self._post_json(
                "/vln/step",
                {
                    "instruction": instruction,
                    "agent_direction_hint": agent_direction_hint,
                    "image_jpeg_b64": img_b64,
                    "force_reset": force_reset,
                },
            )
            with self._lock:
                self._last_rpc_mono = time.monotonic()
                self._cached_step = res
                self._force_next_reset = False
            return res
        except Exception as exc:
            logger.warning("Remote VLN /vln/step failed (%s): %s", self.base_url, exc)
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
            logger.warning("Remote VLN /vln/unload failed (%s): %s", self.base_url, exc)
            return None
