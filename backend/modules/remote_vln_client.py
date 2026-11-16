"""Persistent HTTP/HTTPS client for remote VLN inference (supports RustDesk 127.0.0.1 & Cloudflare HTTPS Tunnel)."""

from __future__ import annotations

import base64
import logging
import os
from typing import Any, Dict, Optional

import httpx
import numpy as np

from backend.modules.cloud_agent import compress_frame_320x240_jpeg

logger = logging.getLogger("fyp.vln.remote")


class RemoteVLNClient:
    """Sends ~15KB 320x240 JPEG frames over HTTP/HTTPS tunnel to the school 32GB GPU VLN server."""

    def __init__(self, base_url: Optional[str] = None, timeout_s: float = 2.0) -> None:
        raw_url = (base_url if base_url is not None else os.getenv("VLN_REMOTE_URL", "")).strip()
        self.base_url: str = raw_url.rstrip("/")
        self.timeout_s: float = timeout_s
        is_local = ("127.0.0.1" in self.base_url) or ("localhost" in self.base_url)
        self._client = httpx.Client(timeout=self.timeout_s, trust_env=not is_local)

    @property
    def enabled(self) -> bool:
        return bool(self.base_url)

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        resp = self._client.post(f"{self.base_url}{path}", json=payload)
        resp.raise_for_status()
        return resp.json()

    def health(self) -> Dict[str, Any]:
        resp = self._client.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def load(self) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        try:
            return self._post_json("/vln/load", {})
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
        img_b64: Optional[str] = None
        if bgr_frame is not None:
            jpeg_bytes, _ = compress_frame_320x240_jpeg(bgr_frame)
            img_b64 = base64.b64encode(jpeg_bytes).decode("ascii")
        try:
            return self._post_json(
                "/vln/step",
                {
                    "instruction": instruction,
                    "agent_direction_hint": agent_direction_hint,
                    "image_jpeg_b64": img_b64,
                },
            )
        except Exception as exc:
            logger.warning("Remote VLN /vln/step failed (%s): %s", self.base_url, exc)
            return None

    def unload(self) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        try:
            return self._post_json("/vln/unload", {})
        except Exception as exc:
            logger.warning("Remote VLN /vln/unload failed (%s): %s", self.base_url, exc)
            return None
