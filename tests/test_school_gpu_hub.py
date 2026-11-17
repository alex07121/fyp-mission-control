"""Unit tests for All-in-One School 32GB GPU Compute Hub & Thin Client Routing."""

from __future__ import annotations

import base64
import json
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

import remote_vln_server
from backend.modules.remote_vln_client import RemoteVLNClient, _should_bypass_proxy, resolve_school_gpu_url


def _sample_bgr() -> np.ndarray:
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    img[:, :] = (40, 80, 200)
    return img


def test_proxy_bypass_for_localhost_and_tailscale(monkeypatch) -> None:
    assert _should_bypass_proxy("http://127.0.0.1:9001") is True
    assert _should_bypass_proxy("http://localhost:9001") is True
    assert _should_bypass_proxy("http://100.106.0.106:9001") is True
    assert _should_bypass_proxy("http://192.168.1.50:9001") is True
    assert _should_bypass_proxy("https://abc-xyz.trycloudflare.com") is False

    monkeypatch.setenv("SCHOOL_GPU_URL", "http://100.106.0.106:9001/")
    assert resolve_school_gpu_url() == "http://100.106.0.106:9001"


def test_school_gpu_hub_all_four_stage_endpoints() -> None:
    hub_client = TestClient(remote_vln_server.app)

    # Health check
    h = hub_client.get("/health")
    assert h.status_code == 200
    assert h.json()["ok"] is True

    # Stage 1: /agent/vlm (mocked local Ollama response)
    fake_ollama_resp = MagicMock()
    fake_ollama_resp.read.return_value = json.dumps(
        {"message": {"content": '{"direction": "LEFT"}'}}
    ).encode("utf-8")
    fake_ollama_cm = MagicMock()
    fake_ollama_cm.__enter__.return_value = fake_ollama_resp
    with patch.object(remote_vln_server._direct_opener, "open", return_value=fake_ollama_cm):
        r1 = hub_client.post(
            "/agent/vlm",
            json={"prompt": "Where is red mug?", "expect_json": True, "max_tokens": 32},
        )
    assert r1.status_code == 200
    assert r1.json()["ok"] is True
    assert '"LEFT"' in r1.json()["text"]

    # Stage 2: /vln/step (fallback stub when LightNav checkpoint absent locally)
    ok, buf = cv2.imencode(".jpg", _sample_bgr())
    assert ok
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    r2 = hub_client.post(
        "/vln/step",
        json={"instruction": "Go to red mug", "agent_direction_hint": "LEFT", "image_jpeg_b64": b64},
    )
    assert r2.status_code == 200
    assert r2.json()["ok"] is True
    assert "wp0" in r2.json()

    # Stage 4: /vla/act
    r4 = hub_client.post(
        "/vla/act",
        json={"model_name": "SmolVLA", "instruction": "Grasp red mug", "joint_state": [0.1] * 6},
    )
    assert r4.status_code == 200
    assert r4.json()["ok"] is True
    assert r4.json()["model_name"] == "SmolVLA"
    assert len(r4.json()["action_chunk"][0]) == 6


def test_remote_vln_client_stop_short_circuit() -> None:
    cli = RemoteVLNClient(base_url="http://100.106.0.106:9001")
    out = cli.step(_sample_bgr(), "Go forward", agent_direction_hint="STOP")
    assert out is not None
    assert out["ok"] is True
    assert out["vx"] == 0.0 and out["wz"] == 0.0
