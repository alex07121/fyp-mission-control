"""Unit tests for Dynamic 32GB School GPU Slot Scheduler (<1s CPU-RAM Sleep/Wake & Mutual Exclusion)."""

from __future__ import annotations

import base64
import json
import os
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


def test_cuda_anti_fragmentation_and_vllm_sleep_env_flags() -> None:
    assert "expandable_segments:True" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    assert os.environ.get("VLLM_SERVER_DEV_MODE") == "1"


def test_proxy_bypass_for_localhost_and_tailscale(monkeypatch) -> None:
    assert _should_bypass_proxy("http://127.0.0.1:9001") is True
    assert _should_bypass_proxy("http://localhost:9001") is True
    assert _should_bypass_proxy("http://100.106.0.106:9001") is True
    assert _should_bypass_proxy("http://192.168.1.50:9001") is True
    assert _should_bypass_proxy("https://abc-xyz.trycloudflare.com") is False

    monkeypatch.setenv("SCHOOL_GPU_URL", "http://100.106.0.106:9001/")
    assert resolve_school_gpu_url() == "http://100.106.0.106:9001"


def test_ollama_resident_8b_auto_selection_and_27b_eviction() -> None:
    mgr = remote_vln_server.ResidentOllamaManager()

    calls = []

    def _fake_open(req, timeout=2.0):
        url = req.full_url
        calls.append((url, req.data))
        resp = MagicMock()
        if url.endswith("/api/ps"):
            resp.read.return_value = json.dumps(
                {"models": [{"name": "qwen3.8:27b", "size_vram": int(17.7 * (1024**3))}]}
            ).encode("utf-8")
        elif url.endswith("/api/generate"):
            resp.read.return_value = b"{}"
        elif url.endswith("/api/tags"):
            resp.read.return_value = json.dumps(
                {"models": [{"name": "qwen3-vl:4b-instruct"}, {"name": "qwen3-vl:8b-instruct"}]}
            ).encode("utf-8")
        cm = MagicMock()
        cm.__enter__.return_value = resp
        return cm

    with patch.object(remote_vln_server._direct_opener, "open", side_effect=_fake_open):
        mgr.startup_optimize()

    assert "qwen3.8:27b" in mgr.evicted_models
    assert mgr.active_model == "qwen3-vl:8b-instruct"


def test_dynamic_slot_mutual_exclusion_vln_vs_vla_cpu_sleep() -> None:
    hub_client = TestClient(remote_vln_server.app)

    # Plain wrapper class matching real LightNav TrackingAgent (whose .llm is the vLLM engine)
    fake_vllm = MagicMock()

    class _DummyTrackingAgent:
        def __init__(self, llm: object) -> None:
            self.llm = llm

        def reset(self, instruction: str = "") -> None:
            pass

        def observe(self, rgb_arr: np.ndarray) -> None:
            pass

        def predict_waypoints(self, instruction: str, task_type: str = "vlnce_traj"):
            return [[0.08, 0.0, 0.0]], "wp_ok", 18.5

    fake_agent = _DummyTrackingAgent(fake_vllm)

    sched = remote_vln_server._scheduler
    sched.vln_agent = fake_agent
    sched.vln_slot_state = "sleeping_cpu"
    sched.vln_backend_active = "LightNav-0:vllm_local"
    sched.is_stop_centroid_fn = lambda wp: False
    sched.vla_slot_state = "unloaded"

    try:
        # 1. /vln/load wakes LightNav-0 from CPU-RAM sleep via vLLM .wake_up()
        r_vln = hub_client.post("/vln/load", json={"instruction": "Go to red mug"})
        assert r_vln.status_code == 200
        assert r_vln.json()["vln_slot_state"] == "awake_gpu"
        fake_vllm.wake_up.assert_called_once()

        # 2. /vla/load (`pi0.5`) AUTOMATICALLY puts LightNav-0 to CPU-RAM sleep (`sleep(level=1)`) first!
        r_vla = hub_client.post("/vla/load", json={"model_name": "pi0.5"})
        assert r_vla.status_code == 200
        j_vla = r_vla.json()
        assert j_vla["vln_slot_state"] == "sleeping_cpu"
        assert j_vla["vla_slot_state"] == "awake_gpu"
        assert j_vla["vla_loaded"] == "pi0.5"
        fake_vllm.sleep.assert_called_once_with(level=1)

        # 3. /vla/act runs action chunk while keeping VLN asleep
        r_act = hub_client.post("/vla/act", json={"model_name": "pi0.5", "instruction": "Grasp mug", "joint_state": [0.2] * 6})
        assert r_act.status_code == 200
        assert len(r_act.json()["action_chunk"][0]) == 6

        # 4. Next /vln/step AUTOMATICALLY puts `pi0.5` to CPU-RAM sleep and wakes LightNav-0 back!
        ok, buf = cv2.imencode(".jpg", _sample_bgr())
        assert ok
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        r_step = hub_client.post(
            "/vln/step",
            json={"instruction": "Go to doorway", "agent_direction_hint": "STRAIGHT", "image_jpeg_b64": b64},
        )
        assert r_step.status_code == 200
        j_step = r_step.json()
        assert j_step["vla_slot_state"] == "sleeping_cpu"
        assert j_step["vln_slot_state"] == "awake_gpu"
        assert fake_vllm.wake_up.call_count == 2

        # 5. /vln/unload (default soft sleep) puts LightNav-0 back into CPU-RAM sleep
        r_un = hub_client.post("/vln/unload", json={"hard_unload": False})
        assert r_un.status_code == 200
        assert r_un.json()["vln_slot_state"] == "sleeping_cpu"
        assert fake_vllm.sleep.call_count == 2
    finally:
        sched.vln_agent = None
        sched.vln_slot_state = "unloaded"
        sched.vla_slot_state = "unloaded"
        sched.vla_model_name = None


def test_agent_vlm_locks_num_ctx_and_keep_alive() -> None:
    hub_client = TestClient(remote_vln_server.app)
    remote_vln_server._ollama_mgr._probed = True
    captured_payloads = []

    def _fake_chat(req, timeout=15.0):
        captured_payloads.append(json.loads(req.data.decode("utf-8")))
        resp = MagicMock()
        resp.read.return_value = json.dumps({"message": {"content": '{"direction": "STRAIGHT"}'}}).encode("utf-8")
        cm = MagicMock()
        cm.__enter__.return_value = resp
        return cm

    with patch.object(remote_vln_server._direct_opener, "open", side_effect=_fake_chat):
        r = hub_client.post("/agent/vlm", json={"prompt": "Check ahead", "expect_json": True, "max_tokens": 32})

    assert r.status_code == 200
    assert r.json()["ok"] is True
    sent = captured_payloads[0]
    assert sent["keep_alive"] == -1
    assert sent["options"]["num_ctx"] == 4096
