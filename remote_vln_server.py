"""
Standalone School 32GB GPU Remote VLN Server (for RustDesk TCP Tunnel 9001 -> 127.0.0.1:9001).

Usage on School Computer (32GB VRAM):
    pip install fastapi uvicorn pillow numpy
    python remote_vln_server.py
"""

from __future__ import annotations

import base64
import gc
import io
import os
import platform
import socket
import threading
import time
from typing import Any, Dict, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel, Field

HOST = os.getenv("VLN_HOST", "127.0.0.1")
PORT = int(os.getenv("VLN_PORT", "9001"))

app = FastAPI(title="FYP School 32GB Remote VLN Server")
_lock = threading.RLock()
_vln_model: Any = None
_step_count: int = 0


class VLNStepRequest(BaseModel):
    instruction: str = Field(..., max_length=256)
    agent_direction_hint: str = Field(default="STRAIGHT", max_length=32)
    image_jpeg_b64: Optional[str] = None


def _gpu_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cuda_available": False,
        "gpu_name": "CPU (torch/cuda not detected)",
        "vram_allocated_gb": 0.0,
        "vram_total_gb": 0.0,
    }
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["cuda_available"] = True
            info["gpu_name"] = str(props.name)
            info["vram_allocated_gb"] = round(torch.cuda.memory_allocated(0) / (1024**3), 2)
            info["vram_total_gb"] = round(props.total_memory / (1024**3), 2)
    except Exception:
        pass
    return info


@app.get("/health")
def health() -> Dict[str, Any]:
    with _lock:
        loaded = _vln_model is not None
        steps = _step_count
    return {"ok": True, "vln_loaded": loaded, "steps_served": steps, **_gpu_info()}


@app.post("/vln/load")
def load_vln() -> Dict[str, Any]:
    global _vln_model
    with _lock:
        if _vln_model is None:
            # TODO(SCHOOL_VLN): Replace `True` with your Light-VLN / StreamVLN model loader on CUDA:
            # _vln_model = LightVLNPolicy.from_pretrained(os.getenv("VLN_CKPT", "./checkpoints/light_vln")).to("cuda")
            _vln_model = True
    return {"ok": True, "vln_loaded": True, **_gpu_info()}


@app.post("/vln/step")
def step_vln(req: VLNStepRequest) -> Dict[str, Any]:
    global _step_count
    t0 = time.perf_counter()

    rgb_shape = None
    if req.image_jpeg_b64:
        raw_bytes = base64.b64decode(req.image_jpeg_b64)
        rgb_arr = np.asarray(Image.open(io.BytesIO(raw_bytes)).convert("RGB"), dtype=np.uint8)
        rgb_shape = list(rgb_arr.shape)

    hint = req.agent_direction_hint.strip().upper()
    if hint == "STOP":
        vx, wz, arrived = 0.0, 0.0, False
    else:
        # TODO(SCHOOL_VLN): Replace below 3 lines with real `_vln_model.act(rgb_arr, req.instruction)`:
        wz_map = {"LEFT": 0.25, "RIGHT": -0.25, "STRAIGHT": 0.0}
        vx = 0.18
        wz = wz_map.get(hint, 0.0)
        arrived = False

    with _lock:
        _step_count += 1
        cur_step = _step_count

    infer_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return {
        "ok": True,
        "vx": vx,
        "vy": 0.0,
        "wz": wz,
        "arrived": arrived,
        "infer_ms": infer_ms,
        "frame_shape": rgb_shape,
        "step": cur_step,
    }


@app.post("/vln/unload")
def unload_vln() -> Dict[str, Any]:
    global _vln_model
    with _lock:
        _vln_model = None
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return {"ok": True, "vln_loaded": False, **_gpu_info()}


if __name__ == "__main__":
    print(f"[Remote VLN Server] Listening on http://{HOST}:{PORT} | GPU: {_gpu_info()['gpu_name']}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
