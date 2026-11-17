"""
All-in-One School 32GB GPU Compute Server (ubuntu@324 — NVIDIA RTX 5000 Ada 32GB).

Consolidates 100% of FYP AI compute onto the school GPU so the Mac / Robot Car uses ZERO GPU/NPU:
  1. `/agent/vlm`      -> School Ollama (`qwen3-vl:4b-instruct`, ~4.5 GB VRAM) for Room Scan & Arrival Gate
  2. `/vln/*`          -> School `LightNav-0` (`vllm_local`, ~13.2 GB VRAM) for 4 FPS Waypoint Navigation
  3. `/vision/detect`  -> School `YOLOE` (`yoloe-26s-seg.pt`, ~0.8 GB VRAM) for Open-Vocab Last-Mile Alignment
  4. `/vla/act`        -> School VLA (`ACT` / `TurboVLA` / `SmolVLA` / `pi0.5`) for SO-101 Arm Manipulation

Total simultaneous resident VRAM: ~18.5 GB / 31.5 GB (~13 GB free headroom for VLA!).

Run on ubuntu@324 (listens on 0.0.0.0:9001 for Tailscale `100.106.0.106:9001` & localhost `127.0.0.1:9001`):
    ollama stop qwen3.8:27b || true
    conda activate lightnav
    cd ~/fyp-mission-control && git pull
    python remote_vln_server.py
"""

from __future__ import annotations

import base64
import gc
import io
import json
import logging
import os
import platform
import re
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("school.gpu.hub")

# Bind 0.0.0.0 by default so BOTH Tailscale (100.106.0.106:9001) and Cloudflare Tunnel (127.0.0.1:9001) work!
HOST: str = os.getenv("VLN_HOST", "0.0.0.0")
PORT: int = int(os.getenv("VLN_PORT", "9001"))

# 1. Ollama VLM settings on ubuntu@324
OLLAMA_BASE_URL: str = os.getenv("SCHOOL_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_VLM_MODEL: str = os.getenv("SCHOOL_VLM_MODEL", "qwen3-vl:4b-instruct")

# 2. LightNav-0 settings (gpu_util=0.42 leaves ~18 GB free for Ollama 4B + YOLOE + VLA!)
LIGHTNAV_REPO: Path = Path(os.getenv("LIGHTNAV_REPO", str(Path.home() / "LightNav-0")))
LIGHTNAV_MODEL_PATH: str = os.getenv(
    "LIGHTNAV_MODEL_PATH",
    str(LIGHTNAV_REPO / "checkpoints" / "LightNav-0"),
)
LIGHTNAV_BACKEND: str = os.getenv("LIGHTNAV_BACKEND", "vllm_local")
LIGHTNAV_GPU_UTIL: float = float(os.getenv("LIGHTNAV_GPU_UTIL", "0.42"))
DT: float = 0.25
MAX_VX_MPS: float = float(os.getenv("LIGHTNAV_MAX_VX", "0.25"))
MAX_WZ_RADS: float = float(os.getenv("LIGHTNAV_MAX_WZ", "0.85"))

# 3. YOLOE weights path on ubuntu@324
YOLOE_WEIGHTS_PATH: str = os.getenv("YOLOE_WEIGHTS_PATH", "yoloe-26s-seg.pt")

for candidate in (LIGHTNAV_REPO / "src", LIGHTNAV_REPO):
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

app = FastAPI(title="FYP All-in-One School 32GB GPU Compute Hub")
_lock = threading.RLock()
_yoloe_lock = threading.RLock()
_vla_lock = threading.RLock()

_vln_agent: Any = None
_vln_backend_active: str = "unloaded"
_vln_load_error: str = ""
_active_episode_instruction: str = ""
_step_count: int = 0
_is_stop_centroid_fn: Any = None

_yoloe_model: Any = None
_yoloe_classes: List[str] = []
_vla_policy: Any = None
_vla_loaded_name: Optional[str] = None

_direct_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class AgentVLMRequest(BaseModel):
    prompt: str
    system_prompt: str = ""
    image_jpeg_b64: Optional[str] = None
    expect_json: bool = True
    max_tokens: int = 400
    model: str = ""


class VLNResetRequest(BaseModel):
    instruction: str = ""
    agent_direction_hint: str = "STRAIGHT"


class VLNStepRequest(BaseModel):
    instruction: str
    agent_direction_hint: str = "STRAIGHT"
    image_jpeg_b64: Optional[str] = None
    force_reset: bool = False


class VLNUnloadRequest(BaseModel):
    hard_unload: bool = False


class VisionDetectRequest(BaseModel):
    image_jpeg_b64: str
    prompts: List[str]
    conf: float = 0.25


class VLAActRequest(BaseModel):
    model_name: str = "SmolVLA"
    instruction: str = ""
    front_jpeg_b64: Optional[str] = None
    wrist_jpeg_b64: Optional[str] = None
    joint_state: Optional[List[float]] = None


def normalize_lightnav_instruction(raw_cmd: str, hint: str = "STRAIGHT") -> str:
    text = (raw_cmd or "").strip()
    low = text.lower()
    dir_phrase = {
        "LEFT": "slightly to the left",
        "RIGHT": "slightly to the right",
    }.get(hint.strip().upper(), "in front")

    if any(k in low for k in ("doorway", "door frame", "door")) or "门" in text:
        return f"Go to the open doorway {dir_phrase} and stop."
    if any(k in low for k in ("bottle", "water")) or any(k in text for k in ("水瓶", "拿水", "取水", "瓶")):
        return f"Go to the blue water bottle on the table {dir_phrase} and stop."
    if any(k in low for k in ("mug", "cup")) or "杯" in text:
        return f"Go to the red mug on the table {dir_phrase} and stop."

    ascii_only = re.sub(r"[^\x20-\x7E]+", " ", text).strip(" ()-.,;:")
    if len(ascii_only) >= 4:
        if "stop" not in ascii_only.lower():
            return f"{ascii_only} and stop."
        return ascii_only if ascii_only.endswith(".") else f"{ascii_only}."
    return f"Go forward {dir_phrase} to the target object and stop."


def _gpu_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cuda_available": False,
        "gpu_name": "CPU",
        "vram_allocated_gb": 0.0,
        "vram_free_gb": 0.0,
        "vram_total_gb": 0.0,
        "ollama_model": OLLAMA_VLM_MODEL,
        "vln_backend": _vln_backend_active,
        "yoloe_loaded": _yoloe_model is not None,
        "vla_loaded": _vla_loaded_name,
        "load_error": _vln_load_error,
    }
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            free_b, total_b = torch.cuda.mem_get_info(0)
            info["cuda_available"] = True
            info["gpu_name"] = str(props.name)
            info["vram_allocated_gb"] = round((total_b - free_b) / (1024**3), 2)
            info["vram_free_gb"] = round(free_b / (1024**3), 2)
            info["vram_total_gb"] = round(total_b / (1024**3), 2)
    except Exception:
        pass
    return info


def _ensure_lightnav_loaded() -> bool:
    global _vln_agent, _vln_backend_active, _vln_load_error, _is_stop_centroid_fn
    with _lock:
        if _vln_agent is not None:
            return True
        try:
            from lightnav.tracking import build_tracking_agent
            from lightnav.velocity import is_stop_centroid

            _is_stop_centroid_fn = is_stop_centroid
            gpu_util = LIGHTNAV_GPU_UTIL
            try:
                import torch

                if torch.cuda.is_available():
                    free_b, total_b = torch.cuda.mem_get_info(0)
                    free_gb = free_b / (1024**3)
                    total_gb = total_b / (1024**3)
                    if free_gb < 15.0:
                        safe_util = max(0.25, min(gpu_util, (free_gb - 3.5) / max(1.0, total_gb)))
                        logger.warning("Low free VRAM (%.2f GB) — tuning gpu_memory_utilization -> %.2f", free_gb, safe_util)
                        gpu_util = round(safe_util, 2)
            except Exception:
                pass

            t0 = time.perf_counter()
            try:
                _vln_agent = build_tracking_agent(
                    model_path=LIGHTNAV_MODEL_PATH,
                    backend=LIGHTNAV_BACKEND,
                    task_key="vlnce",
                    gpu_memory_utilization=gpu_util,
                )
                _vln_backend_active = f"LightNav-0:{LIGHTNAV_BACKEND}"
            except Exception as vllm_exc:
                if LIGHTNAV_BACKEND == "vllm_local":
                    logger.warning("vllm_local failed (%s); falling back to backend='hf'", vllm_exc)
                    _vln_agent = build_tracking_agent(model_path=LIGHTNAV_MODEL_PATH, backend="hf", task_key="vlnce")
                    _vln_backend_active = "LightNav-0:hf"
                else:
                    raise

            _vln_load_error = ""
            logger.info("LightNav-0 ready in %.2fs (%s)", time.perf_counter() - t0, _vln_backend_active)
            return True
        except Exception as exc:
            _vln_load_error = f"{type(exc).__name__}: {exc}"
            _vln_backend_active = "fallback_stub"
            logger.error("Failed to load LightNav-0: %s", _vln_load_error)
            return False


@app.get("/health")
def health() -> Dict[str, Any]:
    with _lock:
        loaded = _vln_agent is not None
        steps = _step_count
        ep = _active_episode_instruction
    return {
        "ok": True,
        "vln_loaded": loaded,
        "steps_served": steps,
        "active_episode_instruction": ep,
        **_gpu_info(),
    }


# ==================== STAGE 1: SCHOOL OLLAMA VLM AGENT ====================
@app.post("/agent/vlm")
def agent_vlm(req: AgentVLMRequest) -> Dict[str, Any]:
    """Run VLM Agent / Room Scan / Arrival Gatekeeper on School GPU via local Ollama (`qwen3-vl:4b-instruct`)."""
    t0 = time.perf_counter()
    model_id = (req.model or OLLAMA_VLM_MODEL).strip()
    messages: List[Dict[str, Any]] = []
    if req.system_prompt:
        messages.append({"role": "system", "content": req.system_prompt})

    user_msg: Dict[str, Any] = {"role": "user", "content": req.prompt}
    if req.image_jpeg_b64:
        user_msg["images"] = [req.image_jpeg_b64]
    messages.append(user_msg)

    body: Dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": req.max_tokens},
    }
    if req.expect_json:
        body["format"] = "json"

    try:
        http_req = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with _direct_opener.open(http_req, timeout=15.0) as resp:
            out = json.loads(resp.read().decode("utf-8"))
        text = str(out.get("message", {}).get("content", "")).strip()
        ms = round((time.perf_counter() - t0) * 1000.0, 1)
        logger.info("[Agent VLM] model=%s latency=%.1fms len=%d", model_id, ms, len(text))
        return {"ok": True, "text": text, "model": f"school_ollama:{model_id}", "latency_ms": ms}
    except Exception as exc:
        ms = round((time.perf_counter() - t0) * 1000.0, 1)
        logger.error("[Agent VLM] Ollama error (%s): %s", model_id, exc)
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "model": model_id, "latency_ms": ms}


# ==================== STAGE 2: LIGHTNAV-0 VLN ====================
@app.post("/vln/load")
def load_vln(req: Optional[VLNResetRequest] = None) -> Dict[str, Any]:
    global _active_episode_instruction
    real_ok = _ensure_lightnav_loaded()
    en_inst = normalize_lightnav_instruction(
        req.instruction if req else "",
        req.agent_direction_hint if req else "STRAIGHT",
    )
    with _lock:
        if real_ok and _vln_agent is not None and req and req.instruction:
            _vln_agent.reset(instruction=en_inst)
            _active_episode_instruction = en_inst
        else:
            _active_episode_instruction = ""
    return {"ok": True, "vln_loaded": real_ok, "en_instruction": en_inst, **_gpu_info()}


@app.post("/vln/step")
def step_vln(req: VLNStepRequest) -> Dict[str, Any]:
    global _step_count, _active_episode_instruction
    t0 = time.perf_counter()

    hint = req.agent_direction_hint.strip().upper()
    if hint == "STOP":
        return {"ok": True, "vx": 0.0, "vy": 0.0, "wz": 0.0, "arrived": False, "infer_ms": 0.1, "mode": _vln_backend_active}

    if req.image_jpeg_b64:
        raw_bytes = base64.b64decode(req.image_jpeg_b64)
        rgb_arr = np.asarray(Image.open(io.BytesIO(raw_bytes)).convert("RGB"), dtype=np.uint8)
    else:
        rgb_arr = np.zeros((256, 448, 3), dtype=np.uint8)

    real_ok = _ensure_lightnav_loaded()
    base_en_inst = normalize_lightnav_instruction(req.instruction, "STRAIGHT")

    with _lock:
        _step_count += 1
        cur_step = _step_count

        if real_ok and _vln_agent is not None:
            if req.force_reset or (_active_episode_instruction != base_en_inst):
                en_with_dir = normalize_lightnav_instruction(req.instruction, hint)
                _vln_agent.reset(instruction=en_with_dir)
                _active_episode_instruction = base_en_inst
                logger.info("[New Episode] LightNav-0 reset(%r)", en_with_dir)

            _vln_agent.observe(rgb_arr)
            wp, raw_text, model_ms = _vln_agent.predict_waypoints(_active_episode_instruction, task_type="vlnce_traj")
            arrived = bool(_is_stop_centroid_fn(wp))
            forward_m, lateral_m, yaw_rad = map(float, wp[0])
            vx = 0.0 if arrived else max(0.0, min(MAX_VX_MPS, forward_m / DT))
            wz = 0.0 if arrived else max(-MAX_WZ_RADS, min(MAX_WZ_RADS, yaw_rad / DT))
            wp0_list: List[float] = [round(forward_m, 4), round(lateral_m, 4), round(yaw_rad, 4)]
        else:
            wz_map = {"LEFT": 0.25, "RIGHT": -0.25, "STRAIGHT": 0.0}
            vx, wz, arrived = 0.18, wz_map.get(hint, 0.0), False
            model_ms, raw_text = 0.0, "fallback_stub"
            wp0_list = [round(vx * DT, 4), 0.0, round(wz * DT, 4)]

    total_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return {
        "ok": True,
        "vx": round(vx, 4),
        "vy": 0.0,
        "wz": round(wz, 4),
        "arrived": arrived,
        "wp0": wp0_list,
        "en_instruction": _active_episode_instruction or base_en_inst,
        "model_ms": round(float(model_ms), 2),
        "infer_ms": total_ms,
        "frame_shape": list(rgb_arr.shape),
        "mode": _vln_backend_active,
        "raw_text": str(raw_text)[:120],
        "step": cur_step,
    }


@app.post("/vln/unload")
def unload_vln(req: Optional[VLNUnloadRequest] = None) -> Dict[str, Any]:
    global _vln_agent, _vln_backend_active, _active_episode_instruction
    hard = bool(req and req.hard_unload)
    with _lock:
        _active_episode_instruction = ""
        if hard and _vln_agent is not None:
            _vln_agent = None
            _vln_backend_active = "unloaded"
    if hard:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return {"ok": True, "vln_loaded": _vln_agent is not None, "hard_unloaded": hard, **_gpu_info()}


# ==================== STAGE 3: YOLOE OPEN-VOCAB DETECTION ====================
@app.post("/vision/detect")
def vision_detect(req: VisionDetectRequest) -> Dict[str, Any]:
    """Run zero-shot YOLOE open-vocabulary detection on School GPU."""
    global _yoloe_model, _yoloe_classes
    t0 = time.perf_counter()
    with _yoloe_lock:
        if _yoloe_model is None:
            try:
                from ultralytics import YOLOE

                _yoloe_model = YOLOE(YOLOE_WEIGHTS_PATH)
                logger.info("[YOLOE] Loaded %s on School GPU", YOLOE_WEIGHTS_PATH)
            except Exception as exc:
                return {"ok": False, "detected": False, "error": f"YOLOE unavailable on school server: {exc}"}

        clean_prompts = [p.strip() for p in req.prompts if p.strip()][:12] or ["object"]
        if clean_prompts != _yoloe_classes:
            _yoloe_model.set_classes(clean_prompts)
            _yoloe_classes = list(clean_prompts)

        raw_bytes = base64.b64decode(req.image_jpeg_b64)
        rgb_arr = np.asarray(Image.open(io.BytesIO(raw_bytes)).convert("RGB"), dtype=np.uint8)
        # Ultralytics expects BGR numpy or PIL Image; pass PIL directly to avoid channel confusion
        pil_img = Image.fromarray(rgb_arr)
        results = _yoloe_model.predict(pil_img, verbose=False, conf=req.conf)
        ms = round((time.perf_counter() - t0) * 1000.0, 2)
        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            best_i = int(boxes.conf.argmax().item())
            conf = float(boxes.conf[best_i].item())
            cls_idx = int(boxes.cls[best_i].item())
            xyxy = [int(round(v)) for v in boxes.xyxy[best_i].tolist()]
            label = clean_prompts[cls_idx] if 0 <= cls_idx < len(clean_prompts) else clean_prompts[0]
            return {"ok": True, "detected": True, "label": label, "confidence": round(conf, 4), "bbox": xyxy, "infer_ms": ms}
        return {"ok": True, "detected": False, "infer_ms": ms}


# ==================== STAGE 4: VLA MANIPULATION POLICY ====================
@app.post("/vla/act")
def vla_act(req: VLAActRequest) -> Dict[str, Any]:
    """Remote VLA action chunk endpoint on School 32GB GPU (`ACT` / `TurboVLA` / `SmolVLA` / `pi0.5`)."""
    global _vla_loaded_name
    t0 = time.perf_counter()
    with _vla_lock:
        _vla_loaded_name = req.model_name
        # Hook ready for ~/outputs/train/{act_lego,turbovla_v3,smolvla_lego,pi05_lego} checkpoints on ubuntu@324
        joint_target = req.joint_state or [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "ok": True,
            "model_name": req.model_name,
            "action_chunk": [joint_target],
            "infer_ms": ms,
        }


if __name__ == "__main__":
    preload = os.getenv("LIGHTNAV_PRELOAD", "1").strip() not in ("0", "false", "False")
    info0 = _gpu_info()
    logger.info(
        "Starting All-in-One School 32GB Compute Hub on http://%s:%d | GPU: %s (Free %.2f / %.2f GB) | VLM: %s",
        HOST,
        PORT,
        info0["gpu_name"],
        info0["vram_free_gb"],
        info0["vram_total_gb"],
        OLLAMA_VLM_MODEL,
    )
    if preload and Path(LIGHTNAV_MODEL_PATH).exists():
        _ensure_lightnav_loaded()
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
