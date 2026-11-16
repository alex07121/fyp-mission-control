"""
School 32GB GPU Remote VLN Server — Real LightNav-0 (vLLM / HF) Integration.

Run inside the `lightnav` conda environment on ubuntu@324:
    conda activate lightnav
    cd ~/fyp-mission-control && git pull
    python remote_vln_server.py
"""

from __future__ import annotations

import base64
import gc
import io
import logging
import os
import platform
import re
import socket
import sys
import threading
import time
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
logger = logging.getLogger("lightnav.remote")

HOST: str = os.getenv("VLN_HOST", "127.0.0.1")
PORT: int = int(os.getenv("VLN_PORT", "9001"))

# LightNav-0 paths & inference constants (from LightNav-0 docs/DEPLOYMENT.md)
LIGHTNAV_REPO: Path = Path(os.getenv("LIGHTNAV_REPO", "/home/ubuntu/LightNav-0"))
LIGHTNAV_MODEL_PATH: str = os.getenv(
    "LIGHTNAV_MODEL_PATH",
    str(LIGHTNAV_REPO / "checkpoints" / "LightNav-0"),
)
LIGHTNAV_BACKEND: str = os.getenv("LIGHTNAV_BACKEND", "vllm_local")  # "vllm_local" or "hf"
LIGHTNAV_GPU_UTIL: float = float(os.getenv("LIGHTNAV_GPU_UTIL", "0.45"))
DT: float = 0.25  # LightNav-0 trained on 4 fps video (0.25s control period)
MAX_VX_MPS: float = float(os.getenv("LIGHTNAV_MAX_VX", "0.25"))
MAX_WZ_RADS: float = float(os.getenv("LIGHTNAV_MAX_WZ", "0.85"))

# Ensure LightNav-0 source tree is importable even if not pip-installed editable
for candidate in (LIGHTNAV_REPO / "src", LIGHTNAV_REPO):
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

app = FastAPI(title="FYP Remote LightNav-0 Server (RTX 5000 Ada 32GB)")
_lock = threading.RLock()
_vln_agent: Any = None
_vln_backend_active: str = "unloaded"
_vln_load_error: str = ""
_active_episode_instruction: str = ""
_step_count: int = 0
_is_stop_centroid_fn: Any = None


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


def normalize_lightnav_instruction(raw_cmd: str, hint: str = "STRAIGHT") -> str:
    """
    Convert bilingual/Chinese Mission Control commands into LightNav-0's trained English format:
    'Action + Direction + Explicit Object + and stop.'
    """
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

    # Custom user text: strip non-ASCII if bilingual, otherwise preserve English & ensure 'and stop.'
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
        "model_path_exists": Path(LIGHTNAV_MODEL_PATH).exists(),
        "vln_backend": _vln_backend_active,
        "load_error": _vln_load_error,
    }
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            free_b, total_b = torch.cuda.mem_get_info(0)
            info["cuda_available"] = True
            info["gpu_name"] = str(props.name)
            info["vram_allocated_gb"] = round(torch.cuda.memory_allocated(0) / (1024**3), 2)
            info["vram_free_gb"] = round(free_b / (1024**3), 2)
            info["vram_total_gb"] = round(total_b / (1024**3), 2)
    except Exception:
        pass
    return info


def _ensure_lightnav_loaded() -> bool:
    """Lazy-load LightNav-0 once and keep warm in 32GB VRAM across missions."""
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
                    if free_gb < 15.5:
                        # Another process (e.g. lightnav-serve on :8051) is holding ~15GB VRAM; cap safely!
                        safe_util = max(0.25, min(gpu_util, (free_gb - 2.0) / max(1.0, total_gb)))
                        logger.warning(
                            "Low free VRAM (%.2f / %.2f GB) — adjusting gpu_memory_utilization %.2f -> %.2f",
                            free_gb,
                            total_gb,
                            gpu_util,
                            safe_util,
                        )
                        gpu_util = round(safe_util, 2)
            except Exception:
                pass

            logger.info(
                "Loading LightNav-0 from %s (backend=%s, task_key=vlnce, gpu_util=%.2f) ...",
                LIGHTNAV_MODEL_PATH,
                LIGHTNAV_BACKEND,
                gpu_util,
            )
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
                    logger.warning("vllm_local init failed (%s); retrying with backend='hf' ...", vllm_exc)
                    _vln_agent = build_tracking_agent(
                        model_path=LIGHTNAV_MODEL_PATH,
                        backend="hf",
                        task_key="vlnce",
                    )
                    _vln_backend_active = "LightNav-0:hf"
                else:
                    raise

            _vln_load_error = ""
            logger.info("LightNav-0 ready in %.2fs (%s)", time.perf_counter() - t0, _vln_backend_active)
            return True
        except Exception as exc:
            _vln_load_error = f"{type(exc).__name__}: {exc}"
            _vln_backend_active = "fallback_stub"
            logger.error("Failed to load LightNav-0 (%s) — make sure `conda activate lightnav` is active!", _vln_load_error)
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


@app.post("/vln/load")
def load_vln(req: Optional[VLNResetRequest] = None) -> Dict[str, Any]:
    """Ensure LightNav-0 is resident in VRAM and reset episode history for the new mission."""
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
            logger.info("Episode reset -> %r", en_inst)
        else:
            # Force reset on first incoming /vln/step of the new mission
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
        # PIL converts standard JPEG directly into HxWx3 uint8 RGB (exact format required by agent.observe!)
        rgb_arr = np.asarray(Image.open(io.BytesIO(raw_bytes)).convert("RGB"), dtype=np.uint8)
    else:
        rgb_arr = np.zeros((256, 448, 3), dtype=np.uint8)

    real_ok = _ensure_lightnav_loaded()
    # Keep instruction stable within an episode so we don't reset history mid-trajectory when hint changes
    base_en_inst = normalize_lightnav_instruction(req.instruction, "STRAIGHT")

    with _lock:
        _step_count += 1
        cur_step = _step_count

        if real_ok and _vln_agent is not None:
            if req.force_reset or (_active_episode_instruction != base_en_inst):
                en_with_dir = normalize_lightnav_instruction(req.instruction, hint)
                _vln_agent.reset(instruction=en_with_dir)
                _active_episode_instruction = base_en_inst
                logger.info("[New Episode] LightNav-0 reset(instruction=%r)", en_with_dir)

            _vln_agent.observe(rgb_arr)
            wp, raw_text, model_ms = _vln_agent.predict_waypoints(
                _active_episode_instruction,
                task_type="vlnce_traj",
            )
            arrived = bool(_is_stop_centroid_fn(wp))
            forward_m, lateral_m, yaw_rad = map(float, wp[0])
            vx = 0.0 if arrived else max(0.0, min(MAX_VX_MPS, forward_m / DT))
            wz = 0.0 if arrived else max(-MAX_WZ_RADS, min(MAX_WZ_RADS, yaw_rad / DT))
            wp0_list: List[float] = [round(forward_m, 4), round(lateral_m, 4), round(yaw_rad, 4)]
        else:
            # Fallback if server was started outside `conda activate lightnav`
            wz_map = {"LEFT": 0.25, "RIGHT": -0.25, "STRAIGHT": 0.0}
            vx = 0.18
            wz = wz_map.get(hint, 0.0)
            arrived = False
            model_ms = 0.0
            raw_text = "fallback_stub"
            wp0_list = [round(vx * DT, 4), 0.0, round(wz * DT, 4)]

    total_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    logger.info(
        "Step #%d | vx=%.3f wz=%.3f arrived=%s | wp0=%s | model=%.1fms total=%.1fms",
        cur_step,
        vx,
        wz,
        arrived,
        wp0_list,
        float(model_ms),
        total_ms,
    )
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
    """
    On the dedicated 32GB school GPU server, keep vLLM warm across episodes by default
    (resetting episode KV/frame history in <2ms); pass `hard_unload=True` to release VRAM completely.
    """
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


if __name__ == "__main__":
    preload = os.getenv("LIGHTNAV_PRELOAD", "1").strip() not in ("0", "false", "False")
    info0 = _gpu_info()
    logger.info(
        "Starting Remote LightNav-0 Server on http://%s:%d | GPU: %s (Free %.2f / %.2f GB)",
        HOST,
        PORT,
        info0["gpu_name"],
        info0["vram_free_gb"],
        info0["vram_total_gb"],
    )
    if preload and info0["model_path_exists"]:
        _ensure_lightnav_loaded()
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
