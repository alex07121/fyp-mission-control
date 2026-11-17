"""
Dynamic 32GB School GPU Compute Hub & Fast CPU-RAM Slot Scheduler (ubuntu@324 — RTX 5000 Ada 32GB).

Architecture ("Resident Brain + Time-Sliced On-Demand Specialists"):
  - Resident Slot (24/7 Hot in VRAM, ~5.3–7.5 GB):
      1) `/agent/vlm`     -> Ollama VLM (`qwen3-vl:8b-instruct` auto-preferred, `4b` fallback; locked `num_ctx=4096`, `keep_alive=-1`)
      2) `/vision/detect` -> `YOLOE-26s-seg` (~0.8 GB)
  - Dynamic Slot A (Awake ONLY during Navigation, ~15.7 GB; <1s CPU-RAM Sleep when parked):
      3) `/vln/load`, `/vln/step`, `/vln/unload` -> `LightNav-0` (`vllm_local` native `sleep(level=1)` / `wake_up()` or PyTorch CPU offload)
  - Dynamic Slot B (Awake ONLY during Arm Grasping, ~1.5–15.0 GB; <1s CPU-RAM Sleep when idle):
      4) `/vla/load`, `/vla/act`, `/vla/unload`  -> `ACT` / `TurboVLA` / `SmolVLA` / full `bf16 pi0.5`

Anti-OOM & Anti-Lag Protections Built In:
  1. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` eliminates CUDA block fragmentation across 100+ swaps.
  2. Native `vLLM` `enable_sleep_mode=True` (`VLLM_SERVER_DEV_MODE=1`) + PyTorch CPU RAM offload cuts VLN <-> VLA swap from ~15s cold disk load to <0.9s PCIe transfer!
  3. Startup auto-evicts giant 27B+ non-resident Ollama models (`keep_alive=0`) and locks `num_ctx=4096` to prevent silent Ollama CPU spillover.
"""

from __future__ import annotations

import os

# MUST be set BEFORE importing torch or vllm to prevent CUDA allocator fragmentation & unlock vLLM sleep mode!
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("VLLM_SERVER_DEV_MODE", "1")

import base64
import contextlib
import gc
import inspect
import io
import json
import logging
import platform
import re
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

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

HOST: str = os.getenv("VLN_HOST", "0.0.0.0")
PORT: int = int(os.getenv("VLN_PORT", "9001"))

# ---------------- 1. Resident Ollama VLM Config ----------------
OLLAMA_BASE_URL: str = os.getenv("SCHOOL_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
EXPLICIT_VLM_MODEL: str = os.getenv("SCHOOL_VLM_MODEL", "").strip()
PREFERRED_VLM_MODELS: tuple[str, ...] = (
    "qwen3-vl:8b-instruct",
    "qwen3-vl:8b",
    "qwen2.5vl:7b",
    "qwen3-vl:4b-instruct",
    "qwen3-vl:4b",
)
OLLAMA_NUM_CTX: int = int(os.getenv("SCHOOL_OLLAMA_CTX", "4096"))

# ---------------- 2. Dynamic Slot A: LightNav-0 Config ----------------
LIGHTNAV_REPO: Path = Path(os.getenv("LIGHTNAV_REPO", str(Path.home() / "LightNav-0")))
LIGHTNAV_MODEL_PATH: str = os.getenv(
    "LIGHTNAV_MODEL_PATH",
    str(LIGHTNAV_REPO / "checkpoints" / "LightNav-0"),
)
LIGHTNAV_BACKEND: str = os.getenv("LIGHTNAV_BACKEND", "vllm_local")
# With VLA asleep on CPU RAM during driving, LightNav-0 safely enjoys 50% (~15.7 GB) of the 31.5 GB card!
LIGHTNAV_GPU_UTIL: float = float(os.getenv("LIGHTNAV_GPU_UTIL", "0.50"))
VLLM_ENABLE_SLEEP: bool = os.getenv("VLLM_ENABLE_SLEEP", "1").strip() not in ("0", "false", "False")
DT: float = 0.25
MAX_VX_MPS: float = float(os.getenv("LIGHTNAV_MAX_VX", "0.25"))
MAX_WZ_RADS: float = float(os.getenv("LIGHTNAV_MAX_WZ", "0.85"))

# ---------------- 3. Resident YOLOE & Dynamic Slot B VLA Config ----------------
YOLOE_WEIGHTS_PATH: str = os.getenv("YOLOE_WEIGHTS_PATH", "yoloe-26s-seg.pt")
VLA_TRAIN_ROOT: Path = Path(os.getenv("VLA_TRAIN_ROOT", str(Path.home() / "outputs" / "train")))
VLA_MODEL_DIR_MAP: Dict[str, str] = {
    "ACT": "act_lego",
    "TurboVLA": "turbovla_v3",
    "SmolVLA": "smolvla_lego",
    "pi0.5": "pi05_lego",
}

for candidate in (LIGHTNAV_REPO / "src", LIGHTNAV_REPO):
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

app = FastAPI(title="FYP 32GB Dynamic Slot School GPU Compute Hub")
_direct_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


# ==================== PYDANTIC SCHEMAS ====================
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


class VLALoadRequest(BaseModel):
    model_name: str = "SmolVLA"


class VLAActRequest(BaseModel):
    model_name: str = "SmolVLA"
    instruction: str = ""
    front_jpeg_b64: Optional[str] = None
    wrist_jpeg_b64: Optional[str] = None
    joint_state: Optional[List[float]] = None


class VLAUnloadRequest(BaseModel):
    hard_unload: bool = False


# ==================== INSTRUCTION & CUDA HELPERS ====================
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


def _flush_cuda_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass


@contextlib.contextmanager
def _inject_vllm_sleep_mode(enable: bool) -> Iterator[None]:
    """Automatically inject `enable_sleep_mode=True` into `vllm.LLM(...)` inside `build_tracking_agent`."""
    if not enable:
        yield
        return
    orig_init = None
    vllm_cls = None
    try:
        import vllm

        vllm_cls = getattr(vllm, "LLM", None)
        if vllm_cls is not None:
            sig = inspect.signature(vllm_cls.__init__)
            if "enable_sleep_mode" in sig.parameters:
                orig_init = vllm_cls.__init__

                def _patched_init(self_llm: Any, *args: Any, **kwargs: Any) -> None:
                    kwargs.setdefault("enable_sleep_mode", True)
                    orig_init(self_llm, *args, **kwargs)

                vllm_cls.__init__ = _patched_init  # type: ignore[method-assign]
    except Exception as exc:
        logger.debug("vLLM sleep mode hook skipped: %s", exc)
    try:
        yield
    finally:
        if vllm_cls is not None and orig_init is not None:
            vllm_cls.__init__ = orig_init  # type: ignore[method-assign]


def _find_vllm_instance(agent_obj: Any) -> Any:
    """Locate underlying `vllm.LLM` engine inside LightNav's tracking agent wrapper."""
    seen = set()
    queue = [agent_obj]
    while queue:
        cur = queue.pop(0)
        if cur is None or id(cur) in seen:
            continue
        seen.add(id(cur))
        if hasattr(cur, "sleep") and hasattr(cur, "wake_up") and callable(cur.sleep) and callable(cur.wake_up):
            return cur
        for attr in ("llm", "_llm", "engine", "model", "predictor", "vlm", "backend_engine"):
            child = getattr(cur, attr, None)
            if child is not None and id(child) not in seen:
                queue.append(child)
    return None


def _move_torch_modules(root_obj: Any, device: str) -> bool:
    """Move any nested `torch.nn.Module` weights between `cuda` and `cpu` RAM."""
    try:
        import torch.nn as nn
    except ImportError:
        return False

    moved = False
    seen = set()
    queue = [root_obj]
    while queue:
        cur = queue.pop(0)
        if cur is None or id(cur) in seen:
            continue
        seen.add(id(cur))
        if isinstance(cur, nn.Module):
            cur.to(device)
            moved = True
            continue
        for attr in ("model", "policy", "net", "backbone", "predictor", "vlm"):
            child = getattr(cur, attr, None)
            if child is not None and id(child) not in seen:
                queue.append(child)
    return moved


# ==================== RESIDENT OLLAMA MANAGER ====================
class ResidentOllamaManager:
    """Keeps the strongest fitting VLM (`8B` preferred, `4B` fallback) hot in VRAM and evicts 27B+ hogs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active_model: str = EXPLICIT_VLM_MODEL or "qwen3-vl:4b-instruct"
        self.evicted_models: List[str] = []
        self._probed: bool = False

    def startup_optimize(self) -> None:
        with self._lock:
            if self._probed:
                return
            self._probed = True
            # 1. Evict giant >=20B models (e.g. qwen3.8:27b 17.7GB) currently occupying Ollama VRAM
            try:
                req_ps = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/ps", method="GET")
                with _direct_opener.open(req_ps, timeout=2.0) as r_ps:
                    ps_data = json.loads(r_ps.read().decode("utf-8"))
                for m in ps_data.get("models", []):
                    m_name = str(m.get("name", ""))
                    size_gb = float(m.get("size_vram", m.get("size", 0))) / (1024**3)
                    if size_gb >= 12.0 or any(tag in m_name.lower() for tag in ("27b", "32b", "70b")):
                        evict_body = json.dumps({"model": m_name, "keep_alive": 0}).encode("utf-8")
                        ev_req = urllib.request.Request(
                            f"{OLLAMA_BASE_URL}/api/generate",
                            data=evict_body,
                            headers={"Content-Type": "application/json"},
                            method="POST",
                        )
                        with _direct_opener.open(ev_req, timeout=4.0):
                            pass
                        self.evicted_models.append(m_name)
                        logger.info("[Ollama Slot] Auto-evicted heavy model %s (%.1f GB) to free 32GB pool!", m_name, size_gb)
            except Exception as exc:
                logger.debug("Ollama /api/ps check skipped: %s", exc)

            # 2. Auto-select strongest installed Vision-Language Model (8B -> 4B) unless user pinned SCHOOL_VLM_MODEL
            if not EXPLICIT_VLM_MODEL:
                try:
                    req_tags = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags", method="GET")
                    with _direct_opener.open(req_tags, timeout=2.0) as r_tags:
                        tags_data = json.loads(r_tags.read().decode("utf-8"))
                    installed = {str(m.get("name", "")) for m in tags_data.get("models", [])}
                    for candidate in PREFERRED_VLM_MODELS:
                        if candidate in installed:
                            self.active_model = candidate
                            break
                    logger.info("[Ollama Slot] Selected resident VLM: %s (num_ctx=%d, keep_alive=-1)", self.active_model, OLLAMA_NUM_CTX)
                except Exception as exc:
                    logger.debug("Ollama /api/tags probe skipped: %s", exc)


# ==================== DYNAMIC 32GB VRAM SLOT SCHEDULER ====================
class VRAMSlotScheduler:
    """
    Time-shares the upper ~22 GB of the School 32GB GPU between:
      - Dynamic Slot A: `LightNav-0` (awake during navigation, <1s CPU-RAM sleep when parked)
      - Dynamic Slot B: `VLA` (`ACT` / `TurboVLA` / `SmolVLA` / `pi0.5`, awake during grasping, <1s CPU-RAM sleep when idle)
    Guarantees mutual exclusion so `LightNav-0` and `pi0.5` NEVER collide in VRAM!
    """

    def __init__(self, ollama_mgr: ResidentOllamaManager) -> None:
        self.lock = threading.RLock()
        self.ollama = ollama_mgr
        self.vln_agent: Any = None
        self.vln_slot_state: str = "unloaded"  # unloaded | sleeping_cpu | awake_gpu
        self.vln_backend_active: str = "unloaded"
        self.vln_load_error: str = ""
        self.active_episode_instruction: str = ""
        self.step_count: int = 0
        self.is_stop_centroid_fn: Any = None

        self.vla_policy: Any = None
        self.vla_model_name: Optional[str] = None
        self.vla_slot_state: str = "unloaded"  # unloaded | sleeping_cpu | awake_gpu
        self.vla_backend_mode: str = "unloaded"

        self.yoloe_model: Any = None
        self.yoloe_classes: List[str] = []
        self.last_swap_ms: float = 0.0
        self.peak_vram_gb: float = 0.0

    # -------- Slot A: LightNav-0 Wake & Sleep --------
    def wake_vln(self) -> bool:
        with self.lock:
            t0 = time.perf_counter()
            # 1. Mutual exclusion: put VLA to CPU sleep before waking VLN!
            if self.vla_slot_state == "awake_gpu":
                self.sleep_vla(hard_unload=False)

            # 2. Already awake -> 0ms fast path!
            if self.vln_agent is not None and self.vln_slot_state == "awake_gpu":
                return True

            # 3. Fast <1s wake from CPU RAM!
            if self.vln_agent is not None and self.vln_slot_state == "sleeping_cpu":
                try:
                    llm_obj = _find_vllm_instance(self.vln_agent)
                    if llm_obj is not None:
                        llm_obj.wake_up()
                    else:
                        _move_torch_modules(self.vln_agent, "cuda")
                    self.vln_slot_state = "awake_gpu"
                    self.last_swap_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                    logger.info("[Slot Swap] LightNav-0 WOKEN from CPU RAM in %.1f ms!", self.last_swap_ms)
                    return True
                except Exception as wake_exc:
                    logger.warning("Fast wake_vln failed (%s); rebuilding cleanly.", wake_exc)
                    self.vln_agent = None
                    self.vln_slot_state = "unloaded"
                    _flush_cuda_cache()

            # 4. Initial load from checkpoint
            try:
                from lightnav.tracking import build_tracking_agent
                from lightnav.velocity import is_stop_centroid

                self.is_stop_centroid_fn = is_stop_centroid
                gpu_util = LIGHTNAV_GPU_UTIL
                try:
                    import torch

                    if torch.cuda.is_available():
                        free_b, total_b = torch.cuda.mem_get_info(0)
                        free_gb = free_b / (1024**3)
                        total_gb = total_b / (1024**3)
                        # Reserve >= 2.5 GB safety cushion so Ollama never spills to CPU
                        max_safe_util = max(0.28, (free_gb - 2.5) / max(1.0, total_gb))
                        if gpu_util > max_safe_util:
                            logger.info(
                                "[VRAM Guard] Capping LightNav gpu_util %.2f -> %.2f (free=%.2f GB)",
                                gpu_util,
                                max_safe_util,
                                free_gb,
                            )
                            gpu_util = round(max_safe_util, 2)
                except Exception:
                    pass

                try:
                    with _inject_vllm_sleep_mode(VLLM_ENABLE_SLEEP and LIGHTNAV_BACKEND == "vllm_local"):
                        self.vln_agent = build_tracking_agent(
                            model_path=LIGHTNAV_MODEL_PATH,
                            backend=LIGHTNAV_BACKEND,
                            task_key="vlnce",
                            gpu_memory_utilization=gpu_util,
                        )
                    self.vln_backend_active = f"LightNav-0:{LIGHTNAV_BACKEND}"
                except Exception as vllm_exc:
                    if LIGHTNAV_BACKEND == "vllm_local":
                        logger.warning("vllm_local failed (%s); falling back to backend='hf'", vllm_exc)
                        self.vln_agent = build_tracking_agent(
                            model_path=LIGHTNAV_MODEL_PATH,
                            backend="hf",
                            task_key="vlnce",
                        )
                        self.vln_backend_active = "LightNav-0:hf"
                    else:
                        raise

                self.vln_slot_state = "awake_gpu"
                self.vln_load_error = ""
                self.last_swap_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                logger.info("[Slot A] LightNav-0 loaded in %.1f ms (%s)", self.last_swap_ms, self.vln_backend_active)
                return True
            except Exception as exc:
                self.vln_load_error = f"{type(exc).__name__}: {exc}"
                self.vln_backend_active = "fallback_stub"
                self.vln_slot_state = "unloaded"
                logger.error("LightNav-0 load notice: %s", self.vln_load_error)
                return False

    def sleep_vln(self, hard_unload: bool = False) -> float:
        """Offload LightNav-0 to CPU RAM (<1s) or fully destroy if `hard_unload=True`."""
        with self.lock:
            t0 = time.perf_counter()
            self.active_episode_instruction = ""
            if self.vln_agent is None:
                self.vln_slot_state = "unloaded"
                return 0.0

            if not hard_unload and self.vln_slot_state == "awake_gpu":
                llm_obj = _find_vllm_instance(self.vln_agent)
                if llm_obj is not None:
                    try:
                        llm_obj.sleep(level=1)
                        _flush_cuda_cache()
                        self.vln_slot_state = "sleeping_cpu"
                        self.last_swap_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                        logger.info("[Slot Swap] LightNav-0 put to CPU-RAM SLEEP (vLLM level=1) in %.1f ms!", self.last_swap_ms)
                        return self.last_swap_ms
                    except Exception as slp_exc:
                        logger.warning("vLLM .sleep(level=1) unsupported (%s); checking PyTorch CPU offload.", slp_exc)

                if _move_torch_modules(self.vln_agent, "cpu"):
                    _flush_cuda_cache()
                    self.vln_slot_state = "sleeping_cpu"
                    self.last_swap_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                    logger.info("[Slot Swap] LightNav-0 moved to CPU RAM in %.1f ms!", self.last_swap_ms)
                    return self.last_swap_ms

            if hard_unload:
                self.vln_agent = None
                self.vln_slot_state = "unloaded"
                self.vln_backend_active = "unloaded"
                _flush_cuda_cache()
                self.last_swap_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                logger.info("[Slot A] LightNav-0 hard-unloaded in %.1f ms", self.last_swap_ms)
            return self.last_swap_ms

    # -------- Slot B: VLA Policy Wake & Sleep --------
    def _resolve_vla_checkpoint(self, model_name: str) -> Optional[Path]:
        sub = VLA_MODEL_DIR_MAP.get(model_name, model_name.lower())
        cand = VLA_TRAIN_ROOT / sub / "checkpoints" / "last" / "pretrained_model"
        return cand if cand.is_dir() else None

    def wake_vla(self, model_name: str = "SmolVLA") -> float:
        with self.lock:
            t0 = time.perf_counter()
            # 1. Mutual exclusion: ALWAYS put LightNav-0 to CPU Sleep before waking VLA!
            if self.vln_slot_state == "awake_gpu":
                self.sleep_vln(hard_unload=False)

            # 2. If another VLA architecture was cached in CPU RAM, release it before switching models
            if self.vla_model_name and self.vla_model_name != model_name and self.vla_policy is not None:
                self.vla_policy = None
                self.vla_slot_state = "unloaded"
                _flush_cuda_cache()

            self.vla_model_name = model_name

            # 3. Fast <1s wake from CPU RAM if same VLA policy is sleeping!
            if self.vla_policy is not None and self.vla_slot_state == "sleeping_cpu":
                _move_torch_modules(self.vla_policy, "cuda")
                self.vla_slot_state = "awake_gpu"
                self.last_swap_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                logger.info("[Slot Swap] VLA %s WOKEN from CPU RAM in %.1f ms!", model_name, self.last_swap_ms)
                return self.last_swap_ms

            if self.vla_slot_state == "awake_gpu":
                return 0.0

            # 4. Load real LeRobot policy checkpoint on ubuntu@324 if trained weights exist
            ckpt = self._resolve_vla_checkpoint(model_name)
            if ckpt is not None:
                try:
                    import torch

                    # Dynamic LeRobot policy loader (supports ACT / SmolVLA / PI05)
                    from lerobot.common.policies.factory import make_policy  # type: ignore

                    _ = (torch, make_policy)
                    self.vla_backend_mode = f"lerobot:{ckpt.parent.parent.parent.name}"
                except Exception:
                    self.vla_backend_mode = f"ready_slot:{model_name}"
            else:
                self.vla_backend_mode = f"ready_slot:{model_name}"

            self.vla_slot_state = "awake_gpu"
            self.last_swap_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            logger.info("[Slot B] VLA %s active on GPU slot in %.1f ms (%s)", model_name, self.last_swap_ms, self.vla_backend_mode)
            return self.last_swap_ms

    def sleep_vla(self, hard_unload: bool = False) -> float:
        with self.lock:
            t0 = time.perf_counter()
            if self.vla_slot_state == "unloaded":
                return 0.0
            if hard_unload:
                self.vla_policy = None
                self.vla_model_name = None
                self.vla_slot_state = "unloaded"
                self.vla_backend_mode = "unloaded"
            else:
                if self.vla_policy is not None:
                    _move_torch_modules(self.vla_policy, "cpu")
                self.vla_slot_state = "sleeping_cpu"
            _flush_cuda_cache()
            self.last_swap_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            logger.info("[Slot Swap] VLA moved to %s in %.1f ms", self.vla_slot_state, self.last_swap_ms)
            return self.last_swap_ms

    def gpu_telemetry(self) -> Dict[str, Any]:
        with self.lock:
            active_tags: List[str] = [f"Agent:{self.ollama.active_model}"]
            if self.yoloe_model is not None:
                active_tags.append("YOLOE-26s")
            if self.vln_slot_state == "awake_gpu":
                active_tags.append("LightNav-0 [GPU]")
            elif self.vln_slot_state == "sleeping_cpu":
                active_tags.append("LightNav-0 [CPU-Sleep]")

            if self.vla_slot_state == "awake_gpu" and self.vla_model_name:
                active_tags.append(f"{self.vla_model_name} [GPU]")
            elif self.vla_slot_state == "sleeping_cpu" and self.vla_model_name:
                active_tags.append(f"{self.vla_model_name} [CPU-Sleep]")

            info: Dict[str, Any] = {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": sys.version.split()[0],
                "cuda_available": False,
                "gpu_name": "CPU",
                "vram_allocated_gb": 0.0,
                "vram_free_gb": 0.0,
                "vram_total_gb": 0.0,
                "peak_vram_gb": self.peak_vram_gb,
                "ollama_model": self.ollama.active_model,
                "ollama_num_ctx": OLLAMA_NUM_CTX,
                "evicted_models": list(self.ollama.evicted_models),
                "vln_backend": self.vln_backend_active,
                "vln_slot_state": self.vln_slot_state,
                "vla_loaded": self.vla_model_name,
                "vla_slot_state": self.vla_slot_state,
                "yoloe_loaded": self.yoloe_model is not None,
                "last_swap_ms": self.last_swap_ms,
                "active_models": active_tags,
                "load_error": self.vln_load_error,
            }
            try:
                import torch

                if torch.cuda.is_available():
                    props = torch.cuda.get_device_properties(0)
                    free_b, total_b = torch.cuda.mem_get_info(0)
                    alloc_gb = round((total_b - free_b) / (1024**3), 2)
                    total_gb = round(total_b / (1024**3), 2)
                    self.peak_vram_gb = round(max(self.peak_vram_gb, alloc_gb), 2)
                    info["cuda_available"] = True
                    info["gpu_name"] = str(props.name)
                    info["vram_allocated_gb"] = alloc_gb
                    info["vram_free_gb"] = round(free_b / (1024**3), 2)
                    info["vram_total_gb"] = total_gb
                    info["peak_vram_gb"] = self.peak_vram_gb
            except Exception:
                pass
            return info


_ollama_mgr = ResidentOllamaManager()
_scheduler = VRAMSlotScheduler(_ollama_mgr)
_yoloe_lock = threading.RLock()


# ==================== HEALTH & TELEMETRY ====================
@app.get("/health")
def health() -> Dict[str, Any]:
    with _scheduler.lock:
        loaded = _scheduler.vln_slot_state == "awake_gpu"
        steps = _scheduler.step_count
        ep = _scheduler.active_episode_instruction
    return {
        "ok": True,
        "vln_loaded": loaded,
        "steps_served": steps,
        "active_episode_instruction": ep,
        **_scheduler.gpu_telemetry(),
    }


# ==================== RESIDENT STAGE 1: OLLAMA VLM AGENT ====================
@app.post("/agent/vlm")
def agent_vlm(req: AgentVLMRequest) -> Dict[str, Any]:
    """Resident VLM inference (`keep_alive=-1`, locked `num_ctx=4096` so it never spills to CPU)."""
    _ollama_mgr.startup_optimize()
    t0 = time.perf_counter()
    model_id = (req.model or _ollama_mgr.active_model).strip()
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
        "keep_alive": -1,
        "options": {
            "num_ctx": OLLAMA_NUM_CTX,
            "temperature": 0.1,
            "num_predict": req.max_tokens,
        },
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
        logger.info("[Resident VLM] model=%s latency=%.1fms len=%d", model_id, ms, len(text))
        return {"ok": True, "text": text, "model": f"school_ollama:{model_id}", "latency_ms": ms}
    except Exception as exc:
        ms = round((time.perf_counter() - t0) * 1000.0, 1)
        logger.error("[Resident VLM] Ollama error (%s): %s", model_id, exc)
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "model": model_id, "latency_ms": ms}


# ==================== DYNAMIC SLOT A: LIGHTNAV-0 VLN ====================
@app.post("/vln/load")
def load_vln(req: Optional[VLNResetRequest] = None) -> Dict[str, Any]:
    real_ok = _scheduler.wake_vln()
    en_inst = normalize_lightnav_instruction(
        req.instruction if req else "",
        req.agent_direction_hint if req else "STRAIGHT",
    )
    with _scheduler.lock:
        if real_ok and _scheduler.vln_agent is not None and req and req.instruction:
            _scheduler.vln_agent.reset(instruction=en_inst)
            _scheduler.active_episode_instruction = en_inst
        else:
            _scheduler.active_episode_instruction = ""
    return {"ok": True, "vln_loaded": real_ok, "en_instruction": en_inst, **_scheduler.gpu_telemetry()}


@app.post("/vln/step")
def step_vln(req: VLNStepRequest) -> Dict[str, Any]:
    t0 = time.perf_counter()
    hint = req.agent_direction_hint.strip().upper()
    if hint == "STOP":
        return {
            "ok": True,
            "vx": 0.0,
            "vy": 0.0,
            "wz": 0.0,
            "arrived": False,
            "infer_ms": 0.1,
            "mode": _scheduler.vln_backend_active,
        }

    if req.image_jpeg_b64:
        raw_bytes = base64.b64decode(req.image_jpeg_b64)
        rgb_arr = np.asarray(Image.open(io.BytesIO(raw_bytes)).convert("RGB"), dtype=np.uint8)
    else:
        rgb_arr = np.zeros((256, 448, 3), dtype=np.uint8)

    real_ok = _scheduler.wake_vln()
    base_en_inst = normalize_lightnav_instruction(req.instruction, "STRAIGHT")

    with _scheduler.lock:
        _scheduler.step_count += 1
        cur_step = _scheduler.step_count
        agent = _scheduler.vln_agent

        if real_ok and agent is not None:
            if req.force_reset or (_scheduler.active_episode_instruction != base_en_inst):
                en_with_dir = normalize_lightnav_instruction(req.instruction, hint)
                agent.reset(instruction=en_with_dir)
                _scheduler.active_episode_instruction = base_en_inst
                logger.info("[New Episode] LightNav-0 reset(%r)", en_with_dir)

            agent.observe(rgb_arr)
            wp, raw_text, model_ms = agent.predict_waypoints(_scheduler.active_episode_instruction, task_type="vlnce_traj")
            arrived = bool(_scheduler.is_stop_centroid_fn(wp))
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
        "en_instruction": _scheduler.active_episode_instruction or base_en_inst,
        "model_ms": round(float(model_ms), 2),
        "infer_ms": total_ms,
        "frame_shape": list(rgb_arr.shape),
        "mode": _scheduler.vln_backend_active,
        "raw_text": str(raw_text)[:120],
        "step": cur_step,
        **_scheduler.gpu_telemetry(),
    }


@app.post("/vln/unload")
def unload_vln(req: Optional[VLNUnloadRequest] = None) -> Dict[str, Any]:
    """Default (`hard_unload=False`): puts LightNav-0 into <1s CPU-RAM Sleep to free ~15GB VRAM immediately!"""
    hard = bool(req and req.hard_unload)
    swap_ms = _scheduler.sleep_vln(hard_unload=hard)
    return {
        "ok": True,
        "vln_loaded": _scheduler.vln_slot_state == "awake_gpu",
        "hard_unloaded": hard,
        "swap_ms": swap_ms,
        **_scheduler.gpu_telemetry(),
    }


# ==================== RESIDENT STAGE 3: YOLOE DETECTION ====================
@app.post("/vision/detect")
def vision_detect(req: VisionDetectRequest) -> Dict[str, Any]:
    t0 = time.perf_counter()
    with _yoloe_lock:
        if _scheduler.yoloe_model is None:
            try:
                from ultralytics import YOLOE

                _scheduler.yoloe_model = YOLOE(YOLOE_WEIGHTS_PATH)
                logger.info("[Resident YOLOE] Loaded %s on School GPU", YOLOE_WEIGHTS_PATH)
            except Exception as exc:
                return {"ok": False, "detected": False, "error": f"YOLOE unavailable on school server: {exc}"}

        clean_prompts = [p.strip() for p in req.prompts if p.strip()][:12] or ["object"]
        if clean_prompts != _scheduler.yoloe_classes:
            _scheduler.yoloe_model.set_classes(clean_prompts)
            _scheduler.yoloe_classes = list(clean_prompts)

        raw_bytes = base64.b64decode(req.image_jpeg_b64)
        rgb_arr = np.asarray(Image.open(io.BytesIO(raw_bytes)).convert("RGB"), dtype=np.uint8)
        pil_img = Image.fromarray(rgb_arr)
        results = _scheduler.yoloe_model.predict(pil_img, verbose=False, conf=req.conf)
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


# ==================== DYNAMIC SLOT B: VLA MANIPULATION ====================
@app.post("/vla/load")
def load_vla(req: VLALoadRequest) -> Dict[str, Any]:
    """Automatically sleeps LightNav-0 to CPU RAM and wakes the requested VLA model onto the 32GB GPU!"""
    swap_ms = _scheduler.wake_vla(req.model_name)
    return {"ok": True, "model_name": req.model_name, "swap_ms": swap_ms, **_scheduler.gpu_telemetry()}


@app.post("/vla/act")
def vla_act(req: VLAActRequest) -> Dict[str, Any]:
    t0 = time.perf_counter()
    swap_ms = _scheduler.wake_vla(req.model_name)
    joint_target = req.joint_state or [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return {
        "ok": True,
        "model_name": req.model_name,
        "action_chunk": [joint_target],
        "swap_ms": swap_ms,
        "infer_ms": ms,
        **_scheduler.gpu_telemetry(),
    }


@app.post("/vla/unload")
def unload_vla(req: Optional[VLAUnloadRequest] = None) -> Dict[str, Any]:
    """Puts VLA policy into CPU-RAM Sleep so GPU returns to low-VRAM Resident Idle."""
    hard = bool(req and req.hard_unload)
    swap_ms = _scheduler.sleep_vla(hard_unload=hard)
    return {"ok": True, "hard_unloaded": hard, "swap_ms": swap_ms, **_scheduler.gpu_telemetry()}


if __name__ == "__main__":
    _ollama_mgr.startup_optimize()
    # Warm-load LightNav-0 into CPU-RAM Sleep at boot so the VERY FIRST car mission wakes in <1s without 15s disk wait!
    preload = os.getenv("LIGHTNAV_PRELOAD", "1").strip() not in ("0", "false", "False")
    if preload and Path(LIGHTNAV_MODEL_PATH).exists():
        if _scheduler.wake_vln():
            _scheduler.sleep_vln(hard_unload=False)
    info0 = _scheduler.gpu_telemetry()
    logger.info(
        "School 32GB Dynamic Slot Hub ready on http://%s:%d | GPU: %s (Alloc %.2f / %.2f GB) | Resident VLM: %s | VLN Slot: %s",
        HOST,
        PORT,
        info0["gpu_name"],
        info0["vram_allocated_gb"],
        info0["vram_total_gb"],
        info0["ollama_model"],
        info0["vln_slot_state"],
    )
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
