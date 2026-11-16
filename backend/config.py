"""Cross-platform configuration and constants for FYP Mission Control."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

# Cross-platform project paths via pathlib.Path
BACKEND_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = BACKEND_DIR.parent
FRONTEND_DIR: Path = PROJECT_ROOT / "frontend"
DATA_DIR: Path = Path(os.getenv("FYP_DATA_DIR", str(PROJECT_ROOT / "data"))).resolve()
WEIGHTS_DIR: Path = PROJECT_ROOT / "weights"

# Ensure runtime data directory exists safely
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Network & runtime settings
DEFAULT_HOST: str = os.getenv("HOST", "0.0.0.0")
DEFAULT_PORT: int = int(os.getenv("PORT", "8080"))
DEFAULT_UDP_PORT: int = int(os.getenv("UDP_PORT", "9999"))
DEFAULT_ROBOT_MODE: str = os.getenv("ROBOT_MODE", "mock").strip().lower()

# Google Cloud Vertex AI configuration
GCP_PROJECT_ID: str = os.getenv("GCP_PROJECT_ID", "project-b04a0005-9614-47d1-996")
GCP_LOCATION: str = os.getenv("GCP_LOCATION", "global")
GEMINI_MODEL_ID: str = os.getenv("GEMINI_MODEL_ID", "gemini-3.5-flash-lite")

# Safety & timing thresholds
MAX_BCI_PAYLOAD_BYTES: int = 4096
AGENT_JPEG_WIDTH: int = 320
AGENT_JPEG_HEIGHT: int = 240
AGENT_WATCHDOG_HARD_LIMIT_S: float = 2.5
# Trigger internal brake strictly <= 2.35s so wall-clock stop latency (incl. resize/JPEG) is 100% < 2.50s
DEFAULT_WATCHDOG_TIMEOUT_S: float = min(2.35, max(0.50, float(os.getenv("AGENT_WATCHDOG_TIMEOUT_S", "2.30"))))

# Depth lens hardware profiles
LENS_PROFILES: Dict[str, Dict[str, float]] = {
    "80cm": {
        "blind_zone_m": 0.80,
        "stop_distance_m": 0.85,
        "slow_zone_m": 1.25,
        "blind_walk_distance_m": 0.40,
        "blind_walk_speed_mps": 0.20,
        "blind_walk_duration_s": 2.00,
    },
    "20cm": {
        "blind_zone_m": 0.20,
        "stop_distance_m": 0.45,
        "slow_zone_m": 0.85,
        "blind_walk_distance_m": 0.0,
        "blind_walk_speed_mps": 0.0,
        "blind_walk_duration_s": 0.0,
    },
}

# 12GB Laptop GPU VRAM budget constants (RTX 5070 Laptop 12GB VRAM)
TOTAL_VRAM_GB: float = 12.0
VRAM_IDLE_BASE_GB: float = 1.4  # OS + lightweight YOLOE resident
VRAM_VLN_DELTA_GB: float = 5.4  # Light-VLN navigation backbone (~6.8 GB total)
VRAM_POST_UNLOAD_GB: float = 1.6  # After unloading VLN + gc + empty_cache
VRAM_VLA_MODELS_GB: Dict[str, float] = {
    "SmolVLA": 4.8,  # 1.6 + 4.8 = 6.4 GB total
    "pi0.5": 5.6,    # 1.6 + 5.6 = 7.2 GB total
}


@dataclass(frozen=True)
class SSVEPOption:
    option_id: int
    frequency_hz: float
    label_zh: str
    command_text: str
    task_type: str  # "navigate_and_pick" | "navigate_only" | "estop"
    target_prompts: List[str]
    landmark_prompts: List[str]


SSVEP_OPTIONS: Dict[int, SSVEPOption] = {
    1: SSVEPOption(
        option_id=1,
        frequency_hz=8.0,
        label_zh="抓红色水杯",
        command_text="去桌边拿红色水杯 (Pick up red mug)",
        task_type="navigate_and_pick",
        target_prompts=["red mug", "mug", "cup"],
        landmark_prompts=["table", "desk"],
    ),
    2: SSVEPOption(
        option_id=2,
        frequency_hz=10.0,
        label_zh="抓蓝色水瓶",
        command_text="去桌边拿水瓶 (Pick up water bottle)",
        task_type="navigate_and_pick",
        target_prompts=["water bottle", "bottle"],
        landmark_prompts=["table", "desk"],
    ),
    3: SSVEPOption(
        option_id=3,
        frequency_hz=12.0,
        label_zh="前往敞开门口",
        command_text="前往敞开的门口 (Go to open doorway)",
        task_type="navigate_only",
        target_prompts=["open doorway", "door frame", "doorway", "door"],
        landmark_prompts=["open doorway", "door frame", "doorway"],
    ),
    4: SSVEPOption(
        option_id=4,
        frequency_hz=14.0,
        label_zh="经门口拿水杯",
        command_text="先看门口方向再去拿红色水杯 (Navigate & pick mug)",
        task_type="navigate_and_pick",
        target_prompts=["red mug", "mug", "cup"],
        landmark_prompts=["doorway", "door frame", "table"],
    ),
    5: SSVEPOption(
        option_id=5,
        frequency_hz=15.0,
        label_zh="紧急急停 E-STOP",
        command_text="紧急停止所有运动 (BCI E-STOP)",
        task_type="estop",
        target_prompts=[],
        landmark_prompts=[],
    ),
}

DOORWAY_KEYWORDS: tuple[str, ...] = (
    "doorway",
    "door frame",
    "doorframe",
    "open doorway",
    "door",
    "entrance",
    "gate",
)
