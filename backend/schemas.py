"""Pydantic schemas for FYP Mission Control REST APIs."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator

_ANSI_ESCAPE_RE = re.compile(r"\x1b(?:\[[0-9;?]*[a-zA-Z]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[()][A-Z0-9]|[a-zA-Z])")


def sanitize_printable_token(raw: str, max_len: int = 64) -> str:
    no_ansi = _ANSI_ESCAPE_RE.sub(" ", raw)
    cleaned = "".join(ch if (ch.isprintable() and ch not in "\r\n\t\x00") else " " for ch in no_ansi)
    return re.sub(r"\s+", " ", cleaned).strip()[:max_len]


class AgentBenchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(
        default="Identify target direction (LEFT, STRAIGHT, or RIGHT) and main visible object.",
        min_length=1,
        max_length=500,
    )
    mode: Optional[Literal["cloud", "mock", "auto", "school"]] = None
    simulate_delay_s: float = Field(default=0.0, ge=0.0, le=15.0)
    simulate_error: bool = Field(default=False)


class RuntimeConfigUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lens_mode: Optional[Literal["80cm", "20cm"]] = None
    blind_walk_enabled: Optional[bool] = None
    agent_mode: Optional[Literal["cloud", "mock", "auto", "school"]] = None
    cruise_speed_mps: Optional[float] = Field(default=None, ge=0.05, le=0.45)
    agent_interval_s: Optional[float] = Field(default=None, ge=0.8, le=5.0)
    watchdog_timeout_s: Optional[float] = Field(default=None, ge=0.5, le=2.35)
    vla_model: Optional[Literal["ACT", "TurboVLA", "SmolVLA", "pi0.5"]] = None
    detection_prompts: Optional[List[str]] = Field(default=None, min_length=1, max_length=12)

    @field_validator("detection_prompts")
    @classmethod
    def _validate_detection_prompts(cls, val: Optional[List[str]]) -> Optional[List[str]]:
        if val is None:
            return None
        if not (1 <= len(val) <= 12):
            raise ValueError("detection_prompts must contain 1..12 items")
        cleaned_list: List[str] = []
        for idx, item in enumerate(val):
            if not isinstance(item, str):
                raise ValueError(f"detection_prompts[{idx}] must be a string")
            no_ansi = _ANSI_ESCAPE_RE.sub(" ", item)
            printable = "".join(ch if (ch.isprintable() and ch not in "\r\n\t\x00") else " " for ch in no_ansi)
            norm = re.sub(r"\s+", " ", printable).strip()
            if not (1 <= len(norm) <= 48):
                raise ValueError(f"detection_prompts[{idx}] must be 1..48 printable characters (got {len(norm)})")
            cleaned_list.append(norm)
        return cleaned_list


class FaultInjectionUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    depth_fault: Optional[Literal["none", "all_zero", "all_nan", "penetrate_4m"]] = None
    simulate_agent_delay_s: Optional[float] = Field(default=None, ge=0.0, le=15.0)
    simulate_agent_error: Optional[bool] = None
    force_vln_early_stop: Optional[bool] = None


class RobotPoseResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: float = Field(default=0.0, ge=-5.0, le=5.0)
    y: float = Field(default=0.0, ge=-5.0, le=5.0)
    theta: float = Field(default=0.0, ge=-3.15, le=3.15)
    scenario: Optional[Literal["default", "near_blind_06m", "doorway_front", "far_search"]] = None


class ManualStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: str = Field(..., min_length=1, max_length=256)
    source: str = Field(default="web_manual", max_length=64)

    @field_validator("source")
    @classmethod
    def _clean_source(cls, val: str) -> str:
        return sanitize_printable_token(val, max_len=64) or "web_manual"


class StandardResponse(BaseModel):
    ok: bool
    message: str
    data: Optional[Dict[str, Any]] = None
