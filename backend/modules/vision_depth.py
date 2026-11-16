"""Module 3: YOLOE Open-Vocabulary Detector + 80cm/20cm Depth Calibrator & Poison Guard."""

from __future__ import annotations

import importlib.util
import logging
import math
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from adapters.base import RGBDFrame
from backend.config import DOORWAY_KEYWORDS, LENS_PROFILES, PROJECT_ROOT, WEIGHTS_DIR

logger = logging.getLogger("fyp.vision")
_ANSI_ESCAPE_RE = re.compile(r"\x1b(?:\[[0-9;?]*[a-zA-Z]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[()][A-Z0-9]|[a-zA-Z])")


@dataclass
class DetectionMeasurement:
    detected: bool
    label: str
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]]
    is_doorway: bool
    roi_mode: str
    median_depth_m: Optional[float]
    yaw_error_deg: float
    steering_hint: str
    blind_zone_triggered: bool
    penetration_anomaly: bool
    depth_status: str
    roi_valid_ratio: float
    global_valid_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected": self.detected,
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "bbox": list(self.bbox) if self.bbox else None,
            "is_doorway": self.is_doorway,
            "roi_mode": self.roi_mode,
            "median_depth_m": round(self.median_depth_m, 3) if self.median_depth_m is not None else None,
            "yaw_error_deg": round(self.yaw_error_deg, 2),
            "steering_hint": self.steering_hint,
            "blind_zone_triggered": self.blind_zone_triggered,
            "penetration_anomaly": self.penetration_anomaly,
            "depth_status": self.depth_status,
            "roi_valid_ratio": round(self.roi_valid_ratio, 3),
            "global_valid_ratio": round(self.global_valid_ratio, 3),
        }


class VisionDepthCalibrator:
    """
    Thread-safe YOLOE (`yoloe-26s-seg.pt`) + dual-ROI depth extractor
    (40% center vs 15% doorway side jambs) + strict <80cm blind-zone & continuous pinhole / anchor poison defense.
    """

    # R2-CRIT-02: Require >=60% physically valid pixels inside target ROI (<40% blind/invalid) to defeat survivorship bias
    MIN_ROI_VALID_RATIO: float = 0.60

    def __init__(self, initial_lens_mode: str = "80cm") -> None:
        self._lock = threading.RLock()
        self._model_lock = threading.Lock()
        self.lens_mode: str = initial_lens_mode if initial_lens_mode in LENS_PROFILES else "80cm"
        self._prompts: List[str] = ["red mug", "mug", "cup"]
        self._yoloe_model: Any = None
        self._detector_backend: str = "builtin_open_vocab"
        self._ultralytics_available: bool = self._check_ultralytics_spec()
        self._last_valid_depth_m: Optional[float] = None
        self._min_approached_depth_m: Optional[float] = None
        self._last_measurement: Optional[DetectionMeasurement] = None
        self._try_load_local_yoloe()

    @staticmethod
    def _check_ultralytics_spec() -> bool:
        try:
            import sys

            if sys.modules.get("ultralytics") is not None:
                return True
            return importlib.util.find_spec("ultralytics") is not None
        except (ImportError, ValueError):
            return False

    def _resolve_yoloe_weight_path(self) -> Optional[Path]:
        env_p = os.getenv("YOLOE_WEIGHTS_PATH", "").strip()
        candidates = []
        if env_p:
            candidates.append(Path(env_p))
        candidates.extend([WEIGHTS_DIR / "yoloe-26s-seg.pt", PROJECT_ROOT / "yoloe-26s-seg.pt"])
        for p in candidates:
            if p.is_file():
                return p
        return None

    def _try_load_local_yoloe(self) -> None:
        weight_path = self._resolve_yoloe_weight_path()
        if weight_path is None:
            # Never auto-download large weights over WAN when local checkpoint is absent; instant <1ms fallback
            return
        try:
            from ultralytics import YOLOE  # type: ignore

            with self._model_lock:
                model = YOLOE(str(weight_path))
                model.set_classes(self._prompts)
                self._yoloe_model = model
                self._detector_backend = "ultralytics_yoloe_26s"
                self._ultralytics_available = True
            logger.info("Loaded local YOLOE weights from %s", weight_path)
        except Exception as exc:
            logger.warning("Local YOLOE unavailable (%s), using fast builtin detector.", exc)
            self._yoloe_model = None
            self._detector_backend = "builtin_open_vocab"

    def set_classes(self, prompts: List[str]) -> List[str]:
        cleaned: List[str] = []
        for p in prompts:
            if not isinstance(p, str):
                continue
            no_ansi = _ANSI_ESCAPE_RE.sub(" ", p)
            printable = "".join(ch if (ch.isprintable() and ch not in "\r\n\t\x00") else " " for ch in no_ansi)
            norm = re.sub(r"\s+", " ", printable).strip()[:48]
            if norm:
                cleaned.append(norm)
        if not cleaned:
            cleaned = ["red mug", "mug", "cup"]
        new_prompts = cleaned[:12]
        with self._lock:
            self._prompts = new_prompts
            self._last_valid_depth_m = None
            self._min_approached_depth_m = None
        if self._yoloe_model is not None:
            with self._model_lock:
                try:
                    self._yoloe_model.set_classes(new_prompts)
                except Exception as exc:
                    logger.warning("YOLOE set_classes fallback: %s", exc)
        return list(new_prompts)

    def reset_tracking_history(self) -> None:
        with self._lock:
            self._last_valid_depth_m = None
            self._min_approached_depth_m = None

    @staticmethod
    def _is_doorway_label(label: str, prompts: List[str]) -> bool:
        if any(k in label.lower() for k in DOORWAY_KEYWORDS):
            return True
        return bool(prompts) and all(any(k in p.lower() for k in DOORWAY_KEYWORDS) for p in prompts if p)

    @staticmethod
    def _is_table_label(label: str) -> bool:
        low = label.lower()
        return bool(re.search(r"\b(?:table|tables|desk|desks)\b", low)) or ("桌" in low)

    def _detect_builtin(self, bgr: np.ndarray, prompts: List[str]) -> Optional[Tuple[str, float, Tuple[int, int, int, int]]]:
        """
        R3-HIGH-03 + R4-LOW-01: Assign each visual mask its TRUE category-matched prompt label, support wood desk/table,
        and strictly return `None` when custom open-vocabulary prompts match no known simulated object (zero hallucination).
        """
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        door_idx: Optional[int] = None
        door_lbl = "open doorway"
        bot_idx: Optional[int] = None
        bot_lbl = "water bottle"
        mug_idx: Optional[int] = None
        mug_lbl = "red mug"
        tbl_idx: Optional[int] = None
        tbl_lbl = "table"

        for idx, p in enumerate(prompts):
            low_p = p.lower()
            if door_idx is None and (any(k in low_p for k in DOORWAY_KEYWORDS) or ("门" in low_p)):
                door_idx = idx
                door_lbl = p
            if bot_idx is None and (bool(re.search(r"\b(?:bottle|bottles|water)\b", low_p)) or ("瓶" in low_p)):
                bot_idx = idx
                bot_lbl = p
            if mug_idx is None and (bool(re.search(r"\b(?:mug|mugs|cup|cups|red|teacup|teacups)\b", low_p)) or ("杯" in low_p)):
                mug_idx = idx
                mug_lbl = p
            if tbl_idx is None and (bool(re.search(r"\b(?:table|tables|desk|desks)\b", low_p)) or ("桌" in low_p)):
                tbl_idx = idx
                tbl_lbl = p

        ranked: List[Tuple[int, str, np.ndarray, int]] = []
        if door_idx is not None:
            m_door = cv2.inRange(hsv, np.array([14, 110, 120]), np.array([36, 255, 255]))
            ranked.append((door_idx, door_lbl, m_door, 100))
        if bot_idx is not None:
            m_bot = cv2.inRange(hsv, np.array([96, 100, 80]), np.array([130, 255, 255]))
            ranked.append((bot_idx, bot_lbl, m_bot, 24))
        if mug_idx is not None:
            m1 = cv2.inRange(hsv, np.array([0, 130, 90]), np.array([10, 255, 255]))
            m2 = cv2.inRange(hsv, np.array([170, 130, 90]), np.array([180, 255, 255]))
            ranked.append((mug_idx, mug_lbl, cv2.bitwise_or(m1, m2), 24))
        if tbl_idx is not None:
            m_tbl = cv2.inRange(hsv, np.array([10, 120, 95]), np.array([17, 195, 165]))
            ranked.append((tbl_idx, tbl_lbl, m_tbl, 80))

        if not ranked:
            return None

        ranked.sort(key=lambda item: item[0])

        for _, label, mask, min_px in ranked:
            ys, xs = np.where(mask > 0)
            if len(xs) >= min_px:
                x1, x2 = int(np.min(xs)), int(np.max(xs)) + 1
                y1, y2 = int(np.min(ys)), int(np.max(ys)) + 1
                if (x2 - x1) >= 5 and (y2 - y1) >= 5:
                    conf = min(0.98, 0.78 + min(0.20, len(xs) / 8000.0))
                    return label, conf, (x1, y1, x2, y2)
        return None

    def _detect_target(self, bgr: np.ndarray, prompts: List[str]) -> Optional[Tuple[str, float, Tuple[int, int, int, int]]]:
        if self._yoloe_model is not None:
            with self._model_lock:
                try:
                    results = self._yoloe_model.predict(bgr, verbose=False, conf=0.25)
                    if results and getattr(results[0], "boxes", None) is not None and len(results[0].boxes) > 0:
                        boxes = results[0].boxes
                        idx = int(boxes.conf.argmax().item())
                        xyxy = boxes.xyxy[idx].cpu().numpy().astype(int)
                        cls_id = int(boxes.cls[idx].item())
                        conf = float(boxes.conf[idx].item())
                        lbl = results[0].names.get(cls_id, prompts[0])
                        return str(lbl), conf, (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                    # R4-HIGH-01: When YOLOE is loaded and predict() returns 0 boxes, do NOT fall back to HSV thresholding!
                    return None
                except Exception as exc:
                    logger.warning("YOLOE inference fallback: %s", exc)
        return self._detect_builtin(bgr, prompts)

    def analyze_frame(
        self,
        frame: RGBDFrame,
        lens_mode_override: Optional[str] = None,
        update_history: bool = True,
    ) -> DetectionMeasurement:
        with self._lock:
            lens = lens_mode_override if lens_mode_override in LENS_PROFILES else self.lens_mode
            prompts = list(self._prompts)
            prev_depth = self._last_valid_depth_m if update_history else None
            min_anchor = self._min_approached_depth_m if update_history else None

        lens_min_m = LENS_PROFILES[lens]["blind_zone_m"]
        depth = frame.depth_m
        H, W = depth.shape[:2]

        # Physically valid depth pixels MUST be >= lens hardware minimum (0.80m on 80cm, 0.20m on 20cm)
        valid_global = np.isfinite(depth) & (depth >= lens_min_m) & (depth < 10.0)
        global_valid_ratio = float(np.mean(valid_global))

        # Global wipeout check (all-0, all-NaN, or whole frame inside <0.80m blind zone)
        if global_valid_ratio < 0.12:
            return self._store_and_return(
                DetectionMeasurement(
                    detected=False,
                    label="none",
                    confidence=0.0,
                    bbox=None,
                    is_doorway=False,
                    roi_mode="none",
                    median_depth_m=None,
                    yaw_error_deg=0.0,
                    steering_hint="BRAKE",
                    blind_zone_triggered=True,
                    penetration_anomaly=False,
                    depth_status="GLOBAL_DEPTH_BLIND_OR_POISON_GUARD",
                    roi_valid_ratio=0.0,
                    global_valid_ratio=global_valid_ratio,
                )
            )

        # Global constant far-depth injection check (e.g., entire frame overwritten with 4.0m)
        g_vals = depth[valid_global]
        if g_vals.size > 1000 and float(np.std(g_vals)) < 0.035 and float(np.median(g_vals)) >= 3.0:
            return self._store_and_return(
                DetectionMeasurement(
                    detected=False,
                    label="none",
                    confidence=0.0,
                    bbox=None,
                    is_doorway=False,
                    roi_mode="none",
                    median_depth_m=float(np.median(g_vals)),
                    yaw_error_deg=0.0,
                    steering_hint="BRAKE",
                    blind_zone_triggered=True,
                    penetration_anomaly=True,
                    depth_status="GLOBAL_PENETRATION_4M_POISON_GUARD",
                    roi_valid_ratio=0.0,
                    global_valid_ratio=global_valid_ratio,
                )
            )

        det = self._detect_target(frame.rgb, prompts)
        if det is None:
            return self._store_and_return(
                DetectionMeasurement(
                    detected=False,
                    label="none",
                    confidence=0.0,
                    bbox=None,
                    is_doorway=False,
                    roi_mode="none",
                    median_depth_m=None,
                    yaw_error_deg=0.0,
                    steering_hint="SEARCH",
                    blind_zone_triggered=False,
                    penetration_anomaly=False,
                    depth_status="NO_TARGET_IN_VIEW",
                    roi_valid_ratio=0.0,
                    global_valid_ratio=global_valid_ratio,
                )
            )

        label, conf, (x1, y1, x2, y2) = det
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        u_center = 0.5 * (x1 + x2)
        yaw_deg = math.degrees(math.atan2(u_center - frame.cx, frame.fx))

        is_door = self._is_doorway_label(label, prompts)
        if is_door:
            roi_mode = "edges_15pct"
            edge_w = max(1, int(round(bw * 0.15)))
            vy1 = y1 + int(bh * 0.15)
            vy2 = max(vy1 + 1, y2 - int(bh * 0.15))
            left_patch = depth[vy1:vy2, x1 : min(W, x1 + edge_w)]
            right_patch = depth[vy1:vy2, max(0, x2 - edge_w) : x2]
            roi_pixels = np.concatenate([left_patch.ravel(), right_patch.ravel()])
        else:
            roi_mode = "center_40pct"
            rx1 = x1 + int(bw * 0.30)
            rx2 = max(rx1 + 1, x2 - int(bw * 0.30))
            ry1 = y1 + int(bh * 0.30)
            ry2 = max(ry1 + 1, y2 - int(bh * 0.30))
            roi_pixels = depth[ry1:ry2, rx1:rx2].ravel()

        valid_roi_mask = np.isfinite(roi_pixels) & (roi_pixels >= lens_min_m) & (roi_pixels < 10.0)
        roi_valid_ratio = float(np.mean(valid_roi_mask)) if roi_pixels.size > 0 else 0.0
        valid_vals = roi_pixels[valid_roi_mask]

        # Defense 1 (R2-CRIT-02): Eliminate survivorship bias! Both center 40% and doorway 15% side ROIs
        # MUST maintain roi_valid_ratio >= 0.60 (blind/invalid pixel ratio <= 0.40), else IMMEDIATE BRAKE!
        if valid_vals.size == 0 or roi_valid_ratio < self.MIN_ROI_VALID_RATIO:
            return self._store_and_return(
                DetectionMeasurement(
                    detected=True,
                    label=label,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    is_doorway=is_door,
                    roi_mode=roi_mode,
                    median_depth_m=None,
                    yaw_error_deg=yaw_deg,
                    steering_hint="BRAKE",
                    blind_zone_triggered=True,
                    penetration_anomaly=False,
                    depth_status=f"BLIND_ZONE_GUARD(<{lens_min_m:.2f}m,valid={roi_valid_ratio:.2f})",
                    roi_valid_ratio=roi_valid_ratio,
                    global_valid_ratio=global_valid_ratio,
                )
            )

        # Open doorway side jambs use 25th-percentile of validated jamb pixels; solid objects use median.
        measured_d = float(np.percentile(valid_vals, 25)) if is_door else float(np.median(valid_vals))

        # Defense 2 (R2-CRIT-02): Continuous pinhole geometry physical upper-bound constraint
        # max_plausible_d = (fx * max_expected_real_width * 1.45) / max(1, bw)
        is_table = (not is_door) and self._is_table_label(label)
        if is_door:
            max_expected_real_width = 0.95
            max_expected_real_height = 1.75
        elif is_table:
            max_expected_real_width = 0.95
            max_expected_real_height = 0.55
        else:
            max_expected_real_width = 0.14
            max_expected_real_height = 0.25

        max_plausible_d_w = (frame.fx * max_expected_real_width * 1.45) / max(1, bw)
        max_plausible_d_h = (frame.fy * max_expected_real_height * 1.45) / max(1, bh)
        max_plausible_d = min(max_plausible_d_w, max_plausible_d_h)
        scale_poison = measured_d > max_plausible_d

        # Defense 3 (R2-CRIT-02): Monotonic near-approach anchor guard (`_min_approached_depth_m`)
        # Once robot approaches <= 1.45m, any subsequent upward depth rebound > +0.40m is gradual penetration poisoning!
        jump_poison = prev_depth is not None and prev_depth <= 1.45 and (measured_d - prev_depth) > 0.40
        anchor_poison = min_anchor is not None and (measured_d - min_anchor) > 0.40

        if jump_poison or anchor_poison or scale_poison:
            return self._store_and_return(
                DetectionMeasurement(
                    detected=True,
                    label=label,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    is_doorway=is_door,
                    roi_mode=roi_mode,
                    median_depth_m=measured_d,
                    yaw_error_deg=yaw_deg,
                    steering_hint="BRAKE",
                    blind_zone_triggered=True,
                    penetration_anomaly=True,
                    depth_status="PENETRATION_ANOMALY_BRAKE",
                    roi_valid_ratio=roi_valid_ratio,
                    global_valid_ratio=global_valid_ratio,
                )
            )

        steer = "LEFT" if yaw_deg < -4.0 else ("RIGHT" if yaw_deg > 4.0 else "STRAIGHT")
        meas = DetectionMeasurement(
            detected=True,
            label=label,
            confidence=conf,
            bbox=(x1, y1, x2, y2),
            is_doorway=is_door,
            roi_mode=roi_mode,
            median_depth_m=measured_d,
            yaw_error_deg=yaw_deg,
            steering_hint=steer,
            blind_zone_triggered=False,
            penetration_anomaly=False,
            depth_status="OK",
            roi_valid_ratio=roi_valid_ratio,
            global_valid_ratio=global_valid_ratio,
        )
        with self._lock:
            if update_history:
                self._last_valid_depth_m = measured_d
                if measured_d <= 1.45:
                    if self._min_approached_depth_m is None or measured_d < self._min_approached_depth_m:
                        self._min_approached_depth_m = measured_d
            self._last_measurement = meas
        return meas

    def _store_and_return(self, meas: DetectionMeasurement) -> DetectionMeasurement:
        with self._lock:
            self._last_measurement = meas
        return meas

    def render_annotated_jpeg(self, frame: RGBDFrame, meas: DetectionMeasurement, fsm_state: str = "IDLE") -> bytes:
        vis = frame.rgb.copy()
        H, W = vis.shape[:2]
        cx, cy = int(frame.cx), int(frame.cy)
        cv2.line(vis, (cx - 24, cy), (cx + 24, cy), (210, 210, 210), 1)
        cv2.line(vis, (cx, cy - 24), (cx, cy + 24), (210, 210, 210), 1)

        if meas.bbox is not None:
            x1, y1, x2, y2 = meas.bbox
            box_col = (0, 60, 255) if meas.blind_zone_triggered else (60, 235, 80)
            cv2.rectangle(vis, (x1, y1), (x2, y2), box_col, 2)
            bw, bh = max(1, x2 - x1), max(1, y2 - y1)
            if meas.is_doorway:
                ew = max(1, int(round(bw * 0.15)))
                cv2.rectangle(vis, (x1, y1), (x1 + ew, y2), (255, 220, 0), 1)
                cv2.rectangle(vis, (x2 - ew, y1), (x2, y2), (255, 220, 0), 1)
            else:
                rx1, rx2 = x1 + int(bw * 0.30), x2 - int(bw * 0.30)
                ry1, ry2 = y1 + int(bh * 0.30), y2 - int(bh * 0.30)
                cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (0, 230, 255), 1)

            d_txt = f"{meas.median_depth_m:.2f}m" if meas.median_depth_m is not None else "BLIND"
            tag = f"{meas.label} ({meas.confidence:.2f}) | {d_txt} [{meas.roi_mode}]"
            cv2.putText(vis, tag, (max(6, x1), max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, box_col, 2)

        hud = f"FSM:{fsm_state} | LENS:{self.lens_mode} | STEER:{meas.steering_hint} ({meas.yaw_error_deg:+.1f}deg) | {meas.depth_status}"
        cv2.rectangle(vis, (0, 0), (W, 28), (18, 20, 24), -1)
        hud_col = (70, 90, 255) if meas.blind_zone_triggered else (220, 245, 220)
        cv2.putText(vis, hud, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.44, hud_col, 1)

        ok, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return buf.tobytes() if ok else b""

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            prof = LENS_PROFILES[self.lens_mode]
            return {
                "lens_mode": self.lens_mode,
                "stop_distance_m": prof["stop_distance_m"],
                "blind_zone_m": prof["blind_zone_m"],
                "detector_backend": self._detector_backend,
                "ultralytics_available": self._ultralytics_available,
                "prompts": list(self._prompts),
                "min_approached_depth_m": round(self._min_approached_depth_m, 3) if self._min_approached_depth_m is not None else None,
                "last_measurement": self._last_measurement.to_dict() if self._last_measurement else None,
            }
