"""Module 5: Leak-Free SQLite Experiment Data Logger + Cached Summary & CSV/PNG/SVG Exporters."""

from __future__ import annotations

import contextlib
import copy
import csv
import io
import json
import math
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import matplotlib

matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.backends.backend_svg import FigureCanvasSVG  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from backend.config import DATA_DIR

_CSV_FORMULA_LEAD_CHARS = ("=", "+", "-", "@", "\t", "\r", "\n")
_ANSI_ESCAPE_RE = re.compile(r"\x1b(?:\[[0-9;?]*[a-zA-Z]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[()][A-Z0-9]|[a-zA-Z])")


def _clean_log_str(val: Any, default: str = "", max_len: int = 160) -> str:
    if val is None:
        raw = default
    else:
        raw = str(val)
    no_ansi = _ANSI_ESCAPE_RE.sub(" ", raw)
    printable = "".join(ch if (ch.isprintable() and ch not in "\r\n\t\x00") else " " for ch in no_ansi)
    norm = re.sub(r"\s+", " ", printable).strip()
    return (norm or default)[:max_len]


def _clean_float(val: Any, default: float = 0.0, ndigits: int = 3) -> float:
    try:
        f = float(val)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(f):
        return default
    return round(f, ndigits)


def _clean_opt_float(val: Any, ndigits: int = 3) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return round(f, ndigits)


def _sanitize_csv_cell(val: Any) -> Any:
    """R2-MED-04 + R3-LOW-01: Strip embedded CR/LF/ANSI control chars & neutralize spreadsheet formula injection."""
    if val is None or isinstance(val, (int, float, bool)):
        return val
    raw = _ANSI_ESCAPE_RE.sub(" ", str(val))
    s = "".join(ch if (ch.isprintable() and ch not in "\r\n\x00") else " " for ch in raw)
    if not s:
        return s
    stripped = s.lstrip(" ")
    if s[0] in _CSV_FORMULA_LEAD_CHARS or (stripped and stripped[0] in ("=", "+", "-", "@", "\t")):
        return "'" + s
    return s


class MissionDataLogger:
    """Persists every mission trial with guaranteed connection closing (`contextlib.closing`)."""

    CSV_COLUMNS = [
        "id",
        "created_at",
        "command",
        "source",
        "lens_mode",
        "blind_walk_enabled",
        "vla_model",
        "total_duration_s",
        "stage_durations_json",
        "vln_early_stop_rejections",
        "agent_avg_latency_ms",
        "agent_avg_jpeg_kb",
        "final_stop_distance_m",
        "final_yaw_error_deg",
        "peak_vram_gb",
        "success",
        "abort_reason",
    ]

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path: Path = (db_path or (DATA_DIR / "mission_logs.sqlite3")).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._cached_summary: Optional[Dict[str, Any]] = None
        self._init_db()

    @contextlib.contextmanager
    def _managed_conn(self) -> Iterator[sqlite3.Connection]:
        """Guarantees SQLite FD is explicitly closed on scope exit (zero FD leak!)."""
        conn = sqlite3.connect(str(self.db_path), timeout=10.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._lock, self._managed_conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS mission_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    command TEXT NOT NULL,
                    source TEXT NOT NULL,
                    lens_mode TEXT NOT NULL,
                    blind_walk_enabled INTEGER NOT NULL DEFAULT 0,
                    vla_model TEXT NOT NULL DEFAULT 'SmolVLA',
                    total_duration_s REAL NOT NULL,
                    stage_durations_json TEXT NOT NULL,
                    vln_early_stop_rejections INTEGER NOT NULL,
                    agent_avg_latency_ms REAL NOT NULL,
                    agent_avg_jpeg_kb REAL NOT NULL,
                    final_stop_distance_m REAL,
                    final_yaw_error_deg REAL,
                    peak_vram_gb REAL NOT NULL,
                    success INTEGER NOT NULL,
                    abort_reason TEXT NOT NULL
                )
                """
            )
            # R2-MED-04: Automatic non-destructive ALTER TABLE migration for pre-existing SQLite databases
            existing_cols = {str(r["name"]) for r in conn.execute("PRAGMA table_info(mission_runs)").fetchall()}
            if "blind_walk_enabled" not in existing_cols:
                conn.execute("ALTER TABLE mission_runs ADD COLUMN blind_walk_enabled INTEGER NOT NULL DEFAULT 0")
            if "vla_model" not in existing_cols:
                conn.execute("ALTER TABLE mission_runs ADD COLUMN vla_model TEXT NOT NULL DEFAULT 'SmolVLA'")

    def record_run(self, record: Dict[str, Any]) -> int:
        now_iso = time.strftime("%Y-%m-%d %H:%M:%S")
        raw_stages = record.get("stage_durations")
        clean_stages: Dict[str, float] = {}
        if isinstance(raw_stages, dict):
            for k, v in raw_stages.items():
                clean_stages[str(k)[:40]] = _clean_float(v, 0.0, 3)
        stages_json = json.dumps(clean_stages, ensure_ascii=False, allow_nan=False)

        stop_d = _clean_opt_float(record.get("final_stop_distance_m"), 3)
        yaw_e = _clean_opt_float(record.get("final_yaw_error_deg"), 2)
        lens_clean = _clean_log_str(record.get("lens_mode", "80cm"), default="80cm", max_len=24)
        # R3-MED-03: Blind-walk applies exclusively to 80cm lens runs; never pollute 20cm runs with `+BW`!
        bw_flag = 1 if (record.get("blind_walk_enabled") and lens_clean == "80cm") else 0
        vla_name = _clean_log_str(record.get("vla_model") or "SmolVLA", default="SmolVLA", max_len=48)
        cmd_clean = _clean_log_str(record.get("command", "unknown"), default="unknown", max_len=200)
        src_clean = _clean_log_str(record.get("source", "manual"), default="manual", max_len=64)
        abort_clean = _clean_log_str(record.get("abort_reason", "completed"), default="completed", max_len=160)

        try:
            rej_int = int(record.get("vln_early_stop_rejections", 0))
        except (TypeError, ValueError):
            rej_int = 0

        with self._lock:
            with self._managed_conn() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO mission_runs (
                        created_at, command, source, lens_mode,
                        blind_walk_enabled, vla_model, total_duration_s,
                        stage_durations_json, vln_early_stop_rejections,
                        agent_avg_latency_ms, agent_avg_jpeg_kb,
                        final_stop_distance_m, final_yaw_error_deg,
                        peak_vram_gb, success, abort_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        now_iso,
                        cmd_clean,
                        src_clean,
                        lens_clean,
                        bw_flag,
                        vla_name,
                        _clean_float(record.get("total_duration_s", 0.0), 0.0, 3),
                        stages_json,
                        rej_int,
                        _clean_float(record.get("agent_avg_latency_ms", 0.0), 0.0, 1),
                        _clean_float(record.get("agent_avg_jpeg_kb", 0.0), 0.0, 2),
                        stop_d,
                        yaw_e,
                        _clean_float(record.get("peak_vram_gb", 1.4), 1.4, 2),
                        1 if record.get("success") else 0,
                        abort_clean,
                    ),
                )
                row_id = int(cur.lastrowid or 0)
            self._cached_summary = None
            return row_id

    def list_runs(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock, self._managed_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM mission_runs ORDER BY id DESC LIMIT ?", (int(limit),)
            ).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            d["success"] = bool(d["success"])
            d["blind_walk_enabled"] = bool(d.get("blind_walk_enabled", 0))
            d["vla_model"] = str(d.get("vla_model") or "SmolVLA")
            try:
                d["stage_durations"] = json.loads(d.get("stage_durations_json") or "{}")
            except Exception:
                d["stage_durations"] = {}
            out.append(d)
        return out

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            if self._cached_summary is not None:
                return copy.deepcopy(self._cached_summary)

            runs = self.list_runs(limit=500)
            total = len(runs)
            if total == 0:
                summary: Dict[str, Any] = {
                    "total_runs": 0,
                    "success_count": 0,
                    "aborted_count": 0,
                    "success_rate_pct": 0.0,
                    "avg_agent_latency_ms": 0.0,
                    "avg_stop_distance_m": 0.0,
                    "avg_yaw_error_deg": 0.0,
                    "max_peak_vram_gb": 0.0,
                    "total_vln_rejections": 0,
                    "lens_comparison": {
                        "80cm": {"count": 0, "success_count": 0, "avg_stop_m": 0.0},
                        "80cm_blind_walk": {"count": 0, "success_count": 0, "avg_stop_m": 0.0},
                        "20cm": {"count": 0, "success_count": 0, "avg_stop_m": 0.0},
                    },
                    "recent_runs": [],
                }
                self._cached_summary = summary
                return copy.deepcopy(summary)

            succ_runs = [r for r in runs if r["success"]]
            succ = len(succ_runs)
            lats = [r["agent_avg_latency_ms"] for r in runs if r["agent_avg_latency_ms"] > 0]
            # Only compute terminal parking precision over completed runs (exclude mid-way E-Stop aborts!)
            dists = [r["final_stop_distance_m"] for r in succ_runs if r["final_stop_distance_m"] is not None]
            yaws = [abs(r["final_yaw_error_deg"]) for r in succ_runs if r["final_yaw_error_deg"] is not None]

            # R2-MED-04: Separate pure 80cm (~0.85m) from 80cm + 40cm blind-walk (~0.45m) so averages never skew!
            buckets = (
                ("80cm", lambda r: r["lens_mode"] == "80cm" and not r.get("blind_walk_enabled")),
                ("80cm_blind_walk", lambda r: r["lens_mode"] == "80cm" and bool(r.get("blind_walk_enabled"))),
                ("20cm", lambda r: r["lens_mode"] == "20cm"),
            )
            lens_comp: Dict[str, Any] = {}
            for key, pred in buckets:
                sub_all = [r for r in runs if pred(r)]
                sub_ok = [r for r in succ_runs if pred(r)]
                sub_d = [r["final_stop_distance_m"] for r in sub_ok if r["final_stop_distance_m"] is not None]
                lens_comp[key] = {
                    "count": len(sub_all),
                    "success_count": len(sub_ok),
                    "avg_stop_m": round(sum(sub_d) / len(sub_d), 3) if sub_d else 0.0,
                }

            summary = {
                "total_runs": total,
                "success_count": succ,
                "aborted_count": total - succ,
                "success_rate_pct": round(succ * 100.0 / total, 1),
                "avg_agent_latency_ms": round(sum(lats) / len(lats), 1) if lats else 0.0,
                "avg_stop_distance_m": round(sum(dists) / len(dists), 3) if dists else 0.0,
                "avg_yaw_error_deg": round(sum(yaws) / len(yaws), 2) if yaws else 0.0,
                "max_peak_vram_gb": round(max(r["peak_vram_gb"] for r in runs), 2),
                "total_vln_rejections": sum(r["vln_early_stop_rejections"] for r in runs),
                "lens_comparison": lens_comp,
                "recent_runs": runs[:25],
            }
            self._cached_summary = summary
            return copy.deepcopy(summary)

    def export_csv_bytes(self) -> bytes:
        runs = list(reversed(self.list_runs(limit=2000)))
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=self.CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in runs:
            row: Dict[str, Any] = {}
            for col in self.CSV_COLUMNS:
                if col == "success":
                    row[col] = 1 if r.get("success") else 0
                elif col == "blind_walk_enabled":
                    row[col] = 1 if r.get("blind_walk_enabled") else 0
                else:
                    row[col] = _sanitize_csv_cell(r.get(col))
            writer.writerow(row)
        return buf.getvalue().encode("utf-8-sig")

    def export_figure_bytes(self, fmt: str = "png") -> bytes:
        runs = list(reversed(self.list_runs(limit=50)))
        fig = Figure(figsize=(11.5, 4.2), dpi=160, facecolor="#0f172a")
        axes = fig.subplots(1, 3)
        for ax in axes:
            ax.set_facecolor("#1e293b")
            ax.tick_params(colors="#cbd5e1", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#475569")

        if not runs:
            for idx, title in enumerate(("Outcome Breakdown", "Stop Distance (m)", "VRAM Peak & Latency")):
                axes[idx].set_title(title, color="#f8fafc", fontsize=10, fontweight="bold")
                axes[idx].text(0.5, 0.5, "No Runs Recorded Yet", color="#94a3b8", ha="center", va="center")
        else:
            succ_n = sum(1 for r in runs if r["success"])
            fail_n = len(runs) - succ_n
            axes[0].bar(["Success", "Aborted/Guarded"], [succ_n, fail_n], color=["#10b981", "#ef4444"], width=0.5)
            axes[0].set_title("Mission Reliability", color="#f8fafc", fontsize=10, fontweight="bold")
            axes[0].set_ylabel("Trials", color="#cbd5e1", fontsize=9)

            xs = [r["id"] for r in runs if r["success"] and r["final_stop_distance_m"] is not None]
            ys = [r["final_stop_distance_m"] for r in runs if r["success"] and r["final_stop_distance_m"] is not None]
            if xs:
                axes[1].plot(xs, ys, marker="o", color="#38bdf8", linewidth=1.8, label="Completed Stop (m)")
                axes[1].axhline(0.85, color="#f59e0b", linestyle="--", linewidth=1.0, label="80cm Target (0.85m)")
                axes[1].axhline(0.45, color="#22c55e", linestyle=":", linewidth=1.0, label="20cm / BW Target (0.45m)")
                axes[1].legend(fontsize=7, facecolor="#0f172a", labelcolor="#e2e8f0")
            axes[1].set_title("Terminal Stop Precision (m)", color="#f8fafc", fontsize=10, fontweight="bold")
            axes[1].set_xlabel("Run ID", color="#cbd5e1", fontsize=8)

            v_ids = [r["id"] for r in runs]
            v_peaks = [r["peak_vram_gb"] for r in runs]
            axes[2].bar(v_ids, v_peaks, color="#a855f7", width=0.6, label="Peak VRAM (GB)")
            axes[2].axhline(12.0, color="#ef4444", linestyle="--", linewidth=1.2, label="12GB GPU Limit")
            axes[2].set_ylim(0, 13.2)
            axes[2].legend(fontsize=7, facecolor="#0f172a", labelcolor="#e2e8f0")
            axes[2].set_title("RTX 5070 12GB VRAM Budget", color="#f8fafc", fontsize=10, fontweight="bold")
            axes[2].set_xlabel("Run ID", color="#cbd5e1", fontsize=8)

        fig.suptitle("FYP Embodied BCI-VLN-YOLOE-VLA Mission Telemetry", color="#f8fafc", fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.92))

        out = io.BytesIO()
        if fmt.lower() == "svg":
            FigureCanvasSVG(fig).print_svg(out)
        else:
            FigureCanvasAgg(fig).print_png(out)
        return out.getvalue()
