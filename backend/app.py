"""FastAPI application and lifespan manager for FYP Mission Control."""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from adapters.base import RobotAdapter
from adapters.mock_robot import MockRobotAdapter
from adapters.xlerobot_real import XLeRobotRealAdapter
from backend.config import DEFAULT_ROBOT_MODE, DEFAULT_UDP_PORT, FRONTEND_DIR
from backend.modules.bci_gateway import BCIGateway
from backend.modules.cloud_agent import CloudAgentEngine
from backend.modules.data_logger import MissionDataLogger
from backend.modules.fsm_controller import FSMController
from backend.modules.room_explorer import RoomExplorer
from backend.modules.vision_depth import VisionDepthCalibrator
from backend.schemas import (
    AgentBenchRequest,
    FaultInjectionUpdate,
    ManualStartRequest,
    RobotPoseResetRequest,
    RuntimeConfigUpdate,
    sanitize_printable_token,
)

# Signaled immediately on SIGTERM/SIGINT so active MJPEG streams exit in <30ms
GLOBAL_SHUTDOWN_EVENT = threading.Event()


class MissionRuntime:
    """Container owning all hardware adapters, threads, sockets, and isolated executors."""

    def __init__(
        self,
        robot_mode: Optional[str] = None,
        udp_port: Optional[int] = None,
        db_path: Optional[Path] = None,
        default_agent_mode: Optional[str] = None,
    ) -> None:
        self.shutdown_evt = threading.Event()
        # R2-MED-01: Strictly isolate slow Agent benchmark pool from CSV/PNG/SVG academic export pool!
        self.agent_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="AgentBenchPool")
        self.slow_pool = self.agent_pool
        self.export_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="ExportPool")

        mode = (robot_mode or os.getenv("ROBOT_MODE", DEFAULT_ROBOT_MODE)).strip().lower()
        self.robot_mode = "real" if mode == "real" else "mock"
        self.adapter: RobotAdapter = XLeRobotRealAdapter() if self.robot_mode == "real" else MockRobotAdapter()

        self.vision = VisionDepthCalibrator(initial_lens_mode="80cm")
        ag_mode = default_agent_mode or os.getenv("AGENT_MODE", "auto")
        self.agent = CloudAgentEngine(
            emergency_brake_cb=lambda reason: self.adapter.stop(),
            default_mode=ag_mode,
        )
        self.data_logger = MissionDataLogger(db_path=db_path)
        self.fsm = FSMController(
            adapter=self.adapter,
            vision=self.vision,
            agent=self.agent,
            data_logger=self.data_logger,
        )
        self.room_explorer = RoomExplorer(adapter=self.adapter, data_logger=self.data_logger)

        eff_udp_port = udp_port if udp_port is not None else int(os.getenv("UDP_PORT", str(DEFAULT_UDP_PORT)))
        self.bci = BCIGateway(
            dispatch_callback=self.fsm.submit_task,
            estop_callback=self._trigger_all_estop,
            is_estop_active=lambda: self.adapter.estop_active,
            udp_port=eff_udp_port,
        )

    def _trigger_all_estop(self, reason: str) -> Dict[str, Any]:
        self.room_explorer.abort()
        return self.fsm.trigger_estop(reason)

    def start(self) -> None:
        GLOBAL_SHUTDOWN_EVENT.clear()
        self.shutdown_evt.clear()
        self.bci.start_udp_server()
        self.agent.warmup_in_background()

    def stop(self) -> None:
        self.shutdown_evt.set()
        GLOBAL_SHUTDOWN_EVENT.set()
        self.room_explorer.abort()
        self.bci.stop_udp_server()
        self.fsm.shutdown()
        self.adapter.shutdown()
        self.agent_pool.shutdown(wait=False, cancel_futures=True)
        self.export_pool.shutdown(wait=False, cancel_futures=True)

    def snapshot_status(self) -> Dict[str, Any]:
        faults: Dict[str, Any] = {
            "simulate_agent_delay_s": self.agent.simulate_delay_s,
            "simulate_agent_error": self.agent.simulate_error,
        }
        if isinstance(self.adapter, MockRobotAdapter):
            faults["depth_fault"] = self.adapter.depth_fault
            faults["force_vln_early_stop"] = self.adapter.force_vln_early_stop
            faults["collision_occurred"] = self.adapter.collision_occurred
            faults["true_mug_distance_m"] = round(self.adapter.get_true_target_distance("mug"), 3)
            faults["true_doorway_distance_m"] = round(self.adapter.get_true_target_distance("doorway"), 3)

        return {
            "robot_mode": self.robot_mode,
            "camera_index": getattr(self.adapter, "active_camera_index", 0),
            "pose": self.adapter.get_pose().to_dict(),
            "gatekeeper": self.adapter.get_gatekeeper_status(),
            "vram": self.adapter.get_vram_status().to_dict(),
            "fsm": self.fsm.get_status(),
            "room": self.room_explorer.get_snapshot(),
            "vision": self.vision.get_status(),
            "agent": self.agent.get_status(),
            "bci": self.bci.get_status(),
            "faults": faults,
            "logs_summary": self.data_logger.get_summary(),
        }


def create_app(
    robot_mode: Optional[str] = None,
    udp_port: Optional[int] = None,
    db_path: Optional[Path] = None,
    default_agent_mode: Optional[str] = None,
) -> FastAPI:
    runtime = MissionRuntime(
        robot_mode=robot_mode,
        udp_port=udp_port,
        db_path=db_path,
        default_agent_mode=default_agent_mode,
    )

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI) -> AsyncGenerator[None, None]:
        app_instance.state.runtime = runtime
        runtime.start()
        try:
            yield
        finally:
            runtime.stop()

    app = FastAPI(title="FYP Mission Control", version="1.2.0", lifespan=lifespan)
    app.state.runtime = runtime

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if FRONTEND_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.middleware("http")
    async def disable_static_cache(request: Request, call_next: Any) -> Response:
        resp = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return resp

    @app.get("/", include_in_schema=False)
    async def index() -> Response:
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return JSONResponse({"ok": True, "service": "FYP Mission Control"})

    @app.get("/api/health")
    async def health() -> Dict[str, Any]:
        return {"ok": True, "robot_mode": runtime.robot_mode, "udp_port": runtime.bci.udp_port}

    @app.get("/api/status")
    async def get_status() -> Dict[str, Any]:
        return runtime.snapshot_status()

    @app.post("/api/bci/command")
    async def bci_command(request: Request) -> JSONResponse:
        # Pure non-blocking in-memory execution — immune to threadpool starvation (<2ms E-Stop!)
        chunks = []
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > 8192:
                code, resp = runtime.bci.process_raw_bytes(b"X" * 5000, protocol="HTTP", sender_hint="http")
                return JSONResponse(status_code=code, content=resp)
            chunks.append(chunk)
        raw = b"".join(chunks)
        client_host = request.client.host if request.client else "http_client"
        code, resp = runtime.bci.process_raw_bytes(raw, "HTTP", client_host)
        return JSONResponse(status_code=code, content=resp)

    @app.post("/api/estop")
    @app.post("/api/estop/trigger")
    async def trigger_estop(request: Request) -> Dict[str, Any]:
        reason = "Web UI Manual E-STOP"
        try:
            body = await request.json()
            if isinstance(body, dict) and body.get("reason"):
                reason = sanitize_printable_token(str(body["reason"]), max_len=100) or reason
        except Exception:
            pass
        gate = runtime._trigger_all_estop(reason)
        return {"ok": True, "gatekeeper": gate}

    @app.post("/api/camera/select")
    async def select_camera(request: Request) -> Dict[str, Any]:
        body = await request.json()
        idx = int(body.get("index", 0))
        if hasattr(runtime.adapter, "set_camera_index"):
            runtime.adapter.set_camera_index(idx)
        return {"ok": True, "requested_index": idx}

    @app.post("/api/room/scan")
    async def room_scan(request: Request) -> JSONResponse:
        body = await request.json()
        full_360 = bool(body.get("full_360", False))
        res = runtime.room_explorer.start_scan(full_360=full_360)
        return JSONResponse(status_code=200 if res.get("ok") else 409, content=res)

    @app.post("/api/room/navigate")
    async def room_navigate(request: Request) -> JSONResponse:
        body = await request.json()
        res = runtime.room_explorer.start_navigate_to(
            target_en=str(body.get("target_en", "")),
            target_zh=str(body.get("target_zh", "")),
            bearing_deg=float(body.get("bearing_deg", 0.0)),
        )
        return JSONResponse(status_code=200 if res.get("ok") else 409, content=res)

    @app.post("/api/car/test_turn")
    async def car_test_turn(request: Request) -> JSONResponse:
        if runtime.adapter.estop_active:
            return JSONResponse(status_code=409, content={"ok": False, "error": "E-Stop 已锁死"})
        body = await request.json()
        deg = float(body.get("deg", 90.0))
        loop = asyncio.get_running_loop()
        if hasattr(runtime.adapter, "turn_in_place_deg"):
            ok = await loop.run_in_executor(runtime.agent_pool, runtime.adapter.turn_in_place_deg, deg, runtime.adapter.estop_epoch)
            return JSONResponse(status_code=200, content={"ok": ok, "deg": deg})
        return JSONResponse(status_code=200, content={"ok": True, "deg": deg, "mode": "mock"})

    @app.post("/api/estop/reset")
    async def reset_estop() -> JSONResponse:
        gate = await runtime.fsm.reset_estop_async()
        if gate.get("reset_aborted_due_to_new_estop"):
            return JSONResponse(status_code=409, content={"ok": False, "gatekeeper": gate})
        return JSONResponse(status_code=200, content={"ok": True, "gatekeeper": gate})

    @app.post("/api/mission/start")
    async def manual_start(req: ManualStartRequest) -> JSONResponse:
        code, payload = runtime.bci.process_payload(
            {"command": req.command, "source": req.source},
            "HTTP",
            req.source,
        )
        return JSONResponse(status_code=code, content=payload)

    @app.post("/api/robot/reset")
    async def reset_robot(req: RobotPoseResetRequest) -> Dict[str, Any]:
        # R2-CRIT-04: Cancel any active mission cleanly WITHOUT ever unlocking an operator's latched E-Stop!
        await runtime.fsm.abort_active_mission_for_pose_reset()
        pose = runtime.adapter.reset_pose(x=req.x, y=req.y, theta=req.theta, scenario=req.scenario)
        runtime.vision.reset_tracking_history()
        return {
            "ok": True,
            "pose": pose.to_dict(),
            "estop_active": runtime.adapter.estop_active,
        }

    @app.post("/api/agent/benchmark")
    async def benchmark_agent(req: AgentBenchRequest) -> Dict[str, Any]:
        frame = runtime.adapter.get_rgbd_frame(runtime.vision.lens_mode)
        loop = asyncio.get_running_loop()
        # R3-MED-01: Only allow debug benchmark to actuate chassis brake when idle or cruising —
        # NEVER disrupt an active local-vision `YOLO_LAST_MILE` or `VRAM_SWAP_AND_VLA` stage!
        allow_brake = runtime.fsm.state in ("IDLE", "VLN_CRUISE")
        allow_auto_fb = req.simulate_delay_s <= 0.0 and not req.simulate_error
        res = await loop.run_in_executor(
            runtime.agent_pool,
            runtime.agent.infer_with_watchdog,
            frame.rgb,
            req.prompt,
            req.mode,
            req.simulate_delay_s,
            req.simulate_error,
            allow_brake,
            None,
            allow_auto_fb,
        )
        return res

    @app.post("/api/config")
    async def update_config(cfg: RuntimeConfigUpdate) -> Dict[str, Any]:
        if cfg.lens_mode is not None:
            runtime.vision.lens_mode = cfg.lens_mode
            runtime.vision.reset_tracking_history()
        if cfg.blind_walk_enabled is not None:
            runtime.fsm.blind_walk_enabled = cfg.blind_walk_enabled
        if cfg.agent_mode is not None:
            runtime.agent.mode = cfg.agent_mode
        if cfg.cruise_speed_mps is not None:
            runtime.fsm.cruise_speed_mps = cfg.cruise_speed_mps
        if cfg.agent_interval_s is not None:
            runtime.fsm.agent_interval_s = cfg.agent_interval_s
        if cfg.watchdog_timeout_s is not None:
            runtime.agent.watchdog_timeout_s = cfg.watchdog_timeout_s
        if cfg.vla_model is not None:
            runtime.fsm.vla_model = cfg.vla_model
        if cfg.detection_prompts is not None:
            runtime.vision.set_classes(cfg.detection_prompts)
        return {"ok": True, "status": runtime.snapshot_status()}

    @app.post("/api/faults")
    async def update_faults(upd: FaultInjectionUpdate) -> Dict[str, Any]:
        runtime.agent.set_fault_injection(
            simulate_delay_s=upd.simulate_agent_delay_s,
            simulate_error=upd.simulate_agent_error,
        )
        if isinstance(runtime.adapter, MockRobotAdapter):
            runtime.adapter.set_fault_injection(
                depth_fault=upd.depth_fault,
                force_vln_early_stop=upd.force_vln_early_stop,
            )
        return {"ok": True, "faults": runtime.snapshot_status()["faults"]}

    @app.get("/api/camera/snapshot.jpg")
    async def camera_snapshot() -> Response:
        jpg = runtime.fsm.get_latest_jpeg()
        return Response(
            content=jpg,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
        )

    @app.get("/api/camera/stream.mjpg")
    async def camera_mjpeg(request: Request) -> StreamingResponse:
        async def _gen() -> AsyncGenerator[bytes, None]:
            while not (runtime.shutdown_evt.is_set() or GLOBAL_SHUTDOWN_EVENT.is_set()):
                if await request.is_disconnected():
                    break
                jpg = runtime.fsm.get_latest_jpeg()
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                for _ in range(4):
                    if runtime.shutdown_evt.is_set() or GLOBAL_SHUTDOWN_EVENT.is_set():
                        return
                    await asyncio.sleep(0.02)

        return StreamingResponse(_gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/logs/summary")
    async def logs_summary() -> Dict[str, Any]:
        return runtime.data_logger.get_summary()

    @app.get("/api/logs/export/csv")
    async def export_csv() -> Response:
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(runtime.export_pool, runtime.data_logger.export_csv_bytes)
        return Response(
            content=data,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="fyp_mission_logs.csv"'},
        )

    @app.get("/api/logs/export/png")
    async def export_png() -> Response:
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(runtime.export_pool, runtime.data_logger.export_figure_bytes, "png")
        return Response(
            content=data,
            media_type="image/png",
            headers={"Content-Disposition": 'attachment; filename="fyp_mission_report.png"'},
        )

    @app.get("/api/logs/export/svg")
    async def export_svg() -> Response:
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(runtime.export_pool, runtime.data_logger.export_figure_bytes, "svg")
        return Response(
            content=data,
            media_type="image/svg+xml",
            headers={"Content-Disposition": 'attachment; filename="fyp_mission_report.svg"'},
        )

    return app


_lazy_default_app: Optional[FastAPI] = None


def __getattr__(name: str) -> Any:
    global _lazy_default_app
    if name == "app":
        if _lazy_default_app is None:
            _lazy_default_app = create_app()
        return _lazy_default_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
