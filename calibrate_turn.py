"""
BT04-A BLE Car Turn Calibration & Angle Verification Script (Cable-Safe Left/Right Paired).

Usage:
  1. Run standard multi-angle test table (100ms..400ms pulses + 30°/45°/90° verification):
       .venv/bin/python calibrate_turn.py
  2. Unwind USB cable (e.g. turn right 360°):
       .venv/bin/python calibrate_turn.py --unwind-right-ms 2800
  3. Interactive manual test (test any ms or degree & adjust scale with your eyes):
       .venv/bin/python calibrate_turn.py --interactive
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from bleak import BleakClient, BleakScanner

PROJECT_ROOT = Path(__file__).resolve().parent
CALIB_PATH = PROJECT_ROOT / "data" / "car_calibration.json"

# Solved from Step-16.2 full-circle closure: Cam 0 FHD wide-angle lens HFOV ≈ 115°, 200ms ≈ 22° (~9.1 ms/deg)
DEFAULT_TRUE_HFOV_DEG = 115.0
DEFAULT_MS_PER_DEG = 9.1

BLE_WRITE_UUID = "0000ffe2-0000-1000-8000-00805f9b34fb"
BLE_CMDS = {
    "left": b"ONC",
    "right": b"OND",
    "forward": b"ONA",
    "backward": b"ONB",
    "stop": b"ONF",
    "explicit_stop": b"ONE",
}


def measure_rotation_deg(before_bgr: np.ndarray, after_bgr: np.ndarray, hfov_deg: float = DEFAULT_TRUE_HFOV_DEG) -> Optional[float]:
    """Measure physical rotation (positive = left, negative = right) using Cam 0 wide-angle HFOV."""
    g1 = cv2.cvtColor(before_bgr, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(after_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(1800)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    if len(matches) < 15:
        return None
    p1 = np.float32([k1[m.queryIdx].pt for m in matches])
    p2 = np.float32([k2[m.trainIdx].pt for m in matches])
    _, inliers = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=4.5)
    if inliers is None or int(inliers.sum()) < 12:
        return None
    keep = inliers.ravel().astype(bool)
    w = float(g1.shape[1])
    cx = w / 2.0
    fx = cx / math.tan(math.radians(hfov_deg) / 2.0)
    a1 = np.degrees(np.arctan((p1[keep, 0] - cx) / fx))
    a2 = np.degrees(np.arctan((p2[keep, 0] - cx) / fx))
    return float(np.median(a2 - a1))


def grab_fresh_frame(cap: cv2.VideoCapture) -> np.ndarray:
    for _ in range(6):
        cap.grab()
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("Failed to read frame from Cam 0")
    if frame.shape[1] != 640:
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (640, int(round(h * 640 / w))), interpolation=cv2.INTER_AREA)
    return frame


async def send_exact_pulse(client: BleakClient, direction: str, pulse_ms: int) -> None:
    await client.write_gatt_char(BLE_WRITE_UUID, BLE_CMDS[direction], response=False)
    await asyncio.sleep(pulse_ms / 1000.0)
    await client.write_gatt_char(BLE_WRITE_UUID, BLE_CMDS["stop"], response=False)


async def turn_calibrated_deg(
    client: BleakClient,
    cap: Optional[cv2.VideoCapture],
    target_deg: float,
    ms_per_deg: float,
    hfov_deg: float = DEFAULT_TRUE_HFOV_DEG,
) -> Tuple[float, int]:
    """Turn in short <=220ms pulses (prevents DC motor high-RPM runaway!) with Cam 0 visual feedback."""
    max_sub_deg = 22.0
    turned_total = 0.0
    steps = 0
    before = grab_fresh_frame(cap) if cap is not None else None

    for _ in range(12):
        rem = target_deg - turned_total
        if abs(rem) <= 3.0:
            break
        sub = max(-max_sub_deg, min(max_sub_deg, rem))
        pulse_ms = int(round(max(65.0, min(230.0, abs(sub) * ms_per_deg))))
        await send_exact_pulse(client, "left" if sub > 0 else "right", pulse_ms)
        await asyncio.sleep(0.28)
        steps += 1
        if cap is not None and before is not None:
            after = grab_fresh_frame(cap)
            meas = measure_rotation_deg(before, after, hfov_deg=hfov_deg)
            if meas is not None and abs(meas) >= 1.5 and (meas > 0) == (sub > 0):
                turned_total += meas
            else:
                turned_total += sub
            before = after
        else:
            turned_total += sub

    return round(turned_total, 2), steps


async def run_main(args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    for _ in range(8):
        cap.read()

    dev = await BleakScanner.find_device_by_name(args.name, timeout=8.0)
    if dev is None:
        cap.release()
        raise RuntimeError(f"BLE car '{args.name}' not found! Stop run.py first so BLE is free.")

    async with BleakClient(dev) as client:
        await client.write_gatt_char(BLE_WRITE_UUID, BLE_CMDS["stop"], response=False)
        print(f"[OK] Connected to {dev.name} ({dev.address}) + Cam {args.camera} (广角 HFOV={args.hfov}°)")

        if args.unwind_right_ms > 0:
            print(f"[UNWIND] 正在向右退绕松开 USB 线 ({args.unwind_right_ms} ms) ...")
            rem = args.unwind_right_ms
            while rem > 0:
                chunk = min(300, rem)
                await send_exact_pulse(client, "right", chunk)
                await asyncio.sleep(0.20)
                rem -= chunk
            await asyncio.sleep(0.4)

        if args.interactive:
            ms_per_deg = args.ms_per_deg
            print("\n=== 手动校准测试模式 ===")
            print("  输入 `l 200` 或 `r 200`  -> 测试单次毫秒脉冲并显示实测转角")
            print("  输入 `deg 45` 或 `deg 90` -> 测试目标角度 (正=左转, 负=右转, 如 deg -90)")
            print("  输入 `set 9.5`           -> 手动微调每度毫秒数 (当前 %.2f ms/°)" % ms_per_deg)
            print("  输入 `q`                 -> 保存配置并退出\n")
            while True:
                line = (await asyncio.to_thread(input, "calib> ")).strip().lower()
                if line in ("q", "quit", "exit"):
                    break
                sp = line.split()
                if len(sp) == 2 and sp[0] in ("l", "r"):
                    ms = int(sp[1])
                    b = grab_fresh_frame(cap)
                    await send_exact_pulse(client, "left" if sp[0] == "l" else "right", ms)
                    await asyncio.sleep(0.32)
                    a = grab_fresh_frame(cap)
                    d = measure_rotation_deg(b, a, args.hfov)
                    print(f"  脉冲 {ms}ms -> 实测转角: {d:+.1f}°" if d is not None else "  纹理不足未能测量")
                elif len(sp) == 2 and sp[0] == "deg":
                    tgt = float(sp[1])
                    act, st = await turn_calibrated_deg(client, cap, tgt, ms_per_deg, args.hfov)
                    print(f"  指令 {tgt:+.1f}° -> 实测转角: {act:+.1f}° ({st} 步短脉冲)")
                elif len(sp) == 2 and sp[0] == "set":
                    ms_per_deg = float(sp[1])
                    print(f"  已更新 ms_per_deg = {ms_per_deg:.2f} ms/°")
            CALIB_PATH.write_text(
                json.dumps({"true_camera_hfov_deg": args.hfov, "pulse_ms_per_deg": round(ms_per_deg, 3)}, indent=2),
                encoding="utf-8",
            )
            cap.release()
            return

        print("\n[1/2] 实测【单次脉冲时长 -> 真实车体转角】对照表 (左转一次 + 右转回正，零缠线)：")
        print("      脉冲时长(ms) | 左转真实度数 | 右转真实度数 | 平均度数 | 每度耗时(ms/°)")
        print("      -------------+--------------+--------------+----------+---------------")
        sweep_rows: List[Dict[str, Any]] = []
        mpd_samples: List[float] = []

        for p_ms in (100, 150, 200, 300, 400):
            await asyncio.sleep(0.30)
            b1 = grab_fresh_frame(cap)
            await send_exact_pulse(client, "left", p_ms)
            await asyncio.sleep(0.35)
            a1 = grab_fresh_frame(cap)
            dl = measure_rotation_deg(b1, a1, args.hfov)

            await send_exact_pulse(client, "right", p_ms)
            await asyncio.sleep(0.35)
            a2 = grab_fresh_frame(cap)
            dr = measure_rotation_deg(a1, a2, args.hfov)

            vals = [abs(x) for x in (dl, dr) if x is not None and abs(x) >= 2.0]
            avg_d = float(np.mean(vals)) if vals else 0.0
            mpd = (p_ms / avg_d) if avg_d > 1.0 else args.ms_per_deg
            if 140 <= p_ms <= 260 and avg_d > 3.0:
                mpd_samples.append(mpd)
            dl_s = f"{dl:+9.1f}°" if dl is not None else "    N/A   "
            dr_s = f"{dr:+9.1f}°" if dr is not None else "    N/A   "
            print(f"      {p_ms:9d} ms |   {dl_s}  |   {dr_s}  |  {avg_d:6.1f}° |    {mpd:6.2f} ms/°")
            sweep_rows.append({"pulse_ms": p_ms, "left_deg": dl, "right_deg": dr, "avg_deg": round(avg_d, 2), "ms_per_deg": round(mpd, 2)})

        best_mpd = float(np.median(mpd_samples)) if mpd_samples else args.ms_per_deg
        print(f"\n      ==> 最佳短脉冲换算系数: {best_mpd:.2f} ms/° (30°≈{30*best_mpd:.0f}ms | 45°≈{45*best_mpd:.0f}ms | 90°≈{90*best_mpd:.0f}ms)")
        print("      ==> 之前 90° 转过头的根本原因:")
        print("          1) 之前沿用 70° 窄视场旧参数 (22.3 ms/°)，而车头广角镜头真实视场为 ~115°；")
        print("          2) 之前连续通电 2010ms 不停顿使电机冲上高速，导致下发 90° 实际狂转了 ~220°！")

        print("\n[2/2] 实测校准后的【指令角度 vs 真实车体转角】验证 (左转后立即右转回原点)：")
        verify_rows = []
        for tgt in (30.0, -30.0, 45.0, -45.0, 90.0, -90.0):
            await asyncio.sleep(0.30)
            act, n_s = await turn_calibrated_deg(client, cap, tgt, best_mpd, args.hfov)
            verify_rows.append({"command_deg": tgt, "actual_deg": act, "error_deg": round(act - tgt, 2), "pulses": n_s})
            print(f"      下发指令 {tgt:+6.1f}°  ==>  车体实测转角 {act:+6.1f}°  (误差 {act - tgt:+5.1f}°, 分 {n_s} 次短脉冲)")

        CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
        CALIB_PATH.write_text(
            json.dumps(
                {
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "true_camera_hfov_deg": args.hfov,
                    "pulse_ms_per_deg": round(best_mpd, 3),
                    "deg_per_200ms_pulse": round(200.0 / best_mpd, 2),
                    "pulse_table": sweep_rows,
                    "angle_verification": verify_rows,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n[OK] 标定表已写入 {CALIB_PATH}，主程序将自动加载该参数！")
        await client.write_gatt_char(BLE_WRITE_UUID, BLE_CMDS["stop"], response=False)

    cap.release()


def main() -> None:
    p = argparse.ArgumentParser(description="BT04-A BLE car angle calibration tool.")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--hfov", type=float, default=DEFAULT_TRUE_HFOV_DEG)
    p.add_argument("--ms-per-deg", type=float, default=DEFAULT_MS_PER_DEG)
    p.add_argument("--unwind-right-ms", type=int, default=0)
    p.add_argument("--name", default="BT04-A")
    p.add_argument("--interactive", action="store_true")
    asyncio.run(run_main(p.parse_args()))


if __name__ == "__main__":
    main()
