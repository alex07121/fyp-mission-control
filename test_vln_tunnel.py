"""One-command Mac-side verifier for Cloudflare/RustDesk -> School 32GB LightNav-0 Server."""

from __future__ import annotations

import os
import time

import numpy as np

from backend.modules.remote_vln_client import RemoteVLNClient


def main() -> None:
    url = os.getenv("VLN_REMOTE_URL", "https://clone-loads-celebrities-brake.trycloudflare.com")
    client = RemoteVLNClient(base_url=url, timeout_s=25.0)
    print(f"[1/3] Probing Remote VLN Server -> {url}/health ...")
    try:
        info = client.health()
    except Exception as exc:
        print(f"\n[!] 隧道探测失败 ({type(exc).__name__}: {exc})！")
        raise SystemExit(1)

    print(
        f"      Connected! Host={info.get('hostname')} | Python={info.get('python')} | "
        f"GPU={info.get('gpu_name')} (Free {info.get('vram_free_gb')} / {info.get('vram_total_gb')} GB) | "
        f"Backend={info.get('vln_backend')}"
    )

    print("[2/3] Resetting / Loading LightNav-0 episode ...")
    load_res = client.load(instruction="去桌边拿红色水杯 (Pick up red mug)")
    print(f"      Backend={load_res.get('vln_backend') if load_res else None} | Prompt={load_res.get('en_instruction') if load_res else None}")

    dummy_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_bgr[180:300, 260:380] = (35, 40, 220)
    print("[3/3] Sending 3 frames at 4 FPS (448x256 RGB) ...")
    for i in range(1, 4):
        t0 = time.perf_counter()
        out = client.step(dummy_bgr, "去桌边拿红色水杯 (Pick up red mug)", "STRAIGHT")
        rtt_ms = (time.perf_counter() - t0) * 1000.0
        if out:
            print(
                f"      Step {i}: RTT={rtt_ms:.1f}ms | vLLM={out.get('model_ms')}ms | "
                f"mode={out.get('mode')} | wp0={out.get('wp0')} -> vx={out.get('vx')}, wz={out.get('wz')}, arrived={out.get('arrived')}"
            )
        time.sleep(0.26)


if __name__ == "__main__":
    main()
