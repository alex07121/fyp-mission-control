"""One-command Mac-side verifier for RustDesk 9001 -> School 127.0.0.1:9001 VLN TCP Tunnel."""

from __future__ import annotations

import os
import time

import numpy as np

from backend.modules.remote_vln_client import RemoteVLNClient


def main() -> None:
    url = os.getenv("VLN_REMOTE_URL", "http://127.0.0.1:9001")
    client = RemoteVLNClient(base_url=url, timeout_s=2.0)
    print(f"[1/3] Probing RustDesk TCP Tunnel -> {url}/health ...")
    info = client.health()
    print(f"      Connected! Host={info.get('hostname')} | GPU={info.get('gpu_name')} ({info.get('vram_total_gb')} GB)")

    print("[2/3] Loading VLN policy on remote GPU ...")
    print("     ", client.load())

    dummy_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_bgr[180:300, 260:380] = (35, 40, 220)
    print("[3/3] Sending 3 compressed 320x240 RGB frames over tunnel ...")
    for i in range(1, 4):
        t0 = time.perf_counter()
        out = client.step(dummy_bgr, "Go to open doorway", "STRAIGHT")
        rtt_ms = (time.perf_counter() - t0) * 1000.0
        print(f"      Step {i}: RTT={rtt_ms:.1f} ms | Server infer={out.get('infer_ms')} ms | Action={out}")


if __name__ == "__main__":
    main()
