# FYP Mission Control — Embodied BCI · VLN · YOLOE · VLA

Cross-platform ground control station for **XLeRobot + Dual SO-ARM101** (macOS & Ubuntu Linux `RTX 5070 Laptop 12GB VRAM`).

## 1. Quick Start (macOS / Ubuntu Linux)

```bash
python3 -m venv .venv
# (Optional on Ubuntu RTX 5070: install CUDA 12.8 PyTorch first)
# .venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -r requirements.txt
.venv/bin/python run.py
```

Place `yoloe-26s-seg.pt` under `weights/yoloe-26s-seg.pt` (or set `YOLOE_WEIGHTS_PATH`) for zero-shot `YOLOE` open-vocabulary segmentation; when local weights are absent, the system instantaneously falls back to the deterministic open-vocabulary HSV detector without blocking on network downloads.

Environment variables (all optional):
- `HOST` (default `0.0.0.0`)
- `PORT` (default `8080`)
- `UDP_PORT` (default `9999`)
- `ROBOT_MODE` (`mock` default simulator | `real` for `adapters/xlerobot_real.py`)
- `AGENT_MODE` (`auto` | `cloud` | `mock`)

## 2. BCI Teammate Quick Integration

Run the 17-line dual-protocol demo while the server is running:

```bash
.venv/bin/python bci_sender_example.py
```

## 3. Dynamic 32GB School GPU Slot Scheduler (`ubuntu@324` — RTX 5000 Ada 32GB)

Maximizes the school `RTX 5000 Ada 32GB` via **Resident Brain + `<1s` CPU-RAM Sleep/Wake Specialists**:
- **Resident Slot (`24/7` Hot)**: `qwen3-vl:8b-instruct` (auto-selected, `num_ctx=4096`, `keep_alive=-1`) + `YOLOE-26s-seg` (`~7.5 GB`); auto-evicts `27b+` models at startup.
- **Dynamic Slot A (Driving)**: `LightNav-0` (`gpu_util=0.50`, `~15.7 GB`) wakes on navigation and sleeps to pinned CPU RAM (`vLLM sleep level=1`) on arrival.
- **Dynamic Slot B (Grasping)**: `ACT` / `TurboVLA` / `SmolVLA` / full `bf16 pi0.5` wakes after braking (auto-sleeping `LightNav-0` first) and sleeps back to CPU RAM after grasp.

1. **On School Server (`ubuntu@324`)**:
   ```bash
   ollama pull qwen3-vl:8b-instruct  # optional one-time upgrade from 4b -> 8b
   conda activate lightnav && git pull && python remote_vln_server.py
   ```
2. **On Local Mac / Car Edge Client** (bypasses Clash proxy automatically for `127.0.0.1` and Tailscale `100.*`):
   ```bash
   NO_PROXY="127.0.0.1,localhost,100.106.0.106" \
   SCHOOL_GPU_URL="http://100.106.0.106:9001" \
   ROBOT_MODE=real AGENT_MODE=school CAMERA_INDEX=0 \
   .venv/bin/python run.py
   ```
   *(Or set `SCHOOL_GPU_URL="https://<your-trycloudflare-url>"` when connecting over Cloudflare Quick Tunnel.)*

## 4. Automated Test Suite

```bash
.venv/bin/pytest -v
```
