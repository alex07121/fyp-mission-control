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

## 3. Switching to Real XLeRobot Hardware on Ubuntu

Fill in the `TODO(REAL_ROBOT)` hooks inside `adapters/xlerobot_real.py` and run:

```bash
ROBOT_MODE=real .venv/bin/python run.py
```

## 4. Automated Test Suite

```bash
.venv/bin/pytest -v
```
