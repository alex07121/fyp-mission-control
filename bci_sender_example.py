"""Minimal 17-line BCI sender demo (HTTP POST + LAN UDP, env-configurable & repeatable)."""
import json, os, socket, urllib.error, urllib.request

HOST = os.getenv("BCI_HOST", "127.0.0.1")
HTTP_URL = f"http://{HOST}:{int(os.getenv('PORT', '8080'))}/api/bci/command"
UDP_ADDR = (HOST, int(os.getenv("UDP_PORT", "9999")))

req = urllib.request.Request(HTTP_URL, data=json.dumps({"option_id": 1, "source": "bci_http_demo"}).encode(), headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=3.0) as resp:
        print("[HTTP]", resp.status, resp.read().decode())
except urllib.error.HTTPError as exc:
    print("[HTTP]", exc.code, exc.read().decode())

with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
    sock.sendto(json.dumps({"option_id": 3, "source": "bci_udp_demo"}).encode(), UDP_ADDR)
    print("[UDP] Sent option_id=3 (non-latching doorway task) to", UDP_ADDR)
