"""Red/Blue Stress Test 1: BCI HTTP & UDP malformed packets, bounds, concurrency, and E-Stop gating."""

from __future__ import annotations

import concurrent.futures
import json
import socket
import time
from fastapi.testclient import TestClient


def test_bci_malformed_and_out_of_bounds_http(client: TestClient) -> None:
    # Binary garbage, broken JSON, oversized payload, wrong root types
    bad_raws = [
        b"",
        b"   ",
        b"\xff\xfe\x00\x81\x99garbage",
        b'{"option_id": ',
        b"[1, 2, 3]",
        b'"just_a_string"',
        b"null",
        b"X" * 6000,
    ]
    for raw in bad_raws:
        res = client.post("/api/bci/command", content=raw, headers={"Content-Type": "application/octet-stream"})
        assert res.status_code in (400, 422), f"Unexpected {res.status_code} for {raw[:20]!r}"

    # Out-of-bounds & tricky Python bool/float/null option_ids
    bad_json_payloads = [
        {},
        {"option_id": 0},
        {"option_id": -5},
        {"option_id": 99},
        {"option_id": "abc"},
        {"option_id": None},
        {"option_id": True},   # isinstance(True, int) trap!
        {"option_id": False},  # isinstance(False, int) trap!
        {"option_id": 1.5},
        {"command": ""},
        {"command": "   \n\t  "},
        {"command": "A" * 400},
    ]
    for payload in bad_json_payloads:
        res = client.post("/api/bci/command", json=payload)
        assert res.status_code in (400, 422), f"Payload {payload!r} was not rejected: {res.status_code}"

    # Ensure server is healthy and still in IDLE
    st = client.get("/api/status").json()
    assert st["fsm"]["state"] == "IDLE"


def test_bci_udp_fuzzing_and_valid_dispatch(client: TestClient) -> None:
    st = client.get("/api/status").json()
    udp_port = st["bci"]["udp_port"]
    assert udp_port > 0

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(1.0)
        # Send garbage & out-of-range packets over UDP
        for pkt in (b"\x80\xff\x00", b"{bad_json", b'{"option_id": 99}', b'{"option_id": -1}', b"0", b"abc"):
            sock.sendto(pkt, ("127.0.0.1", udp_port))
            ack, _ = sock.recvfrom(2048)
            ack_obj = json.loads(ack.decode("utf-8"))
            assert ack_obj["ok"] is False

        # Send Option 5 (BCI E-Stop) via UDP -> must immediately latch E-Stop!
        sock.sendto(json.dumps({"option_id": 5, "source": "pytest_udp"}).encode(), ("127.0.0.1", udp_port))
        ack, _ = sock.recvfrom(2048)
        ack_obj = json.loads(ack.decode("utf-8"))
        assert ack_obj["status"] == 200 and ack_obj["ok"] is True

    # Movement commands while E-Stop latched MUST be rejected with 409
    r_move = client.post("/api/bci/command", json={"option_id": 1})
    assert r_move.status_code == 409
    assert r_move.json()["ok"] is False

    # Option 5 while E-Stop latched is still safely accepted (200)
    r_estop_again = client.post("/api/bci/command", json={"option_id": 5})
    assert r_estop_again.status_code == 200

    # Reset E-Stop
    client.post("/api/estop/reset")
    assert client.get("/api/status").json()["gatekeeper"]["estop_active"] is False


def test_bci_high_concurrency_storm_single_admission(client: TestClient) -> None:
    def _fire(idx: int) -> int:
        res = client.post("/api/bci/command", json={"option_id": 1, "source": f"burst_{idx}"})
        return res.status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        codes = list(pool.map(_fire, range(30)))

    # Exactly 1 request must win admission (200), all other concurrent bursts get 409 Conflict!
    assert codes.count(200) == 1
    assert codes.count(409) == 29

    # Stop running mission cleanly via E-Stop + Reset
    client.post("/api/estop/trigger")
    client.post("/api/estop/reset")
    time.sleep(0.05)
    assert client.get("/api/status").json()["fsm"]["state"] == "IDLE"
