"""Module 1: Dual-protocol (HTTP + UDP) BCI Gateway with strict anti-crash validation."""

from __future__ import annotations

import collections
import json
import logging
import math
import re
import socket
import threading
import time
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from backend.config import DEFAULT_UDP_PORT, MAX_BCI_PAYLOAD_BYTES, SSVEP_OPTIONS

logger = logging.getLogger("fyp.bci")

# Canonical emergency stop natural-language keywords (case-insensitive, punctuation-stripped)
ESTOP_EXACT_WORDS = {
    "stop",
    "estop",
    "e-stop",
    "e_stop",
    "emergency",
    "emergency stop",
    "emergencystop",
    "halt",
    "cancel",
    "abort",
    "brake",
    "freeze",
    "急停",
    "停",
    "停下",
    "停住",
    "停吧",
    "暂停",
    "算了",
    "取消",
    "中止",
    "终止",
    "快停",
    "快点停",
    "给我停",
    "请停",
    "停止",
    "刹车",
    "别动",
    "不要动",
    "紧急停止",
    "马上停",
    "立刻停",
    "立即停",
}

_ANSI_ESCAPE_RE = re.compile(r"\x1b(?:\[[0-9;?]*[a-zA-Z]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[()][A-Z0-9]|[a-zA-Z])")

_EN_STOP_NOUN_MASK_RE = re.compile(
    r"\b(?:(?:bus|rest|pit|truck|train|subway|tram|metro|next|full|door|non)\s+stop"
    r"|nonstop|doorstop|backstop|pitstop"
    r"|no\s+stopping"
    r"|stop\s+(?:sign|signs|light|lights|watch|watches))\b",
    re.IGNORECASE,
)
_EN_ESTOP_TOKEN_RE = re.compile(
    r"\b(stop|estop|halt|abort|brake|brakes|freeze|cancel|emergency)\b",
    re.IGNORECASE,
)
# R3-HIGH-01: Tighten English prohibition/inspection prefix so spoken `"go to doorway no stop"` or `"no robot stop"`
# triggers E-STOP immediately, while `"do not stop"`, `"don't stop"`, `"never stop"`, `"check brake"` stay excluded!
_EN_NEG_OR_CHECK_PREFIX_RE = re.compile(
    r"\b(?:(?:do\s+not|dont|never|without)"
    r"|(?:check|checking|inspect|inspecting|test|testing|verify|verifying)(?:\s+(?:the|a|an|my|our|robot|system))?)\s*$",
    re.IGNORECASE,
)
_EN_FREEZE_MOVE_RE = re.compile(r"\b(?:do\s+not|dont|never|stop)\s+mov(?:e|ing)\b", re.IGNORECASE)
_CLAUSE_SPLIT_CHARS_RE = re.compile(r"[,.;!?，。；！？、\n]+")
_ZH_SUBCLAUSE_SPLIT_RE = re.compile(r"[,.;!?，。；！？、\s]+")
_ZH_TRAILING_PARTICLES_RE = re.compile(r"[吧啊呀啦呐哦嘛]+$")

_ZH_NEG_STOP_EXCLUSIONS = (
    "不要停",
    "别停",
    "不用停",
    "无需停",
    "不停",
    "未停",
    "没停",
    "不要刹",
    "别刹",
    "不要急停",
    "检查刹车",
    "测试刹车",
    "公交站",
    "停靠站",
    "停车场",
)
_ZH_ESTOP_PHRASES = (
    "急停",
    "紧急停止",
    "停下",
    "停住",
    "停吧",
    "停一",
    "暂停",
    "快停",
    "快点停",
    "给我停",
    "请停",
    "立即停",
    "立刻停",
    "马上停",
    "停止",
    "刹车",
    "紧急制动",
    "立刻制动",
    "别动",
    "不要动",
    "不许动",
    "取消",
    "中止",
    "终止",
)
_ZH_EXACT_CLAUSE_ESTOP = {"停", "取消", "暂停", "算了", "中止", "终止", "急停", "停止", "刹车", "先停", "不对停"}

_TASK_NEGATION_EN_RE = re.compile(
    r"\b(?:do\s+not|don['’`]?t|dont|never|avoid|no|not|none|neither|without|wrong|cancel|wait|hold\s+on|forget\s+it|skip|instead\s+of)\b",
    re.IGNORECASE,
)
_TASK_BENIGN_BU_WORDS = ("不锈钢", "不倒翁", "不错")
_TASK_NEGATION_ZH = ("不", "不对", "别", "勿", "没", "非", "莫", "禁止", "切勿", "不用", "无需", "算了", "取消", "暂停", "慢着", "等一下", "等等", "稍等", "放弃")
_TASK_FALSE_TARGET_COMPOUNDS = (
    "bus stop",
    "doorstop",
    "door stop",
    "non-stop",
    "nonstop",
    "油门",
    "阀门",
    "部门",
    "专门",
    "热门",
    "入门",
    "水果",
    "水泥",
    "水平",
    "墨水",
    "胶水",
    "薪水",
)


def sanitize_bci_field(raw: Any, default: str = "client", max_len: int = 64) -> str:
    """R3-LOW-01: Strip ANSI escape sequences and control characters (`\\r\\n\\t\\x00`), enforcing <= max_len."""
    if raw is None:
        base = default
    else:
        base = str(raw)
    no_ansi = _ANSI_ESCAPE_RE.sub(" ", base)
    cleaned = "".join(ch if (ch.isprintable() and ch not in "\r\n\t\x00") else " " for ch in no_ansi)
    norm = re.sub(r"\s+", " ", cleaned).strip()
    if not norm:
        if default == base:
            return "client"[:max_len]
        return sanitize_bci_field(default, default="client", max_len=max_len)
    return norm[:max_len]


def _reject_nonfinite_constant(val: str) -> Any:
    raise ValueError(f"Non-standard JSON constant rejected: {val}")


def _normalize_estop_text(cmd: str) -> str:
    low = cmd.strip().lower()
    low = re.sub(r"\be[-_]stop\b", "estop", low)
    low = re.sub(r"\bnon[-_]stop\b", "nonstop", low)
    low = re.sub(r"\bdoor[-_]stop\b", "doorstop", low)
    low = re.sub(r"\bdon['’`]t\b", "dont", low)
    low = re.sub(r"\bcan['’`]t\b", "cant", low)
    return low


def _is_emergency_stop_text(cmd: str) -> bool:
    if not isinstance(cmd, str) or not cmd.strip():
        return False

    low = _normalize_estop_text(cmd)
    stripped_exact = re.sub(r"[\s*!！?？.。,，;；:：\"'`~()\[\]{}<>_—\-]+", " ", low).strip()
    if stripped_exact in ESTOP_EXACT_WORDS:
        return True

    # Explicit "do not move / don't move / stop moving" is an imperative freeze command
    if _EN_FREEZE_MOVE_RE.search(stripped_exact):
        return True

    # Clause-aware English E-Stop scan (masks out "bus stop", "non-stop", "no stopping", and negated/check prefixes)
    en_masked = _EN_STOP_NOUN_MASK_RE.sub(" ", low)
    for clause in _CLAUSE_SPLIT_CHARS_RE.split(en_masked):
        clean_clause = re.sub(r"[\s*\"'`~()\[\]{}<>_—\-]+", " ", clause).strip()
        if not clean_clause:
            continue
        for m in _EN_ESTOP_TOKEN_RE.finditer(clean_clause):
            prefix = clean_clause[: m.start()]
            if not _EN_NEG_OR_CHECK_PREFIX_RE.search(prefix):
                return True

    # Chinese E-Stop phrase & sub-clause scan after removing negated forms ("不要停", "别停", "不停", etc.)
    zh_scan = low
    for excl in _ZH_NEG_STOP_EXCLUSIONS:
        zh_scan = zh_scan.replace(excl, " ")
    if any(k in zh_scan for k in _ZH_ESTOP_PHRASES):
        return True

    # R3-HIGH-01: Sub-clause check catches compound corrections like `"去门口拿水杯，不对，停！"` or `"算了！"`
    for sub in _ZH_SUBCLAUSE_SPLIT_RE.split(zh_scan):
        sub_clean = re.sub(r"[\s*!！?？.。,，;；:：\"'`~()\[\]{}<>_—\-]+", "", sub)
        if not sub_clean:
            continue
        sub_core = _ZH_TRAILING_PARTICLES_RE.sub("", sub_clean)
        if sub_clean in _ZH_EXACT_CLAUSE_ESTOP or sub_core in _ZH_EXACT_CLAUSE_ESTOP:
            return True

    return False


def _classify_natural_command(clean_cmd: str) -> Optional[Tuple[str, List[str], List[str]]]:
    """
    Strictly classify natural-language commands against an explicit target/task whitelist.
    Returns `(task_type, target_prompts, landmark_prompts)` or `None` if unrecognized/negated.
    """
    scan = clean_cmd.strip().lower()
    for comp in _TASK_FALSE_TARGET_COMPOUNDS:
        scan = scan.replace(comp, " ")

    zh_neg_scan = scan
    for benign in _TASK_BENIGN_BU_WORDS:
        zh_neg_scan = zh_neg_scan.replace(benign, " ")

    if _TASK_NEGATION_EN_RE.search(scan) or any(neg in zh_neg_scan for neg in _TASK_NEGATION_ZH):
        return None

    has_mug = bool(re.search(r"\b(?:mug|mugs|cup|cups|teacup)\b", scan)) or ("杯" in scan)
    explicit_bottle = bool(re.search(r"\b(?:bottle|bottles)\b", scan)) or ("瓶" in scan)
    has_bottle = (
        explicit_bottle
        or bool(re.search(r"\bwater\b", scan))
        or any(w in scan for w in ("拿水", "取水", "抓水", "喝水", "端水"))
    )
    has_door = bool(re.search(r"\b(?:door|doors|doorway|doorways|doorframe|entrance)\b", scan)) or ("门" in scan)

    if has_mug and explicit_bottle:
        return None
    if not (has_mug or has_bottle or has_door):
        return None

    if has_door and not (has_mug or has_bottle):
        return (
            "navigate_only",
            ["open doorway", "door frame", "doorway", "door"],
            ["open doorway", "door frame", "doorway"],
        )
    if has_bottle and not has_mug:
        return (
            "navigate_and_pick",
            ["water bottle", "bottle"],
            ["doorway", "table"] if has_door else ["table", "desk"],
        )
    return (
        "navigate_and_pick",
        ["red mug", "mug", "cup"],
        ["doorway", "table"] if has_door else ["table", "desk"],
    )


class BCIGateway:
    """Validates SSVEP / natural-language BCI packets from HTTP POST and LAN UDP."""

    def __init__(
        self,
        dispatch_callback: Callable[[Dict[str, Any]], Tuple[int, str]],
        estop_callback: Callable[[str], Dict[str, Any]],
        is_estop_active: Callable[[], bool],
        udp_host: str = "0.0.0.0",
        udp_port: int = DEFAULT_UDP_PORT,
    ) -> None:
        self._dispatch_cb = dispatch_callback
        self._estop_cb = estop_callback
        self._is_estop_cb = is_estop_active
        self.udp_host = udp_host
        self.udp_port = int(udp_port)

        self._lock = threading.RLock()
        self._logs: Deque[Dict[str, Any]] = collections.deque(maxlen=120)
        self._stats = {"accepted": 0, "rejected": 0, "udp_packets": 0, "http_requests": 0}

        self._stop_evt = threading.Event()
        self._udp_sock: Optional[socket.socket] = None
        self._udp_thread: Optional[threading.Thread] = None

    def _record_log(
        self,
        protocol: str,
        status_code: int,
        accepted: bool,
        summary: str,
        option_id: Optional[int] = None,
        source: str = "unknown",
    ) -> None:
        clean_src = sanitize_bci_field(source, default="unknown", max_len=64)
        clean_sum = sanitize_bci_field(summary, default="event", max_len=200)
        entry = {
            "ts": round(time.time(), 3),
            "time_str": time.strftime("%H:%M:%S"),
            "protocol": protocol,
            "status_code": status_code,
            "accepted": accepted,
            "option_id": option_id,
            "source": clean_src,
            "summary": clean_sum,
        }
        with self._lock:
            if accepted:
                self._stats["accepted"] += 1
            else:
                self._stats["rejected"] += 1
            self._logs.appendleft(entry)

    def process_raw_bytes(
        self,
        raw_bytes: bytes,
        protocol: str = "HTTP",
        sender_hint: str = "client",
    ) -> Tuple[int, Dict[str, Any]]:
        """Validate raw bytes from HTTP body or UDP datagram without ever raising."""
        with self._lock:
            if protocol.upper() == "UDP":
                self._stats["udp_packets"] += 1
            else:
                self._stats["http_requests"] += 1

        if not isinstance(raw_bytes, (bytes, bytearray)) or len(raw_bytes) == 0:
            self._record_log(protocol, 400, False, "Empty BCI payload rejected", source=sender_hint)
            return 400, {"ok": False, "error": "Empty payload"}

        if len(raw_bytes) > MAX_BCI_PAYLOAD_BYTES:
            msg = f"Payload too large ({len(raw_bytes)}B > {MAX_BCI_PAYLOAD_BYTES}B)"
            self._record_log(protocol, 400, False, msg, source=sender_hint)
            return 400, {"ok": False, "error": msg}

        try:
            text = raw_bytes.decode("utf-8", errors="strict").strip()
        except UnicodeDecodeError:
            self._record_log(protocol, 400, False, "Invalid UTF-8 binary garbage rejected", source=sender_hint)
            return 400, {"ok": False, "error": "Malformed UTF-8 byte stream"}

        if not text:
            self._record_log(protocol, 400, False, "Blank payload rejected", source=sender_hint)
            return 400, {"ok": False, "error": "Blank payload"}

        parsed: Any
        if protocol.upper() == "UDP" and text in {"1", "2", "3", "4", "5"}:
            parsed = {"option_id": int(text), "source": sender_hint}
        else:
            try:
                parsed = json.loads(text, parse_constant=_reject_nonfinite_constant)
            except (json.JSONDecodeError, ValueError):
                self._record_log(protocol, 400, False, f"Malformed/Non-finite JSON rejected: {text[:40]!r}", source=sender_hint)
                return 400, {"ok": False, "error": "Malformed or non-standard JSON payload"}

        if not isinstance(parsed, dict):
            self._record_log(protocol, 422, False, "JSON root must be an object", source=sender_hint)
            return 422, {"ok": False, "error": "JSON payload must be an object"}

        # Reject any nested NaN/Infinity floats passed via in-process dicts
        for k, v in parsed.items():
            if isinstance(v, float) and not math.isfinite(v):
                self._record_log(protocol, 400, False, f"Non-finite float in field {k!r}", source=sender_hint)
                return 400, {"ok": False, "error": f"Non-finite value in {k!r}"}

        return self.process_payload(parsed, protocol=protocol, sender_hint=sender_hint)

    def process_payload(
        self,
        payload: Dict[str, Any],
        protocol: str = "HTTP",
        sender_hint: str = "client",
    ) -> Tuple[int, Dict[str, Any]]:
        clean_hint = sanitize_bci_field(sender_hint, default="client", max_len=64)
        raw_src = payload.get("source")
        source = sanitize_bci_field(raw_src, default=clean_hint, max_len=64)
        raw_opt = payload.get("option_id", None)
        raw_cmd = payload.get("command", None)

        # R2-CRIT-01: Hardware BCI Option #5 holds ABSOLUTE HIGHEST SAFETY PREEMPTION!
        # Regardless of whether `command` or auxiliary fields are empty, >256 chars, or wrong type, brake IMMEDIATELY!
        if not isinstance(raw_opt, bool) and isinstance(raw_opt, int) and raw_opt == 5:
            cmd_hint = sanitize_bci_field(raw_cmd, default="Option #5", max_len=64) if isinstance(raw_cmd, str) else "Option #5"
            gate = self._estop_cb(f"BCI {protocol} E-STOP ({cmd_hint} from {source})")
            self._record_log(protocol, 200, True, "BCI Option #5 E-STOP preempted immediately", option_id=5, source=source)
            return 200, {"ok": True, "action": "estop", "option_id": 5, "gatekeeper": gate}

        option_id: Optional[int] = None
        if raw_opt is not None:
            if isinstance(raw_opt, bool) or not isinstance(raw_opt, int):
                msg = f"Invalid option_id type: {raw_opt!r} (must be integer 1..5)"
                self._record_log(protocol, 422, False, msg, source=source)
                return 422, {"ok": False, "error": msg}
            if raw_opt not in SSVEP_OPTIONS:
                msg = f"Out-of-range SSVEP option_id={raw_opt} (valid: 1..5)"
                self._record_log(protocol, 422, False, msg, source=source)
                return 422, {"ok": False, "error": msg}
            option_id = raw_opt

        clean_cmd: Optional[str] = None
        if raw_cmd is not None:
            if not isinstance(raw_cmd, str):
                msg = "Field 'command' must be a string"
                self._record_log(protocol, 422, False, msg, source=source)
                return 422, {"ok": False, "error": msg}
            no_ansi_cmd = _ANSI_ESCAPE_RE.sub(" ", raw_cmd)
            stripped = re.sub(
                r"\s+",
                " ",
                "".join(ch if (ch.isprintable() and ch not in "\r\n\t\x00") else " " for ch in no_ansi_cmd),
            ).strip()
            if not stripped or len(stripped) > 256:
                msg = "Field 'command' must be 1..256 printable characters"
                self._record_log(protocol, 422, False, msg, source=source)
                return 422, {"ok": False, "error": msg}
            clean_cmd = stripped

        if option_id is None and clean_cmd is None:
            msg = "Either valid 'option_id' (1..5) or non-empty 'command' is required"
            self._record_log(protocol, 422, False, msg, source=source)
            return 422, {"ok": False, "error": msg}

        # Natural-language emergency/stop phrase triggers E-STOP immediately!
        if clean_cmd is not None and _is_emergency_stop_text(clean_cmd):
            gate = self._estop_cb(f"BCI {protocol} E-STOP ({clean_cmd[:64]} from {source})")
            self._record_log(protocol, 200, True, "BCI E-STOP triggered immediately", option_id=option_id or 5, source=source)
            return 200, {"ok": True, "action": "estop", "gatekeeper": gate}

        # Block any movement BCI command while E-Stop is latched
        if self._is_estop_cb():
            msg = "E-Stop is latched! Please reset E-Stop before sending movement commands."
            self._record_log(protocol, 409, False, msg, option_id=option_id, source=source)
            return 409, {"ok": False, "error": msg, "estop_active": True}

        if option_id is not None:
            opt = SSVEP_OPTIONS[option_id]
            effective_cmd = opt.command_text
            if clean_cmd is not None:
                norm_cmd_low = clean_cmd.lower()
                official_forms = {opt.command_text.strip().lower(), opt.label_zh.strip().lower()}
                if norm_cmd_low in official_forms:
                    effective_cmd = opt.command_text if norm_cmd_low == opt.label_zh.strip().lower() else clean_cmd
                else:
                    # R4-MED-01: Prevent hybrid payload (`option_id` + `command`) from bypassing negation filter
                    # or causing System-1 / System-2 split-brain target mismatch!
                    classified_mix = _classify_natural_command(clean_cmd)
                    if classified_mix is None:
                        msg = f"Rejected negated or unrecognized command for option_id={option_id}: {clean_cmd[:80]!r}"
                        self._record_log(protocol, 422, False, msg, option_id=option_id, source=source)
                        return 422, {"ok": False, "error": msg}
                    cmd_task_type, cmd_targets, _ = classified_mix
                    opt_target_set = {t.lower() for t in opt.target_prompts}
                    cmd_target_set = {t.lower() for t in cmd_targets}
                    if cmd_task_type != opt.task_type or not (opt_target_set & cmd_target_set):
                        msg = f"Conflicting option_id={option_id} and command: {clean_cmd[:80]!r}"
                        self._record_log(protocol, 422, False, msg, option_id=option_id, source=source)
                        return 422, {"ok": False, "error": msg}
                    effective_cmd = clean_cmd

            task_spec = {
                "option_id": opt.option_id,
                "frequency_hz": opt.frequency_hz,
                "command": effective_cmd,
                "task_type": opt.task_type,
                "target_prompts": list(opt.target_prompts),
                "landmark_prompts": list(opt.landmark_prompts),
                "source": f"bci_{protocol.lower()}:{source}",
            }
        else:
            assert clean_cmd is not None
            classified = _classify_natural_command(clean_cmd)
            if classified is None:
                msg = f"Unrecognized task command: {clean_cmd[:80]!r}"
                self._record_log(protocol, 422, False, msg, source=source)
                return 422, {"ok": False, "error": msg}
            t_type, targets, landmarks = classified
            task_spec = {
                "option_id": None,
                "frequency_hz": None,
                "command": clean_cmd,
                "task_type": t_type,
                "target_prompts": targets,
                "landmark_prompts": landmarks,
                "source": f"bci_{protocol.lower()}:{source}",
            }

        code, dispatch_msg = self._dispatch_cb(task_spec)
        accepted = code == 200
        self._record_log(protocol, code, accepted, dispatch_msg, option_id=option_id, source=source)
        return code, {
            "ok": accepted,
            "message": dispatch_msg,
            "option_id": option_id,
            "command": task_spec["command"],
        }

    def start_udp_server(self) -> None:
        if self._udp_thread and self._udp_thread.is_alive():
            return
        self._stop_evt.clear()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except OSError:
                pass
        sock.bind((self.udp_host, self.udp_port))
        self.udp_port = int(sock.getsockname()[1])
        sock.settimeout(0.10)
        self._udp_sock = sock
        self._udp_thread = threading.Thread(target=self._udp_loop, name="BCIUDPListener", daemon=True)
        self._udp_thread.start()
        logger.info("BCI UDP gateway listening on %s:%d", self.udp_host, self.udp_port)

    def _udp_loop(self) -> None:
        assert self._udp_sock is not None
        while not self._stop_evt.is_set():
            try:
                data, addr = self._udp_sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            except Exception as exc:  # pragma: no cover
                logger.warning("UDP recv error: %s", exc)
                continue

            try:
                sender = f"{addr[0]}:{addr[1]}"
                code, resp = self.process_raw_bytes(data, protocol="UDP", sender_hint=sender)
                ack = json.dumps({"status": code, **resp}).encode("utf-8")
                self._udp_sock.sendto(ack[:1024], addr)
            except Exception as exc:
                logger.warning("UDP packet handling guarded error: %s", exc)

    def stop_udp_server(self) -> None:
        self._stop_evt.set()
        if self._udp_sock is not None:
            try:
                self._udp_sock.close()
            except OSError:
                pass
            self._udp_sock = None
        if self._udp_thread and self._udp_thread.is_alive():
            self._udp_thread.join(timeout=0.5)
        self._udp_thread = None

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "udp_port": self.udp_port,
                "udp_running": bool(self._udp_thread and self._udp_thread.is_alive()),
                "stats": dict(self._stats),
                "options": [
                    {
                        "option_id": o.option_id,
                        "frequency_hz": o.frequency_hz,
                        "label_zh": o.label_zh,
                        "command_text": o.command_text,
                        "task_type": o.task_type,
                    }
                    for o in SSVEP_OPTIONS.values()
                ],
                "recent_logs": list(self._logs)[:30],
            }
