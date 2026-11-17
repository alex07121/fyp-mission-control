/* FYP Mission Control — Hardened XSS-Safe Real-Time Telemetry & Control Client */
(() => {
  const $ = (id) => document.getElementById(id);

  function escapeHtml(raw) {
    return String(raw ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  async function apiPost(url, body = {}) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    return { status: res.status, data };
  }

  function showManualFeedback(text, isError = false) {
    const box = $("manualFeedback");
    if (!box) return;
    box.textContent = text;
    box.classList.toggle("err", Boolean(isError && text));
    box.classList.toggle("ok", Boolean(!isError && text));
  }

  // R2-HIGH-01: Only allow Spacebar through when actively typing into a genuine text field;
  // focus on checkbox, range slider, select, button, or anywhere else triggers immediate hard E-STOP!
  function isTextEntryElement(el) {
    if (!el) return false;
    if (el.isContentEditable) return true;
    const tag = (el.tagName || "").toUpperCase();
    if (tag === "TEXTAREA") return true;
    if (tag === "INPUT") {
      const inputType = (el.type || "text").toLowerCase();
      return ["text", "search", "password", "url", "email", "number"].includes(inputType);
    }
    return false;
  }

  // 1. Global Hard E-STOP + Capture-Phase Spacebar Preemption
  async function triggerEstop(reason = "Web UI Manual E-STOP") {
    await apiPost("/api/estop/trigger", { reason });
    await pollStatus();
  }

  $("btnEstop").addEventListener("click", () => triggerEstop("Topbar Red E-STOP Button"));
  $("btnResetEstop").addEventListener("click", async () => {
    const { status, data } = await apiPost("/api/estop/reset", {});
    if (status === 200) {
      showManualFeedback("已解除急停锁死，系统恢复待命 (IDLE)", false);
    } else {
      showManualFeedback(`急停解除被拒 (${status}): ${data.error || "存在更新的急停信号"}`, true);
    }
    await pollStatus();
  });

  window.addEventListener(
    "keydown",
    (ev) => {
      if (ev.code === "Space" || ev.key === " ") {
        const el = document.activeElement;
        if (!isTextEntryElement(el)) {
          ev.preventDefault();
          ev.stopPropagation();
          if (el && typeof el.blur === "function") {
            el.blur();
          }
          triggerEstop("Keyboard SPACEBAR Emergency Stop");
        }
      }
    },
    { capture: true }
  );

  // 2. SSVEP 5-Option BCI Grid (`ssvepGrid` bound via event delegation) & Manual Input
  $("ssvepGrid").addEventListener("click", async (ev) => {
    const btn = ev.target.closest(".ssvep-btn");
    if (!btn) return;
    const optId = parseInt(btn.dataset.opt, 10);
    const { status, data } = await apiPost("/api/bci/command", { option_id: optId, source: "web_ssvep_panel" });
    if (status !== 200) {
      showManualFeedback(`BCI #${optId} 被拦截 (${status}): ${data.error || data.message || "拒绝发车"}`, true);
    } else {
      showManualFeedback(optId === 5 ? "⛔ BCI #5 紧急急停已立即锁死电机！" : `已下发 BCI #${optId}: ${data.command || ""}`, optId === 5);
    }
    await pollStatus();
  });

  async function submitManualCommand() {
    const inp = $("inpManualCmd");
    const cmd = inp.value.trim();
    if (!cmd) {
      showManualFeedback("请输入有效的任务或急停指令", true);
      return;
    }
    const { status, data } = await apiPost("/api/mission/start", { command: cmd, source: "web_manual" });
    // R2-MED-02: Clear input ONLY on HTTP 200 OK; preserve user input and alert prominently on 409/422!
    if (status === 200) {
      inp.value = "";
      if (data.action === "estop") {
        showManualFeedback("⛔ 自然语言急停已触发，底盘已锁死！", true);
      } else {
        showManualFeedback(`✅ 任务已受理: ${data.command || cmd}`, false);
      }
    } else {
      showManualFeedback(`⚠️ 指令被拒 (HTTP ${status}): ${data.error || data.message || "无法识别或急停锁死中"}`, true);
    }
    await pollStatus();
  }

  $("btnSendManual").addEventListener("click", submitManualCommand);
  $("inpManualCmd").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && !ev.isComposing) {
      ev.preventDefault();
      submitManualCommand();
    }
  });

  // 2B. Whole-Room Scanner, Camera Switcher & One-Click Navigate-to-Object
  document.querySelectorAll("[data-cam]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const idx = parseInt(btn.dataset.cam, 10);
      await apiPost("/api/camera/select", { index: idx });
      await pollStatus();
    });
  });

  $("btnScanView").addEventListener("click", async () => {
    $("roomStatusText").textContent = "⏳ 正在抓拍 Cam 0 分析眼前物体...";
    const { status, data } = await apiPost("/api/room/scan", { full_360: false });
    if (status !== 200) {
      $("roomStatusText").textContent = `⚠️ 无法启动扫描: ${data.error || status}`;
    }
    await pollStatus();
  });

  $("btnScan360").addEventListener("click", async () => {
    $("roomStatusText").textContent = "🔄 已发送 360° 全屋旋转扫描指令，小车即将分 8 步旋转...";
    const { status, data } = await apiPost("/api/room/scan", { full_360: true });
    if (status !== 200) {
      $("roomStatusText").textContent = `⚠️ 无法启动 360° 扫描: ${data.error || status}`;
    }
    await pollStatus();
  });

  $("btnGoCustomObj").addEventListener("click", async () => {
    const val = $("inpCustomObj").value.trim();
    if (!val) return;
    await apiPost("/api/room/navigate", { target_en: val, target_zh: val, bearing_deg: 0.0 });
    await pollStatus();
  });

  $("btnTestTurnL90")?.addEventListener("click", async () => {
    $("roomStatusText").textContent = "📐 正在测试原地左转 +90° (短脉冲 + Cam 0 视觉闭环)...";
    const { status } = await apiPost("/api/car/test_turn", { deg: 90.0 });
    $("roomStatusText").textContent = status === 200 ? "✅ 左转 +90° 测试完成！请肉眼核对车头是否刚好转了约 90°" : "⚠️ 急停锁死中，请先解除急停";
  });

  $("btnTestTurnR90")?.addEventListener("click", async () => {
    $("roomStatusText").textContent = "📐 正在测试原地右转 -90° (短脉冲 + Cam 0 视觉闭环)...";
    const { status } = await apiPost("/api/car/test_turn", { deg: -90.0 });
    $("roomStatusText").textContent = status === 200 ? "✅ 右转 -90° 测试完成！请肉眼核对车头是否回正" : "⚠️ 急停锁死中，请先解除急停";
  });

  $("roomObjectsGrid").addEventListener("click", async (ev) => {
    const btn = ev.target.closest(".room-obj-btn");
    if (!btn) return;
    await apiPost("/api/room/navigate", {
      target_en: btn.dataset.en || "",
      target_zh: btn.dataset.zh || "",
      bearing_deg: parseFloat(btn.dataset.bearing || "0"),
    });
    await pollStatus();
  });

  // 3. Hardware Lens Profile & Blind-Walk Controls
  $("btnLens80").addEventListener("click", async () => {
    await apiPost("/api/config", { lens_mode: "80cm" });
    await pollStatus();
  });
  $("btnLens20").addEventListener("click", async () => {
    await apiPost("/api/config", { lens_mode: "20cm" });
    await pollStatus();
  });
  $("chkBlindWalk").addEventListener("change", async (e) => {
    await apiPost("/api/config", { blind_walk_enabled: e.target.checked });
    await pollStatus();
  });

  // 4. Open-Vocab Prompts & Simulation Scenario Presets
  async function applyCustomPrompts() {
    const list = $("inpPrompts").value.split(",").map((s) => s.trim()).filter(Boolean);
    if (list.length) {
      await apiPost("/api/config", { detection_prompts: list });
      await pollStatus();
    }
  }

  $("btnSetPrompts").addEventListener("click", applyCustomPrompts);
  $("inpPrompts").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && !ev.isComposing) {
      ev.preventDefault();
      applyCustomPrompts();
    }
  });

  document.querySelectorAll("[data-scenario]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      await apiPost("/api/robot/reset", { scenario: btn.dataset.scenario });
      await pollStatus();
    });
  });

  // 5. Cloud Agent Mode, VLA Selector, Sliders & Single-Step Benchmark
  $("selAgentMode").addEventListener("change", async (e) => {
    await apiPost("/api/config", { agent_mode: e.target.value });
    await pollStatus();
  });
  $("selVlaModel").addEventListener("change", async (e) => {
    await apiPost("/api/config", { vla_model: e.target.value });
    await pollStatus();
  });

  $("rngSpeed").addEventListener("input", (e) => {
    $("lblSpeed").textContent = parseFloat(e.target.value).toFixed(2);
  });
  $("rngSpeed").addEventListener("change", async (e) => {
    await apiPost("/api/config", { cruise_speed_mps: parseFloat(e.target.value) });
    await pollStatus();
  });

  $("rngInterval").addEventListener("input", (e) => {
    $("lblInterval").textContent = parseFloat(e.target.value).toFixed(2);
  });
  $("rngInterval").addEventListener("change", async (e) => {
    await apiPost("/api/config", { agent_interval_s: parseFloat(e.target.value) });
    await pollStatus();
  });

  async function runAgentBenchmark(simulateDelayS = 0.0) {
    const benchBtn = $("btnBenchAgent");
    const slowBtn = $("btnTestSlowAgent");
    benchBtn.disabled = true;
    slowBtn.disabled = true;
    benchBtn.textContent = simulateDelayS > 0 ? "看门狗测试中..." : "测速中...";
    try {
      const { data } = await apiPost("/api/agent/benchmark", {
        prompt: "Identify red mug or doorway direction: LEFT, STRAIGHT, or RIGHT.",
        simulate_delay_s: simulateDelayS,
      });
      $("agKb").textContent = `${data.jpeg_kb ?? "--"} KB`;
      $("agDir").textContent = `${data.direction ?? "--"} (${data.provider ?? ""})`;
      $("agMs").textContent = `${data.latency_ms ?? "--"} ms`;
    } finally {
      benchBtn.disabled = false;
      slowBtn.disabled = false;
      benchBtn.textContent = "单次抓拍测速";
      await pollStatus();
    }
  }

  $("btnBenchAgent").addEventListener("click", () => runAgentBenchmark(0.0));

  // 6. Safety Fault Injection Controls
  $("selDepthFault").addEventListener("change", async (e) => {
    await apiPost("/api/faults", { depth_fault: e.target.value });
    await pollStatus();
  });
  // R2-MED-02: One-shot 5.0s watchdog test via `/api/agent/benchmark` ONLY — NEVER pollutes persistent `/api/faults`!
  $("btnTestSlowAgent").addEventListener("click", () => runAgentBenchmark(5.0));

  $("btnClearFaults").addEventListener("click", async () => {
    $("selDepthFault").value = "none";
    await apiPost("/api/faults", {
      depth_fault: "none",
      simulate_agent_delay_s: 0.0,
      simulate_agent_error: false,
      force_vln_early_stop: true,
    });
    await pollStatus();
  });

  // 7. Live Telemetry Polling (with `pollBusy` re-entrancy mutex) & Camera Refresh
  let camBusy = false;
  function refreshCamera() {
    if (camBusy) return;
    camBusy = true;
    const img = $("camFeed");
    const nextUrl = `/api/camera/snapshot.jpg?t=${Date.now()}`;
    const loader = new Image();
    loader.onload = () => {
      img.src = nextUrl;
      camBusy = false;
    };
    loader.onerror = () => {
      camBusy = false;
    };
    loader.src = nextUrl;
  }

  async function renderStatusOnce() {
    try {
      const res = await fetch("/api/status");
      if (!res.ok) throw new Error("HTTP " + res.status);
      const s = await res.json();

      // Connection indicator (`connDot`)
      $("connDot").classList.remove("offline");
      $("connDot").title = "Backend Connected";

      // Gatekeeper, Topbar & SSVEP grid visual lock hint
      const estop = Boolean(s.gatekeeper.estop_active);
      $("topbar").classList.toggle("estop-latched", estop);
      $("ssvepGrid").classList.toggle("estop-disabled", estop);
      const gb = $("gateBadge");
      gb.classList.toggle("latched", estop);
      gb.textContent = estop
        ? `⛔ E-STOP LATCHED (Epoch #${s.gatekeeper.estop_epoch}) | Blocked: ${s.gatekeeper.blocked_move_attempts}`
        : `GATEKEEPER: UNLOCKED (Epoch #${s.gatekeeper.estop_epoch}) | Blocked: ${s.gatekeeper.blocked_move_attempts}`;

      // FSM Pipeline Track (`fsmTrack`) + IDLE E-STOP LATCHED highlight (`R2-HIGH-02`)
      const fsmTrack = $("fsmTrack");
      fsmTrack.classList.toggle("running", s.fsm.state !== "IDLE");
      fsmTrack.querySelectorAll(".fsm-node").forEach((el) => {
        const isCurrent = el.dataset.state === s.fsm.state;
        el.classList.toggle("active", isCurrent && !(estop && el.dataset.state === "IDLE"));
        if (el.dataset.state === "IDLE") {
          const latchedIdle = estop && s.fsm.state === "IDLE";
          el.classList.toggle("estop-idle", latchedIdle);
          el.textContent = latchedIdle ? "1. IDLE (⛔ E-STOP LATCHED)" : "1. IDLE";
        }
      });

      // Dynamic 12GB / 32GB School GPU Slot VRAM Bar
      if ($("vramTitleLabel")) {
        $("vramTitleLabel").textContent = s.vram.total_gb > 16.0 ? "School RTX 5000 32GB Slot" : "RTX 5070 Laptop VRAM";
      }
      $("vramModels").textContent = s.vram.active_models.join(" + ");
      $("vramText").textContent = `${s.vram.allocated_gb.toFixed(2)} / ${s.vram.total_gb.toFixed(1)} GB (Peak ${s.vram.peak_gb.toFixed(2)} GB)`;
      $("vramFill").style.width = `${Math.min(100, s.vram.utilization_pct)}%`;

      // Vision, Pose & Non-intrusive Control Sync (`R2-MED-02` + `R3-MED-03`)
      $("visionBackendBadge").textContent = s.vision.detector_backend;
      const isLens20 = s.vision.lens_mode === "20cm";
      $("btnLens80").classList.toggle("active", !isLens20);
      $("btnLens20").classList.toggle("active", isLens20);
      const bwEl = $("chkBlindWalk");
      bwEl.disabled = isLens20;
      if (bwEl.parentElement) {
        bwEl.parentElement.style.opacity = isLens20 ? "0.52" : "1";
        bwEl.parentElement.title = isLens20 ? "20cm 新镜头支持 0.45m 视觉直停，无需末端盲走" : "";
      }
      if (document.activeElement !== bwEl) {
        bwEl.checked = Boolean(s.fsm.blind_walk_enabled);
      }
      if (document.activeElement !== $("selAgentMode")) {
        $("selAgentMode").value = s.agent.mode;
      }
      if (document.activeElement !== $("selVlaModel")) {
        $("selVlaModel").value = s.fsm.vla_model;
      }
      if (document.activeElement !== $("rngSpeed") && typeof s.fsm.cruise_speed_mps === "number") {
        $("rngSpeed").value = String(s.fsm.cruise_speed_mps);
        $("lblSpeed").textContent = Number(s.fsm.cruise_speed_mps).toFixed(2);
      }
      if (document.activeElement !== $("rngInterval") && typeof s.fsm.agent_interval_s === "number") {
        $("rngInterval").value = String(s.fsm.agent_interval_s);
        $("lblInterval").textContent = Number(s.fsm.agent_interval_s).toFixed(2);
      }
      if (document.activeElement !== $("inpPrompts") && Array.isArray(s.vision.prompts)) {
        $("inpPrompts").value = s.vision.prompts.join(", ");
      }

      // Fault injection visibility banner
      const flt = s.faults || {};
      const activeFaults = [];
      if (flt.depth_fault && flt.depth_fault !== "none") activeFaults.push(`depth=${flt.depth_fault}`);
      if (flt.simulate_agent_delay_s > 0) activeFaults.push(`delay=${flt.simulate_agent_delay_s}s`);
      if (flt.simulate_agent_error) activeFaults.push("agent_err=ON");
      const fb = $("faultStatusBadge");
      fb.classList.toggle("fault-active", activeFaults.length > 0);
      fb.textContent = activeFaults.length > 0 ? `⚠️ ACTIVE: ${activeFaults.join(", ")}` : "FAULTS: NONE";
      if (document.activeElement !== $("selDepthFault") && flt.depth_fault) {
        $("selDepthFault").value = flt.depth_fault;
      }

      const m = s.vision.last_measurement;
      if (m) {
        $("valDepth").textContent = m.median_depth_m !== null ? `${m.median_depth_m.toFixed(2)} m` : m.depth_status;
        $("valYaw").textContent = `${m.yaw_error_deg.toFixed(1)}°`;
        $("valSteer").textContent = m.steering_hint;
      }
      $("valPose").textContent = `(${s.pose.x.toFixed(2)}, ${s.pose.y.toFixed(2)}, ${s.pose.theta_deg.toFixed(0)}°)`;

      // Agent & BCI
      $("udpPortBadge").textContent = `UDP :${s.bci.udp_port}`;
      if ($("camIdxBadge") && s.camera_index !== undefined) {
        $("camIdxBadge").textContent = `当前镜头: Cam ${s.camera_index}${s.camera_index === 0 ? " (车头广角)" : ""}`;
      }
      if (s.room) {
        $("roomStatusText").textContent = `${s.room.progress_msg}${s.room.scene_summary ? " | 场景: " + s.room.scene_summary : ""}`;
        const objs = s.room.objects || [];
        const sig = JSON.stringify(objs);
        if ($("roomObjectsGrid").dataset.sig !== sig) {
          $("roomObjectsGrid").dataset.sig = sig;
          $("roomObjectsGrid").innerHTML = objs.map((o) => {
            const b = Number(o.bearing_deg || 0);
            const dirStr = Math.abs(b) <= 12 ? "正前" : (b > 0 ? `左 ${b.toFixed(0)}°` : `右 ${Math.abs(b).toFixed(0)}°`);
            return `<button class="btn-opt room-obj-btn" style="padding:7px 12px; border-color:#3b82f6;" data-en="${escapeHtml(o.name_en)}" data-zh="${escapeHtml(o.name_zh)}" data-bearing="${b}">🎯 <b>${escapeHtml(o.name_zh)}</b> <small>(${escapeHtml(o.name_en)} · ${dirStr} · ${escapeHtml(o.distance)})</small></button>`;
          }).join("");
        }
      }
      $("watchdogBadge").textContent = `Watchdog Trips: ${s.agent.watchdog_trips}`;
      $("vlnRejCount").textContent = String(s.fsm.vln_early_stop_rejections);
      if (s.agent.last_call) {
        $("agKb").textContent = `${s.agent.last_call.jpeg_kb} KB`;
        $("agDir").textContent = `${s.agent.last_call.direction} (${s.agent.last_call.provider})`;
        $("agMs").textContent = `${s.agent.last_call.latency_ms} ms`;
      }

      // Summary Stats & XSS-Escaped Runs Table
      const sum = s.logs_summary;
      $("stTotal").textContent = String(sum.total_runs);
      $("stRate").textContent = `${sum.success_rate_pct}%`;
      $("stLat").textContent = `${sum.avg_agent_latency_ms} ms`;
      $("stStop").textContent = `${sum.avg_stop_distance_m.toFixed(2)} m`;

      if (sum.recent_runs && sum.recent_runs.length > 0) {
        $("runsTbody").innerHTML = sum.recent_runs.slice(0, 12).map((r) => {
          const lensTag = r.lens_mode === "80cm" && r.blind_walk_enabled ? "80cm+BW" : r.lens_mode;
          return `
          <tr>
            <td>#${Number(r.id)}</td>
            <td>${escapeHtml(String(r.command).slice(0, 22))}</td>
            <td>${escapeHtml(lensTag)}</td>
            <td>${r.final_stop_distance_m !== null ? Number(r.final_stop_distance_m).toFixed(2) + "m" : "--"}</td>
            <td>${Number(r.peak_vram_gb).toFixed(1)}G</td>
            <td class="${r.success ? "log-ok" : "log-err"}">${r.success ? "SUCCESS" : escapeHtml(String(r.abort_reason).slice(0, 24))}</td>
          </tr>
        `;
        }).join("");
      }

      // XSS-Escaped Live Consoles
      $("fsmConsole").innerHTML = s.fsm.recent_events.map((e) => {
        const cls = e.level === "CRITICAL" || e.level === "ERROR" ? "log-err" : (e.level === "GUARD" || e.level === "WARN" ? "log-guard" : "log-ok");
        return `<div class="log-line ${cls}">[${escapeHtml(e.time_str)}] ${escapeHtml(e.msg)}</div>`;
      }).join("");

      $("bciConsole").innerHTML = s.bci.recent_logs.map((l) => {
        const cls = l.accepted ? "log-ok" : "log-err";
        return `<div class="log-line ${cls}">[${escapeHtml(l.time_str)}][${escapeHtml(l.protocol)}:${Number(l.status_code)}] ${escapeHtml(l.summary)}</div>`;
      }).join("");
    } catch (_) {
      $("connDot").classList.add("offline");
      $("connDot").title = "Backend Disconnected";
    }
  }

  let pollBusy = false;
  let pollPending = false;
  async function pollStatus() {
    if (pollBusy) {
      pollPending = true;
      return;
    }
    pollBusy = true;
    try {
      do {
        pollPending = false;
        await renderStatusOnce();
      } while (pollPending);
    } finally {
      pollBusy = false;
    }
  }

  setInterval(refreshCamera, 120);
  setInterval(pollStatus, 260);
  pollStatus();
})();
