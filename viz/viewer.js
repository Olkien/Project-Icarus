// Icarus run viewer: replay and live
(function () {
  const P = window.IcarusParse;
  const MAX_RUNS = 4;
  const WORLD_MM = 820;
  const ARENA_MM = 800;
  const TARGET_BOX_MM = 500;
  const HIT_MM = 10;
  const TRAIL_S = 1.5;
  const STRIP_TICKS = 120;
  const RANGE_BAR_MM = 400;

  const el = function (id) { return document.getElementById(id); };
  const canvas = el("view");
  const ctx = canvas.getContext("2d");
  const ui = {
    file: el("file"), play: el("play"), restart: el("restart"), scrub: el("scrub"),
    clock: el("clock"), speed: el("speed"), png: el("png"), legend: el("legend"),
    compare: el("compare"),
    status: el("status"), live: el("live"), liveScenario: el("live-scenario"),
    liveController: el("live-controller"), liveSeed: el("live-seed"), liveStart: el("live-start"),
    liveWind: el("live-wind"), liveWindVal: el("live-wind-val"),
    liveNoise: el("live-noise"), liveNoiseVal: el("live-noise-val"),
  };

  const state = {
    runs: [],
    primary: 0,
    t: 0,
    playing: false,
    speed: 1,
    last: null,
    live: null,
    source: null,
    compare: "overlay",
  };

  // theme
  function css(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function controllerColor(name) {
    return css("--ctrl-" + name) || css("--ctrl-other");
  }

  function setStatus(text) {
    ui.status.textContent = text;
  }

  // runs
  function runLabel(entry) {
    const h = entry.run.header || {};
    return (h.controller || "?") + " " + (h.scenario || "?") + " s" + (h.seed !== undefined ? h.seed : "?");
  }

  function addRun(run, name) {
    const h = run.header || {};
    const entry = { run: run, name: name, color: controllerColor(h.controller || "other") };
    state.runs.push(entry);
    while (state.runs.length > MAX_RUNS) state.runs.shift();
    state.primary = state.runs.length - 1;
    checkCompare();
    renderLegend();
    return entry;
  }

  function checkCompare() {
    const first = state.runs[0] && state.runs[0].run.header;
    const notes = [];
    for (const entry of state.runs) {
      const h = entry.run.header;
      if (first && h && (h.seed !== first.seed || h.scenario !== first.scenario)) {
        notes.push("runs differ in scenario or seed");
        break;
      }
    }
    for (const entry of state.runs) {
      if (entry.run.errors.length) notes.push(entry.name + ": " + entry.run.errors.length + " bad lines");
      if (entry.run.truncated && entry !== state.live) notes.push(entry.name + ": truncated");
    }
    setStatus(notes.join(" | "));
  }

  function renderLegend() {
    ui.legend.textContent = "";
    state.runs.forEach(function (entry, i) {
      const item = document.createElement("span");
      item.className = "item" + (i === state.primary ? " primary" : "");
      const sw = document.createElement("span");
      sw.className = "swatch";
      sw.style.background = entry.color;
      item.appendChild(sw);
      item.appendChild(document.createTextNode(runLabel(entry)));
      item.title = "Show this run in the panel";
      item.onclick = function () { state.primary = i; renderLegend(); };
      const drop = document.createElement("span");
      drop.className = "drop";
      drop.textContent = "x";
      drop.title = "Remove";
      drop.onclick = function (ev) {
        ev.stopPropagation();
        state.runs.splice(i, 1);
        state.primary = Math.max(0, Math.min(state.primary, state.runs.length - 1));
        renderLegend();
        checkCompare();
      };
      item.appendChild(drop);
      ui.legend.appendChild(item);
    });
  }

  function maxDuration() {
    let d = 0;
    for (const entry of state.runs) d = Math.max(d, P.duration(entry.run));
    return d;
  }

  function loadText(text, name) {
    const run = P.parseRun(text);
    if (!run.ticks.length) {
      setStatus(name + ": no ticks");
      return;
    }
    addRun(run, name);
  }

  function fetchText(url) {
    return fetch(url)
      .then(function (r) {
        if (!r.ok) throw new Error(r.status + " " + url);
        return r.text();
      })
      .catch(function (e) {
        setStatus("load failed: " + e.message);
        return null;
      });
  }

  function loadUrls(urls) {
    return Promise.all(urls.map(fetchText)).then(function (texts) {
      texts.forEach(function (text, i) {
        if (text !== null) loadText(text, urls[i].split("/").pop());
      });
    });
  }

  // playback
  function setPlaying(on) {
    state.playing = on;
    ui.play.textContent = on ? "Pause" : "Play";
    state.last = null;
  }

  function tickFrame(now) {
    const total = maxDuration();
    if (state.playing && state.last !== null) {
      state.t += ((now - state.last) / 1000) * state.speed;
      if (state.t >= total) {
        state.t = total;
        if (!state.source) setPlaying(false);
      }
    }
    if (total > 0 && state.t > total) state.t = total;
    state.last = state.playing ? now : null;
    ui.scrub.value = total > 0 ? Math.round((state.t / total) * 1000) : 0;
    ui.clock.textContent = state.t.toFixed(2) + " / " + total.toFixed(2) + " s";
    draw();
    requestAnimationFrame(tickFrame);
  }

  // drawing helpers
  function layout(w, h) {
    const wide = w >= 760;
    const panelW = wide ? Math.min(340, Math.max(260, w * 0.28)) : w;
    const arenaH = wide ? h : Math.min(w, h * 0.6);
    const arena = { x: 0, y: 0, w: wide ? w - panelW : w, h: arenaH };
    const panel = wide
      ? { x: w - panelW, y: 0, w: panelW, h: h }
      : { x: 0, y: arenaH, w: w, h: h - arenaH };
    return { arena: arena, panel: panel };
  }

  function toPx(a, x, y) {
    return [a.cx + x * a.s, a.cy - y * a.s];
  }

  function drawArena(a, tick) {
    const s = a.s;
    ctx.fillStyle = css("--bg-arena");
    ctx.fillRect(a.x, a.y, a.w, a.h);
    ctx.lineWidth = 1;
    ctx.strokeStyle = css("--grid");
    ctx.beginPath();
    const gxMax = Math.ceil(a.halfW / 100) * 100;
    const gyMax = Math.ceil(a.halfH / 100) * 100;
    for (let g = -gxMax; g <= gxMax; g += 100) {
      const [gx] = toPx(a, g, 0);
      ctx.moveTo(gx, a.y);
      ctx.lineTo(gx, a.y + a.h);
    }
    for (let g = -gyMax; g <= gyMax; g += 100) {
      const [, gy] = toPx(a, 0, g);
      ctx.moveTo(a.x, gy);
      ctx.lineTo(a.x + a.w, gy);
    }
    ctx.stroke();
    ctx.strokeStyle = css("--edge");
    const [ex, ey] = toPx(a, -ARENA_MM, ARENA_MM);
    ctx.strokeRect(ex, ey, 2 * ARENA_MM * s, 2 * ARENA_MM * s);
    ctx.setLineDash([4, 6]);
    const [bx, by] = toPx(a, -TARGET_BOX_MM, TARGET_BOX_MM);
    ctx.strokeRect(bx, by, 2 * TARGET_BOX_MM * s, 2 * TARGET_BOX_MM * s);
    ctx.setLineDash([]);
    if (tick) drawWind(a, tick.wind);
  }

  function drawWind(a, wind) {
    const wx = wind[0];
    const wy = wind[1];
    const speed = Math.hypot(wx, wy);
    if (speed < 0.5) return;
    const lenMm = Math.min(90, 12 + speed * 0.35);
    const ux = wx / speed;
    const uy = wy / speed;
    ctx.strokeStyle = css("--wind");
    ctx.globalAlpha = Math.min(0.8, 0.25 + speed / 300);
    ctx.lineWidth = 1;
    ctx.beginPath();
    const wxMax = Math.floor(a.halfW / 200) * 200;
    const wyMax = Math.floor(a.halfH / 200) * 200;
    for (let gx = -wxMax; gx <= wxMax; gx += 200) {
      for (let gy = -wyMax; gy <= wyMax; gy += 200) {
        const [x0, y0] = toPx(a, gx - ux * lenMm / 2, gy - uy * lenMm / 2);
        const [x1, y1] = toPx(a, gx + ux * lenMm / 2, gy + uy * lenMm / 2);
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        const ang = Math.atan2(y1 - y0, x1 - x0);
        ctx.moveTo(x1, y1);
        ctx.lineTo(x1 - 6 * Math.cos(ang - 0.45), y1 - 6 * Math.sin(ang - 0.45));
        ctx.moveTo(x1, y1);
        ctx.lineTo(x1 - 6 * Math.cos(ang + 0.45), y1 - 6 * Math.sin(ang + 0.45));
      }
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  function drawTargets(a, run, t) {
    const s = a.s;
    run.targets.forEach(function (target, i) {
      if (!target || target.appearT > t) return;
      const [x, y] = toPx(a, target.x, target.y);
      const hit = target.hitT !== null && target.hitT <= t;
      if (hit) {
        ctx.strokeStyle = css("--target-hit");
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(x - 5, y - 5); ctx.lineTo(x + 5, y + 5);
        ctx.moveTo(x + 5, y - 5); ctx.lineTo(x - 5, y + 5);
        ctx.stroke();
        return;
      }
      const r = Math.max(4, HIT_MM * s);
      const pulse = 0.5 + 0.5 * Math.sin(t * 6);
      ctx.strokeStyle = css("--target");
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.globalAlpha = 0.25 + 0.35 * pulse;
      ctx.beginPath();
      ctx.arc(x, y, r + 6 + 4 * pulse, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillStyle = css("--target");
      ctx.font = "11px " + css("--font");
      ctx.fillText(String(i + 1), x + r + 6, y - r - 4);
    });
  }

  function drawFly(a, entry, idx) {
    const run = entry.run;
    const ticks = run.ticks;
    const back = Math.round(TRAIL_S / run.tickS);
    const start = Math.max(0, idx - back);
    ctx.strokeStyle = entry.color;
    ctx.lineWidth = 2;
    for (let i = start + 1; i <= idx; i++) {
      const p0 = arenaPx(a, ticks[i - 1].state);
      const p1 = arenaPx(a, ticks[i].state);
      ctx.globalAlpha = 0.08 + 0.7 * ((i - start) / (idx - start + 1));
      ctx.beginPath();
      ctx.moveTo(p0[0], p0[1]);
      ctx.lineTo(p1[0], p1[1]);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    const st = ticks[idx].state;
    const [x, y] = arenaPx(a, st);
    const size = Math.max(7, Math.min(a.w, a.h) / 90);
    const end = P.timeline([run], state.t).ends[0];
    const clamped = P.clampToArena(st.x, st.y, ARENA_MM);
    if ((end.ended && end.oob) || clamped.outside) {
      drawOutOfBounds(a, entry, x, y);
      return;
    }
    const c = Math.cos(-st.psi);
    const s = Math.sin(-st.psi);
    const pts = [[1.4, 0], [-0.8, 0.7], [-0.4, 0], [-0.8, -0.7]].map(function (p) {
      return [x + size * (p[0] * c - p[1] * s), y + size * (p[0] * s + p[1] * c)];
    });
    ctx.fillStyle = entry.color;
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (const p of pts.slice(1)) ctx.lineTo(p[0], p[1]);
    ctx.closePath();
    ctx.fill();
  }

  function arenaPx(a, st) {
    const c = P.clampToArena(st.x, st.y, ARENA_MM);
    return toPx(a, c.x, c.y);
  }

  function drawOutOfBounds(a, entry, x, y) {
    const r = Math.max(6, Math.min(a.w, a.h) / 110);
    ctx.strokeStyle = entry.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.moveTo(x - r * 0.6, y - r * 0.6);
    ctx.lineTo(x + r * 0.6, y + r * 0.6);
    ctx.moveTo(x + r * 0.6, y - r * 0.6);
    ctx.lineTo(x - r * 0.6, y + r * 0.6);
    ctx.stroke();
    ctx.font = "11px " + css("--font");
    ctx.fillStyle = entry.color;
    const cx = a.cx;
    const cy = a.cy;
    ctx.textAlign = x > cx ? "right" : "left";
    ctx.fillText("out of bounds", x + (x > cx ? -1 : 1) * (r + 6), y + (y > cy ? -r - 6 : r + 14));
    ctx.textAlign = "left";
  }

  function drawArenaHud(a, entry, tick, idx, withLabel) {
    const run = entry.run;
    const pad = 10;
    ctx.font = "12px " + css("--font");
    if (withLabel) {
      ctx.fillStyle = entry.color;
      ctx.fillText(runLabel(entry), a.x + pad, a.y + pad + 12);
      a = { ...a, y: a.y + 16 };
    }
    ctx.fillStyle = css("--text");
    const wind = tick.wind;
    const ws = Math.hypot(wind[0], wind[1]);
    const wdeg = (Math.atan2(wind[1], wind[0]) * 180) / Math.PI;
    const m = run.summary && run.summary.metrics;
    const count = m ? m.targets_hit + m.misses : Math.max(run.targets.length, 5);
    const tl = P.timeline(state.runs.map(function (e) { return e.run; }), state.t);
    const own = P.timeline([run], state.t).ends[0];
    const lines = [
      "t " + tl.t.toFixed(2) + " / " + tl.total.toFixed(2) + " s   " + state.speed + "x",
      "target " + Math.min(run.hitsBefore[idx] + 1, count) + " / " + count,
      "wind " + ws.toFixed(0) + " mm/s  " + (ws > 0.5 ? wdeg.toFixed(0) + " deg" : ""),
    ];
    if (own.ended) {
      lines.push("this run ended at " + own.duration.toFixed(2) + " s" + (own.oob ? ", out of bounds" : ""));
    }
    lines.forEach(function (line, i) {
      ctx.fillText(line, a.x + pad, a.y + pad + 12 + i * 16);
    });
  }

  function bar(p, y, label, value, lo, hi, centered) {
    const labelW = 86;
    const w = p.w - labelW - 70;
    const x = p.x + 16 + labelW;
    ctx.fillStyle = css("--muted");
    ctx.fillText(label, p.x + 16, y + 9);
    ctx.fillStyle = css("--line");
    ctx.fillRect(x, y, w, 10);
    const f = Math.max(0, Math.min(1, (value - lo) / (hi - lo)));
    ctx.fillStyle = css("--bar");
    if (centered) {
      const mid = x + w / 2;
      ctx.fillRect(Math.min(mid, x + f * w), y, Math.abs(x + f * w - mid), 10);
      ctx.fillStyle = css("--edge");
      ctx.fillRect(mid, y - 2, 1, 14);
    } else {
      ctx.fillRect(x, y, f * w, 10);
    }
    ctx.fillStyle = css("--text");
    ctx.fillText(value.toFixed(2), x + w + 8, y + 9);
  }

  function runningStats(run, t) {
    let hits = 0;
    let sum = 0;
    for (const target of run.targets) {
      if (target && target.hitT !== null && target.hitT <= t) {
        hits += 1;
        sum += target.hitT - target.appearT;
      }
    }
    return { hits: hits, mean: hits ? sum / hits : null };
  }

  function drawPanel(p, entry, idx) {
    const run = entry.run;
    const tick = run.ticks[idx];
    const h = run.header || {};
    ctx.fillStyle = css("--panel");
    ctx.fillRect(p.x, p.y, p.w, p.h);
    ctx.font = "12px " + css("--font");
    let y = p.y + 22;
    ctx.fillStyle = entry.color;
    ctx.fillText((h.controller || "?").toUpperCase(), p.x + 16, y);
    ctx.fillStyle = css("--muted");
    ctx.fillText((h.scenario || "?") + "  seed " + (h.seed !== undefined ? h.seed : "?"), p.x + 16, y + 16);
    y += 40;

    const obs = tick.obs || {};
    bar(p, y, "thrust", tick.action.thrust, 0, 1, false); y += 20;
    bar(p, y, "yaw torque", tick.action.yaw_torque, -1, 1, true); y += 20;
    if (typeof obs.bearing === "number") { bar(p, y, "bearing", obs.bearing, -Math.PI, Math.PI, true); y += 20; }
    if (typeof obs.range === "number") { bar(p, y, "range mm", obs.range, 0, RANGE_BAR_MM, false); y += 20; }
    y += 12;

    ctx.fillStyle = css("--muted");
    ctx.fillText("activity", p.x + 16, y);
    y += 10;
    if (!run.groups.length) {
      ctx.fillText("no info.groups in this run", p.x + 16, y + 14);
      y += 30;
    } else {
      y = drawActivity(p, run, idx, y);
    }

    y += 14;
    const stats = runningStats(run, state.t);
    ctx.fillStyle = css("--muted");
    ctx.fillText("running", p.x + 16, y);
    ctx.fillStyle = css("--text");
    ctx.fillText("hits " + stats.hits + "   time/target " + (stats.mean === null ? "-" : stats.mean.toFixed(2) + " s"), p.x + 16, y + 16);
    y += 36;
    const atEnd = idx === run.ticks.length - 1 && run.summary;
    if (atEnd) {
      const m = run.summary.metrics || {};
      const rows = [
        ["success", String(m.episode_success)],
        ["targets hit", String(m.targets_hit)],
        ["time/target s", m.time_per_target_s == null ? "-" : m.time_per_target_s.toFixed(3)],
        ["rms cross mm", m.rms_cross_track_mm == null ? "-" : m.rms_cross_track_mm.toFixed(1)],
        ["path eff", m.path_efficiency == null ? "-" : m.path_efficiency.toFixed(3)],
      ];
      ctx.fillStyle = css("--muted");
      ctx.fillText("summary", p.x + 16, y);
      rows.forEach(function (r, i) {
        ctx.fillStyle = css("--muted");
        ctx.fillText(r[0], p.x + 16, y + 16 + i * 15);
        ctx.fillStyle = css("--text");
        ctx.fillText(r[1], p.x + 140, y + 16 + i * 15);
      });
    }
  }

  function drawActivity(p, run, idx, y) {
    const tick = run.ticks[idx];
    const groups = (tick.info && tick.info.groups) || {};
    const labelW = 86;
    const x = p.x + 16 + labelW;
    const w = p.w - labelW - 70;
    const barW = w * 0.35;
    const stripX = x + barW + 8;
    const stripW = w - barW - 8;
    const cell = stripW / STRIP_TICKS;
    const start = Math.max(0, idx - STRIP_TICKS + 1);
    for (const name of run.groups) {
      const max = run.groupMax[name] || 1;
      const v = groups[name] || 0;
      ctx.fillStyle = css("--muted");
      ctx.fillText(name, p.x + 16, y + 9);
      ctx.fillStyle = css("--line");
      ctx.fillRect(x, y, barW, 10);
      ctx.fillStyle = css("--activity");
      ctx.fillRect(x, y, barW * Math.min(1, v / max), 10);
      for (let i = start; i <= idx; i++) {
        const g = run.ticks[i].info && run.ticks[i].info.groups;
        const gv = g ? g[name] || 0 : 0;
        if (!gv) continue;
        ctx.globalAlpha = Math.min(1, 0.15 + gv / max);
        ctx.fillRect(stripX + (i - start) * cell, y, Math.max(1, cell), 10);
      }
      ctx.globalAlpha = 1;
      ctx.fillStyle = css("--text");
      ctx.fillText(String(v), x + w + 8, y + 9);
      y += 16;
    }
    return y;
  }

  function draw() {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = css("--bg");
    ctx.fillRect(0, 0, w, h);
    const L = layout(w, h);
    const primary = state.runs[state.primary];
    const pIdx = primary ? P.indexAt(primary.run, state.t) : -1;
    const side = state.compare === "side" && state.runs.length > 1;
    const views = P.tileRects(L.arena, state.runs.length, side ? "side" : "overlay")
      .map(function (r) { return P.arenaView(r, WORLD_MM); });
    if (!primary || pIdx < 0) {
      drawArena(views[0], null);
      ctx.fillStyle = css("--muted");
      ctx.font = "13px " + css("--font");
      ctx.fillText("Open a run file, or start a live run.", L.arena.x + 20, L.arena.y + 30);
      return;
    }
    if (side) {
      state.runs.slice(0, views.length).forEach(function (entry, i) {
        const v = views[i];
        const idx = P.indexAt(entry.run, state.t);
        ctx.save();
        ctx.beginPath();
        ctx.rect(v.x, v.y, v.w, v.h);
        ctx.clip();
        drawArena(v, entry.run.ticks[idx]);
        drawTargets(v, entry.run, state.t);
        drawFly(v, entry, idx);
        drawArenaHud(v, entry, entry.run.ticks[idx], idx, true);
        ctx.restore();
      });
    } else {
      const v = views[0];
      drawArena(v, primary.run.ticks[pIdx]);
      drawTargets(v, primary.run, state.t);
      state.runs.forEach(function (entry, i) {
        if (i === state.primary) return;
        const idx = P.indexAt(entry.run, state.t);
        if (idx >= 0) drawFly(v, entry, idx);
      });
      drawFly(v, primary, pIdx);
      drawArenaHud(v, primary, primary.run.ticks[pIdx], pIdx, false);
      drawLegendOnCanvas(v);
    }
    drawPanel(L.panel, primary, pIdx);
  }

  function drawLegendOnCanvas(a) {
    if (state.runs.length < 2) return;
    const tl = P.timeline(state.runs.map(function (e) { return e.run; }), state.t);
    ctx.font = "11px " + css("--font");
    state.runs.forEach(function (entry, i) {
      const y = a.y + a.h - 14 - (state.runs.length - 1 - i) * 15;
      const end = tl.ends[i];
      const note = end.duration.toFixed(2) + " s" + (end.ended ? (end.oob ? "  ended, out of bounds" : "  ended") : "");
      ctx.fillStyle = entry.color;
      ctx.fillRect(a.x + 10, y - 8, 9, 9);
      ctx.fillStyle = end.ended ? css("--muted") : css("--text");
      ctx.fillText(runLabel(entry) + "  " + note, a.x + 24, y);
    });
  }

  // live mode
  function setupLive(options) {
    const fill = function (select, items, chosen) {
      select.textContent = "";
      for (const item of items) {
        const opt = document.createElement("option");
        opt.value = item;
        opt.textContent = item;
        if (item === chosen) opt.selected = true;
        select.appendChild(opt);
      }
    };
    fill(ui.liveScenario, options.scenarios, "s3_gust");
    fill(ui.liveController, options.controllers, "pid");
    ui.live.hidden = false;
    ui.liveStart.onclick = startLive;
    ui.liveWind.oninput = function () { ui.liveWindVal.textContent = Number(ui.liveWind.value).toFixed(1); };
    ui.liveNoise.oninput = function () { ui.liveNoiseVal.textContent = Number(ui.liveNoise.value).toFixed(2); };
  }

  function startLive() {
    if (state.source) state.source.close();
    state.runs = [];
    state.live = null;
    const q = new URLSearchParams({
      scenario: ui.liveScenario.value,
      controller: ui.liveController.value,
      seed: String(Math.max(0, parseInt(ui.liveSeed.value, 10) || 0)),
      wind: ui.liveWind.value,
      noise: ui.liveNoise.value,
    });
    const run = P.emptyRun();
    run.truncated = true;
    let entry = null;
    const source = new EventSource("api/live?" + q.toString());
    state.source = source;
    state.t = 0;
    setPlaying(true);
    setStatus("live: running");
    source.onmessage = function (ev) {
      const res = P.parseLine(ev.data);
      if (!res.ok) return;
      P.addRecord(run, res.record);
      if (res.record.type === "header" && !entry) {
        entry = addRun(run, "live");
        state.live = entry;
      }
      if (res.record.type === "summary") {
        run.truncated = false;
        source.close();
        state.source = null;
        setStatus("live: done, run file written");
        renderLegend();
      }
    };
    source.onerror = function () {
      source.close();
      state.source = null;
      if (!run.summary) setStatus("live: stream closed early");
    };
  }

  // controls
  ui.play.onclick = function () {
    if (!state.playing && state.t >= maxDuration()) state.t = 0;
    setPlaying(!state.playing);
  };
  ui.restart.onclick = function () { state.t = 0; setPlaying(true); };
  ui.scrub.oninput = function () {
    state.t = (ui.scrub.value / 1000) * maxDuration();
    state.last = null;
  };
  ui.speed.onchange = function () { state.speed = parseFloat(ui.speed.value); };
  function setCompare(mode) {
    state.compare = mode;
    ui.compare.textContent = mode === "side" ? "Overlay" : "Side by side";
  }
  ui.compare.onclick = function () { setCompare(state.compare === "side" ? "overlay" : "side"); };
  ui.png.onclick = function () {
    const link = document.createElement("a");
    const primary = state.runs[state.primary];
    link.download = (primary ? runLabel(primary).replace(/ /g, "_") : "frame") + "_t" + state.t.toFixed(2) + ".png";
    link.href = canvas.toDataURL("image/png");
    link.click();
  };
  ui.file.onchange = function () {
    for (const file of Array.from(ui.file.files).slice(0, MAX_RUNS)) {
      file.text().then(function (text) { loadText(text, file.name); });
    }
    ui.file.value = "";
    state.t = 0;
  };
  document.addEventListener("keydown", function (ev) {
    if (ev.code === "Space" && ev.target.tagName !== "INPUT" && ev.target.tagName !== "SELECT") {
      ev.preventDefault();
      ui.play.click();
    }
  });

  // start
  const params = new URLSearchParams(location.search);
  const urls = params.getAll("run");
  const served = location.protocol.startsWith("http");
  const options = served
    ? fetch("api/options").then(function (r) { return r.ok ? r.json() : null; }).catch(function () { return null; })
    : Promise.resolve(null);
  options.then(function (opts) {
    if (opts) setupLive(opts);
    const list = urls.length ? urls : opts ? opts.samples : [];
    return loadUrls(list.slice(0, MAX_RUNS));
  }).then(function () {
    state.primary = 0;
    state.t = Math.max(0, parseFloat(params.get("t")) || 0);
    if (params.get("compare") === "side") setCompare("side");
    renderLegend();
  });
  requestAnimationFrame(tickFrame);
})();
