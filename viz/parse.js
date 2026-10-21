// Run file parser: pure functions, no DOM
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  } else {
    root.IcarusParse = api;
  }
})(typeof self !== "undefined" ? self : this, function () {
  const KINDS = ["header", "tick", "summary"];

  function isNum(v) {
    return typeof v === "number" && Number.isFinite(v);
  }

  function checkRecord(r) {
    if (r === null || typeof r !== "object" || Array.isArray(r)) return "not an object";
    if (!KINDS.includes(r.type)) return "unknown type " + JSON.stringify(r.type);
    if (r.type === "tick") {
      if (!Number.isInteger(r.k)) return "tick without integer k";
      if (!isNum(r.t)) return "tick without t";
      if (!r.state || !isNum(r.state.x) || !isNum(r.state.y) || !isNum(r.state.psi)) {
        return "tick without state x, y, psi";
      }
      if (!r.action || !isNum(r.action.thrust) || !isNum(r.action.yaw_torque)) {
        return "tick without action";
      }
    }
    if (r.type === "header" && typeof r.controller !== "string") return "header without controller";
    return null;
  }

  function parseLine(line) {
    const text = line.trim();
    if (!text) return { ok: false, empty: true, error: "empty line" };
    let record;
    try {
      record = JSON.parse(text);
    } catch (e) {
      return { ok: false, error: "bad JSON" };
    }
    const problem = checkRecord(record);
    if (problem) return { ok: false, error: problem };
    return { ok: true, record: record };
  }

  function emptyRun() {
    return {
      header: null, ticks: [], summary: null, errors: [], truncated: false,
      groups: [], groupMax: {}, targets: [], hitsBefore: [], hits: 0, tickS: 0.005,
    };
  }

  function appendTick(run, t) {
    run.ticks.push(t);
    run.hitsBefore.push(run.hits);
    if (t.target && !run.targets[t.target.i]) {
      run.targets[t.target.i] = { x: t.target.x, y: t.target.y, appearT: t.t, hitT: null };
    }
    for (const e of t.events || []) {
      if (e !== "hit") continue;
      if (run.targets[run.hits]) run.targets[run.hits].hitT = t.t + run.tickS;
      run.hits += 1;
    }
    const groups = t.info && t.info.groups;
    if (groups && typeof groups === "object") {
      for (const name of Object.keys(groups)) {
        if (!run.groups.includes(name)) {
          run.groups.push(name);
          run.groupMax[name] = 0;
        }
        const v = groups[name];
        if (isNum(v) && v > run.groupMax[name]) run.groupMax[name] = v;
      }
    }
  }

  function addRecord(run, record) {
    if (record.type === "header") {
      run.header = record;
      if (isNum(record.tick_s)) run.tickS = record.tick_s;
    } else if (record.type === "tick") appendTick(run, record);
    else run.summary = record;
  }

  function parseRun(text) {
    const run = emptyRun();
    const lines = text.replace(/\r\n/g, "\n").split("\n");
    const endsClean = text.length === 0 || /\n$/.test(text);
    const records = [];
    lines.forEach(function (line, i) {
      const res = parseLine(line);
      if (res.ok) records.push(res.record);
      else if (!res.empty) {
        const last = i === lines.length - 1 && !endsClean;
        if (last) run.truncated = true;
        else run.errors.push({ line: i + 1, error: res.error });
      }
    });
    const header = records.find(function (r) { return r.type === "header"; });
    if (header) addRecord(run, header);
    for (const r of records) if (r.type !== "header") addRecord(run, r);
    if (!run.summary) run.truncated = true;
    return run;
  }

  function groupNames(ticks) {
    const names = [];
    for (const t of ticks) {
      const groups = t.info && t.info.groups;
      if (!groups || typeof groups !== "object") continue;
      for (const name of Object.keys(groups)) if (!names.includes(name)) names.push(name);
    }
    return names;
  }

  function lineBuffer() {
    let rest = "";
    return {
      push: function (chunk) {
        const text = rest + chunk;
        const lines = text.split("\n");
        rest = lines.pop();
        return lines.map(parseLine).filter(function (r) { return !r.empty; });
      },
      end: function () {
        const out = rest.trim() ? [parseLine(rest)] : [];
        rest = "";
        return out;
      },
    };
  }

  function indexAt(run, t) {
    const ticks = run.ticks;
    if (!ticks.length) return -1;
    let lo = 0;
    let hi = ticks.length - 1;
    if (t <= ticks[0].t) return 0;
    if (t >= ticks[hi].t) return hi;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (ticks[mid].t <= t) lo = mid;
      else hi = mid - 1;
    }
    return lo;
  }

  function duration(run) {
    if (!run.ticks.length) return 0;
    return run.ticks[run.ticks.length - 1].t + run.tickS;
  }

  return {
    parseLine: parseLine,
    parseRun: parseRun,
    emptyRun: emptyRun,
    addRecord: addRecord,
    groupNames: groupNames,
    lineBuffer: lineBuffer,
    indexAt: indexAt,
    duration: duration,
  };
});
