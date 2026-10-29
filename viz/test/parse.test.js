const test = require("node:test");
const assert = require("node:assert");
const P = require("../parse.js");

const header = {
  type: "header", schema: 1, scenario: "s3_gust", scenario_hash: "abc", controller: "pid",
  seed: 7, mode: "xy", physics_hash: "def", dt_s: 0.001, tick_s: 0.005, commit: "x",
};

function tick(k, opts) {
  const o = opts || {};
  return {
    type: "tick", k: k, t: k * 0.005,
    state: { x: k, y: 0, z: 0, psi: 0.1, vx: 0, vy: 0, vz: 0, omega: 0 },
    wind: [0, 0, 0],
    target: { i: o.i || 0, x: 100 + 100 * (o.i || 0), y: 0 },
    obs: {}, action: { thrust: 0.5, yaw_torque: 0.1 },
    info: o.info || {}, events: o.events || [],
  };
}

function lines(records) {
  return records.map(function (r) { return JSON.stringify(r); }).join("\n") + "\n";
}

const summary = { type: "summary", metrics: { targets_hit: 1 } };

test("parses header, ticks and summary", function () {
  const run = P.parseRun(lines([header, tick(0), tick(1, { events: ["hit"] }),
    tick(2, { i: 1 }), summary]));
  assert.strictEqual(run.header.controller, "pid");
  assert.strictEqual(run.ticks.length, 3);
  assert.deepStrictEqual(run.summary, summary);
  assert.strictEqual(run.truncated, false);
  assert.deepStrictEqual(run.errors, []);
  assert.strictEqual(run.tickS, 0.005);
  assert.strictEqual(run.targets.length, 2);
  assert.strictEqual(run.targets[0].hitT, 0.005 * 2);
  assert.strictEqual(run.targets[1].hitT, null);
  assert.deepStrictEqual(run.hitsBefore, [0, 0, 1]);
});

test("parseLine accepts a tick and rejects bad lines", function () {
  assert.strictEqual(P.parseLine(JSON.stringify(tick(3))).ok, true);
  assert.strictEqual(P.parseLine("{not json").ok, false);
  assert.strictEqual(P.parseLine("[1, 2]").ok, false);
  assert.strictEqual(P.parseLine(JSON.stringify({ type: "other" })).ok, false);
  const noState = tick(1);
  delete noState.state;
  assert.strictEqual(P.parseLine(JSON.stringify(noState)).ok, false);
  assert.strictEqual(P.parseLine("   ").empty, true);
});

test("bad lines are reported and skipped", function () {
  const text = lines([header, tick(0)]) + "garbage\n" + lines([tick(1), summary]);
  const run = P.parseRun(text);
  assert.strictEqual(run.ticks.length, 2);
  assert.strictEqual(run.errors.length, 1);
  assert.strictEqual(run.errors[0].line, 3);
  assert.strictEqual(run.truncated, false);
});

test("truncated file keeps complete ticks", function () {
  const full = lines([header, tick(0), tick(1), tick(2)]);
  const cut = full.slice(0, full.length - 20);
  const run = P.parseRun(cut);
  assert.strictEqual(run.truncated, true);
  assert.strictEqual(run.ticks.length, 2);
  assert.deepStrictEqual(run.errors, []);
  assert.strictEqual(run.summary, null);
});

test("windows line endings parse", function () {
  const run = P.parseRun(lines([header, tick(0), summary]).replace(/\n/g, "\r\n"));
  assert.strictEqual(run.ticks.length, 1);
  assert.strictEqual(run.truncated, false);
});

test("run without info.groups has no groups", function () {
  const run = P.parseRun(lines([header, tick(0, { info: { e_bearing: 0.1 } }), summary]));
  assert.deepStrictEqual(run.groups, []);
  assert.deepStrictEqual(run.groupMax, {});
});

test("run with info.groups collects names and max", function () {
  const run = P.parseRun(lines([
    header,
    tick(0, { info: { groups: { vis_L: 3, out_L: 1 }, spikes_total: 40 } }),
    tick(1, { info: { groups: { vis_L: 7, out_R: 2 }, spikes_total: 50 } }),
    summary,
  ]));
  assert.deepStrictEqual(run.groups, ["vis_L", "out_L", "out_R"]);
  assert.strictEqual(run.groupMax.vis_L, 7);
  assert.strictEqual(run.groupMax.out_R, 2);
});

test("line buffer handles split chunks", function () {
  const buf = P.lineBuffer();
  const text = lines([header, tick(0)]);
  const a = buf.push(text.slice(0, 50));
  const b = buf.push(text.slice(50));
  assert.strictEqual(a.length + b.length, 2);
  assert.ok(a.concat(b).every(function (r) { return r.ok; }));
  assert.deepStrictEqual(buf.end(), []);
});

test("indexAt finds the tick at a time", function () {
  const run = P.parseRun(lines([header, tick(0), tick(1), tick(2), summary]));
  assert.strictEqual(P.indexAt(run, 0), 0);
  assert.strictEqual(P.indexAt(run, 0.007), 1);
  assert.strictEqual(P.indexAt(run, 99), 2);
  assert.strictEqual(P.indexAt(P.emptyRun(), 1), -1);
  assert.ok(Math.abs(P.duration(run) - 0.015) < 1e-12);
});

test("header after ticks still sets tick length", function () {
  const run = P.parseRun(lines([tick(0), { ...header, tick_s: 0.01 }, summary]));
  assert.strictEqual(run.tickS, 0.01);
  assert.strictEqual(run.header.seed, 7);
});

test("timeline uses the longest run and marks ended runs", function () {
  const short = P.parseRun(lines([header, tick(0), tick(1), summary]));
  const long = P.parseRun(lines([header, tick(0), tick(1), tick(2), tick(3), summary]));
  const tl = P.timeline([short, long], 0.012);
  assert.ok(Math.abs(tl.total - 0.02) < 1e-12);
  assert.strictEqual(tl.ends[0].ended, true);
  assert.strictEqual(tl.ends[1].ended, false);
  assert.ok(Math.abs(tl.ends[0].duration - 0.01) < 1e-12);
  assert.strictEqual(P.timeline([short, long], 99).t, tl.total);
  assert.strictEqual(P.timeline([], 1).total, 0);
});

test("out of bounds end is detected", function () {
  const oob = P.parseRun(lines([header, tick(0), tick(1, { events: ["oob", "done"] }), summary]));
  const ok = P.parseRun(lines([header, tick(0), summary]));
  assert.strictEqual(P.endedOutOfBounds(oob), true);
  assert.strictEqual(P.endedOutOfBounds(ok), false);
  assert.strictEqual(P.timeline([oob], 1).ends[0].oob, true);
});

test("clampToArena keeps points on the frame", function () {
  assert.deepStrictEqual(P.clampToArena(100, -50, 800), { x: 100, y: -50, outside: false });
  assert.deepStrictEqual(P.clampToArena(950, -900, 800), { x: 800, y: -800, outside: true });
  assert.strictEqual(P.clampToArena(800, 0, 800).outside, false);
});

test("tileRects overlay is one tile", function () {
  const region = { x: 0, y: 0, w: 1000, h: 800 };
  assert.deepStrictEqual(P.tileRects(region, 3, "overlay"), [region]);
  assert.deepStrictEqual(P.tileRects(region, 1, "side"), [region]);
});

test("tileRects splits two runs along the long side", function () {
  const wide = P.tileRects({ x: 0, y: 0, w: 1006, h: 500 }, 2, "side");
  assert.deepStrictEqual(wide, [{ x: 0, y: 0, w: 500, h: 500 }, { x: 506, y: 0, w: 500, h: 500 }]);
  const tall = P.tileRects({ x: 0, y: 0, w: 400, h: 806 }, 2, "side");
  assert.strictEqual(tall[1].y, 406);
  assert.strictEqual(tall[0].w, 400);
});

test("tileRects puts three and four runs in a 2 by 2 grid", function () {
  const region = { x: 10, y: 20, w: 806, h: 606 };
  const four = P.tileRects(region, 4, "side");
  assert.strictEqual(four.length, 4);
  assert.deepStrictEqual(four[3], { x: 416, y: 326, w: 400, h: 300 });
  assert.strictEqual(P.tileRects(region, 3, "side").length, 3);
  assert.strictEqual(P.tileRects(region, 9, "side").length, 4);
  for (const t of four) {
    assert.ok(t.x >= region.x && t.y >= region.y);
    assert.ok(t.x + t.w <= region.x + region.w + 1e-9 && t.y + t.h <= region.y + region.h + 1e-9);
  }
});

test("arenaView fits the world on the short side", function () {
  const v = P.arenaView({ x: 100, y: 0, w: 1000, h: 820 }, 820);
  assert.strictEqual(v.s, 0.5);
  assert.strictEqual(v.cx, 600);
  assert.strictEqual(v.cy, 410);
  assert.strictEqual(v.halfH, 820);
  assert.strictEqual(v.halfW, 1000);
});
