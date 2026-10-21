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
