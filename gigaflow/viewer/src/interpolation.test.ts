import { describe, expect, it } from "vitest";
import {
  REPLAY_DELAY_SECONDS,
  TickInterpolator,
  interpolateCar,
  shortestAngleDelta,
} from "./interpolation";
import { PROTOCOL_VERSION, type CarPose, type TickMessage } from "./protocol";

function pose(
  x: number,
  y: number,
  yaw: number,
  active = 1,
  vx = 0,
  vy = 0,
  yawRate = 0,
): CarPose {
  const speed = Math.hypot(vx, vy);
  return [x, y, yaw, active, speed, vx, vy, yawRate, 0, 0, 0, 0];
}

function tick(
  step: number,
  t: number,
  cars: CarPose[],
  extras: Partial<TickMessage> = {},
): TickMessage {
  return {
    v: PROTOCOL_VERSION,
    type: "tick",
    step,
    t,
    control_dt: 0.1,
    sim_fps: 10,
    paused: false,
    cars,
    ...extras,
  };
}

describe("linear pose interpolation", () => {
  it("wraps yaw across ±π without using velocity tangents", () => {
    const a = pose(0, 0, Math.PI - 0.1, 1, 0, 0, 1);
    const b = pose(0, 0, -Math.PI + 0.1, 1, 0, 0, 1);
    const mid = interpolateCar(a, b, 0.5);
    const err = Math.abs(shortestAngleDelta(Math.PI, mid[2]));
    expect(err).toBeLessThan(0.2);
    expect(Math.abs(shortestAngleDelta(a[2], mid[2]))).toBeLessThan(
      Math.abs(shortestAngleDelta(a[2], b[2])) + 1e-9,
    );
  });
});

describe("TickInterpolator", () => {
  it("uses a fixed three-second authoritative replay delay", () => {
    expect(REPLAY_DELAY_SECONDS).toBe(3);
    const buf = new TickInterpolator();
    buf.push(tick(1, 0.1, [pose(0, 0, 0)]), 0);
    expect(buf.delaySeconds).toBeCloseTo(3, 9);
  });

  it("samples midpoint linearly between known endpoints", () => {
    const buf = new TickInterpolator();
    const a = pose(0, 0, 0, 1, 10, 0, 0);
    const b = pose(1, 0, 0, 1, 10, 0, 0);
    buf.push(tick(1, 0.1, [a]), 1000);
    buf.push(tick(2, 0.2, [b]), 1100);
    const cars = buf.sampleAt(0.15);
    expect(cars).not.toBeNull();
    expect(cars![0][0]).toBeCloseTo(0.5, 5);
    expect(cars![0][1]).toBeCloseTo(0, 5);
  });

  it("drains only authoritative samples for trails", () => {
    const buf = new TickInterpolator();
    buf.push(tick(1, 0.1, [pose(1, 0, 0)]), 0);
    buf.push(tick(2, 0.2, [pose(2, 0, 0)]), 100);
    expect(buf.drainKnownThrough(0.15).map((sample) => sample.t)).toEqual([0.1]);
    expect(buf.drainKnownThrough(0.19)).toEqual([]);
    expect(buf.drainKnownThrough(0.2).map((sample) => sample.t)).toEqual([0.2]);
  });

  it("holds at newest pose — never extrapolates past known ticks", () => {
    const buf = new TickInterpolator();
    const car = pose(0, 0, 0, 1, 10, 0, 0);
    buf.push(tick(5, 0.5, [car]), 0);
    const atNewest = buf.sampleAt(0.5)!;
    const beyond = buf.sampleAt(0.5 + 1.0)!;
    expect(atNewest[0][0]).toBeCloseTo(0, 5);
    expect(beyond[0][0]).toBeCloseTo(0, 5);
    expect(beyond[0][0]).toBeCloseTo(atNewest[0][0], 5);
  });

  it("snaps across inactive/respawn discontinuity", () => {
    const buf = new TickInterpolator();
    const dead = pose(0, 0, 0, 0, 0, 0, 0);
    const spawned = pose(5, 5, 1.2, 1, 2, 0, 0);
    buf.push(tick(1, 0.1, [dead]), 0);
    buf.push(tick(2, 0.2, [spawned]), 100);
    const before = buf.sampleAt(0.19)!;
    const after = buf.sampleAt(0.2)!;
    expect(before[0][3]).toBe(0);
    expect(before[0][0]).toBeCloseTo(0, 5);
    expect(after[0][3]).toBe(1);
    expect(after[0][0]).toBeCloseTo(5, 5);
  });

  it("resets buffer on scene rewind / reset", () => {
    const buf = new TickInterpolator();
    buf.push(tick(10, 1.0, [pose(3, 0, 0, 1, 1, 0, 0)]), 0);
    buf.push(tick(11, 1.1, [pose(4, 0, 0, 1, 1, 0, 0)]), 100);
    buf.reset();
    expect(buf.sampleAt(1.05)).toBeNull();
    buf.push(tick(0, 0.0, [pose(0, 0, 0, 1, 0, 0, 0)]), 200);
    const cars = buf.sampleAt(0.0)!;
    expect(cars[0][0]).toBeCloseTo(0, 5);
  });

  it("treats step rewind as a buffer snap", () => {
    const buf = new TickInterpolator();
    buf.push(tick(8, 0.8, [pose(8, 0, 0)]), 0);
    buf.push(tick(0, 0.0, [pose(0, 0, 0)]), 50);
    const cars = buf.sampleAt(0.0)!;
    expect(cars[0][0]).toBeCloseTo(0, 5);
  });

  it("holds playhead while paused (no unbounded wall-clock drift)", () => {
    const buf = new TickInterpolator();
    // Need span covering two-tick delay before pause sampling.
    buf.push(tick(1, 0.1, [pose(0, 0, 0, 1, 5, 0, 0)]), 800);
    buf.push(tick(2, 0.2, [pose(0.5, 0, 0, 1, 5, 0, 0)]), 900);
    buf.push(
      tick(3, 0.3, [pose(1, 2, 0.3, 1, 5, 0, 0)], { paused: true }),
      1000,
    );
    const a = buf.sample(1000)!;
    const b = buf.sample(5000)!;
    expect(a[0][0]).toBeCloseTo(b[0][0], 5);
    expect(a[0][1]).toBeCloseTo(b[0][1], 5);
  });

  it("replays continuously behind irregular authoritative arrivals", () => {
    const buf = new TickInterpolator();
    let wall = 1000;
    for (let i = 1; i <= 31; i += 1) {
      wall += 100;
      buf.push(tick(i, i * 0.1, [pose(i, 0, 0, 1, 10, 0, 0)]), wall);
    }

    const gapsMs = [70, 180, 110, 90, 160, 75, 140, 100, 170];
    const xs: number[] = [];
    const heads: number[] = [];
    for (let i = 0; i < gapsMs.length; i += 1) {
      wall += gapsMs[i];
      const step = 32 + i;
      const t = step * 0.1;
      buf.push(tick(step, t, [pose(step, 0, 0, 1, 10, 0, 0)]), wall);
      const cars = buf.sample(wall);
      const ph = buf.getPlayhead();
      expect(cars).not.toBeNull();
      expect(ph).not.toBeNull();
      xs.push(cars![0][0]);
      heads.push(ph!);
    }

    // Monotonic playhead (non-decreasing) and no snap-back.
    for (let i = 1; i < heads.length; i += 1) {
      expect(heads[i]).toBeGreaterThanOrEqual(heads[i - 1] - 1e-12);
    }
    for (let i = 1; i < xs.length; i += 1) {
      expect(xs[i]).toBeGreaterThanOrEqual(xs[i - 1] - 1e-6);
      expect(xs[i] - xs[i - 1]).toBeLessThanOrEqual(1.01);
    }
    expect(xs[xs.length - 1]).toBeLessThanOrEqual(40);
  });

  it("rebases on reset without bridging old history", () => {
    const buf = new TickInterpolator();
    buf.push(tick(5, 0.5, [pose(5, 0, 0, 1, 10, 0, 0)]), 0);
    buf.push(tick(6, 0.6, [pose(6, 0, 0, 1, 10, 0, 0)]), 100);
    buf.sample(300);
    buf.push(tick(0, 0.0, [pose(0, 0, 0, 1, 0, 0, 0)]), 400);
    buf.push(tick(1, 0.1, [pose(0, 0, 0, 1, 0, 0, 0)]), 500);
    buf.push(tick(2, 0.2, [pose(0, 0, 0, 1, 0, 0, 0)]), 600);
    const cars = buf.sample(700)!;
    expect(cars[0][0]).toBeCloseTo(0, 5);
    expect(buf.getPlayhead()!).toBeLessThanOrEqual(0.2 + 1e-9);
  });
});
