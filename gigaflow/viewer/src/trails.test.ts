import { describe, expect, it } from "vitest";
import {
  TRAIL_MAX_POINTS,
  TRAIL_SECONDS,
  CarTrail,
  TrailField,
} from "./trails";

describe("CarTrail", () => {
  it("retains about TRAIL_SECONDS of sim-time history and fades", () => {
    const trail = new CarTrail();
    for (let i = 0; i <= 40; i += 1) {
      const t = i * 0.1;
      trail.push(t, t * 2, 0, true);
    }
    const pts = trail.snapshot();
    expect(pts[0].t).toBeGreaterThanOrEqual(4.0 - TRAIL_SECONDS - 1e-9);
    expect(pts[pts.length - 1].t).toBeCloseTo(4.0, 9);
    expect(trail.fadeAt(0, 4.0)).toBeLessThan(0.15);
    expect(trail.fadeAt(pts.length - 1, 4.0)).toBeCloseTo(1, 5);
  });

  it("clears on inactive and respawn jump", () => {
    const trail = new CarTrail();
    trail.push(0.1, 0, 0, true);
    trail.push(0.2, 0.2, 0, true);
    expect(trail.length).toBe(2);
    trail.push(0.3, 0.3, 0, false);
    expect(trail.length).toBe(0);
    trail.push(0.4, 0, 0, true);
    trail.push(0.5, 5, 5, true); // snap
    expect(trail.length).toBe(1);
    expect(trail.snapshot()[0].x).toBeCloseTo(5, 5);
  });

  it("bounds memory to TRAIL_MAX_POINTS", () => {
    const trail = new CarTrail();
    for (let i = 0; i < TRAIL_MAX_POINTS + 50; i += 1) {
      trail.push(i * 0.01, i * 0.01, 0, true);
    }
    expect(trail.length).toBeLessThanOrEqual(TRAIL_MAX_POINTS);
  });
});

describe("TrailField", () => {
  it("clears all on reset and drops inactive cars independently", () => {
    const field = new TrailField();
    field.pushAll(0.1, [
      [0, 0, 0, 1],
      [1, 0, 0, 1],
    ]);
    field.pushAll(0.2, [
      [0.2, 0, 0, 1],
      [1.2, 0, 0, 0],
    ]);
    expect(field.get(0).length).toBe(2);
    expect(field.get(1).length).toBe(0);
    field.clearAll();
    expect(field.get(0).length).toBe(0);
    expect(field.get(1).length).toBe(0);
  });

  it("supports dense 8-car updates without exceeding caps", () => {
    const field = new TrailField();
    for (let step = 0; step < 80; step += 1) {
      const t = step * 0.05;
      const cars = [];
      for (let c = 0; c < 8; c += 1) {
        cars.push([c + t, 0, 0, 1]);
      }
      field.pushAll(t, cars);
    }
    for (let c = 0; c < 8; c += 1) {
      expect(field.get(c).length).toBeGreaterThan(2);
      expect(field.get(c).length).toBeLessThanOrEqual(TRAIL_MAX_POINTS);
    }
  });
});
