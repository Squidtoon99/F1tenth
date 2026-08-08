import { describe, expect, it } from "vitest";
import {
  ANALYSIS_MAX_SAMPLES,
  TelemetryHistory,
  chartPath,
  currentControls,
} from "./analysis";
import type { CarPose } from "./protocol";

function car(
  speed: number,
  steer: number,
  effort: number,
  active = 1,
  collision = 0,
): CarPose {
  return [0, 0, 0, active, speed, 0, 0, 0, 0, effort, steer, collision];
}

describe("TelemetryHistory", () => {
  it("derives throttle and brake from raw longitudinal action", () => {
    const history = new TelemetryHistory();
    history.push(0, car(2, 0.1, 0.7));
    history.push(0.1, car(3, -0.2, -0.4));
    expect(history.snapshot()).toEqual([
      { t: 0, speed: 2, steering: 0.1, throttle: 0.7, brake: 0, collisions: 0 },
      { t: 0.1, speed: 3, steering: -0.2, throttle: 0, brake: 0.4, collisions: 0 },
    ]);
    expect(history.topSpeed).toBe(3);
  });

  it("counts debounced collision rising edges", () => {
    const history = new TelemetryHistory();
    history.push(0, car(1, 0, 0, 1, 1));
    history.push(0.1, car(1, 0, 0, 1, 1));
    history.push(0.2, car(1, 0, 0, 1, 0));
    history.push(0.3, car(1, 0, 0, 1, 1));
    expect(history.collisionCount).toBe(1);
    history.push(0.4, car(1, 0, 0, 1, 0));
    history.push(1.0, car(1, 0, 0, 1, 0));
    history.push(1.1, car(1, 0, 0, 1, 1));
    expect(history.collisionCount).toBe(2);
    history.reset();
    expect(history.collisionCount).toBe(0);
  });

  it("ignores absent agents and keeps bounded history", () => {
    const history = new TelemetryHistory();
    history.push(0, undefined);
    history.push(0, car(1, 0, 0, 0));
    for (let i = 0; i < ANALYSIS_MAX_SAMPLES + 20; i += 1) {
      history.push(i * 0.01, car(i, 0, 0));
    }
    expect(history.snapshot()).toHaveLength(ANALYSIS_MAX_SAMPLES);
  });

  it("uses the supplied fixed chart range instead of observed extrema", () => {
    const history = new TelemetryHistory();
    history.push(0, car(1, 0, 0.2));
    history.push(1, car(3, 0, 0.8));
    expect(chartPath(history.snapshot(), "speed", 100, 40, [0, 4])).toBe(
      "M 0.0 30.0 C 33.3 30.0 66.7 10.0 100.0 10.0",
    );
  });

  it("maps normalized steering against the semantic -1 to 1 range", () => {
    const history = new TelemetryHistory();
    history.push(0, car(1, -0.5, 0));
    history.push(1, car(1, 0.5, 0));
    expect(chartPath(history.snapshot(), "steering", 100, 40, [-1, 1])).toBe(
      "M 0.0 30.0 C 33.3 30.0 66.7 10.0 100.0 10.0",
    );
  });

  it("derives bounded live values for current control gauges", () => {
    const pose = car(2, 0.4, 1.4);
    pose[10] = -1.2;
    expect(currentControls(pose)).toEqual({
      steering: -1,
      throttle: 1,
      brake: 0,
    });
    expect(currentControls(car(2, 0, -0.45))).toEqual({
      steering: 0,
      throttle: 0,
      brake: 0.45,
    });
    expect(currentControls(car(2, 0, 0, 0))).toBeNull();
  });
});
