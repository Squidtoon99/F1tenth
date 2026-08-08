import { describe, expect, it } from "vitest";
import { advanceFollowTarget } from "./camera";

describe("follow camera translation", () => {
  it("moves target and camera by the same delta, preserving user offset", () => {
    const camera = [8, 4, 5] as const;
    const current = [2, 0, 1] as const;
    const { target, delta } = advanceFollowTarget(
      [...current],
      [6, 0, 3],
      0.25,
    );
    const movedCamera = camera.map((value, i) => value + delta[i]);
    const beforeOffset = camera.map((value, i) => value - current[i]);
    const afterOffset = movedCamera.map((value, i) => value - target[i]);
    expect(afterOffset).toEqual(beforeOffset);
    expect(target).toEqual([3, 0, 1.5]);
  });
});
