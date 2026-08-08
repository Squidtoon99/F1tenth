import { describe, expect, it } from "vitest";
import {
  PROTOCOL_VERSION,
  encodeClientMessage,
  parseServerMessage,
  worldToThree,
} from "./protocol";

describe("protocol", () => {
  it("encodes client commands with version", () => {
    expect(JSON.parse(encodeClientMessage("pause"))).toEqual({
      v: PROTOCOL_VERSION,
      type: "pause",
    });
    expect(
      JSON.parse(encodeClientMessage({ type: "set_suite", suite: "dense" })),
    ).toEqual({
      v: PROTOCOL_VERSION,
      type: "set_suite",
      suite: "dense",
    });
    expect(
      JSON.parse(
        encodeClientMessage({
          type: "set_environment_count",
          environment_count: 4,
        }),
      ),
    ).toEqual({
      v: PROTOCOL_VERSION,
      type: "set_environment_count",
      environment_count: 4,
    });
    expect(
      JSON.parse(
        encodeClientMessage({
          type: "set_obstacle_preset",
          obstacle_preset: "light",
        }),
      ),
    ).toEqual({
      v: PROTOCOL_VERSION,
      type: "set_obstacle_preset",
      obstacle_preset: "light",
    });
    expect(
      JSON.parse(
        encodeClientMessage({
          type: "set_checkpoint",
          checkpoint: "/runs/rtx4080/actor_004300.pt",
        }),
      ),
    ).toEqual({
      v: PROTOCOL_VERSION,
      type: "set_checkpoint",
      checkpoint: "/runs/rtx4080/actor_004300.pt",
    });
  });

  it("parses tick messages with velocity channels", () => {
    const msg = parseServerMessage(
      JSON.stringify({
        v: PROTOCOL_VERSION,
        type: "tick",
        step: 3,
        t: 0.3,
        control_dt: 0.1,
        sim_fps: 9.5,
        paused: false,
        cars: [[1, 2, 0.5, 1, 1.2, 1.1, 0.2, 0.05, 0.1, -0.4, 0.2]],
      }),
    );
    expect(msg.type).toBe("tick");
    if (msg.type === "tick") {
      expect(msg.step).toBe(3);
      expect(msg.t).toBeCloseTo(0.3);
      expect(msg.control_dt).toBeCloseTo(0.1);
      expect(msg.cars[0][4]).toBe(1.2);
      expect(msg.cars[0][5]).toBeCloseTo(1.1);
      expect(msg.cars[0][6]).toBeCloseTo(0.2);
      expect(msg.cars[0][7]).toBeCloseTo(0.05);
      expect(msg.cars[0][8]).toBeCloseTo(0.1);
      expect(msg.cars[0][9]).toBeCloseTo(-0.4);
      expect(msg.cars[0][11]).toBe(0);
    }
  });

  it("pads legacy short car rows and derives t", () => {
    const msg = parseServerMessage(
      JSON.stringify({
        v: PROTOCOL_VERSION,
        type: "tick",
        step: 2,
        sim_fps: 1,
        paused: false,
        cars: [[1, 0, 0, 1, 0.5]],
      }),
    );
    expect(msg.type).toBe("tick");
    if (msg.type === "tick") {
      expect(msg.control_dt).toBeCloseTo(0.1);
      expect(msg.t).toBeCloseTo(0.2);
      expect(msg.cars[0]).toEqual([1, 0, 0, 1, 0.5, 0, 0, 0, 0, 0, 0, 0]);
    }
  });

  it("maps world XY into Three Y-up", () => {
    expect(worldToThree(3, 4, 1.5)).toEqual([3, 1.5, 4]);
  });
});
