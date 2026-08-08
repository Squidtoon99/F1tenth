export const PROTOCOL_VERSION = 4;

/** [x, y, yaw, active, speed_mps, vx, vy, yaw_rate, steer_rad, long_action, steer_action, collision] */
export type CarPose = [
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
];

export const CAR_POSE_DIM = 12;

export type HelloMessage = {
  v: number;
  type: "hello";
  track: string;
  track_id: number;
  suite: string;
  checkpoint: string;
  seed: number;
  device: string;
  control_hz: number;
  car_length: number;
  car_width: number;
  car_height: number;
  speed_axis_max_mps: number;
  num_cars: number;
  num_obstacles: number;
  obstacle_preset: string;
  obstacle_nonce: number;
  obstacle_presets: string[];
  obstacles: [number, number, number][];
  num_environments: number;
  cars_per_environment: number;
  min_dense_environments: number;
  max_dense_environments: number;
  tracks: string[];
  suites: string[];
};

export type TrackMessage = {
  v: number;
  type: "track";
  center: [number, number][];
  left: [number, number][];
  right: [number, number][];
};

export type TickMessage = {
  v: number;
  type: "tick";
  step: number;
  t: number;
  control_dt: number;
  sim_fps: number;
  paused: boolean;
  cars: CarPose[];
};

export type ErrorMessage = {
  v: number;
  type: "error";
  code: string;
  message: string;
};

export type ActorUpdateMessage = {
  v: number;
  type: "actor_update";
  checkpoint: string;
  sha256: string;
  loaded_at: string;
  hidden_state_reset: true;
};

export type ServerMessage =
  | HelloMessage
  | TrackMessage
  | TickMessage
  | ActorUpdateMessage
  | ErrorMessage;

export type ClientMessage =
  | {
      v: number;
      type: "pause" | "resume" | "reset" | "ping" | "respawn_obstacles";
    }
  | { v: number; type: "set_track"; track: string }
  | { v: number; type: "set_track"; track_id: number }
  | { v: number; type: "set_suite"; suite: string }
  | { v: number; type: "set_obstacle_preset"; obstacle_preset: string }
  | { v: number; type: "set_environment_count"; environment_count: number }
  | { v: number; type: "set_checkpoint"; checkpoint: string };

export type CheckpointEntry = {
  name: string;
  path: string;
  condition_dim?: number;
};

export type CheckpointIndex = {
  active: CheckpointEntry;
  runs: { run: string; checkpoints: CheckpointEntry[] }[];
  condition_dim: number;
  incompatible: { count: number; reasons: string[] };
};

export function encodeClientMessage(
  msg:
    | "pause"
    | "resume"
    | "reset"
    | "ping"
    | "respawn_obstacles"
    | { type: "set_track"; track: string }
    | { type: "set_track"; track_id: number }
    | { type: "set_suite"; suite: string }
    | { type: "set_obstacle_preset"; obstacle_preset: string }
    | { type: "set_environment_count"; environment_count: number }
    | { type: "set_checkpoint"; checkpoint: string },
): string {
  if (typeof msg === "string") {
    return JSON.stringify({ v: PROTOCOL_VERSION, type: msg });
  }
  return JSON.stringify({ v: PROTOCOL_VERSION, ...msg });
}

export function normalizeCarPose(row: number[]): CarPose {
  return [
    row[0] ?? 0,
    row[1] ?? 0,
    row[2] ?? 0,
    row[3] ?? 0,
    row[4] ?? 0,
    row[5] ?? 0,
    row[6] ?? 0,
    row[7] ?? 0,
    row[8] ?? 0,
    row[9] ?? 0,
    row[10] ?? 0,
    row[11] ?? 0,
  ];
}

export function parseServerMessage(raw: string): ServerMessage {
  const payload = JSON.parse(raw) as ServerMessage;
  if (payload.v !== PROTOCOL_VERSION) {
    throw new Error(`unsupported protocol version: ${payload.v}`);
  }
  if (
    payload.type !== "hello" &&
    payload.type !== "track" &&
    payload.type !== "tick" &&
    payload.type !== "actor_update" &&
    payload.type !== "error"
  ) {
    throw new Error("unsupported server message type");
  }
  if (payload.type === "tick") {
    const tick = payload as TickMessage & {
      t?: number;
      control_dt?: number;
    };
    const controlDt =
      typeof tick.control_dt === "number" && tick.control_dt > 0
        ? tick.control_dt
        : 0.1;
    tick.control_dt = controlDt;
    tick.t =
      typeof tick.t === "number" ? tick.t : Number(tick.step) * controlDt;
    tick.cars = tick.cars.map((row) => normalizeCarPose(row as number[]));
  }
  if (payload.type === "hello") {
    payload.tracks = payload.tracks ?? [];
    payload.suites = payload.suites ?? ["solo", "head_to_head", "dense"];
    payload.num_environments = payload.num_environments ?? 1;
    payload.cars_per_environment =
      payload.cars_per_environment ?? payload.num_cars;
    payload.min_dense_environments = payload.min_dense_environments ?? 1;
    payload.max_dense_environments = payload.max_dense_environments ?? 4;
    payload.num_obstacles = payload.num_obstacles ?? 0;
    payload.obstacle_preset = payload.obstacle_preset ?? "off";
    payload.obstacle_nonce = payload.obstacle_nonce ?? 0;
    payload.obstacle_presets = payload.obstacle_presets ?? ["off", "light", "heavy"];
    payload.obstacles = payload.obstacles ?? [];
  }
  return payload;
}

/** World XY (meters) -> Three.js Y-up coordinates. */
export function worldToThree(
  x: number,
  y: number,
  z = 0,
): [number, number, number] {
  return [x, z, y];
}
