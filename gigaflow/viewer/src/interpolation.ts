import type { CarPose, TickMessage } from "./protocol";
import { normalizeCarPose } from "./protocol";

/** Presentation delay keeps display safely behind authoritative samples. */
export const REPLAY_DELAY_SECONDS = 3.0;

/** Position jump (m) treated as a respawn / teleport discontinuity. */
export const SNAP_POS_M = 2.0;

/** Sim-time gap (as multiples of control_dt) treated as a sequence break. */
export const SEQUENCE_GAP_TICKS = 3;

/** Retain enough ticks to cover delay + arrival jitter without dropping the bracket. */
export const MAX_BUFFER_TICKS = 48;
export const MAX_PLAYHEAD_ADVANCE_SECONDS = 0.1;

export type TickSnapshot = {
  t: number;
  control_dt: number;
  step: number;
  paused: boolean;
  cars: CarPose[];
};

export function shortestAngleDelta(from: number, to: number): number {
  let d = to - from;
  while (d > Math.PI) {
    d -= 2 * Math.PI;
  }
  while (d < -Math.PI) {
    d += 2 * Math.PI;
  }
  return d;
}

export function carDiscontinuous(a: CarPose, b: CarPose): boolean {
  const aOn = a[3] > 0;
  const bOn = b[3] > 0;
  if (aOn !== bOn) {
    return true;
  }
  if (!aOn && !bOn) {
    return false;
  }
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  return dx * dx + dy * dy > SNAP_POS_M * SNAP_POS_M;
}

export function interpolateCar(
  a: CarPose,
  b: CarPose,
  u: number,
): CarPose {
  if (carDiscontinuous(a, b)) {
    return u < 1 ? normalizeCarPose([...a]) : normalizeCarPose([...b]);
  }
  if (a[3] <= 0 && b[3] <= 0) {
    return normalizeCarPose([...b]);
  }
  const uu = Math.min(1, Math.max(0, u));
  const x = a[0] + (b[0] - a[0]) * uu;
  const y = a[1] + (b[1] - a[1]) * uu;
  const yaw0 = a[2];
  const yaw1 = yaw0 + shortestAngleDelta(yaw0, b[2]);
  const yaw = yaw0 + (yaw1 - yaw0) * uu;
  const active = uu < 1 ? a[3] : b[3];
  const out = a.map((value, i) => value + (b[i] - value) * uu) as CarPose;
  out[0] = x;
  out[1] = y;
  out[2] = yaw;
  out[3] = active;
  return out;
}

export function tickFromMessage(msg: TickMessage): TickSnapshot {
  return {
    t: msg.t,
    control_dt: msg.control_dt,
    step: msg.step,
    paused: msg.paused,
    cars: msg.cars.map((c) => normalizeCarPose(c)),
  };
}

/**
 * Delayed replay buffer: linear display interpolation between known waypoints.
 * Playhead advances with wall-clock, never extrapolates past newest tick.
 */
export class TickInterpolator {
  private ticks: TickSnapshot[] = [];
  private playhead: number | null = null;
  private lastSampleMs: number | null = null;
  private holdPlayhead: number | null = null;
  private pendingKnown: TickSnapshot[] = [];

  reset(): void {
    this.ticks = [];
    this.playhead = null;
    this.lastSampleMs = null;
    this.holdPlayhead = null;
    this.pendingKnown = [];
  }

  push(msg: TickMessage, _arrivalMs: number): void {
    const snap = tickFromMessage(msg);
    if (this.ticks.length) {
      const last = this.ticks[this.ticks.length - 1];
      const gapLimit = SEQUENCE_GAP_TICKS * Math.max(last.control_dt, 1e-6);
      if (
        snap.step < last.step ||
        snap.t + 1e-9 < last.t ||
        snap.t - last.t > gapLimit
      ) {
        this.ticks = [];
        this.playhead = null;
        this.lastSampleMs = null;
        this.holdPlayhead = null;
        this.pendingKnown = [];
      }
    }
    if (this.ticks.length) {
      const last = this.ticks[this.ticks.length - 1];
      if (Math.abs(snap.t - last.t) < 1e-12 && snap.step === last.step) {
        this.ticks[this.ticks.length - 1] = snap;
        if (this.pendingKnown.length) {
          this.pendingKnown[this.pendingKnown.length - 1] = snap;
        }
      } else {
        this.ticks.push(snap);
        this.pendingKnown.push(snap);
      }
    } else {
      this.ticks.push(snap);
      this.pendingKnown.push(snap);
    }
    while (this.ticks.length > MAX_BUFFER_TICKS) {
      this.ticks.shift();
    }
    while (this.pendingKnown.length > MAX_BUFFER_TICKS) {
      this.pendingKnown.shift();
    }
    // Drop ticks that can no longer bracket the playhead (keep one behind).
    if (this.playhead !== null) {
      while (
        this.ticks.length > 2 &&
        this.ticks[1].t <= this.playhead + 1e-12
      ) {
        this.ticks.shift();
      }
    }
    if (snap.paused) {
      if (this.holdPlayhead === null) {
        this.holdPlayhead =
          this.playhead !== null ? this.playhead : snap.t;
      }
    } else {
      this.holdPlayhead = null;
    }
  }

  get delaySeconds(): number {
    return REPLAY_DELAY_SECONDS;
  }

  /** Current playhead sim-time (tests / diagnostics). */
  getPlayhead(): number | null {
    return this.holdPlayhead !== null ? this.holdPlayhead : this.playhead;
  }

  drainKnownThrough(playhead: number): TickSnapshot[] {
    let count = 0;
    while (
      count < this.pendingKnown.length &&
      this.pendingKnown[count].t <= playhead + 1e-12
    ) {
      count += 1;
    }
    return this.pendingKnown.splice(0, count);
  }

  sample(nowMs: number): CarPose[] | null {
    if (!this.ticks.length) {
      return null;
    }
    if (this.holdPlayhead !== null) {
      this.lastSampleMs = nowMs;
      return this.sampleAt(this.holdPlayhead);
    }

    const newest = this.ticks[this.ticks.length - 1];
    const oldest = this.ticks[0];
    const delay = REPLAY_DELAY_SECONDS;

    if (this.playhead === null || this.lastSampleMs === null) {
      // Hold at oldest until the buffer spans the presentation delay.
      if (newest.t - oldest.t + 1e-12 < delay) {
        this.playhead = oldest.t;
        this.lastSampleMs = nowMs;
        return this.sampleAt(this.playhead);
      }
      this.playhead = Math.max(oldest.t, newest.t - delay);
      this.lastSampleMs = nowMs;
      return this.sampleAt(this.playhead);
    }

    // Still filling the delayed replay buffer: hold, do not advance.
    if (newest.t - oldest.t + 1e-12 < delay) {
      this.playhead = oldest.t;
      this.lastSampleMs = nowMs;
      return this.sampleAt(this.playhead);
    }

    const dtWall = Math.min(
      MAX_PLAYHEAD_ADVANCE_SECONDS,
      Math.max(0, (nowMs - this.lastSampleMs) / 1000),
    );
    this.lastSampleMs = nowMs;
    const target = newest.t - delay;
    this.playhead = Math.min(this.playhead + dtWall, target);
    if (this.playhead < oldest.t) {
      this.playhead = oldest.t;
    }
    return this.sampleAt(this.playhead);
  }

  /** Deterministic sample at an absolute sim time (for tests). */
  sampleAt(playhead: number): CarPose[] | null {
    if (!this.ticks.length) {
      return null;
    }
    const newest = this.ticks[this.ticks.length - 1];
    const oldest = this.ticks[0];
    const clamped = Math.min(Math.max(playhead, oldest.t), newest.t);

    if (this.ticks.length === 1 || clamped <= oldest.t) {
      return oldest.cars.map((c) => normalizeCarPose([...c]));
    }
    if (clamped >= newest.t) {
      return newest.cars.map((c) => normalizeCarPose([...c]));
    }

    let a = this.ticks[0];
    let b = this.ticks[1];
    for (let i = 0; i < this.ticks.length - 1; i += 1) {
      a = this.ticks[i];
      b = this.ticks[i + 1];
      if (clamped <= b.t) {
        break;
      }
    }

    if (clamped <= a.t) {
      return a.cars.map((c) => normalizeCarPose([...c]));
    }
    if (clamped >= b.t) {
      return b.cars.map((c) => normalizeCarPose([...c]));
    }

    const u = (clamped - a.t) / Math.max(b.t - a.t, 1e-9);
    const n = Math.max(a.cars.length, b.cars.length);
    const out: CarPose[] = [];
    for (let i = 0; i < n; i += 1) {
      const ca = a.cars[i] ?? b.cars[i];
      const cb = b.cars[i] ?? a.cars[i];
      if (!ca || !cb) {
        continue;
      }
      out.push(interpolateCar(ca, cb, u));
    }
    return out;
  }
}
