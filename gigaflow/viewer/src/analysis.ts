import type { CarPose } from "./protocol";

export const ANALYSIS_SECONDS = 20;
export const ANALYSIS_MAX_SAMPLES = 240;
export const NORMALIZED_ACTION_RANGE = [-1, 1] as const;
export const PEDAL_RANGE = [0, 1] as const;
export const COLLISION_REARM_SECONDS = 0.5;

export type TelemetrySample = {
  t: number;
  speed: number;
  steering: number;
  throttle: number;
  brake: number;
  collisions: number;
};

export class TelemetryHistory {
  private samples: TelemetrySample[] = [];
  private collisionLatched = false;
  private collisionClearSince: number | null = null;
  private collisionEvents = 0;

  reset(): void {
    this.samples = [];
    this.collisionLatched = false;
    this.collisionClearSince = null;
    this.collisionEvents = 0;
  }

  push(t: number, car: CarPose | undefined): void {
    if (!car || car[3] <= 0) {
      return;
    }
    const effort = car[9];
    const contact = car[11] > 0;
    if (contact) {
      if (
        this.collisionClearSince !== null &&
        t - this.collisionClearSince >= COLLISION_REARM_SECONDS
      ) {
        this.collisionLatched = false;
      }
      if (!this.collisionLatched) {
        this.collisionEvents += 1;
        this.collisionLatched = true;
      }
      this.collisionClearSince = null;
    } else if (this.collisionLatched) {
      this.collisionClearSince ??= t;
      if (t - this.collisionClearSince >= COLLISION_REARM_SECONDS) {
        this.collisionLatched = false;
        this.collisionClearSince = null;
      }
    }
    this.samples.push({
      t,
      speed: car[4],
      steering: Math.max(-1, Math.min(1, car[10])),
      throttle: Math.max(0, Math.min(1, effort)),
      brake: Math.max(0, Math.min(1, -effort)),
      collisions: this.collisionEvents,
    });
    const cutoff = t - ANALYSIS_SECONDS;
    while (this.samples.length && this.samples[0].t < cutoff) {
      this.samples.shift();
    }
    if (this.samples.length > ANALYSIS_MAX_SAMPLES) {
      this.samples.splice(0, this.samples.length - ANALYSIS_MAX_SAMPLES);
    }
  }

  snapshot(): TelemetrySample[] {
    return this.samples.slice();
  }

  get topSpeed(): number {
    let top = 0;
    for (const sample of this.samples) {
      top = Math.max(top, sample.speed);
    }
    return top;
  }

  get collisionCount(): number {
    return this.collisionEvents;
  }
}

export type CurrentControls = {
  steering: number;
  throttle: number;
  brake: number;
};

export function currentControls(car: CarPose | undefined): CurrentControls | null {
  if (!car || car[3] <= 0) {
    return null;
  }
  const effort = Math.max(-1, Math.min(1, car[9]));
  return {
    steering: Math.max(-1, Math.min(1, car[10])),
    throttle: Math.max(0, effort),
    brake: Math.max(0, -effort),
  };
}

export function chartPath(
  samples: TelemetrySample[],
  key: keyof Omit<TelemetrySample, "t">,
  width: number,
  height: number,
  range: readonly [number, number],
): string {
  if (samples.length < 2) {
    return "";
  }
  const [min, max] = range;
  const span = max - min;
  if (!(span > 0)) {
    throw new Error("chart range must have positive span");
  }
  const points = samples.map((sample, i) => {
    const x = (i / (samples.length - 1)) * width;
    const normalized = Math.max(0, Math.min(1, (sample[key] - min) / span));
    const y = height - normalized * height;
    return { x, y };
  });
  let path = `M ${points[0].x.toFixed(1)} ${points[0].y.toFixed(1)}`;
  for (let i = 1; i < points.length; i += 1) {
    const previous = points[i - 1];
    const point = points[i];
    const controlOffset = (point.x - previous.x) / 3;
    path +=
      ` C ${(previous.x + controlOffset).toFixed(1)} ${previous.y.toFixed(1)}` +
      ` ${(point.x - controlOffset).toFixed(1)} ${point.y.toFixed(1)}` +
      ` ${point.x.toFixed(1)} ${point.y.toFixed(1)}`;
  }
  return path;
}
