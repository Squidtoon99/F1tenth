/** Sim-time motion trail history per car (presentation only). */

/** Default trail length in simulation seconds. */
export const TRAIL_SECONDS = 3.0;

/** Hard cap on retained samples per car (memory / GPU bound). */
export const TRAIL_MAX_POINTS = 96;

/** Position jump (m) that clears the trail (respawn / teleport). */
export const TRAIL_SNAP_M = 2.0;

export type TrailPoint = { t: number; x: number; y: number };

/**
 * Bounded ring of (sim_t, x, y) samples for one car.
 * Cleared on inactive, respawn jump, or explicit reset.
 */
export class CarTrail {
  private points: TrailPoint[] = [];

  clear(): void {
    this.points = [];
  }

  get length(): number {
    return this.points.length;
  }

  snapshot(): TrailPoint[] {
    return this.points.slice();
  }

  push(t: number, x: number, y: number, active: boolean): void {
    if (!active) {
      this.clear();
      return;
    }
    if (this.points.length) {
      const last = this.points[this.points.length - 1];
      if (t + 1e-9 < last.t) {
        this.clear();
      } else {
        const dx = x - last.x;
        const dy = y - last.y;
        if (dx * dx + dy * dy > TRAIL_SNAP_M * TRAIL_SNAP_M) {
          this.clear();
        } else if (Math.abs(t - last.t) < 1e-9 && dx * dx + dy * dy < 1e-12) {
          return;
        }
      }
    }
    this.points.push({ t, x, y });
    const cutoff = t - TRAIL_SECONDS;
    let drop = 0;
    while (
      drop < this.points.length &&
      this.points[drop].t < cutoff - 1e-12
    ) {
      drop += 1;
    }
    if (drop > 0) {
      this.points.splice(0, drop);
    }
    if (this.points.length > TRAIL_MAX_POINTS) {
      this.points.splice(0, this.points.length - TRAIL_MAX_POINTS);
    }
  }

  /**
   * Fade weight per point: 0 at TRAIL_SECONDS behind `nowT`, 1 at `nowT`.
   */
  fadeAt(i: number, nowT: number): number {
    const p = this.points[i];
    if (!p) {
      return 0;
    }
    return Math.min(
      1,
      Math.max(0, 1 - (nowT - p.t) / Math.max(TRAIL_SECONDS, 1e-6)),
    );
  }
}

/** Fixed-size pool of per-car trails. */
export class TrailField {
  private trails: CarTrail[] = [];

  clearAll(): void {
    for (const tr of this.trails) {
      tr.clear();
    }
  }

  ensure(count: number): void {
    while (this.trails.length < count) {
      this.trails.push(new CarTrail());
    }
  }

  pushAll(t: number, cars: ArrayLike<ArrayLike<number>>): void {
    this.ensure(cars.length);
    for (let i = 0; i < cars.length; i += 1) {
      const c = cars[i];
      const active = (c[3] ?? 0) > 0;
      this.trails[i].push(t, c[0], c[1], active);
    }
    for (let i = cars.length; i < this.trails.length; i += 1) {
      this.trails[i].clear();
    }
  }

  get(i: number): CarTrail {
    this.ensure(i + 1);
    return this.trails[i];
  }

  get size(): number {
    return this.trails.length;
  }
}
