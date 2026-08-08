import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { advanceFollowTarget } from "./camera";
import type { CarPose, HelloMessage, TrackMessage } from "./protocol";
import { worldToThree } from "./protocol";
import { TRAIL_MAX_POINTS, TrailField } from "./trails";

/** Slightly brighter blue cars on the dark palette. */
const CAR_BLUE = 0x6a9be8;
const CAR_EDGE = 0xd0e4ff;
const OBSTACLE_GREY = 0x777d86;
const OBSTACLE_EDGE = 0xd6d8dc;
const TRACK_BOUNDARY = 0xffffff;
const CENTERLINE = 0x8a9bb0;
const BG = 0x000000;
const TRAIL_RGB = { r: 0.55, g: 0.75, b: 0.95 };
const FOLLOW_RADIUS_M = 5.0;

export type CameraMode = "follow" | "dynamic";

type CarMesh = {
  body: THREE.Mesh;
  heading: THREE.Mesh;
};

type TrailMesh = {
  line: THREE.Line;
  positions: Float32Array;
  colors: Float32Array; // RGB; brightness fades to black on dark bg
};

export class RaceScene {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly controls: OrbitControls;

  private trackGroup = new THREE.Group();
  private cars: CarMesh[] = [];
  private obstacles: THREE.Group[] = [];
  private trails = new TrailField();
  private trailMeshes: TrailMesh[] = [];
  private carLength = 0.568;
  private carWidth = 0.296;
  private carHeight = 0.44 * 0.568;
  private followIndex = 0;
  private wholeTrack = false;
  private cameraMode: CameraMode = "follow";
  private trackCenter = new THREE.Vector3();
  private trackSpan = 20;
  private target = new THREE.Vector3();
  private smoothTarget = new THREE.Vector3();
  private lastCars: CarPose[] = [];
  private hasSmooth = false;
  private cameraDirty = false;
  private cameraDirtyListener: ((dirty: boolean) => void) | null = null;

  constructor(canvas: HTMLCanvasElement) {
    this.scene.background = new THREE.Color(BG);
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: false,
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.05, 2000);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.enablePan = true;
    this.controls.addEventListener("start", () => this.setCameraDirty(true));
    this.scene.add(this.trackGroup);
    const light = new THREE.AmbientLight(0xffffff, 0.85);
    this.scene.add(light);
    const dir = new THREE.DirectionalLight(0xffffff, 0.45);
    dir.position.set(4, 10, 2);
    this.scene.add(dir);
    this.resize();
    window.addEventListener("resize", () => this.resize());
  }

  setVehicleDims(hello: HelloMessage): void {
    this.carLength = hello.car_length;
    this.carWidth = hello.car_width;
    this.carHeight = hello.car_height;
    this.ensureCars(hello.num_cars);
    this.rebuildCarGeometry();
  }

  setTrack(msg: TrackMessage): void {
    while (this.trackGroup.children.length) {
      const child = this.trackGroup.children.pop();
      if (child) {
        this.trackGroup.remove(child);
        disposeObject(child);
      }
    }
    this.trackGroup.add(lineFromPairs(msg.left, TRACK_BOUNDARY, 2.0));
    this.trackGroup.add(lineFromPairs(msg.right, TRACK_BOUNDARY, 2.0));
    this.trackGroup.add(lineFromPairs(msg.center, CENTERLINE, 1.0));

    const pts = [...msg.left, ...msg.right].map(([x, y]) => worldToThree(x, y));
    const box = new THREE.Box3();
    for (const p of pts) {
      box.expandByPoint(new THREE.Vector3(p[0], p[1], p[2]));
    }
    box.getCenter(this.trackCenter);
    const size = new THREE.Vector3();
    box.getSize(size);
    this.trackSpan = Math.max(size.x, size.z, 1);
    this.clearTrails();
    this.hasSmooth = false;
    this.resetCamera();
  }

  setObstacles(poses: [number, number, number][]): void {
    for (const obstacle of this.obstacles) {
      this.scene.remove(obstacle);
      disposeObject(obstacle);
    }
    this.obstacles = poses.map(([x, y, yaw]) => {
      const group = new THREE.Group();
      const bodyGeom = new THREE.BoxGeometry(
        this.carLength,
        this.carHeight,
        this.carWidth,
      );
      const body = new THREE.Mesh(
        bodyGeom,
        new THREE.MeshLambertMaterial({ color: OBSTACLE_GREY }),
      );
      body.add(
        new THREE.LineSegments(
          new THREE.EdgesGeometry(bodyGeom),
          new THREE.LineBasicMaterial({ color: OBSTACLE_EDGE }),
        ),
      );
      const heading = new THREE.Mesh(
        new THREE.BoxGeometry(0.45 * this.carLength, 0.04, 0.04),
        new THREE.MeshBasicMaterial({ color: OBSTACLE_EDGE }),
      );
      heading.position.x = 0.35 * this.carLength;
      heading.position.y = 0.5 * this.carHeight + 0.02;
      group.add(body, heading);
      const [tx, ty, tz] = worldToThree(x, y, this.carHeight * 0.5);
      group.position.set(tx, ty, tz);
      group.rotation.set(0, -yaw, 0);
      this.scene.add(group);
      return group;
    });
  }

  clearTrails(): void {
    this.trails.clearAll();
    for (const tm of this.trailMeshes) {
      tm.line.geometry.setDrawRange(0, 0);
      tm.line.visible = false;
    }
  }

  setFollowIndex(index: number): void {
    this.followIndex = Math.max(0, index);
  }

  setWholeTrack(enabled: boolean): void {
    this.wholeTrack = enabled;
    this.resetCamera();
  }

  setCameraMode(mode: CameraMode): void {
    this.cameraMode = mode;
    this.hasSmooth = false;
    this.resetCamera();
  }

  onCameraDirty(listener: (dirty: boolean) => void): void {
    this.cameraDirtyListener = listener;
    listener(this.cameraDirty);
  }

  resetCamera(): void {
    this.updateFollowTarget();
    const target =
      this.wholeTrack || this.cameraMode === "dynamic"
        ? this.trackCenter
        : this.target;
    this.smoothTarget.copy(target);
    this.controls.target.copy(target);
    this.camera.position.copy(this.defaultCameraPosition(target));
    this.hasSmooth = true;
    this.setCameraDirty(false);
    this.controls.update();
  }

  updateCars(cars: CarPose[]): void {
    this.lastCars = cars;
    this.ensureCars(cars.length);
    for (let i = 0; i < this.cars.length; i += 1) {
      const mesh = this.cars[i];
      const pose = cars[i];
      if (!pose || pose[3] <= 0) {
        mesh.body.visible = false;
        mesh.heading.visible = false;
        continue;
      }
      const [x, y, yaw] = pose;
      const [tx, ty, tz] = worldToThree(x, y, this.carHeight * 0.5);
      mesh.body.visible = true;
      mesh.heading.visible = true;
      mesh.body.position.set(tx, ty, tz);
      mesh.body.rotation.set(0, -yaw, 0);
      const [hx, hy, hz] = worldToThree(
        x + 0.35 * this.carLength * Math.cos(yaw),
        y + 0.35 * this.carLength * Math.sin(yaw),
        this.carHeight + 0.02,
      );
      mesh.heading.position.set(hx, hy, hz);
      mesh.heading.rotation.set(0, -yaw, 0);
    }
    this.updateFollowTarget();
  }

  pushKnownCars(simT: number, cars: CarPose[]): void {
    this.trails.pushAll(simT, cars);
    this.syncTrailMeshes(simT);
  }

  render(): void {
    this.followCameraTarget();
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  private ensureCars(count: number): void {
    while (this.cars.length < count) {
      const bodyGeom = new THREE.BoxGeometry(
        this.carLength,
        this.carHeight,
        this.carWidth,
      );
      const bodyMat = new THREE.MeshLambertMaterial({
        color: CAR_BLUE,
        transparent: true,
        opacity: 0.88,
      });
      const body = new THREE.Mesh(bodyGeom, bodyMat);
      const edge = new THREE.LineSegments(
        new THREE.EdgesGeometry(bodyGeom),
        new THREE.LineBasicMaterial({ color: CAR_EDGE }),
      );
      body.add(edge);

      const headingGeom = new THREE.BoxGeometry(
        0.45 * this.carLength,
        0.04,
        0.04,
      );
      const headingMat = new THREE.MeshBasicMaterial({ color: CAR_EDGE });
      const heading = new THREE.Mesh(headingGeom, headingMat);
      this.scene.add(body);
      this.scene.add(heading);
      this.cars.push({ body, heading });
    }
    this.ensureTrailMeshes(count);
  }

  private ensureTrailMeshes(count: number): void {
    this.trails.ensure(count);
    while (this.trailMeshes.length < count) {
      const cap = TRAIL_MAX_POINTS;
      const positions = new Float32Array(cap * 3);
      const colors = new Float32Array(cap * 3);
      const geom = new THREE.BufferGeometry();
      geom.setAttribute(
        "position",
        new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage),
      );
      geom.setAttribute(
        "color",
        new THREE.BufferAttribute(colors, 3).setUsage(THREE.DynamicDrawUsage),
      );
      geom.setDrawRange(0, 0);
      const mat = new THREE.LineBasicMaterial({
        vertexColors: true,
        depthWrite: false,
        linewidth: 1,
      });
      const line = new THREE.Line(geom, mat);
      line.frustumCulled = false;
      line.visible = false;
      this.scene.add(line);
      this.trailMeshes.push({ line, positions, colors });
    }
  }

  private syncTrailMeshes(simT: number): void {
    for (let i = 0; i < this.trailMeshes.length; i += 1) {
      const tm = this.trailMeshes[i];
      const trail = this.trails.get(i);
      const pts = trail.snapshot();
      if (pts.length < 2) {
        tm.line.geometry.setDrawRange(0, 0);
        tm.line.visible = false;
        continue;
      }
      const n = Math.min(pts.length, TRAIL_MAX_POINTS);
      for (let k = 0; k < n; k += 1) {
        const p = pts[k];
        const [tx, ty, tz] = worldToThree(p.x, p.y, 0.03);
        tm.positions[k * 3] = tx;
        tm.positions[k * 3 + 1] = ty;
        tm.positions[k * 3 + 2] = tz;
        // Fade brightness toward black (visible trail age on dark bg).
        const a = 0.12 + 0.88 * trail.fadeAt(k, simT);
        tm.colors[k * 3] = TRAIL_RGB.r * a;
        tm.colors[k * 3 + 1] = TRAIL_RGB.g * a;
        tm.colors[k * 3 + 2] = TRAIL_RGB.b * a;
      }
      const posAttr = tm.line.geometry.getAttribute(
        "position",
      ) as THREE.BufferAttribute;
      const colAttr = tm.line.geometry.getAttribute(
        "color",
      ) as THREE.BufferAttribute;
      posAttr.needsUpdate = true;
      colAttr.needsUpdate = true;
      tm.line.geometry.setDrawRange(0, n);
      tm.line.visible = true;
    }
  }

  private rebuildCarGeometry(): void {
    for (const car of this.cars) {
      car.body.geometry.dispose();
      car.body.geometry = new THREE.BoxGeometry(
        this.carLength,
        this.carHeight,
        this.carWidth,
      );
      car.heading.geometry.dispose();
      car.heading.geometry = new THREE.BoxGeometry(
        0.45 * this.carLength,
        0.04,
        0.04,
      );
    }
  }

  private activeFollowIndex(): number {
    const cars = this.lastCars;
    if (!cars.length) {
      return 0;
    }
    let idx = this.followIndex;
    if (idx >= cars.length || cars[idx][3] <= 0) {
      idx = cars.findIndex((c) => c[3] > 0);
      if (idx < 0) {
        idx = 0;
      }
    }
    return idx;
  }

  private updateFollowTarget(): void {
    const cars = this.lastCars;
    if (!cars.length) {
      this.target.copy(this.trackCenter);
      return;
    }
    const idx = this.activeFollowIndex();
    const [x, y] = cars[idx];
    const [tx, , tz] = worldToThree(x, y, 0);
    this.target.set(tx, 0, tz);
  }

  private defaultCameraPosition(target: THREE.Vector3): THREE.Vector3 {
    const radius =
      this.wholeTrack || this.cameraMode === "dynamic"
        ? this.trackSpan * 0.9
        : FOLLOW_RADIUS_M;
    return offsetFixed(target, radius);
  }

  private followCameraTarget(): void {
    if (this.wholeTrack || this.cameraMode !== "follow") {
      return;
    }
    if (!this.hasSmooth) {
      this.resetCamera();
      this.hasSmooth = true;
      return;
    }
    const next = advanceFollowTarget(
      [this.smoothTarget.x, this.smoothTarget.y, this.smoothTarget.z],
      [this.target.x, this.target.y, this.target.z],
      0.28,
    );
    this.smoothTarget.fromArray(next.target);
    this.controls.target.add(new THREE.Vector3().fromArray(next.delta));
    this.camera.position.add(new THREE.Vector3().fromArray(next.delta));
  }

  private setCameraDirty(dirty: boolean): void {
    if (this.cameraDirty === dirty) {
      return;
    }
    this.cameraDirty = dirty;
    this.cameraDirtyListener?.(dirty);
  }

  private resize(): void {
    const w = window.innerWidth;
    const h = window.innerHeight;
    this.camera.aspect = w / Math.max(h, 1);
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h, false);
  }
}

function offsetFixed(target: THREE.Vector3, radius: number): THREE.Vector3 {
  const elev = THREE.MathUtils.degToRad(30);
  const azim = THREE.MathUtils.degToRad(45);
  const x = target.x + radius * Math.cos(elev) * Math.cos(azim);
  const y = target.y + radius * Math.sin(elev);
  const z = target.z + radius * Math.cos(elev) * Math.sin(azim);
  return new THREE.Vector3(x, y, z);
}

function lineFromPairs(
  pairs: [number, number][],
  color: number,
  linewidth: number,
): THREE.Line {
  const points = pairs.map(([x, y]) => {
    const [tx, ty, tz] = worldToThree(x, y, 0);
    return new THREE.Vector3(tx, ty, tz);
  });
  if (points.length > 1) {
    points.push(points[0].clone());
  }
  const geom = new THREE.BufferGeometry().setFromPoints(points);
  const mat = new THREE.LineBasicMaterial({ color, linewidth });
  return new THREE.Line(geom, mat);
}

function disposeObject(obj: THREE.Object3D): void {
  obj.traverse((child) => {
    const mesh = child as THREE.Mesh;
    if (mesh.geometry) {
      mesh.geometry.dispose();
    }
    const mat = mesh.material;
    if (Array.isArray(mat)) {
      mat.forEach((m) => m.dispose());
    } else if (mat) {
      mat.dispose();
    }
  });
}
