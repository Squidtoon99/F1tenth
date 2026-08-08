import "./styles.css";
import {
  NORMALIZED_ACTION_RANGE,
  PEDAL_RANGE,
  TelemetryHistory,
  chartPath,
  currentControls,
} from "./analysis";
import { TickInterpolator } from "./interpolation";
import {
  encodeClientMessage,
  parseServerMessage,
  type CarPose,
  type CheckpointIndex,
  type HelloMessage,
} from "./protocol";
import { RaceScene, type CameraMode } from "./scene";

function wsUrl(): string {
  const params = new URLSearchParams(window.location.search);
  const override = params.get("ws");
  if (override) {
    return override;
  }
  const host = window.location.hostname || "127.0.0.1";
  const port = params.get("wsPort") || "8765";
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${host}:${port}`;
}

function requireEl<T extends Element>(sel: string): T {
  const el = document.querySelector<T>(sel);
  if (!el) {
    throw new Error(`viewer DOM missing: ${sel}`);
  }
  return el;
}

const canvas = requireEl<HTMLCanvasElement>("#scene");
const statusEl = requireEl<HTMLDivElement>("#status");
const pauseBtn = requireEl<HTMLButtonElement>("#btn-pause");
const resetBtn = requireEl<HTMLButtonElement>("#btn-reset");
const followSel = requireEl<HTMLSelectElement>("#follow-car");
const trackSel = requireEl<HTMLSelectElement>("#track-sel");
const checkpointSel = requireEl<HTMLSelectElement>("#checkpoint-sel");
const checkpointStatus = requireEl<HTMLDivElement>("#checkpoint-status");
const suiteSel = requireEl<HTMLSelectElement>("#suite-sel");
const obstaclePreset = requireEl<HTMLSelectElement>("#obstacle-preset");
const respawnObstaclesBtn = requireEl<HTMLButtonElement>("#btn-respawn-obstacles");
const cameraSel = requireEl<HTMLSelectElement>("#camera-mode");
const wholeTrack = requireEl<HTMLInputElement>("#whole-track");
const resetCameraBtn = requireEl<HTMLButtonElement>("#btn-reset-camera");
const environmentControl = requireEl<HTMLLabelElement>("#environment-control");
const environmentCount = requireEl<HTMLInputElement>("#environment-count");
const environmentCountValue = requireEl<HTMLOutputElement>(
  "#environment-count-value",
);
const showAnalysis = requireEl<HTMLInputElement>("#show-analysis");
const analysisEl = requireEl<HTMLElement>("#analysis");
const analysisAgent = requireEl<HTMLDivElement>("#analysis-agent");
const topSpeed = requireEl<HTMLElement>("#top-speed");
const collisionCount = requireEl<HTMLElement>("#collision-count");
const speedAxisRange = requireEl<HTMLElement>("#speed-axis-range");
const chartSpeed = requireEl<SVGPathElement>("#chart-speed");
const chartSteering = requireEl<SVGPathElement>("#chart-steering");
const chartThrottle = requireEl<SVGPathElement>("#chart-throttle");
const chartBrake = requireEl<SVGPathElement>("#chart-brake");
const chartCollisions = requireEl<SVGPathElement>("#chart-collisions");
const steeringWheel = requireEl<SVGElement>(".steering-wheel");
const steeringValue = requireEl<HTMLOutputElement>("#steering-value");
const throttleGauge = requireEl<HTMLDivElement>("#throttle-gauge");
const throttleValue = requireEl<HTMLOutputElement>("#throttle-value");
const brakeGauge = requireEl<HTMLDivElement>("#brake-gauge");
const brakeValue = requireEl<HTMLOutputElement>("#brake-value");

const scene = new RaceScene(canvas);
const motion = new TickInterpolator();
const telemetry = new TelemetryHistory();
let hello: HelloMessage | null = null;
let paused = false;
let socket: WebSocket | null = null;
let applyingHello = false;
let analysisDirty = true;
let latestStep = -1;
let selectedAgentActive = false;
let activeCheckpoint = "";
function setStatus(lines: string[]): void {
  statusEl.textContent = lines.join("\n");
}

function fillSelect(
  sel: HTMLSelectElement,
  values: string[],
  selected: string,
): void {
  sel.innerHTML = "";
  for (const value of values) {
    const opt = document.createElement("option");
    opt.value = value;
    opt.textContent = value;
    sel.appendChild(opt);
  }
  sel.value = selected;
}

function setCheckpointStatus(text: string, isError = false): void {
  checkpointStatus.textContent = text;
  checkpointStatus.classList.toggle("error", isError);
}

function fillCheckpointOptions(index: CheckpointIndex): void {
  checkpointSel.innerHTML = "";
  for (const run of index.runs) {
    const group = document.createElement("optgroup");
    group.label = run.run;
    for (const entry of run.checkpoints) {
      const opt = document.createElement("option");
      opt.value = entry.path;
      const width = entry.condition_dim;
      opt.textContent =
        width === undefined || width === index.condition_dim
          ? entry.name
          : `${entry.name} (cond ${width})`;
      group.appendChild(opt);
    }
    checkpointSel.appendChild(group);
  }
  activeCheckpoint = index.active.path;
  const known = Array.from(checkpointSel.options).some(
    (opt) => opt.value === activeCheckpoint,
  );
  if (!known) {
    const opt = document.createElement("option");
    opt.value = activeCheckpoint;
    opt.textContent = index.active.name;
    checkpointSel.insertBefore(opt, checkpointSel.firstChild);
  }
  checkpointSel.value = activeCheckpoint;
  checkpointSel.disabled = false;
  const hidden = index.incompatible.count;
  setCheckpointStatus(
    hidden > 0
      ? `active ${index.active.name} · ${hidden} hidden (architecture mismatch)`
      : `active ${index.active.name}`,
  );
  checkpointStatus.title = index.incompatible.reasons.join("\n");
}

async function refreshCheckpoints(): Promise<void> {
  try {
    const res = await fetch("/api/checkpoints", { cache: "no-store" });
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    fillCheckpointOptions((await res.json()) as CheckpointIndex);
  } catch (err) {
    checkpointSel.disabled = true;
    setCheckpointStatus(`checkpoint list unavailable (${String(err)})`, true);
  }
}

function fillFollowOptions(msg: HelloMessage): void {
  const prev = followSel.value;
  followSel.innerHTML = "";
  for (let i = 0; i < msg.num_cars; i += 1) {
    const opt = document.createElement("option");
    opt.value = String(i);
    const environment = Math.floor(i / msg.cars_per_environment);
    const car = i % msg.cars_per_environment;
    opt.textContent =
      msg.num_environments > 1
        ? `environment ${environment + 1} · car ${car}`
        : `car ${car}`;
    followSel.appendChild(opt);
  }
  if (prev && Number(prev) < msg.num_cars) {
    followSel.value = prev;
  }
  scene.setFollowIndex(Number(followSel.value || 0));
}

function connect(): void {
  const url = wsUrl();
  setStatus([`connecting ${url}`]);
  socket = new WebSocket(url);
  socket.addEventListener("open", () => {
    setStatus([`connected ${url}`]);
  });
  socket.addEventListener("close", () => {
    setStatus(["disconnected — retrying…"]);
    motion.reset();
    scene.clearTrails();
    telemetry.reset();
    analysisDirty = true;
    latestStep = -1;
    selectedAgentActive = false;
    window.setTimeout(connect, 1000);
  });
  socket.addEventListener("message", (ev) => {
    const msg = parseServerMessage(String(ev.data));
    if (msg.type === "error") {
      setStatus([`error: ${msg.message}`]);
      if (msg.code === "checkpoint_error") {
        setCheckpointStatus(msg.message, true);
        checkpointSel.value = activeCheckpoint;
      }
      return;
    }
    if (msg.type === "hello") {
      hello = msg;
      applyingHello = true;
      motion.reset();
      scene.clearTrails();
      telemetry.reset();
      analysisDirty = true;
      latestStep = -1;
      selectedAgentActive = false;
      scene.setVehicleDims(msg);
      scene.setObstacles(msg.obstacles);
      fillFollowOptions(msg);
      fillSelect(trackSel, msg.tracks, msg.track);
      fillSelect(suiteSel, msg.suites, msg.suite);
      fillSelect(obstaclePreset, msg.obstacle_presets, msg.obstacle_preset);
      respawnObstaclesBtn.disabled = msg.obstacle_preset === "off";
      for (const option of obstaclePreset.options) {
        option.textContent =
          option.value.charAt(0).toUpperCase() + option.value.slice(1);
      }
      environmentControl.hidden = msg.suite !== "dense";
      environmentCount.min = String(msg.min_dense_environments);
      environmentCount.max = String(msg.max_dense_environments);
      environmentCount.value = String(msg.num_environments);
      environmentCountValue.value = String(msg.num_environments);
      speedAxisRange.textContent = `0–${msg.speed_axis_max_mps.toFixed(1)}`;
      paused = false;
      pauseBtn.textContent = "Pause";
      applyingHello = false;
      return;
    }
    if (msg.type === "actor_update") {
      if (hello) {
        hello.checkpoint = msg.checkpoint;
      }
      setStatus([
        `actor updated ${msg.checkpoint}`,
        `GRU state reset · ${msg.loaded_at}`,
      ]);
      void refreshCheckpoints();
      return;
    }
    if (msg.type === "track") {
      motion.reset();
      scene.setTrack(msg);
      return;
    }
    if (msg.type === "tick") {
      if (latestStep >= 0 && msg.step < latestStep) {
        telemetry.reset();
        scene.clearTrails();
        analysisDirty = true;
      }
      latestStep = msg.step;
      motion.push(msg, performance.now());
      paused = msg.paused;
      pauseBtn.textContent = paused ? "Resume" : "Pause";
      if (hello) {
        setStatus([
          `track ${hello.track}  suite ${hello.suite}`,
          `${hello.num_environments} environment${
            hello.num_environments === 1 ? "" : "s"
          } · ${hello.cars_per_environment} cars/environment`,
          `ckpt ${hello.checkpoint}  seed ${hello.seed}`,
          `step ${msg.step}  sim ${msg.sim_fps.toFixed(1)} Hz` +
            (paused ? "  paused" : ""),
        ]);
      }
    }
  });
}

function send(
  msg:
    | "pause"
    | "resume"
    | "reset"
    | "respawn_obstacles"
    | { type: "set_track"; track: string }
    | { type: "set_suite"; suite: string }
    | { type: "set_obstacle_preset"; obstacle_preset: string }
    | { type: "set_environment_count"; environment_count: number }
    | { type: "set_checkpoint"; checkpoint: string },
): void {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    return;
  }
  socket.send(encodeClientMessage(msg));
}

pauseBtn.addEventListener("click", () => {
  send(paused ? "resume" : "pause");
});

resetBtn.addEventListener("click", () => {
  scene.clearTrails();
  motion.reset();
  telemetry.reset();
  analysisDirty = true;
  latestStep = -1;
  selectedAgentActive = false;
  send("reset");
});

followSel.addEventListener("change", () => {
  scene.setFollowIndex(Number(followSel.value));
  telemetry.reset();
  analysisDirty = true;
  selectedAgentActive = false;
});

trackSel.addEventListener("change", () => {
  if (applyingHello) {
    return;
  }
  send({ type: "set_track", track: trackSel.value });
});

checkpointSel.addEventListener("change", () => {
  const path = checkpointSel.value;
  if (!path || path === activeCheckpoint) {
    return;
  }
  setCheckpointStatus(`loading ${path.split("/").pop() ?? path}…`);
  send({ type: "set_checkpoint", checkpoint: path });
});

suiteSel.addEventListener("change", () => {
  if (applyingHello) {
    return;
  }
  send({ type: "set_suite", suite: suiteSel.value });
});

obstaclePreset.addEventListener("change", () => {
  respawnObstaclesBtn.disabled = obstaclePreset.value === "off";
  send({
    type: "set_obstacle_preset",
    obstacle_preset: obstaclePreset.value,
  });
});

respawnObstaclesBtn.addEventListener("click", () => {
  scene.clearTrails();
  motion.reset();
  telemetry.reset();
  analysisDirty = true;
  latestStep = -1;
  selectedAgentActive = false;
  send("respawn_obstacles");
});

environmentCount.addEventListener("input", () => {
  environmentCountValue.value = environmentCount.value;
});

environmentCount.addEventListener("change", () => {
  send({
    type: "set_environment_count",
    environment_count: Number(environmentCount.value),
  });
});

cameraSel.addEventListener("change", () => {
  scene.setCameraMode(cameraSel.value as CameraMode);
});

wholeTrack.addEventListener("change", () => {
  scene.setWholeTrack(wholeTrack.checked);
});

scene.onCameraDirty((dirty) => {
  resetCameraBtn.hidden = !dirty;
});

resetCameraBtn.addEventListener("click", () => {
  scene.resetCamera();
});

showAnalysis.addEventListener("change", () => {
  analysisEl.hidden = !showAnalysis.checked;
  analysisDirty = true;
});

function setPedalGauge(
  gauge: HTMLDivElement,
  output: HTMLOutputElement,
  value: number | null,
): void {
  gauge.style.setProperty("--gauge-value", String(value ?? 0));
  output.value = value === null ? "—" : value.toFixed(2);
  if (value === null) {
    gauge.removeAttribute("aria-valuenow");
  } else {
    gauge.setAttribute("aria-valuenow", value.toFixed(2));
  }
}

function renderCurrentControls(car: CarPose | undefined): void {
  const controls = currentControls(car);
  if (!controls) {
    steeringWheel.style.setProperty("--steering-angle", "0deg");
    steeringValue.value = "—";
    steeringValue.setAttribute("aria-label", "Current steering unavailable");
    setPedalGauge(throttleGauge, throttleValue, null);
    setPedalGauge(brakeGauge, brakeValue, null);
    return;
  }
  steeringWheel.style.setProperty(
    "--steering-angle",
    `${(controls.steering * 110).toFixed(1)}deg`,
  );
  const steeringText = controls.steering.toFixed(2);
  steeringValue.value = steeringText;
  steeringValue.setAttribute(
    "aria-label",
    `Current normalized steering ${steeringText}`,
  );
  setPedalGauge(throttleGauge, throttleValue, controls.throttle);
  setPedalGauge(brakeGauge, brakeValue, controls.brake);
}

function renderAnalysis(): void {
  if (!showAnalysis.checked || !analysisDirty) {
    return;
  }
  const samples = telemetry.snapshot();
  const index = Number(followSel.value || 0);
  const latest = samples[samples.length - 1];
  analysisAgent.textContent = selectedAgentActive && latest
    ? `${followSel.options[followSel.selectedIndex]?.textContent ?? `car ${index}`}`
    : "Selected agent inactive or unavailable";
  topSpeed.textContent = samples.length
    ? `${telemetry.topSpeed.toFixed(2)} m/s`
    : "—";
  collisionCount.textContent = samples.length
    ? String(telemetry.collisionCount)
    : "—";
  chartSpeed.setAttribute(
    "d",
    hello
      ? chartPath(samples, "speed", 240, 56, [0, hello.speed_axis_max_mps])
      : "",
  );
  chartSteering.setAttribute(
    "d",
    chartPath(samples, "steering", 240, 56, NORMALIZED_ACTION_RANGE),
  );
  chartThrottle.setAttribute(
    "d",
    chartPath(samples, "throttle", 240, 56, PEDAL_RANGE),
  );
  chartBrake.setAttribute(
    "d",
    chartPath(samples, "brake", 240, 56, PEDAL_RANGE),
  );
  chartCollisions.setAttribute(
    "d",
    chartPath(
      samples,
      "collisions",
      240,
      56,
      [0, Math.max(1, telemetry.collisionCount)],
    ),
  );
  analysisDirty = false;
}

function frame(nowMs: number): void {
  const cars = motion.sample(nowMs);
  const playhead = motion.getPlayhead();
  if (cars && playhead !== null) {
    scene.updateCars(cars);
    const selected = cars[Number(followSel.value || 0)];
    renderCurrentControls(selected);
    const active = Boolean(selected && selected[3] > 0);
    if (active !== selectedAgentActive) {
      selectedAgentActive = active;
      analysisDirty = true;
    }
    for (const known of motion.drainKnownThrough(playhead)) {
      scene.pushKnownCars(known.t, known.cars);
      telemetry.push(known.t, known.cars[Number(followSel.value || 0)]);
      analysisDirty = true;
    }
  } else {
    renderCurrentControls(undefined);
  }
  renderAnalysis();
  scene.render();
  requestAnimationFrame(frame);
}

connect();
void refreshCheckpoints();
requestAnimationFrame(frame);
