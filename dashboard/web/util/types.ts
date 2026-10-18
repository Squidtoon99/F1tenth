export type Stat = {
  label: string;
  value: number | null;
  unit: string;
  delta: number | null;
  versus: string | null;
};

export type Overview = {
  stats: Stat[];
  laps: { label: string; lap: number; verdict: string }[];
  episode: { run_name: string; marks: { step: number; seconds: number }[] } | null;
  gauge: { used: number; budget: number; label: string } | null;
  scenarios: { name: string; fraction: number }[];
  crash: { rate: number | null; label: string | null };
  best: {
    name: string;
    lap: number;
    completion: number;
    checkpoint_id: string | null;
    staged: boolean;
  } | null;
  live?: string[];
};

export type VideoRow = {
  id: string;
  kind: string;
  duration_s: number | null;
  width: number | null;
  height: number | null;
  run_name: string | null;
  step: number | null;
  stream: string;
  run_id?: string;
};

export type LiveState = {
  state: string | null;
  source: string | null;
  transitions: number | null;
  completion: number | null;
  crash: number | null;
  lifespan: number | null;
} | null;

export type TreeNode = {
  id: string;
  slug: string;
  name: string;
  parent_id: string | null;
  parent_slug?: string | null;
  one_change: string;
  verdict: string;
  lap: number | null;
  completion?: number | null;
  run_id: string;
  wandb_run_id?: string;
  live?: LiveState;
  video?: VideoRow | null;
};

export type RunRow = {
  id: string;
  name: string;
  slug: string;
  verdict: string;
  wandb_run_id: string;
  wandb_url: string | null;
  seed: number | null;
  plant: string | null;
  transitions: number | null;
  stop_reason: string | null;
  budget: number | null;
  one_change: string;
  result?: string;
  live?: LiveState;
};

export type EvalRow = {
  id: string;
  scenario: string;
  flying_lap_s: number | null;
  completion: number | null;
  crash: number | null;
  overtakes: number | null;
  steer_reversal: number | null;
  spawn_mps: number | null;
  step: number;
  is_best: boolean;
  is_final: boolean;
  run_name?: string;
  run_id?: string;
};

export type CheckpointRow = {
  id: string;
  step: number;
  sha256: string | null;
  bytes: number | null;
  is_best: boolean;
  is_final: boolean;
  run_name: string;
  run_id: string;
  staged: boolean;
  key: string;
};

export type ModelChannel = {
  name: string;
  size: number | null;
  note: string;
};

export type TimelineItem = {
  slug: string;
  name: string;
  one_change: string;
  result: string;
  verdict: string;
  lap: number | null;
  run_id: string;
};

export type RunDetail = {
  run: RunRow;
  experiment: { name: string; verdict: string; one_change: string; kind: string };
  evals: EvalRow[];
  videos: VideoRow[];
  model_io: {
    kind: string;
    inputs: ModelChannel[];
    outputs: ModelChannel[];
    actor_dim: number | null;
    action_dim: number | null;
  };
  timeline: TimelineItem[];
  series: {
    laps: { step: number; lap: number }[];
    episode: { step: number; seconds: number }[];
  };
  chart: { title: string; src: string; wandb_url: string | null };
};
