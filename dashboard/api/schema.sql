CREATE TABLE IF NOT EXISTS experiments (
    id uuid PRIMARY KEY,
    slug text UNIQUE NOT NULL,
    name text NOT NULL,
    parent_id uuid REFERENCES experiments (id),
    kind text NOT NULL,
    one_change text NOT NULL,
    verdict text NOT NULL,
    created_at timestamptz NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id uuid PRIMARY KEY,
    experiment_id uuid NOT NULL REFERENCES experiments (id),
    wandb_run_id text UNIQUE NOT NULL,
    wandb_url text,
    seed int,
    plant text,
    transitions bigint,
    stop_reason text,
    started_at timestamptz,
    ended_at timestamptz
);

CREATE TABLE IF NOT EXISTS checkpoints (
    id uuid PRIMARY KEY,
    run_id uuid NOT NULL REFERENCES runs (id),
    step bigint NOT NULL,
    s3_key text,
    sha256 text,
    bytes bigint,
    is_best bool NOT NULL DEFAULT false,
    is_final bool NOT NULL DEFAULT false,
    disk_path text,
    UNIQUE (run_id, step)
);

CREATE TABLE IF NOT EXISTS evals (
    id uuid PRIMARY KEY,
    checkpoint_id uuid NOT NULL REFERENCES checkpoints (id),
    scenario text NOT NULL,
    flying_lap_s double precision,
    completion double precision,
    crash double precision,
    overtakes double precision,
    steer_reversal double precision,
    spawn_mps double precision
);

CREATE TABLE IF NOT EXISTS videos (
    id uuid PRIMARY KEY,
    checkpoint_id uuid REFERENCES checkpoints (id),
    eval_id uuid REFERENCES evals (id),
    kind text NOT NULL,
    s3_key text,
    duration_s double precision,
    width int,
    height int,
    disk_path text
);

CREATE TABLE IF NOT EXISTS notes (
    id uuid PRIMARY KEY,
    body text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS experiments_parent_id_idx ON experiments (parent_id);
CREATE INDEX IF NOT EXISTS runs_experiment_id_idx ON runs (experiment_id);
CREATE INDEX IF NOT EXISTS checkpoints_run_id_idx ON checkpoints (run_id);
CREATE INDEX IF NOT EXISTS evals_checkpoint_id_idx ON evals (checkpoint_id);
CREATE INDEX IF NOT EXISTS videos_checkpoint_id_idx ON videos (checkpoint_id);
