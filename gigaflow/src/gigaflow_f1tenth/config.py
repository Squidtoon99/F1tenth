"""Versioned experiment config, layout constants, and startup validation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

CONFIG_VERSION = 1

# Sensor / action layout mirrored locally (not imported from libs/f1tenth_policy).
LIDAR_DIM = 1081
PROPRIO_DIM = 16
SENSOR_OBS_DIM = LIDAR_DIM + PROPRIO_DIM
ACTION_DIM = 2
GRU_HIDDEN_DIM = 512
CNN_PROJECTION_DIM = 256
CONTROL_HZ = 10.0
SIM_DT = 0.005
CONTROL_INTERVAL = 20
CAR_LENGTH_M = 0.568
CAR_WIDTH_M = 0.296

BYTES_PER_FLOAT32 = 4
BYTES_PER_INT64 = 8
BYTES_PER_INT32 = 4
BYTES_PER_BOOL = 1
# Privileged compact agent-state / critic element width (shared packing).
# Includes track-local frenet_segment so pack/restore keeps projection lock, and
# the command history / wheel speeds the proprioception channels are rebuilt from.
AGENT_STATE_DIM = 45
CRITIC_ELEMENT_DIM = 45
# Upstream f1tenth_racetracks pin ships this many centerlines (see track_pin.json).
PINNED_UPSTREAM_TRACK_COUNT = 23

# cuBLAS/cuDNN allocate per-handle workspaces through the torch caching allocator
# on first use. They are library internals, not derivable from the config;
# measured at ~17 MB on sm_89, reserved with margin.
CUDA_LIBRARY_WORKSPACE_BYTES = 32_000_000
# Single-step cuDNN GRU scratch, per row of the step batch. Not analytically
# derivable either; measured at <=20 floats per hidden unit on sm_89.
GRU_WORKSPACE_FLOATS_PER_HIDDEN = 40
# Elements a cuDNN GRU step keeps per row for its backward pass.
GRU_BACKWARD_STATE_FLOATS_PER_HIDDEN = 8
# Margin over the derived terms for cuDNN algorithm scratch and caching-allocator
# block rounding, neither of which is a function of the config.
ESTIMATE_SAFETY_MARGIN = 1.15


@dataclass(frozen=True)
class TracksConfig:
    manifest_path: str | None
    num_tracks: int = PINNED_UPSTREAM_TRACK_COUNT
    sampling: str = "stratified_shuffle"
    include_optional_local: bool = False


@dataclass(frozen=True)
class WorldsConfig:
    num_worlds: int
    max_agents_per_world: int
    solo_world_fraction: float = 0.05
    density_bins: tuple[str, ...] = ("sparse", "medium", "dense")
    device: str = "cuda"
    # Immobile "static opponent" slots reserved per world (paper's obstacle
    # agents); they count against max_agents_per_world rather than growing it.
    static_opponents_per_world: int = 0


@dataclass(frozen=True)
class AgentsConfig:
    car_length_m: float = CAR_LENGTH_M
    car_width_m: float = CAR_WIDTH_M
    control_hz: float = CONTROL_HZ
    sim_dt: float = SIM_DT
    control_interval: int = CONTROL_INTERVAL
    lidar_dim: int = LIDAR_DIM
    proprio_dim: int = PROPRIO_DIM
    sensor_obs_dim: int = SENSOR_OBS_DIM
    action_dim: int = ACTION_DIM
    gru_hidden_dim: int = GRU_HIDDEN_DIM
    cnn_projection_dim: int = CNN_PROJECTION_DIM
    actor_mlp_sizes: tuple[int, ...] = (1024, 1024, 1024)
    condition_dim: int = 10
    target_laps: float = 3.0
    reference_speed_mps: float = 3.5
    episode_seconds_min: float = 120.0
    episode_seconds_max: float = 360.0
    async_respawn: bool = True
    actor_variant: str = "gru"
    frame_stack: int = 1


@dataclass(frozen=True)
class RewardConditioningConfig:
    enabled: bool = True
    x_drive_a: float = 1.25
    x_accel_a: float = 1.5
    enable_finish_rank: bool = False
    enable_blocking: bool = False
    enable_zero_sum: bool = False
    # Ablation seams: randomization without conditioning, or fixed style + conditioning.
    randomize_styles: bool = True
    expose_condition_to_actor: bool = True


@dataclass(frozen=True)
class PPOConfig:
    rollout_length: int = 128
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    ent_coef: float | None = 0.01
    ent_coef_initial: float | None = None
    ent_coef_final: float | None = None
    ent_anneal_updates: int | None = None
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    # The paper's 5e-4 is tuned for a 256k-transition feed-forward batch. Our
    # recurrent minibatches are far smaller and correlated within a trajectory,
    # where 5e-4 collapsed twice; 1e-4 is the rate that trained stably here.
    learning_rate: float = 1.0e-4
    num_epochs: int = 3
    minibatch_size: int = 2048
    adaptive_filter_beta: float = 0.25
    adaptive_filter_eta_scale: float = 0.01
    adaptive_filter_enabled: bool = True
    value_clip: bool = False
    amp: bool = True
    total_updates: int = 10_000
    # 0 disables KL early stopping. The paper discloses none, and the check here
    # runs after optimizer.step(), so it reports a bad update instead of
    # preventing one. Smoke/gate configs set it explicitly to exercise the path.
    target_kl: float = 0.0
    actor_kl_soft: float = 0.0
    actor_kl_hard: float = 0.0
    actor_lr_backoff: float = 0.5
    actor_lr_scale_min: float = 0.125
    actor_kl_warmup_updates: int = 200
    actor_kl_stop_window_updates: int = 500
    actor_kl_stop_window_count: int = 5
    actor_lr_recovery_interval_updates: int = 100
    actor_lr_recovery_multiplier: float = 2.0
    # When false, rollback_stop_requested is never raised (telemetry only).
    actor_kl_stop_enabled: bool = True
    actor_checkpoint_interval_updates: int = 0
    full_checkpoint_interval_updates: int = 0
    # Fail-fast if frozen-weight collect/evaluate disagreement exceeds this.
    # pre_kl is the primary learning-staleness signal; logp_delta allows tiny
    # CUDA float32 residue after matching collect/evaluate GRU paths.
    max_pre_update_kl: float = 1.0e-3
    max_logp_delta: float = 5.0e-3


@dataclass(frozen=True)
class EvaluationConfig:
    sync_no_respawn: bool = True
    seeds: tuple[int, ...] = (0, 1, 2)
    suite: tuple[str, ...] = ("solo", "head_to_head", "dense")
    conservative_deployment_style: str = "centered_high_collision"
    soak_steps: int = 2_000
    viz_enabled: bool = True
    viz_fps: int = 10
    viz_max_frames: int = 240
    incident_window_steps: int = 32
    # Cadenced/offline eval device. None inherits worlds.device. Use "cpu" for
    # async subprocess eval that allocates no CUDA memory during training.
    device: str | None = None
    # Eval-only world count. None inherits worlds.num_worlds. Training scale is
    # always worlds.num_worlds and is never reduced by this field.
    num_worlds: int | None = None
    behavior_feasibility_passes: int = 2
    behavior_no_feasibility_stop_updates: int = 1000
    # When false, behavioral/KL gate signals are logged only; training stops on
    # budget, operator signal, or non-finite/runtime failure.
    quality_stop_enabled: bool = True


@dataclass(frozen=True)
class AblationConfig:
    """Optional architecture / reward / filter seams (defaults = paper baseline)."""

    actor_variant: str = "gru"
    frame_stack: int = 1
    centralized_critic: bool = True
    adaptive_filter_enabled: bool = True
    surprise_braking: bool = False
    reward_randomization: bool = True
    condition_to_actor: bool = True


@dataclass(frozen=True)
class ProfilingConfig:
    enabled: bool = True
    estimate_bytes_budget: int = 12_000_000_000
    report_interval_updates: int = 10


@dataclass(frozen=True)
class WandbConfig:
    """Optional Weights & Biases logging (disabled by default; never stores API keys)."""

    enabled: bool = False
    mode: str = "online"
    entity: str | None = None
    project: str = "f1tenth-gigaflow"
    group: str | None = None
    name: str | None = None
    tags: tuple[str, ...] = ()
    notes: str | None = None
    run_id: str | None = None
    resume: str = "allow"
    # Mid-train eval cadence (0 = train metrics only; evaluate CLI still logs).
    eval_interval_updates: int = 0
    log_artifacts: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    config_version: int
    seed: int
    tracks: TracksConfig
    worlds: WorldsConfig
    agents: AgentsConfig
    reward_conditioning: RewardConditioningConfig
    ppo: PPOConfig
    evaluation: EvaluationConfig
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    ablations: AblationConfig = field(default_factory=AblationConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


class ConfigError(ValueError):
    """Raised when an experiment config fails validation."""


def _as_tuple(value: Any) -> tuple:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    raise ConfigError(f"expected list/tuple, got {type(value).__name__}")


def _from_mapping(cls: type, data: Mapping[str, Any]):
    if not isinstance(data, Mapping):
        raise ConfigError(f"{cls.__name__} requires a mapping")
    kwargs: dict[str, Any] = {}
    valid = {f.name for f in fields(cls)}
    unknown = set(data) - valid
    if unknown:
        raise ConfigError(f"{cls.__name__} unknown keys: {sorted(unknown)}")
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        if f.name in {
            "density_bins",
            "actor_mlp_sizes",
            "seeds",
            "suite",
            "tags",
        }:
            value = _as_tuple(value)
        kwargs[f.name] = value
    return cls(**kwargs)


def config_from_dict(
    data: Mapping[str, Any], *, check_training_budget: bool = True
) -> ExperimentConfig:
    if not isinstance(data, Mapping):
        raise ConfigError("config root must be a mapping")
    required = (
        "config_version",
        "seed",
        "tracks",
        "worlds",
        "agents",
        "reward_conditioning",
        "ppo",
        "evaluation",
    )
    missing = [k for k in required if k not in data]
    if missing:
        raise ConfigError(f"missing config sections: {missing}")
    profiling_raw = data.get("profiling", {})
    ablations_raw = data.get("ablations", {})
    wandb_raw = data.get("wandb", {})
    cfg = ExperimentConfig(
        config_version=int(data["config_version"]),
        seed=int(data["seed"]),
        tracks=_from_mapping(TracksConfig, data["tracks"]),
        worlds=_from_mapping(WorldsConfig, data["worlds"]),
        agents=_from_mapping(AgentsConfig, data["agents"]),
        reward_conditioning=_from_mapping(
            RewardConditioningConfig, data["reward_conditioning"]
        ),
        ppo=_from_mapping(PPOConfig, data["ppo"]),
        evaluation=_from_mapping(EvaluationConfig, data["evaluation"]),
        profiling=_from_mapping(ProfilingConfig, profiling_raw),
        ablations=_from_mapping(AblationConfig, ablations_raw),
        wandb=_from_mapping(WandbConfig, wandb_raw),
    )
    validate_config(cfg, check_training_budget=check_training_budget)
    return cfg


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if raw is None:
        raise ConfigError(f"empty config: {path}")
    return config_from_dict(raw)


def _agent_steps(cfg: ExperimentConfig) -> int:
    n_slots = cfg.worlds.num_worlds * cfg.worlds.max_agents_per_world
    return n_slots * cfg.ppo.rollout_length


def rollout_buffer_bytes(cfg: ExperimentConfig) -> int:
    """Bytes of every tensor ``TensorRolloutBuffer`` preallocates."""
    a = cfg.agents
    n_slots = cfg.worlds.num_worlds * cfg.worlds.max_agents_per_world
    per_agent_step = BYTES_PER_FLOAT32 * (
        2 * AGENT_STATE_DIM  # state, next_state
        + 2 * a.action_dim  # actions, pre_tanh
        + a.condition_dim
        + 2  # rewards, old_logp
    )
    per_agent_step += 4 * BYTES_PER_BOOL  # valid, done, timeout, reset_mask
    per_agent_step += 2 * BYTES_PER_INT64  # sensor_noise_seed, obs_digest
    per_agent_step += 2 * BYTES_PER_INT32  # episode_id, episode_step
    per_agent = BYTES_PER_INT32 + BYTES_PER_FLOAT32 * a.gru_hidden_dim
    filled = cfg.ppo.rollout_length * BYTES_PER_BOOL
    return _agent_steps(cfg) * per_agent_step + n_slots * per_agent + filled


def critic_set_bytes(cfg: ExperimentConfig) -> int:
    """One packed opponent set ``[T, S, N, D]`` plus its ``[T, S, N]`` mask."""
    n_others = max(cfg.worlds.max_agents_per_world - 1, 0)
    cells = _agent_steps(cfg) * n_others
    return cells * (CRITIC_ELEMENT_DIM * BYTES_PER_FLOAT32 + BYTES_PER_BOOL)


def prepared_inputs_bytes(cfg: ExperimentConfig) -> int:
    """Bytes of ``PreparedPPOInputs``: rebuilt observations and critic features.

    Observations are no longer stored per rollout, so this reconstructed
    ``[T, S, sensor_obs_dim]`` tensor is one full-width copy. ``ego_state``
    concatenates an ordered track preview onto the compact state (see
    ``critic.pack_critic_features``), so unlike the compact state it stores, it
    no longer aliases anything and is a second full-width copy.
    """
    arch = _architecture_constants()
    cells = _agent_steps(cfg)
    obs = cells * cfg.agents.sensor_obs_dim * BYTES_PER_FLOAT32
    ego_state = cells * arch["critic_ego_state_dim"] * BYTES_PER_FLOAT32
    bootstrap_values = cells * BYTES_PER_FLOAT32
    return obs + ego_state + critic_set_bytes(cfg) + bootstrap_values


def ppo_update_bytes(cfg: ExperimentConfig) -> int:
    """Bytes one PPO update adds on top of the rollout and prepared inputs.

    ``update`` sanitizes the critic features into fresh copies and keeps four
    ``[T, S]`` float scalars plus a keep mask. Sequence scoring also needs the
    observations laid out as ``[B, T, obs]``: once over every kept slot (the logp
    rescore and the parity gate), and once per minibatch.
    """
    arch = _architecture_constants()
    n_slots = cfg.worlds.num_worlds * cfg.worlds.max_agents_per_world
    cells = _agent_steps(cfg)
    obs_dim = cfg.agents.sensor_obs_dim
    sanitized = (
        cells * arch["critic_ego_state_dim"] * BYTES_PER_FLOAT32
        + critic_set_bytes(cfg)
    )
    scalars = cells * (4 * BYTES_PER_FLOAT32 + BYTES_PER_BOOL)
    all_slot_obs = cells * obs_dim * BYTES_PER_FLOAT32
    mb_agents = min(
        n_slots, max(1, cfg.ppo.minibatch_size // max(1, cfg.ppo.rollout_length))
    )
    minibatch_obs = (
        mb_agents * cfg.ppo.rollout_length * obs_dim * BYTES_PER_FLOAT32
    )
    return sanitized + scalars + all_slot_obs + minibatch_obs


def _architecture_constants() -> dict[str, Any]:
    """Layer geometry read from the modules themselves, never mirrored here."""
    from gigaflow_f1tenth.critic import (
        CRITIC_CONDITION_EMBED,
        CRITIC_EGO_STATE_DIM,
        CRITIC_ELEMENT_HIDDEN,
        DEFAULT_CRITIC_MLP,
    )
    from gigaflow_f1tenth.model import (
        CNN_CONV_CHANNELS,
        CNN_KERNELS,
        CNN_PADDING,
        CNN_STRIDES,
        CONDITION_EMBED_DIM,
        LIDAR_POOL_BINS,
    )

    return {
        "channels": CNN_CONV_CHANNELS,
        "kernels": CNN_KERNELS,
        "strides": CNN_STRIDES,
        "padding": CNN_PADDING,
        "pool_bins": LIDAR_POOL_BINS,
        "condition_embed": CONDITION_EMBED_DIM,
        "critic_hidden": CRITIC_ELEMENT_HIDDEN,
        "critic_condition_embed": CRITIC_CONDITION_EMBED,
        "critic_mlp": DEFAULT_CRITIC_MLP,
        "critic_ego_state_dim": CRITIC_EGO_STATE_DIM,
    }


def _conv_output_lengths(cfg: ExperimentConfig, arch: dict[str, Any]) -> list[int]:
    lengths = []
    length = cfg.agents.lidar_dim
    for kernel, stride, pad in zip(arch["kernels"], arch["strides"], arch["padding"]):
        length = (length + 2 * pad - kernel) // stride + 1
        lengths.append(length)
    return lengths


def _actor_activation_floats(
    cfg: ExperimentConfig, arch: dict[str, Any]
) -> list[int]:
    """Elements of every tensor one agent-step produces in the actor, in order."""
    a = cfg.agents
    embed = arch["condition_embed"]
    floats = [a.lidar_dim]  # conv1 needs the sliced LiDAR window contiguous
    for channels, length in zip(
        arch["channels"], _conv_output_lengths(cfg, arch)
    ):
        floats += [channels * length, channels * length]  # conv, relu
    floats.append(arch["channels"][-1] * arch["pool_bins"])
    floats += [a.cnn_projection_dim, a.cnn_projection_dim]  # projection, relu
    floats += [embed] * 4  # condition encoder: two linear+relu pairs
    floats.append(a.cnn_projection_dim + a.proprio_dim + embed)  # trunk cat
    floats += [
        a.gru_hidden_dim,
        GRU_BACKWARD_STATE_FLOATS_PER_HIDDEN * a.gru_hidden_dim,
    ]
    for width in a.actor_mlp_sizes:
        floats += [width, width]  # linear, relu
    # mu, clamped log_std, std, pre_tanh, log_prob terms, entropy.
    floats += [a.action_dim] * 6
    return floats


def _critic_activation_floats(
    cfg: ExperimentConfig, arch: dict[str, Any]
) -> list[int]:
    """Elements of every tensor one scored row produces in the critic, in order."""
    hidden = arch["critic_hidden"]
    embed = arch["critic_condition_embed"]
    n_others = max(cfg.worlds.max_agents_per_world - 1, 1)
    floats = [n_others * hidden] * 4  # element encoder: two linear+relu pairs
    floats.append(n_others * hidden)  # masked_fill copy before the max pool
    floats += [hidden, hidden]  # pooled, all-inactive where()
    floats += [embed] * 4  # condition encoder
    floats.append(arch["critic_ego_state_dim"] + hidden + embed)  # backbone input cat
    for width in arch["critic_mlp"]:
        floats += [width, width]  # linear, relu
    floats.append(1)
    return floats


def _forward_live_floats(chain: list[int]) -> int:
    """Largest pair of consecutive tensors — the live set of a no-grad forward."""
    return max(x + y for x, y in zip(chain, chain[1:]))


def full_rollout_scoring_bytes(cfg: ExperimentConfig) -> int:
    """Activations of the no-grad passes that score the whole rollout at once.

    Rescoring in ``reconstruct_prepared``, the collect/evaluate parity gate, and
    the critic's ``old_values`` / bootstrap passes all flatten ``[T, S]`` into one
    batch, so their activations scale with every agent-step in the rollout — the
    same dimensions the buffer terms scale with, and the largest single
    allocation training makes.
    """
    arch = _architecture_constants()
    a = cfg.agents
    cells = _agent_steps(cfg)
    n_slots = cfg.worlds.num_worlds * cfg.worlds.max_agents_per_world
    actor_chain = _actor_activation_floats(cfg, arch)
    # The GRU walks the rollout one step at a time, so its scratch scales with
    # the step batch while the CNN sees every agent-step at once.
    actor = cells * (
        _forward_live_floats(actor_chain)
        + a.cnn_projection_dim
        + a.proprio_dim
        + arch["condition_embed"]
        + a.gru_hidden_dim
    ) + n_slots * GRU_WORKSPACE_FLOATS_PER_HIDDEN * a.gru_hidden_dim
    critic_chain = _critic_activation_floats(cfg, arch)
    critic = cells * (
        _forward_live_floats(critic_chain)
        + arch["critic_hidden"]
        + arch["critic_ego_state_dim"]
        + arch["critic_condition_embed"]
    )
    # The bootstrap pass packs a second opponent set from the next states while
    # the one in PreparedPPOInputs is still alive.
    return max(
        BYTES_PER_FLOAT32 * actor,
        BYTES_PER_FLOAT32 * critic + critic_set_bytes(cfg),
    )


def ppo_minibatch_backward_bytes(cfg: ExperimentConfig) -> int:
    """Activations one PPO minibatch keeps alive until its backward pass.

    Epoch count does not enter: each minibatch frees its graph before the next
    one is built, so the peak is set by the minibatch shape alone.
    """
    arch = _architecture_constants()
    n_slots = cfg.worlds.num_worlds * cfg.worlds.max_agents_per_world
    mb_agents = min(
        n_slots, max(1, cfg.ppo.minibatch_size // max(1, cfg.ppo.rollout_length))
    )
    cells = mb_agents * cfg.ppo.rollout_length
    retained = sum(_actor_activation_floats(cfg, arch)) + sum(
        _critic_activation_floats(cfg, arch)
    )
    workspace = (
        GRU_WORKSPACE_FLOATS_PER_HIDDEN * cfg.agents.gru_hidden_dim * mb_agents
    )
    return BYTES_PER_FLOAT32 * (cells * retained + workspace)


def _linear_params(in_dim: int, out_dim: int) -> int:
    return in_dim * out_dim + out_dim


def actor_parameter_count(cfg: ExperimentConfig) -> int:
    """Parameters of the recurrent baseline actor."""
    arch = _architecture_constants()
    a = cfg.agents
    embed = arch["condition_embed"]
    total = 0
    in_channels = 1
    for channels, kernel in zip(arch["channels"], arch["kernels"]):
        total += in_channels * channels * kernel + channels
        in_channels = channels
    total += _linear_params(
        arch["channels"][-1] * arch["pool_bins"], a.cnn_projection_dim
    )
    total += _linear_params(a.condition_dim, embed) + _linear_params(embed, embed)
    gru_in = a.cnn_projection_dim + a.proprio_dim + embed
    total += 3 * a.gru_hidden_dim * (gru_in + a.gru_hidden_dim + 2)
    widths = [a.gru_hidden_dim, *a.actor_mlp_sizes]
    for in_dim, out_dim in zip(widths, widths[1:]):
        total += _linear_params(in_dim, out_dim)
    total += 2 * _linear_params(a.actor_mlp_sizes[-1], a.action_dim)
    return total


def critic_parameter_count(cfg: ExperimentConfig) -> int:
    arch = _architecture_constants()
    hidden = arch["critic_hidden"]
    embed = arch["critic_condition_embed"]
    total = _linear_params(CRITIC_ELEMENT_DIM, hidden) + _linear_params(hidden, hidden)
    total += _linear_params(cfg.agents.condition_dim, embed)
    total += _linear_params(embed, embed)
    widths = [arch["critic_ego_state_dim"] + hidden + embed, *arch["critic_mlp"], 1]
    for in_dim, out_dim in zip(widths, widths[1:]):
        total += _linear_params(in_dim, out_dim)
    return total


def model_state_bytes(cfg: ExperimentConfig) -> int:
    """Parameters, gradients, and both Adam moments for the actor and critic.

    ``nn.GRU`` additionally keeps a flattened copy of its own weights for cuDNN.
    """
    a = cfg.agents
    embed = _architecture_constants()["condition_embed"]
    gru_in = a.cnn_projection_dim + a.proprio_dim + embed
    gru_params = 3 * a.gru_hidden_dim * (gru_in + a.gru_hidden_dim + 2)
    params = actor_parameter_count(cfg) + critic_parameter_count(cfg)
    return BYTES_PER_FLOAT32 * (4 * params + gru_params)


def estimate_memory_bytes(cfg: ExperimentConfig) -> int:
    """Peak allocated bytes of one training update, for the startup budget.

    Resident data tensors and model state are always live; the two activation
    phases are sequential, so only the larger of them can be at the peak.

    The terms predict ``torch.cuda.max_memory_allocated``. Actual VRAM use is
    higher: the CUDA context, the Warp simulator arrays, and caching-allocator
    block rounding all sit outside the torch allocator's accounting, so a budget
    must stay well under the card. ``ESTIMATE_SAFETY_MARGIN`` only covers what
    varies per kernel choice (cuDNN algorithm scratch, allocator rounding);
    everything that scales with worlds, agents, or rollout length is derived.
    """
    resident = (
        rollout_buffer_bytes(cfg)
        + prepared_inputs_bytes(cfg)
        + ppo_update_bytes(cfg)
        + model_state_bytes(cfg)
        + CUDA_LIBRARY_WORKSPACE_BYTES
    )
    activations = max(
        full_rollout_scoring_bytes(cfg), ppo_minibatch_backward_bytes(cfg)
    )
    return int(ESTIMATE_SAFETY_MARGIN * (resident + activations))


def episode_seconds_for_track_length(cfg: ExperimentConfig, track_length_m: float) -> float:
    agents = cfg.agents
    raw = agents.target_laps * track_length_m / agents.reference_speed_mps
    return float(min(agents.episode_seconds_max, max(agents.episode_seconds_min, raw)))


def episode_steps(cfg: ExperimentConfig, track_length_m: float) -> int:
    seconds = episode_seconds_for_track_length(cfg, track_length_m)
    return int(math.ceil(seconds * cfg.agents.control_hz))


def validate_config(cfg: ExperimentConfig, *, check_training_budget: bool = True) -> None:
    """Validate a config; ``check_training_budget`` covers PPO-update sizing only.

    Evaluation builds a simulator from a world layout that never feeds a PPO
    update, so the minibatch/transition and rollout-memory budgets do not apply
    to it. Everything else is checked either way.
    """
    if not is_dataclass(cfg):
        raise ConfigError("config must be an ExperimentConfig dataclass")
    if cfg.config_version != CONFIG_VERSION:
        raise ConfigError(
            f"unsupported config_version={cfg.config_version}; "
            f"expected {CONFIG_VERSION}"
        )
    a = cfg.agents
    w = cfg.worlds
    p = cfg.ppo

    if a.lidar_dim != LIDAR_DIM or a.proprio_dim != PROPRIO_DIM:
        raise ConfigError(
            "sensor layout must remain "
            f"{LIDAR_DIM}+{PROPRIO_DIM}={SENSOR_OBS_DIM} for deploy parity"
        )
    if a.lidar_dim + a.proprio_dim != a.sensor_obs_dim:
        raise ConfigError(
            f"lidar_dim+proprio_dim={a.lidar_dim + a.proprio_dim} != "
            f"sensor_obs_dim={a.sensor_obs_dim}"
        )
    if a.sensor_obs_dim != SENSOR_OBS_DIM or a.action_dim != ACTION_DIM:
        raise ConfigError("sensor_obs_dim/action_dim diverge from package layout")
    if a.control_interval <= 0 or a.sim_dt <= 0.0 or a.control_hz <= 0.0:
        raise ConfigError("control_interval, sim_dt, and control_hz must be positive")
    if a.control_interval > CONTROL_INTERVAL:
        raise ConfigError(
            f"control_interval={a.control_interval} exceeds fused max "
            f"{CONTROL_INTERVAL}"
        )
    control_dt = a.control_interval * a.sim_dt
    expected_dt = 1.0 / a.control_hz
    if abs(control_dt - expected_dt) > 1e-9:
        raise ConfigError(
            f"control_interval*sim_dt={control_dt} != 1/control_hz={expected_dt}"
        )
    if a.car_length_m <= 0.0 or a.car_width_m <= 0.0:
        raise ConfigError("car dimensions must be positive")
    if a.condition_dim <= 0 or a.gru_hidden_dim <= 0:
        raise ConfigError("condition_dim and gru_hidden_dim must be positive")
    if len(a.actor_mlp_sizes) < 1:
        raise ConfigError("actor_mlp_sizes must be non-empty")
    if a.episode_seconds_min <= 0.0 or a.episode_seconds_max < a.episode_seconds_min:
        raise ConfigError("invalid episode_seconds_min/max")
    if a.reference_speed_mps <= 0.0 or a.target_laps <= 0.0:
        raise ConfigError("reference_speed_mps and target_laps must be positive")

    if w.num_worlds <= 0 or w.max_agents_per_world <= 0:
        raise ConfigError("num_worlds and max_agents_per_world must be positive")
    if not 0.0 <= w.solo_world_fraction <= 1.0:
        raise ConfigError("solo_world_fraction must be in [0, 1]")
    if not w.density_bins:
        raise ConfigError("density_bins must be non-empty")
    if w.static_opponents_per_world < 0:
        raise ConfigError("static_opponents_per_world must be non-negative")
    if w.static_opponents_per_world >= w.max_agents_per_world:
        raise ConfigError(
            "static_opponents_per_world must leave room for at least one "
            f"learner slot (max_agents_per_world={w.max_agents_per_world})"
        )

    if cfg.tracks.num_tracks <= 0:
        raise ConfigError("num_tracks must be positive")
    if cfg.tracks.sampling not in {"stratified_shuffle", "balanced_length"}:
        raise ConfigError(f"unsupported track sampling: {cfg.tracks.sampling}")

    rc = cfg.reward_conditioning
    if rc.x_drive_a < 1.0 or rc.x_accel_a < 1.0:
        raise ConfigError("reward conditioning X(a) scale factors must be >= 1")

    if p.rollout_length <= 0 or p.num_epochs <= 0 or p.minibatch_size <= 0:
        raise ConfigError("ppo rollout_length/num_epochs/minibatch_size must be > 0")
    if not 0.0 < p.gamma <= 1.0:
        raise ConfigError("ppo.gamma must be in (0, 1]")
    if not 0.0 <= p.gae_lambda <= 1.0:
        raise ConfigError("ppo.gae_lambda must be in [0, 1]")
    if p.clip_ratio <= 0.0 or p.learning_rate <= 0.0:
        raise ConfigError("ppo clip_ratio and learning_rate must be positive")
    schedule_values = (
        p.ent_coef_initial,
        p.ent_coef_final,
        p.ent_anneal_updates,
    )
    if any(value is not None for value in schedule_values):
        if not all(value is not None for value in schedule_values):
            raise ConfigError(
                "ppo entropy schedule requires ent_coef_initial, "
                "ent_coef_final, and ent_anneal_updates"
            )
        if p.ent_coef is not None:
            raise ConfigError(
                "ppo.ent_coef is a legacy fixed coefficient and cannot be combined "
                "with the entropy schedule"
            )
        assert p.ent_coef_initial is not None
        assert p.ent_coef_final is not None
        assert p.ent_anneal_updates is not None
        if p.ent_coef_initial < 0.0 or p.ent_coef_final < 0.0:
            raise ConfigError("ppo entropy coefficients must be non-negative")
        if p.ent_coef_initial < p.ent_coef_final:
            raise ConfigError("ppo.ent_coef_initial must be >= ent_coef_final")
        if p.ent_anneal_updates <= 0:
            raise ConfigError("ppo.ent_anneal_updates must be positive")
    elif p.ent_coef is None or p.ent_coef < 0.0:
        raise ConfigError(
            "ppo.ent_coef must be a non-negative legacy coefficient when no "
            "entropy schedule is configured"
        )
    if p.value_clip:
        raise ConfigError("value_clip must be false (paper-aligned PPO)")
    if not 0.0 < p.adaptive_filter_beta <= 1.0:
        raise ConfigError("adaptive_filter_beta must be in (0, 1]")
    if p.adaptive_filter_eta_scale < 0.0:
        raise ConfigError("adaptive_filter_eta_scale must be non-negative")
    if p.total_updates <= 0:
        raise ConfigError("ppo.total_updates must be positive")
    if p.target_kl < 0.0:
        raise ConfigError("ppo.target_kl must be non-negative")
    if p.actor_kl_soft < 0.0 or p.actor_kl_hard < 0.0:
        raise ConfigError("ppo actor KL thresholds must be non-negative")
    if (p.actor_kl_soft > 0.0 or p.actor_kl_hard > 0.0) and not (
        0.0 < p.actor_kl_soft <= p.actor_kl_hard
    ):
        raise ConfigError(
            "ppo actor KL thresholds require 0 < actor_kl_soft <= actor_kl_hard"
        )
    if not 0.0 < p.actor_lr_backoff < 1.0:
        raise ConfigError("ppo.actor_lr_backoff must be in (0, 1)")
    if not 0.0 < p.actor_lr_scale_min <= 1.0:
        raise ConfigError("ppo.actor_lr_scale_min must be in (0, 1]")
    if p.actor_kl_warmup_updates < 0:
        raise ConfigError("ppo.actor_kl_warmup_updates must be non-negative")
    if p.actor_kl_stop_window_updates <= 0:
        raise ConfigError("ppo.actor_kl_stop_window_updates must be positive")
    if p.actor_kl_stop_window_count <= 0:
        raise ConfigError("ppo.actor_kl_stop_window_count must be positive")
    if p.actor_lr_recovery_interval_updates <= 0:
        raise ConfigError("ppo.actor_lr_recovery_interval_updates must be positive")
    if not 1.0 < p.actor_lr_recovery_multiplier <= 2.0:
        raise ConfigError(
            "ppo.actor_lr_recovery_multiplier must be in (1, 2]"
        )
    if (
        p.actor_checkpoint_interval_updates < 0
        or p.full_checkpoint_interval_updates < 0
    ):
        raise ConfigError("ppo checkpoint intervals must be non-negative")
    if p.max_pre_update_kl < 0.0:
        raise ConfigError("ppo.max_pre_update_kl must be non-negative")
    if p.max_logp_delta < 0.0:
        raise ConfigError("ppo.max_logp_delta must be non-negative")

    if a.actor_variant not in {"gru", "feedforward", "frame_stack"}:
        raise ConfigError(f"unsupported actor_variant: {a.actor_variant}")
    if a.actor_variant == "frame_stack" and a.frame_stack not in (4, 8):
        raise ConfigError("frame_stack actor_variant requires frame_stack in {4, 8}")
    if a.actor_variant != "frame_stack" and a.frame_stack != 1:
        raise ConfigError("frame_stack must be 1 unless actor_variant=frame_stack")

    ab = cfg.ablations
    if ab.actor_variant not in {"gru", "feedforward", "frame_stack"}:
        raise ConfigError(f"unsupported ablations.actor_variant: {ab.actor_variant}")
    if ab.frame_stack not in (1, 4, 8):
        raise ConfigError("ablations.frame_stack must be 1, 4, or 8")

    budget = cfg.profiling.estimate_bytes_budget
    if budget <= 0:
        raise ConfigError("profiling.estimate_bytes_budget must be positive")

    if check_training_budget:
        n_slots = w.num_worlds * w.max_agents_per_world
        transitions = n_slots * p.rollout_length
        if p.minibatch_size > transitions:
            raise ConfigError(
                f"minibatch_size={p.minibatch_size} exceeds "
                f"worlds*agents*rollout_length={transitions}"
            )
        est = estimate_memory_bytes(cfg)
        if est > budget:
            raise ConfigError(
                f"estimated memory {est} bytes exceeds budget {budget} bytes "
                f"(worlds={w.num_worlds}, agents/world={w.max_agents_per_world}, "
                f"rollout={p.rollout_length}, minibatch={p.minibatch_size})"
            )

    if not cfg.evaluation.seeds:
        raise ConfigError("evaluation.seeds must be non-empty")
    if not cfg.evaluation.suite:
        raise ConfigError("evaluation.suite must be non-empty")
    if cfg.evaluation.soak_steps <= 0:
        raise ConfigError("evaluation.soak_steps must be positive")
    if cfg.evaluation.viz_fps <= 0 or cfg.evaluation.viz_max_frames <= 0:
        raise ConfigError("evaluation viz_fps/viz_max_frames must be positive")
    if cfg.evaluation.num_worlds is not None and cfg.evaluation.num_worlds <= 0:
        raise ConfigError("evaluation.num_worlds must be positive when set")
    if cfg.evaluation.behavior_feasibility_passes <= 0:
        raise ConfigError("evaluation.behavior_feasibility_passes must be positive")
    if cfg.evaluation.behavior_no_feasibility_stop_updates <= 0:
        raise ConfigError(
            "evaluation.behavior_no_feasibility_stop_updates must be positive"
        )
    if cfg.evaluation.device is not None:
        dev = str(cfg.evaluation.device)
        if not (dev == "cpu" or dev == "cuda" or dev.startswith("cuda:")):
            raise ConfigError(
                f"unsupported evaluation.device={dev!r}; expected cpu|cuda|cuda:N"
            )
    known_suites = {
        "solo",
        "head_to_head",
        "dense",
        "surprise_braking",
        "all_tracks",
        "conservative_longform",
    }
    unknown = [s for s in cfg.evaluation.suite if s not in known_suites]
    if unknown:
        raise ConfigError(f"unknown evaluation suite entries: {unknown}")

    wb = cfg.wandb
    if wb.mode not in {"online", "offline", "disabled"}:
        raise ConfigError(
            f"unsupported wandb.mode={wb.mode!r}; expected online|offline|disabled"
        )
    if wb.resume not in {"allow", "must", "never", "auto"}:
        raise ConfigError(
            f"unsupported wandb.resume={wb.resume!r}; expected allow|must|never|auto"
        )
    if wb.enabled and wb.mode != "disabled":
        if not str(wb.project).strip():
            raise ConfigError("wandb.project must be non-empty when wandb is enabled")
    if any(not isinstance(t, str) or not t.strip() for t in wb.tags):
        raise ConfigError("wandb.tags must be non-empty strings")
    if wb.eval_interval_updates < 0:
        raise ConfigError("wandb.eval_interval_updates must be non-negative")


def replace_wandb_config(
    cfg: ExperimentConfig,
    *,
    enabled: bool | None = None,
    mode: str | None = None,
    entity: str | None = None,
    project: str | None = None,
    group: str | None = None,
    name: str | None = None,
    tags: Sequence[str] | None = None,
    notes: str | None = None,
    run_id: str | None = None,
    resume: str | None = None,
) -> ExperimentConfig:
    """Return a copy of cfg with selected wandb fields overridden."""
    raw = config_to_dict(cfg)
    wb = dict(raw.get("wandb", {}))
    if enabled is not None:
        wb["enabled"] = bool(enabled)
    if mode is not None:
        wb["mode"] = str(mode)
    if entity is not None:
        wb["entity"] = entity
    if project is not None:
        wb["project"] = str(project)
    if group is not None:
        wb["group"] = group
    if name is not None:
        wb["name"] = name
    if tags is not None:
        wb["tags"] = list(tags)
    if notes is not None:
        wb["notes"] = notes
    if run_id is not None:
        wb["run_id"] = run_id
    if resume is not None:
        wb["resume"] = str(resume)
    raw["wandb"] = wb
    return config_from_dict(raw)


def config_to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    def _convert(obj: Any) -> Any:
        if is_dataclass(obj):
            return {f.name: _convert(getattr(obj, f.name)) for f in fields(obj)}
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, Mapping):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
            return [_convert(v) for v in obj]
        return obj

    return _convert(cfg)
