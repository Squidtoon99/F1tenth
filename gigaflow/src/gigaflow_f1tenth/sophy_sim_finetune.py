"""KL-clipped PPO fine-tune of a baked Gigaflow actor in the Sophy Warp env."""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from gigaflow_f1tenth.artifacts import FileArtifactStore, bake_actor_artifact
from gigaflow_f1tenth.irl_identity import (
    IRL_F_BRAKE_N,
    IRL_F_DRIVE_N,
    IRL_TIRE_MU,
    baked_normalized_condition,
)
from gigaflow_f1tenth.sophy_compat import (
    BakedConditionActor,
    gaussian_kl,
    load_baked_actor,
)

DEFAULT_KL_COEF = 0.2
DEFAULT_KL_MAX = 0.05
DEFAULT_UPDATES = 1000
DEFAULT_NUM_ENVS = 256
DEFAULT_TRACK = "IV_2026_SIM"


def _ensure_sophy_path() -> Path:
    repo = Path(__file__).resolve().parents[3]
    training = repo / "training"
    path = str(training)
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)
    for name in ("ppo", "config", "f1tenth_env", "f1tenth_sim"):
        mod = sys.modules.get(name)
        origin = getattr(mod, "__file__", None) if mod is not None else None
        if origin and not str(Path(origin).resolve()).startswith(path):
            stale = [
                key
                for key in sys.modules
                if key == name or key.startswith(name + ".")
            ]
            for key in stale:
                del sys.modules[key]
    return repo


def _resolve_device(device: str) -> torch.device:
    requested = str(device)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "cuda was requested but torch.cuda.is_available() is False"
        )
    return torch.device(requested)


def _maybe_wandb(smoke: bool, payload: dict[str, Any]):
    if smoke:
        return None
    project = os.environ.get("WANDB_PROJECT")
    if not project:
        return None
    try:
        import wandb
    except ImportError:
        print("wandb not installed; continuing without it", flush=True)
        return None
    return wandb.init(
        project=project,
        name=os.environ.get("WANDB_NAME", "gigaflow-sophy-sim-ft"),
        config=payload,
    )


def sophy_env_config(*, track: str, smoke: bool) -> dict[str, Any]:
    _ensure_sophy_path()
    from config import DEFAULT_CONFIG
    from f1tenth_env.utils import episode_length_for_track

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    env_cfg = cfg["env"]
    env_cfg["track"] = str(track)
    env_cfg["opponent_strategy"] = None
    env_cfg["f_drive_max"] = float(IRL_F_DRIVE_N)
    env_cfg["f_brake_max"] = float(IRL_F_BRAKE_N)
    env_cfg["tire_friction"] = float(IRL_TIRE_MU)
    dr = env_cfg.setdefault("domain_randomization", {})
    dr["enabled"] = False
    repo = _ensure_sophy_path()
    env_cfg["episode_length"] = episode_length_for_track(
        env_cfg["track"],
        str(repo / "training"),
        ref_lap_speed_mps=float(env_cfg.get("expected_lap_speed_mps", 3.5)),
        lap_multiplier=float(env_cfg.get("episode_lap_multiplier", 3.0)),
    )
    if smoke:
        env_cfg["episode_length"] = min(float(env_cfg["episode_length"]), 8.0)
    return cfg


class KLClippedPPOTrainer:
    """Sophy PPO collect loop with an extra KL clip to a frozen base actor."""

    def __init__(
        self,
        actor: BakedConditionActor,
        base_actor: BakedConditionActor,
        critic_obs_dim: int,
        critic_normalizer,
        device: str | torch.device,
        *,
        kl_coef: float = DEFAULT_KL_COEF,
        kl_max: float = DEFAULT_KL_MAX,
        rollout_steps: int = 128,
        num_epochs: int = 4,
        env_minibatch_size: int = 32,
        actor_lr: float = 3e-5,
        value_lr: float = 1e-4,
    ):
        _ensure_sophy_path()
        from ppo import PPOTrainer

        self.kl_coef = float(kl_coef)
        self.kl_max = float(kl_max)
        self.base_actor = base_actor
        for param in self.base_actor.parameters():
            param.requires_grad_(False)
        self.base_actor.eval()
        self.inner = PPOTrainer(
            actor,
            critic_obs_dim,
            actor_normalizer=None,
            critic_normalizer=critic_normalizer,
            device=device,
            rollout_steps=rollout_steps,
            num_epochs=num_epochs,
            env_minibatch_size=env_minibatch_size,
            actor_lr=actor_lr,
            value_lr=value_lr,
        )
        self.inner.update = self._update_with_kl  # type: ignore[method-assign]
        self.device = self.inner.device

    def initialize(self, actor_obs, critic_obs):
        return self.inner.initialize(actor_obs, critic_obs)

    def act(self, actor_obs, critic_obs):
        return self.inner.act(actor_obs, critic_obs)

    def observe(self, *args, **kwargs):
        return self.inner.observe(*args, **kwargs)

    @property
    def rollout_position(self):
        return self.inner.rollout_position

    def _update_with_kl(self, rollout):
        inner = self.inner
        if rollout.optimized:
            raise ValueError("rollout has already been optimized")
        from ppo import advantage_filter_keep_mask

        raw_advantages = rollout.advantages
        keep_mask, eta = advantage_filter_keep_mask(
            raw_advantages,
            enabled=inner.advantage_filter_enabled,
            discard_fraction=inner.advantage_filter_discard_fraction,
        )
        retained = int(keep_mask.sum().item())
        total = keep_mask.numel()
        discard_rate = 1.0 - (retained / max(total, 1))
        empty = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "approx_kl": 0.0,
            "base_kl": 0.0,
            "policy_entropy": 0.0,
            "kl_skips": 0.0,
            "advantage_filter_discard_rate": discard_rate,
            "advantage_filter_retained": retained,
            "advantage_filter_eta": float(eta),
        }
        if inner.advantage_filter_enabled and retained == 0:
            rollout.optimized = True
            return empty
        if inner.advantage_filter_enabled:
            kept_adv = raw_advantages[keep_mask]
            adv_mean = kept_adv.mean()
            adv_std = kept_adv.std(unbiased=False)
        else:
            adv_mean = raw_advantages.mean()
            adv_std = raw_advantages.std(unbiased=False)
        advantages = (raw_advantages - adv_mean) / (adv_std + 1e-8)
        keep_weights = keep_mask.to(dtype=advantages.dtype)
        steps, num_envs = rollout.rewards.shape
        del steps
        sums = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "approx_kl": 0.0,
            "base_kl": 0.0,
            "policy_entropy": 0.0,
        }
        updates = 0
        skips = 0
        for _epoch in range(inner.num_epochs):
            permutation = torch.randperm(num_envs, device=inner.device)
            for start in range(0, num_envs, inner.env_minibatch_size):
                env_index = permutation[start : start + inner.env_minibatch_size]
                actor_raw = rollout.actor_obs[:, env_index]
                critic_raw = rollout.critic_obs[:, env_index]
                actor_seq = inner._normalize(inner.actor_normalizer, actor_raw)
                actor_seq = actor_seq.transpose(0, 1)
                action_seq = rollout.actions[:, env_index].transpose(0, 1)
                pre_tanh_seq = rollout.pre_tanh_actions[:, env_index].transpose(0, 1)
                reset_seq = rollout.reset_masks[:, env_index].transpose(0, 1)
                hidden = rollout.rollout_start_hidden[env_index]
                new_logp, _, new_entropy = inner._evaluate_actions_sequence(
                    actor_seq,
                    hidden,
                    action_seq,
                    reset_mask=reset_seq,
                    pre_tanh_actions=pre_tanh_seq,
                )
                new_logp = new_logp.transpose(0, 1)
                new_entropy = new_entropy.transpose(0, 1)
                old_logp = rollout.old_logp[:, env_index]
                mb_advantages = advantages[:, env_index]
                log_ratio = new_logp - old_logp
                ratio = log_ratio.exp()
                unclipped = ratio * mb_advantages
                clipped = ratio.clamp(
                    1.0 - inner.clip_ratio, 1.0 + inner.clip_ratio
                ) * mb_advantages
                mb_keep = keep_weights[:, env_index]
                keep_denom = mb_keep.sum().clamp(min=1.0)
                policy_loss_per = -torch.minimum(unclipped, clipped)
                policy_loss = (policy_loss_per * mb_keep).sum() / keep_denom
                policy_entropy = (new_entropy * mb_keep).sum() / keep_denom
                policy_loss = policy_loss - inner.entropy_coef * policy_entropy
                approx_kl = (((ratio - 1.0) - log_ratio) * mb_keep).sum() / keep_denom
                _lp, mean_p, log_std_p = inner.actor.dist_stats(
                    actor_seq, hidden, action_seq, reset_mask=reset_seq
                )
                with torch.no_grad():
                    _lq, mean_q, log_std_q = self.base_actor.dist_stats(
                        actor_seq, hidden, action_seq, reset_mask=reset_seq
                    )
                del _lp, _lq
                base_kl = gaussian_kl(mean_p, log_std_p, mean_q, log_std_q)
                skip_actor = float(base_kl.detach()) > self.kl_max
                if not skip_actor:
                    policy_loss = policy_loss + self.kl_coef * base_kl

                critic_obs = inner._normalize(inner.critic_normalizer, critic_raw)
                new_values = inner._update_value_forward(critic_obs)
                old_values = rollout.old_values[:, env_index]
                mb_returns = rollout.returns[:, env_index]
                clipped_values = old_values + (new_values - old_values).clamp(
                    -inner.value_clip, inner.value_clip
                )
                value_loss_unclipped = (new_values - mb_returns).square()
                value_loss_clipped = (clipped_values - mb_returns).square()
                value_loss_per = 0.5 * torch.maximum(
                    value_loss_unclipped, value_loss_clipped
                )
                value_loss = (value_loss_per * mb_keep).sum() / keep_denom
                inner.value_optimizer.zero_grad(set_to_none=True)
                value_loss.backward()
                nn.utils.clip_grad_norm_(
                    inner.value_critic.parameters(), inner.max_grad_norm
                )
                inner.value_optimizer.step()

                if skip_actor:
                    skips += 1
                else:
                    inner.actor_optimizer.zero_grad(set_to_none=True)
                    policy_loss.backward()
                    nn.utils.clip_grad_norm_(
                        inner.actor.parameters(), inner.max_grad_norm
                    )
                    inner.actor_optimizer.step()
                sums["policy_loss"] += float(policy_loss.detach())
                sums["value_loss"] += float(value_loss.detach())
                sums["approx_kl"] += float(approx_kl.detach())
                sums["base_kl"] += float(base_kl.detach())
                sums["policy_entropy"] += float(policy_entropy.detach())
                updates += 1
        rollout.optimized = True
        if inner.critic_normalizer is not None:
            inner.critic_normalizer.update(
                rollout.critic_obs.reshape(-1, inner.critic_obs_dim)
            )
        metrics = {key: value / max(updates, 1) for key, value in sums.items()}
        metrics.update(
            {
                "kl_skips": float(skips),
                "advantage_filter_discard_rate": discard_rate,
                "advantage_filter_retained": retained,
                "advantage_filter_eta": float(eta),
                "minibatches": updates,
            }
        )
        return metrics


def unpack_obs(obs) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(obs, dict) or "actor" not in obs or "frenet" not in obs:
        raise KeyError("Sophy WarpEnv must return actor/frenet observations")
    return obs["actor"].to(torch.float32), obs["frenet"].to(torch.float32)


def run_sophy_finetune(
    *,
    checkpoint: str | Path,
    output_dir: str | Path,
    device: str = "cuda",
    num_updates: int = DEFAULT_UPDATES,
    num_envs: int = DEFAULT_NUM_ENVS,
    track: str = DEFAULT_TRACK,
    kl_coef: float = DEFAULT_KL_COEF,
    kl_max: float = DEFAULT_KL_MAX,
    smoke: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    repo = _ensure_sophy_path()
    from f1tenth_env import F1tenthEnv
    from f1tenth_env import runtime as rt
    from f1tenth_policy import ObsNormalizer
    from ppo import pure_timeout_mask

    torch.manual_seed(int(seed))
    torch_device = _resolve_device(device)
    if smoke:
        num_envs = min(int(num_envs), 2)
        num_updates = min(int(num_updates), 1)
        rollout_steps = 8
        env_minibatch_size = min(int(num_envs), 2)
    else:
        rollout_steps = 128
        env_minibatch_size = min(32, int(num_envs))
    rt.configure(dev=torch_device)
    cfg = sophy_env_config(track=track, smoke=smoke)
    env_cfg = cfg["env"]
    obs_cfg = cfg["obs"]
    reward_cfg = cfg["reward"]
    env = F1tenthEnv(
        num_envs=int(num_envs),
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        show_viewer=False,
        enable_recording=False,
    )
    actor, payload = load_baked_actor(checkpoint, device=torch_device)
    live = BakedConditionActor(actor)
    base_actor, _base_payload = load_baked_actor(checkpoint, device=torch_device)
    del _base_payload
    base = BakedConditionActor(base_actor)
    critic_obs_dim = int(obs_cfg["num_obs"])
    critic_normalizer = ObsNormalizer(
        obs_dim=critic_obs_dim,
        device=torch_device,
        eps=float(obs_cfg.get("norm_eps", 1e-8)),
        clip=float(obs_cfg.get("norm_clip", 10.0)),
    )
    trainer = KLClippedPPOTrainer(
        live,
        base,
        critic_obs_dim,
        critic_normalizer,
        device=torch_device,
        kl_coef=kl_coef,
        kl_max=kl_max,
        rollout_steps=rollout_steps,
        env_minibatch_size=env_minibatch_size,
    )
    raw_obs, _ = env.reset(with_sensors=True)
    actor_obs, critic_obs = unpack_obs(raw_obs)
    critic_normalizer.update(critic_obs)
    trainer.initialize(actor_obs, critic_obs)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device_name = (
        torch.cuda.get_device_name(torch_device)
        if torch_device.type == "cuda"
        else "cpu"
    )
    print(
        json.dumps(
            {
                "event": "sophy_finetune_start",
                "device": str(torch_device),
                "device_name": device_name,
                "track": track,
                "num_envs": int(num_envs),
                "num_updates": int(num_updates),
                "kl_coef": float(kl_coef),
                "kl_max": float(kl_max),
                "smoke": bool(smoke),
            }
        ),
        flush=True,
    )
    wandb_run = _maybe_wandb(
        smoke,
        {
            "checkpoint": str(checkpoint),
            "track": track,
            "num_envs": int(num_envs),
            "num_updates": int(num_updates),
            "kl_coef": float(kl_coef),
            "kl_max": float(kl_max),
            "device": str(torch_device),
            "baked_condition": baked_normalized_condition(),
        },
    )
    last_metrics: dict[str, Any] = {}
    t0 = time.perf_counter()
    updates_done = 0
    while updates_done < int(num_updates) or trainer.rollout_position > 0:
        actions = trainer.act(actor_obs, critic_obs)
        next_raw, reward, done, extras = env.step(
            actions, n_steps=int(env.control_interval), with_sensors=True
        )
        next_actor, next_critic = unpack_obs(next_raw)
        timed_out = pure_timeout_mask(extras["termination"])
        timeout_obs = extras["observations"]["terminal_critic"]
        metrics = trainer.observe(
            next_actor,
            next_critic,
            reward.to(torch.float32),
            done,
            timed_out=timed_out,
            timeout_critic_obs=timeout_obs,
        )
        actor_obs, critic_obs = next_actor, next_critic
        if metrics is not None:
            updates_done += 1
            last_metrics = metrics
            if updates_done == 1 or updates_done % 10 == 0:
                print(
                    json.dumps(
                        {
                            "update": updates_done,
                            "base_kl": last_metrics.get("base_kl"),
                            "approx_kl": last_metrics.get("approx_kl"),
                            "policy_loss": last_metrics.get("policy_loss"),
                            "kl_skips": last_metrics.get("kl_skips"),
                        }
                    ),
                    flush=True,
                )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        key: float(value)
                        for key, value in last_metrics.items()
                        if isinstance(value, (int, float))
                    },
                    step=updates_done,
                )
    elapsed = time.perf_counter() - t0
    baked = bake_actor_artifact(payload, actor=live.actor)
    baked["sophy_sim_finetune"] = {
        "track": track,
        "num_updates": updates_done,
        "num_envs": int(num_envs),
        "kl_coef": float(kl_coef),
        "kl_max": float(kl_max),
        "device": str(torch_device),
        "device_name": device_name,
        "baked_condition": baked_normalized_condition(),
        "elapsed_s": elapsed,
        "repo": str(repo),
        "last_metrics": {
            key: float(value)
            for key, value in last_metrics.items()
            if isinstance(value, (int, float))
        },
    }
    artifact_path = output / "actor_sophy_ft.pt"
    FileArtifactStore().write_actor_artifact(str(artifact_path), baked)
    report = {
        "ok": True,
        "output": str(artifact_path),
        "irl_baked": True,
        "irl_identity": baked.get("irl_identity"),
        "deployment_condition_normalized": baked.get(
            "deployment_condition_normalized"
        ),
        "sophy_sim_finetune": baked["sophy_sim_finetune"],
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if wandb_run is not None:
        wandb_run.finish()
    return report
