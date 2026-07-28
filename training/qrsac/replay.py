from __future__ import annotations

import logging

import torch

from qrsac.spinningup.core import GRU_HIDDEN_DIM

LOGGER_NAME = "qrsac.replay"
OPP_OBS_BASE_IDX = 384
OPP_OBS_DIM = 8
OPP_OBS_END_IDX = OPP_OBS_BASE_IDX + OPP_OBS_DIM
REPLAY_BURN_IN = 16
REPLAY_TRAIN_LEN = 32
REPLAY_CHECKPOINT_INTERVAL = 16
REPLAY_HIDDEN_DTYPE = torch.float16
REPLAY_OBS_DTYPE = torch.float16


class TrajectoryReplayBuffer:
    """Per-env circular trajectory replay with checkpointed recurrent state.

    Envs advance in lockstep into ``[num_envs, steps_per_env, ...]`` storage.
    Sample starts are restricted to hidden-checkpoint boundaries (every
    ``checkpoint_interval`` steps). Each sample is a fixed window of
    ``burn_in + train_len + n_step`` contiguous steps with terminal-safe
    n-step target tensors for the optimized segment.
    """

    def __init__(
        self,
        capacity: int,
        actor_obs_dim: int,
        critic_obs_dim: int,
        act_dim: int,
        num_envs: int,
        device: torch.device,
        n_step: int = 7,
        gamma: float = 0.9896,
        burn_in: int = REPLAY_BURN_IN,
        train_len: int = REPLAY_TRAIN_LEN,
        checkpoint_interval: int = REPLAY_CHECKPOINT_INTERVAL,
        hidden_dim: int = GRU_HIDDEN_DIM,
        obs_dtype: torch.dtype = REPLAY_OBS_DTYPE,
    ):
        if actor_obs_dim == critic_obs_dim:
            raise ValueError(
                "Dual replay requires distinct actor/critic observation dimensions; "
                f"got actor_obs_dim={actor_obs_dim} critic_obs_dim={critic_obs_dim}."
            )
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        if checkpoint_interval <= 0:
            raise ValueError(
                f"checkpoint_interval must be positive, got {checkpoint_interval}"
            )
        if burn_in % checkpoint_interval != 0:
            raise ValueError(
                f"burn_in={burn_in} must be a multiple of "
                f"checkpoint_interval={checkpoint_interval}"
            )
        seq_len = int(burn_in) + int(train_len) + int(n_step)
        raw_steps = int(capacity) // int(num_envs)
        steps_per_env = (raw_steps // int(checkpoint_interval)) * int(
            checkpoint_interval
        )
        if steps_per_env < seq_len:
            raise ValueError(
                f"capacity={capacity} with num_envs={num_envs} yields "
                f"steps_per_env={steps_per_env} < seq_len={seq_len}"
            )

        self.num_envs = int(num_envs)
        self.steps_per_env = int(steps_per_env)
        self.capacity = self.steps_per_env * self.num_envs
        self.actor_obs_dim = int(actor_obs_dim)
        self.critic_obs_dim = int(critic_obs_dim)
        self.act_dim = int(act_dim)
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.burn_in = int(burn_in)
        self.train_len = int(train_len)
        self.checkpoint_interval = int(checkpoint_interval)
        self.hidden_dim = int(hidden_dim)
        self.seq_len = seq_len
        self.device = device
        self.obs_dtype = obs_dtype
        self.num_checkpoints = self.steps_per_env // self.checkpoint_interval

        e, t = self.num_envs, self.steps_per_env
        self.actor_obs = torch.zeros(
            e, t, self.actor_obs_dim, device=device, dtype=obs_dtype
        )
        self.critic_obs = torch.zeros(
            e, t, self.critic_obs_dim, device=device, dtype=obs_dtype
        )
        self.action = torch.zeros(
            e, t, self.act_dim, device=device, dtype=torch.float32
        )
        self.reward = torch.zeros(e, t, device=device, dtype=torch.float32)
        self.done = torch.zeros(e, t, device=device, dtype=torch.float32)
        self.reset = torch.zeros(e, t, device=device, dtype=torch.bool)
        self.episode_id = torch.zeros(e, t, device=device, dtype=torch.long)
        self.hidden = torch.zeros(
            e,
            self.num_checkpoints,
            self.hidden_dim,
            device=device,
            dtype=REPLAY_HIDDEN_DTYPE,
        )
        # Compact per-step visibility from critic opponent block [384:392).
        self.opponent_visible = torch.zeros(e, t, device=device, dtype=torch.bool)
        self._has_opp_block = self.critic_obs_dim >= OPP_OBS_END_IDX

        self.ptr = 0
        self._size = 0
        self.size = torch.zeros((), device=device, dtype=torch.long)
        self._insert_count = torch.full(
            (), e, device=device, dtype=torch.long
        )
        self._ready = False
        self._ep_id = torch.zeros(e, device=device, dtype=torch.long)
        self._pending_reset = torch.ones(e, device=device, dtype=torch.bool)
        self._gamma_powers = torch.tensor(
            [self.gamma**k for k in range(self.n_step)],
            device=device,
            dtype=torch.float32,
        )
        self._arange_seq = torch.arange(self.seq_len, device=device, dtype=torch.long)
        self._arange_train = torch.arange(
            self.train_len, device=device, dtype=torch.long
        )
        self._arange_n = torch.arange(self.n_step, device=device, dtype=torch.long)
        self._n_step_idx = (
            self.burn_in
            + self._arange_train.unsqueeze(1)
            + self._arange_n.unsqueeze(0)
        )
        self._boot_lo = self.burn_in + self.n_step
        self._boot_hi = self.burn_in + self.train_len + self.n_step
        # Static checkpoint columns; validity mask is cached across samples that
        # share the same (ptr, size) — the common multi-update-per-tick case.
        self._all_starts = torch.arange(
            0,
            self.steps_per_env,
            self.checkpoint_interval,
            device=device,
            dtype=torch.long,
        )
        self._empty_starts = self._all_starts[:0]
        self._cached_starts: torch.Tensor | None = None
        self._cached_starts_key: tuple[int, int] | None = None
        self._cached_buckets: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ] | None = None
        self._cached_buckets_key: tuple[int, int] | None = None
        # Odd batch sizes: alternate which visibility bucket receives the extra slot.
        self._odd_extra_to_visible = True
        self._fallback_events = 0
        self._sample_count = 0
        self.last_sample_metrics: dict[str, float] = {
            "available_visible_frac": 0.0,
            "available_not_visible_frac": 0.0,
            "sampled_visible_frac": 0.0,
            "sampled_not_visible_frac": 0.0,
            "fallback": 0.0,
            "fallback_rate": 0.0,
        }

    def add(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Append one lockstep transition for every env. Returns inserted count."""
        col = self.ptr
        critic_store = critic_obs.detach().to(self.obs_dtype)
        self.actor_obs[:, col] = actor_obs.detach().to(self.obs_dtype)
        self.critic_obs[:, col] = critic_store
        self.action[:, col] = actions.detach().to(torch.float32)
        self.reward[:, col] = rewards.detach().to(torch.float32)
        dones_b = dones.detach().bool()
        self.done[:, col] = dones_b.to(torch.float32)
        self.reset[:, col] = self._pending_reset
        self.episode_id[:, col] = self._ep_id
        if self._has_opp_block:
            opp = critic_store[:, OPP_OBS_BASE_IDX:OPP_OBS_END_IDX]
            self.opponent_visible[:, col] = (opp != 0).any(dim=-1)
        else:
            self.opponent_visible[:, col] = False

        if col % self.checkpoint_interval == 0:
            ckpt = col // self.checkpoint_interval
            if hidden is None:
                h = torch.zeros(
                    self.num_envs,
                    self.hidden_dim,
                    device=self.device,
                    dtype=torch.float32,
                )
            else:
                h = hidden.detach()
                if h.shape != (self.num_envs, self.hidden_dim):
                    raise ValueError(
                        f"hidden shape {tuple(h.shape)} != "
                        f"({self.num_envs}, {self.hidden_dim})"
                    )
            self.hidden[:, ckpt] = h.to(REPLAY_HIDDEN_DTYPE)

        self._pending_reset = dones_b
        self._ep_id = self._ep_id + dones_b.long()
        self.ptr = (col + 1) % self.steps_per_env
        self._size = min(self._size + 1, self.steps_per_env)
        self.size.fill_(self._size * self.num_envs)
        self._cached_starts = None
        self._cached_starts_key = None
        self._cached_buckets = None
        self._cached_buckets_key = None
        return self._insert_count

    def _checkpoint_starts(self) -> torch.Tensor:
        """Physical columns where a hidden checkpoint exists and a window fits."""
        key = (self.ptr, self._size)
        if self._cached_starts is not None and self._cached_starts_key == key:
            return self._cached_starts
        starts = self._all_starts
        if self._size < self.seq_len:
            out = self._empty_starts
        elif self._size < self.steps_per_env:
            out = starts[starts <= (self._size - self.seq_len)]
        else:
            t_len = self.steps_per_env
            dist = (self.ptr - starts) % t_len
            # Full buffer: dist==0 means the start is exactly at ptr → age is t_len.
            dist = torch.where(dist == 0, t_len, dist)
            out = starts[dist >= self.seq_len]
        self._cached_starts = out
        self._cached_starts_key = key
        return out

    def _visibility_buckets(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Checkpoint-aligned (env, start) pairs split by window visibility."""
        key = (self.ptr, self._size)
        if self._cached_buckets is not None and self._cached_buckets_key == key:
            return self._cached_buckets
        starts = self._checkpoint_starts()
        empty = self._empty_starts
        if starts.numel() == 0:
            buckets = (empty, empty, empty, empty)
            self._cached_buckets = buckets
            self._cached_buckets_key = key
            return buckets
        time_idx = (starts.unsqueeze(1) + self._arange_seq) % self.steps_per_env
        any_vis = self.opponent_visible[:, time_idx].any(dim=-1)
        vis_e, vis_s = torch.where(any_vis)
        not_e, not_s = torch.where(~any_vis)
        buckets = (vis_e, starts[vis_s], not_e, starts[not_s])
        self._cached_buckets = buckets
        self._cached_buckets_key = key
        return buckets

    def _draw_even_indices(
        self,
        vis_e: torch.Tensor,
        vis_s: torch.Tensor,
        not_e: torch.Tensor,
        not_s: torch.Tensor,
        n_seq: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int, bool]:
        """Return (env_idx, start, n_vis_sampled, fallback)."""
        n_vis_avail = int(vis_e.numel())
        n_not_avail = int(not_e.numel())
        fallback = n_vis_avail == 0 or n_not_avail == 0
        env_idx = torch.empty(n_seq, device=self.device, dtype=torch.long)
        start = torch.empty(n_seq, device=self.device, dtype=torch.long)
        if fallback:
            self._fallback_events += 1
            if self._fallback_events <= 3 or self._fallback_events % 100 == 0:
                logging.getLogger(LOGGER_NAME).info(
                    "TrajectoryReplayBuffer: visibility bucket empty "
                    "(visible=%d not_visible=%d); filling %d sequences from "
                    "available bucket (fallback #%d)",
                    n_vis_avail,
                    n_not_avail,
                    n_seq,
                    self._fallback_events,
                )
            src_e = vis_e if n_vis_avail > 0 else not_e
            src_s = vis_s if n_vis_avail > 0 else not_s
            pick = torch.randint(
                0, int(src_e.numel()), (n_seq,), device=self.device, dtype=torch.long
            )
            env_idx.copy_(src_e[pick])
            start.copy_(src_s[pick])
            n_vis_sampled = n_seq if n_vis_avail > 0 else 0
            return env_idx, start, n_vis_sampled, True

        n_vis_sampled = n_seq // 2
        n_not_sampled = n_seq // 2
        if n_seq % 2 == 1:
            if self._odd_extra_to_visible:
                n_vis_sampled += 1
            else:
                n_not_sampled += 1
            self._odd_extra_to_visible = not self._odd_extra_to_visible
        if n_vis_sampled:
            pick_v = torch.randint(
                0,
                n_vis_avail,
                (n_vis_sampled,),
                device=self.device,
                dtype=torch.long,
            )
            env_idx[:n_vis_sampled] = vis_e[pick_v]
            start[:n_vis_sampled] = vis_s[pick_v]
        if n_not_sampled:
            pick_n = torch.randint(
                0,
                n_not_avail,
                (n_not_sampled,),
                device=self.device,
                dtype=torch.long,
            )
            env_idx[n_vis_sampled:] = not_e[pick_n]
            start[n_vis_sampled:] = not_s[pick_n]
        return env_idx, start, n_vis_sampled, False

    def is_ready(self, num_sequences: int) -> bool:
        """Whether at least ``num_sequences`` checkpoint-aligned windows exist."""
        if not self._ready:
            total = int(self._checkpoint_starts().numel()) * self.num_envs
            if total >= int(num_sequences):
                self._ready = True
        return self._ready

    def sample(self, num_sequences: int) -> dict[str, torch.Tensor]:
        """Sample fixed-shape trajectory windows. Caller gates on ``is_ready``."""
        n_seq = int(num_sequences)
        starts = self._checkpoint_starts()
        n_starts = int(starts.numel())
        if n_starts <= 0:
            raise RuntimeError("TrajectoryReplayBuffer.sample called before ready")

        vis_e, vis_s, not_e, not_s = self._visibility_buckets()
        n_vis_avail = int(vis_e.numel())
        n_not_avail = int(not_e.numel())
        n_total_avail = n_vis_avail + n_not_avail
        avail_vis_frac = (
            float(n_vis_avail) / float(n_total_avail) if n_total_avail > 0 else 0.0
        )
        avail_not_frac = (
            float(n_not_avail) / float(n_total_avail) if n_total_avail > 0 else 0.0
        )
        env_idx, start, n_vis_sampled, fallback = self._draw_even_indices(
            vis_e, vis_s, not_e, not_s, n_seq
        )

        self._sample_count += 1
        sampled_vis_frac = float(n_vis_sampled) / float(n_seq) if n_seq > 0 else 0.0
        self.last_sample_metrics = {
            "available_visible_frac": avail_vis_frac,
            "available_not_visible_frac": avail_not_frac,
            "sampled_visible_frac": sampled_vis_frac,
            "sampled_not_visible_frac": 1.0 - sampled_vis_frac,
            "fallback": 1.0 if fallback else 0.0,
            "fallback_rate": (
                float(self._fallback_events) / float(self._sample_count)
            ),
        }

        time_idx = (start.unsqueeze(1) + self._arange_seq) % self.steps_per_env
        env_exp = env_idx.unsqueeze(1).expand(n_seq, self.seq_len)
        # One advanced-index gather each; cast f16 obs in-place via .float().
        actor = self.actor_obs[env_exp, time_idx].float()
        critic = self.critic_obs[env_exp, time_idx].float()
        action = self.action[env_exp, time_idx]
        reward = self.reward[env_exp, time_idx]
        done = self.done[env_exp, time_idx]
        reset = self.reset[env_exp, time_idx].float()
        episode_id = self.episode_id[env_exp, time_idx]

        ckpt_idx = start // self.checkpoint_interval
        hidden = self.hidden[env_idx, ckpt_idx].float()

        rew_g = reward[:, self._n_step_idx]
        done_g = done[:, self._n_step_idx]
        prior_done = torch.cumsum(done_g, dim=-1) - done_g
        alive = (prior_done == 0).to(dtype=reward.dtype)
        n_step_reward = (rew_g * self._gamma_powers * alive).sum(dim=-1)
        n_step_done = (done_g * alive).sum(dim=-1).clamp_max(1.0)

        boot_actor = actor[:, self._boot_lo : self._boot_hi]
        boot_critic = critic[:, self._boot_lo : self._boot_hi]

        return {
            "actor_obs": actor,
            "critic_obs": critic,
            "action": action,
            "reward": reward,
            "done": done,
            "reset": reset,
            "episode_id": episode_id,
            "hidden": hidden,
            "n_step_reward": n_step_reward,
            "n_step_done": n_step_done,
            "bootstrap_actor_obs": boot_actor,
            "bootstrap_critic_obs": boot_critic,
            "env_index": env_idx,
            "start_index": start,
        }
