from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import Adam

from f1tenth_policy.dreamer import LidarDreamerActor, deploy_actor_state_dict
from f1tenth_policy.normalizer import ObsNormalizer

from .world_model import (
    WorldModelHeads,
    imagine_actor_critic,
    world_model_losses,
)

DEFAULT_REPLAY_CAPACITY = 4096
DEFAULT_REPLAY_BATCH_SIZE = 32


class ImaginationCritic(nn.Module):
    def __init__(self, feat_dim, hidden_sizes=(256, 256)):
        super().__init__()
        sizes = [int(feat_dim), *[int(s) for s in hidden_sizes], 1]
        layers = []
        for index in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[index], sizes[index + 1]))
            if index < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, features):
        return self.net(features).squeeze(-1)


class SequenceReplay:
    """On-device FIFO of length-T sequences for RSSM + imagination updates."""

    def __init__(
        self, capacity, seq_len, obs_dim, act_dim, hidden_dim, device
    ):
        self.capacity = int(capacity)
        self.seq_len = int(seq_len)
        device = torch.device(device)
        self.obs = torch.zeros(self.capacity, self.seq_len, obs_dim, device=device)
        self.next_obs = torch.zeros(
            self.capacity, self.seq_len, obs_dim, device=device
        )
        self.actions = torch.zeros(
            self.capacity, self.seq_len, act_dim, device=device
        )
        self.rewards = torch.zeros(self.capacity, self.seq_len, device=device)
        self.dones = torch.zeros(
            self.capacity, self.seq_len, dtype=torch.bool, device=device
        )
        self.reset = torch.zeros(
            self.capacity, self.seq_len, dtype=torch.bool, device=device
        )
        self.start_hidden = torch.zeros(
            self.capacity, hidden_dim, device=device
        )
        self.size = 0
        self.index = 0

    def clear(self):
        self.size = 0
        self.index = 0

    def add(self, obs, next_obs, actions, rewards, dones, reset, start_hidden):
        n = int(obs.shape[0])
        if n <= 0:
            return
        if n > self.capacity:
            obs = obs[-self.capacity :]
            next_obs = next_obs[-self.capacity :]
            actions = actions[-self.capacity :]
            rewards = rewards[-self.capacity :]
            dones = dones[-self.capacity :]
            reset = reset[-self.capacity :]
            start_hidden = start_hidden[-self.capacity :]
            n = self.capacity
        idx = (
            self.index + torch.arange(n, device=obs.device)
        ) % self.capacity
        self.obs[idx] = obs
        self.next_obs[idx] = next_obs
        self.actions[idx] = actions
        self.rewards[idx] = rewards
        self.dones[idx] = dones
        self.reset[idx] = reset
        self.start_hidden[idx] = start_hidden
        self.index = int((self.index + n) % self.capacity)
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size):
        if self.size <= 0:
            raise RuntimeError("sequence replay is empty")
        n = min(int(batch_size), self.size)
        idx = torch.randint(0, self.size, (n,), device=self.obs.device)
        return {
            "obs": self.obs[idx],
            "next_obs": self.next_obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "dones": self.dones[idx],
            "reset": self.reset[idx],
            "start_hidden": self.start_hidden[idx],
        }


class DreamerTrainer:
    """World-model + imagination actor-critic. Not PPO."""

    def __init__(
        self,
        actor: LidarDreamerActor,
        actor_normalizer: ObsNormalizer,
        device="cpu",
        *,
        batch_length=16,
        imagination_horizon=8,
        gamma=0.99,
        lambda_gae=0.95,
        kl_scale=1.0,
        actor_lr=3e-5,
        world_lr=1e-4,
        critic_lr=1e-4,
        entropy_coef=3e-4,
        max_grad_norm=0.5,
        world_hidden_sizes=(256, 256),
        replay_capacity=DEFAULT_REPLAY_CAPACITY,
        replay_batch_size=DEFAULT_REPLAY_BATCH_SIZE,
    ):
        if not isinstance(actor, LidarDreamerActor):
            raise TypeError("DreamerTrainer requires LidarDreamerActor")
        self.device = torch.device(device)
        self.actor = actor.to(self.device)
        self.world = WorldModelHeads(
            self.actor, hidden_sizes=world_hidden_sizes
        ).to(self.device)
        feat_dim = int(self.actor.deter_dim + self.actor.stoch_dim)
        self.critic = ImaginationCritic(
            feat_dim, hidden_sizes=world_hidden_sizes
        ).to(self.device)
        self.actor_normalizer = actor_normalizer
        self.batch_length = int(batch_length)
        self.imagination_horizon = int(imagination_horizon)
        self.gamma = float(gamma)
        self.lambda_gae = float(lambda_gae)
        self.kl_scale = float(kl_scale)
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.actor_obs_dim = int(actor.obs_dim)
        self.act_dim = int(actor.act_dim)
        self.replay_batch_size = int(replay_batch_size)
        wm_params = list(self.actor.encoder.parameters())
        wm_params += list(self.actor.ls3c.parameters())
        wm_params += list(self.actor.post_mean.parameters())
        wm_params += list(self.actor.post_std.parameters())
        wm_params += list(self.world.parameters())
        self.world_optimizer = Adam(wm_params, lr=float(world_lr))
        actor_params = (
            list(self.actor.net.parameters())
            + list(self.actor.mu_layer.parameters())
            + list(self.actor.log_std_layer.parameters())
        )
        self.actor_optimizer = Adam(actor_params, lr=float(actor_lr))
        self.critic_optimizer = Adam(
            self.critic.parameters(), lr=float(critic_lr)
        )
        self._replay = SequenceReplay(
            capacity=replay_capacity,
            seq_len=self.batch_length,
            obs_dim=self.actor_obs_dim,
            act_dim=self.act_dim,
            hidden_dim=int(self.actor.gru_hidden_dim),
            device=self.device,
        )
        self.last_batch_obs_shape = None
        self._initialized = False
        self._awaiting_observe = False
        self._step = 0
        self._num_envs = 0
        self.rollout_position = 0
        self.env_minibatch_size = 1

    def initialize(self, actor_obs, _critic_obs=None):
        actor_obs = actor_obs.to(device=self.device, dtype=torch.float32)
        self._num_envs = int(actor_obs.shape[0])
        t = self.batch_length
        n = self._num_envs
        self._obs = torch.zeros(
            t, n, self.actor_obs_dim, device=self.device
        )
        self._next_obs = torch.zeros(
            t, n, self.actor_obs_dim, device=self.device
        )
        self._actions = torch.zeros(t, n, self.act_dim, device=self.device)
        self._rewards = torch.zeros(t, n, device=self.device)
        self._dones = torch.zeros(t, n, dtype=torch.bool, device=self.device)
        self._reset = torch.zeros(t, n, dtype=torch.bool, device=self.device)
        self._live_hidden = self.actor.initial_hidden(
            n, device=self.device, dtype=torch.float32
        )
        self._start_hidden = self._live_hidden.clone()
        self._reset_mask = torch.ones(n, dtype=torch.bool, device=self.device)
        self._replay.clear()
        self.last_batch_obs_shape = None
        self._step = 0
        self._initialized = True
        self._awaiting_observe = False
        self.rollout_position = 0

    def act(self, actor_obs, _critic_obs=None):
        if not self._initialized:
            raise RuntimeError("initialize must be called before act")
        if self._awaiting_observe:
            raise RuntimeError("observe must be called before the next act")
        actor_obs = actor_obs.to(device=self.device, dtype=torch.float32)
        if self._step == 0:
            self._start_hidden.copy_(self._live_hidden)
        self._obs[self._step].copy_(actor_obs)
        self._reset[self._step].copy_(self._reset_mask)
        self.actor_normalizer.update(actor_obs)
        normalized = self.actor_normalizer.normalize(actor_obs)
        action, _, next_hidden = self.actor.step(
            normalized,
            self._live_hidden,
            reset_mask=self._reset_mask,
            deterministic=False,
            with_logprob=False,
        )
        self._actions[self._step].copy_(action)
        self._live_hidden.copy_(next_hidden)
        self._awaiting_observe = True
        return action

    def observe(
        self,
        next_actor_obs,
        _next_critic_obs,
        reward,
        done,
        *,
        timed_out=None,
        timeout_critic_obs=None,
    ):
        del timed_out, timeout_critic_obs
        if not self._initialized:
            raise RuntimeError("initialize must be called before observe")
        if not self._awaiting_observe:
            raise RuntimeError("act must be called before observe")
        next_actor_obs = next_actor_obs.to(
            device=self.device, dtype=torch.float32
        )
        reward = reward.to(device=self.device, dtype=torch.float32)
        done = done.to(device=self.device, dtype=torch.bool)
        self._next_obs[self._step].copy_(next_actor_obs)
        self._rewards[self._step].copy_(reward)
        self._dones[self._step].copy_(done)
        self._live_hidden.masked_fill_(done.unsqueeze(-1), 0.0)
        self._reset_mask.copy_(done)
        self._step += 1
        self._awaiting_observe = False
        self.rollout_position = self._step
        if self._step < self.batch_length:
            return None
        self._replay.add(
            self._obs.transpose(0, 1).contiguous(),
            self._next_obs.transpose(0, 1).contiguous(),
            self._actions.transpose(0, 1).contiguous(),
            self._rewards.transpose(0, 1).contiguous(),
            self._dones.transpose(0, 1).contiguous(),
            self._reset.transpose(0, 1).contiguous(),
            self._start_hidden.contiguous(),
        )
        metrics = self.update()
        self._step = 0
        self.rollout_position = 0
        return metrics

    def update(self):
        batch = self._replay.sample(self.replay_batch_size)
        obs = batch["obs"]
        self.last_batch_obs_shape = tuple(obs.shape)
        normalized = self.actor_normalizer.normalize(
            obs.reshape(-1, self.actor_obs_dim)
        ).view_as(obs)
        wm = world_model_losses(
            self.actor,
            self.world,
            normalized,
            batch["actions"],
            batch["rewards"],
            batch["dones"],
            batch["start_hidden"],
            reset_mask=batch["reset"],
            sample_latents=True,
        )
        world_loss = wm["world_loss"] + self.kl_scale * wm["kl_loss"]
        self.world_optimizer.zero_grad(set_to_none=True)
        world_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.world_optimizer.param_groups[0]["params"]),
            self.max_grad_norm,
        )
        self.world_optimizer.step()

        imag = imagine_actor_critic(
            self.actor,
            self.world,
            self.critic,
            wm["posterior_hidden"].detach(),
            horizon=self.imagination_horizon,
            gamma=self.gamma,
            lambda_gae=self.lambda_gae,
            entropy_coef=self.entropy_coef,
            sample_latents=True,
        )
        self.actor_optimizer.zero_grad(set_to_none=True)
        imag["actor_loss"].backward(retain_graph=True)
        nn.utils.clip_grad_norm_(
            list(self.actor_optimizer.param_groups[0]["params"]),
            self.max_grad_norm,
        )
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad(set_to_none=True)
        imag["critic_loss"].backward()
        nn.utils.clip_grad_norm_(
            list(self.critic.parameters()), self.max_grad_norm
        )
        self.critic_optimizer.step()
        return {
            "world_loss": float(world_loss.detach()),
            "recon_loss": float(wm["recon_loss"].detach()),
            "reward_loss": float(wm["reward_loss"].detach()),
            "kl_loss": float(wm["kl_loss"].detach()),
            "policy_loss": float(imag["actor_loss"].detach()),
            "value_loss": float(imag["critic_loss"].detach()),
            "imagined_return": float(imag["imagined_return"]),
            "policy_entropy": float(imag["policy_entropy"]),
            "clip_ratio": 0.0,
            "minibatches": 1,
            "actor_updates": 1,
            "value_updates": 1,
            "epochs": 1,
        }

    def deploy_state_dict(self):
        return deploy_actor_state_dict(self.actor)
