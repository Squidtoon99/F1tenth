"""One racing policy per car, across the actor families this repo trains."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from gigaflow_f1tenth.config import ExperimentConfig
from gigaflow_f1tenth.viewer.replay import (
    ViewerError,
    checkpoint_architecture,
    load_checkpoint_actor,
    trunk_mismatch,
)

SENSOR_ARCHITECTURES = frozenset(
    {
        "lidar_cnn_gru",
        "tinylidarnet_gru",
        "lidar_dreamer_ls3c",
    }
)
MAX_LINEUP = 8


def kind_label(arch: Mapping[str, Any] | None) -> str:
    if not isinstance(arch, Mapping):
        return "gigaflow-gru"
    name = str(arch.get("name", ""))
    if name == "lidar_cnn_gru_conditioned":
        return "gigaflow-gru"
    if name == "lidar_cnn_ff_conditioned":
        return "gigaflow-feedforward"
    if name == "lidar_cnn_frame_stack_conditioned":
        return "gigaflow-frame-stack"
    if name == "tinylidarnet_gru":
        return "tinylidarnet"
    if name == "lidar_dreamer_ls3c":
        return "lidar-dreamer"
    if name == "lidar_cnn_gru":
        horizon = int(arch.get("action_horizon", 1) or 1)
        if horizon > 1 or int(arch.get("action_dim", 2)) > 2:
            return "action-chunk"
        return "cnn-gru"
    return name or "unknown"


def is_sensor_architecture(arch: Mapping[str, Any] | None) -> bool:
    return isinstance(arch, Mapping) and str(arch.get("name", "")) in SENSOR_ARCHITECTURES


def sensor_compatible(cfg: ExperimentConfig, arch: Mapping[str, Any] | None) -> bool:
    if not is_sensor_architecture(arch):
        return False
    assert arch is not None
    lidar = int(arch.get("lidar_dim", cfg.agents.lidar_dim))
    proprio = int(arch.get("proprio_dim", cfg.agents.proprio_dim))
    return (
        lidar == int(cfg.agents.lidar_dim)
        and proprio == int(cfg.agents.proprio_dim)
        and lidar + proprio == int(cfg.agents.sensor_obs_dim)
    )


_RACE_KINDS = frozenset(
    {
        "gigaflow-gru",
        "gigaflow-feedforward",
        "gigaflow-frame-stack",
        "cnn-gru",
        "action-chunk",
        "tinylidarnet",
        "lidar-dreamer",
    }
)


def describe_checkpoint(
    checkpoint: Path, cfg: ExperimentConfig
) -> dict[str, Any] | None:
    """Metadata for a checkpoint the live sim can step. Incompatible files are omitted."""
    arch = checkpoint_architecture(checkpoint)
    if is_sensor_architecture(arch):
        if not sensor_compatible(cfg, arch):
            return None
    elif trunk_mismatch(cfg, arch) is not None:
        return None
    kind = kind_label(arch if isinstance(arch, Mapping) else None)
    if kind not in _RACE_KINDS:
        return None
    return {
        "name": checkpoint.name,
        "path": str(checkpoint.resolve()),
        "kind": kind,
        "condition_dim": (
            int(arch["condition_dim"])
            if isinstance(arch, Mapping) and "condition_dim" in arch
            else None
        ),
    }


class RaceDriver:
    """Steps one checkpoint. Hidden state is per sim slot."""

    def __init__(
        self,
        actor: Any,
        *,
        kind: str,
        label: str,
        path: Path,
        condition: torch.Tensor | None,
        normalizer: Any | None,
        frame_stack: int,
        family: str,
    ) -> None:
        self.actor = actor
        self.kind = kind
        self.label = label
        self.path = path
        self.condition = condition
        self._normalizer = normalizer
        self.frame_stack = int(frame_stack)
        self.family = family
        self._family = family
        self._hidden: torch.Tensor | None = None
        self._frames: torch.Tensor | None = None

    def reset(self, n_slots: int, device: str) -> None:
        self._hidden = self.actor.initial_hidden(int(n_slots), device=device)
        self._frames = None

    def act(
        self,
        obs: torch.Tensor,
        slot_ids: torch.Tensor,
        live_cond: torch.Tensor,
    ) -> torch.Tensor:
        if self._hidden is None:
            raise ViewerError("race driver was stepped before reset")
        if self._normalizer is not None:
            obs = self._normalizer.normalize(obs)
        hidden = self._hidden.index_select(0, slot_ids)
        if self._family == "gigaflow":
            model_obs = self._stacked(obs, slot_ids) if self.frame_stack > 1 else obs
            if self.condition is None:
                cond = live_cond
            else:
                cond = self.condition.expand(obs.shape[0], -1)
            out = self.actor.forward(
                model_obs, cond, hidden, deterministic=True
            )
            actions = out.actions
            nxt = out.hidden
        else:
            from f1tenth_policy.layout import executed_plant_action

            actions, _logp, nxt = self.actor.step(
                obs, hidden, deterministic=True
            )
            actions = executed_plant_action(actions)
        updated = self._hidden.clone()
        updated.index_copy_(0, slot_ids, nxt.detach())
        self._hidden = updated
        return actions.detach()

    def _stacked(self, obs: torch.Tensor, slot_ids: torch.Tensor) -> torch.Tensor:
        if self._frames is None or self._frames.shape[0] != self._hidden.shape[0]:
            self._frames = obs.new_zeros(
                self._hidden.shape[0], self.frame_stack, obs.shape[-1]
            )
        buf = self._frames.index_select(0, slot_ids).clone()
        if self.frame_stack > 1:
            buf = torch.roll(buf, shifts=-1, dims=1)
        buf[:, -1] = obs
        self._frames.index_copy_(0, slot_ids, buf)
        return buf


def _frame_stack_of(actor: Any) -> int:
    return int(getattr(actor, "frame_stack", 1) or 1)


def _load_sensor_driver(checkpoint: Path, arch: Mapping[str, Any], device: str) -> RaceDriver:
    from f1tenth_policy.actor import actor_from_architecture
    from f1tenth_policy.normalizer import load_obs_normalizer

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ViewerError(f"unsupported checkpoint payload: {checkpoint}")
    state = payload.get("actor")
    if not isinstance(state, Mapping):
        state = payload.get("actor_state_dict")
    if not isinstance(state, Mapping):
        raise ViewerError("checkpoint missing actor weights")
    try:
        actor = actor_from_architecture(arch)
        actor.load_state_dict(state)
    except Exception as exc:  # noqa: BLE001
        raise ViewerError(f"failed to load actor from {checkpoint}: {exc}") from exc
    actor.eval()
    normalizer = None
    obs_norm = payload.get("obs_norm")
    if isinstance(obs_norm, Mapping) and "mean" in obs_norm and "var" in obs_norm:
        normalizer = load_obs_normalizer(
            obs_norm["mean"],
            obs_norm["var"],
            device=torch.device(device),
            count=float(obs_norm["count"]) if "count" in obs_norm else None,
        )
    return RaceDriver(
        actor.to(device),
        kind=kind_label(arch),
        label=checkpoint.name,
        path=checkpoint,
        condition=None,
        normalizer=normalizer,
        frame_stack=1,
        family="sensor",
    )


def load_race_driver(
    cfg: ExperimentConfig, checkpoint: Path, device: str
) -> RaceDriver:
    checkpoint = Path(checkpoint).resolve()
    arch = checkpoint_architecture(checkpoint)
    if is_sensor_architecture(arch):
        if not sensor_compatible(cfg, arch):
            raise ViewerError(
                f"{checkpoint.name} observation does not match the live sim"
            )
        assert arch is not None
        return _load_sensor_driver(checkpoint, arch, device)
    try:
        actor, condition = load_checkpoint_actor(cfg, checkpoint, device)
    except ViewerError:
        raise
    actor.eval()
    from gigaflow_f1tenth.model import actor_architecture_from_module

    built = actor_architecture_from_module(actor)
    return RaceDriver(
        actor,
        kind=kind_label(built),
        label=checkpoint.name,
        path=checkpoint,
        condition=condition,
        normalizer=None,
        frame_stack=_frame_stack_of(actor),
        family="gigaflow",
    )


def sensor_control_spec() -> tuple[float, int, float]:
    """Control rate and steer increment of an e2e sensor policy step."""
    from f1tenth_policy.layout import CONTROL_HZ, STEERING_DELTA_MAX_RAD
    from gigaflow_f1tenth.config import SIM_DT

    hz = float(CONTROL_HZ)
    interval = int(round((1.0 / hz) / float(SIM_DT)))
    return hz, interval, float(STEERING_DELTA_MAX_RAD)


def install_sensor_command_scale(sim: Any) -> None:
    """Make +1 steer add the training delta, once per training control tick."""
    hz, _interval, delta = sensor_control_spec()
    sim.vehicle_params.steering_action_mode = "delta"
    sim.vehicle_params.steering_delta_max = delta
    sim.sim_params.steering_action_mode = 1
    sim.sim_params.steering_delta_max = delta
    sim.sim_params.control_dt = 1.0 / hz


class StridedCommandHistory:
    """0.30 s steer window and 0.10 s throttle predecessor at 20 Hz."""

    def __init__(self) -> None:
        self._steer: torch.Tensor | None = None
        self._long: torch.Tensor | None = None
        self._stride = 1
        self._steer_len = 4
        self._long_len = 2

    def reset(self, n_slots: int, device: str) -> None:
        from f1tenth_policy.layout import HISTORY_STRIDE, LONG_HISTORY, STEER_HISTORY

        self._stride = int(HISTORY_STRIDE)
        self._steer_len = int(STEER_HISTORY)
        self._long_len = int(LONG_HISTORY)
        self._steer = torch.zeros(int(n_slots), self._steer_len, device=device)
        self._long = torch.zeros(int(n_slots), self._long_len, device=device)

    def push(self, steer: torch.Tensor, effort: torch.Tensor) -> None:
        if self._steer is None or self._long is None:
            raise ViewerError("command history was pushed before reset")
        self._steer = torch.roll(self._steer, shifts=1, dims=1)
        self._steer[:, 0] = steer.detach()
        self._long = torch.roll(self._long, shifts=1, dims=1)
        self._long[:, 0] = effort.detach()

    def project(self, rows: torch.Tensor, slot_ids: torch.Tensor) -> torch.Tensor:
        from f1tenth_policy.layout import (
            STEER_DELTA0,
            STEER_DELTA1,
            STEER_DELTA2,
            STEER_T,
            STEER_T1,
            STEER_T2,
            THROTTLE_CURRENT,
            THROTTLE_PRED,
        )

        if self._steer is None or self._long is None:
            raise ViewerError("command history was read before reset")
        out = rows.clone()
        step = self._stride
        steer = self._steer.index_select(0, slot_ids)
        long = self._long.index_select(0, slot_ids)
        out[:, STEER_T] = steer[:, 0]
        out[:, STEER_T1] = steer[:, step]
        out[:, STEER_T2] = steer[:, 2 * step]
        out[:, STEER_DELTA0] = steer[:, 0] - steer[:, step]
        out[:, STEER_DELTA1] = steer[:, step] - steer[:, 2 * step]
        out[:, STEER_DELTA2] = steer[:, 2 * step] - steer[:, 3 * step]
        out[:, THROTTLE_CURRENT] = long[:, 0]
        out[:, THROTTLE_PRED] = long[:, step]
        return out
