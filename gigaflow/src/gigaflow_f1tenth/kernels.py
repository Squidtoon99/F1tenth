"""Warp simulator kernel stage interfaces (N-agent worlds)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from gigaflow_f1tenth.config import ExperimentConfig
from gigaflow_f1tenth.tracks import PackedTrackAtlasView


@dataclass(frozen=True)
class WorldSlotLayout:
    """Flattened (world_id, agent_slot) indexing with compact masks."""

    num_worlds: int
    max_agents_per_world: int
    num_slots: int


def world_slot_layout(cfg: ExperimentConfig) -> WorldSlotLayout:
    n_worlds = cfg.worlds.num_worlds
    n_agents = cfg.worlds.max_agents_per_world
    return WorldSlotLayout(
        num_worlds=n_worlds,
        max_agents_per_world=n_agents,
        num_slots=n_worlds * n_agents,
    )


@dataclass
class SimulatorState:
    """Fixed-capacity SoA tensors. Concrete dtypes owned by n-agent-sim."""

    layout: WorldSlotLayout
    active: Any
    trainable: Any
    done: Any
    track_id: Any
    episode_id: Any
    # Pose / dynamics / sensors are attached by the simulator implementation.
    arrays: dict[str, Any]


@runtime_checkable
class Simulator(Protocol):
    def state(self) -> SimulatorState:
        ...

    def reset_agents(self, agent_mask: Any, seed: int) -> None:
        """Asynchronous per-agent respawn into verified empty regions."""

    def step(self, actions: Any) -> dict[str, Any]:
        """
        One control tick: fused physics substeps, contact, reward/reset, sensors.

        Returns at least:
          rewards [S], done [S], timeout [S], reset_mask [S],
          contact [S], wall_contact [S] (transition-time, pre-respawn),
          sensor_obs [S, 1097] (or a deferred reconstruction handle),
          critic_state (privileged masked agent sets, including per-slot
          track_id for the critic's ego-relative track preview).
        """


def slot_index(world_id: int, agent_slot: int, max_agents_per_world: int) -> int:
    return int(world_id) * int(max_agents_per_world) + int(agent_slot)


def build_simulator(
    cfg: ExperimentConfig,
    atlas: PackedTrackAtlasView | None,
    device: str,
    *,
    sync_no_respawn: bool | None = None,
) -> Simulator:
    """Construct the production Warp N-agent simulator.

    ``sync_no_respawn`` defaults to False (training async respawn). Evaluation
    suites must pass ``sync_no_respawn=True`` explicitly.
    """
    from gigaflow_f1tenth.sim.runtime import build_n_agent_simulator

    return build_n_agent_simulator(
        cfg, atlas, device, sync_no_respawn=sync_no_respawn
    )


def simulator_interface_gaps() -> tuple[str, ...]:
    from gigaflow_f1tenth.sim.runtime import INTERFACE_GAPS

    return INTERFACE_GAPS
