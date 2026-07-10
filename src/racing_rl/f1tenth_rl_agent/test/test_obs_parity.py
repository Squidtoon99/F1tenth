"""Parity test: obs_core vs the real f1tenth_env observation pipeline.

This guarantees the deployed observation equals the one the policy trained on. It
imports the real ``f1tenth_env.observations`` and ``f1tenth_env.utils`` modules
(genesis-free after configuring the env runtime dtype/device; the lazy package
``__init__`` keeps ``env.py`` out of the import) and compares their
``build_observation`` output against ``obs_core.ObservationBuilder``.

The test is gated on ``genesis`` being importable (it is in the training venv, not
the amd64 ROS dev container) purely to keep the existing skip behaviour there.
"""

import os
import sys
import tempfile
import types

import numpy as np
import pytest

# Keep matplotlib (imported transitively by genesis) from complaining.
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp())

import torch  # noqa: E402

from f1tenth_rl_agent import obs_core  # noqa: E402
from f1tenth_rl_agent.interfaces import default_obs_cfg  # noqa: E402

# NOTE: genesis is only importable in the training venv (natively on macOS), not in
# the amd64 ROS dev container. We gate it inside the ``real_modules`` fixture (not at
# module import) so that a missing genesis skips only these two tests instead of
# aborting collection of the whole f1tenth_rl_agent test session.


def _find_monorepo_root() -> str:
    """Walk up from this test until we find the training/f1tenth_env package.

    In the monorepo the deployed package lives under src/racing_rl/ while the
    training env lives under training/, so the old fixed "up 3" no longer applies.
    """
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        if os.path.isdir(os.path.join(d, "training", "f1tenth_env")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    raise RuntimeError("could not locate training/f1tenth_env from " + __file__)


_REPO_ROOT = _find_monorepo_root()


def _ensure_training_on_path() -> None:
    training_dir = os.path.join(_REPO_ROOT, "training")
    if training_dir not in sys.path:
        sys.path.insert(0, training_dir)


def _stub_requests():
    """Stub `requests` so utils.py module-level load_tracks() returns immediately."""
    if "requests" in sys.modules:
        return

    class _Resp:
        status_code = 404

        def json(self):
            return {}

    stub = types.ModuleType("requests")
    stub.get = lambda *a, **k: _Resp()
    sys.modules["requests"] = stub


@pytest.fixture(scope="module")
def real_modules():
    pytest.importorskip("genesis")
    _ensure_training_on_path()
    _stub_requests()
    from f1tenth_env import runtime as rt

    rt.configure(float_dtype=torch.float32, int_dtype=torch.int32,
                 dev=torch.device("cpu"), eps=1e-12)
    from f1tenth_env import observations as real_obs
    from f1tenth_env import utils as real_utils
    return real_utils, real_obs


def _make_track(n=240):
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False).astype(np.float32)
    cl = np.stack([25.0 * np.cos(th), 14.0 * np.sin(th)], axis=-1).astype(np.float32)
    wl = np.full(n, 1.6, dtype=np.float32)
    wr = np.full(n, 1.4, dtype=np.float32)
    return cl, wl, wr


def _yaw_quat_wxyz(yaw: np.ndarray) -> torch.Tensor:
    half = 0.5 * yaw
    w = np.cos(half)
    z = np.sin(half)
    quat = np.zeros((yaw.shape[0], 4), dtype=np.float32)
    quat[:, 0] = w
    quat[:, 3] = z
    return torch.tensor(quat, dtype=torch.float32)


def test_obs_parity(real_modules):
    real_utils, real_obs = real_modules
    device = torch.device("cpu")
    cl, wl, wr = _make_track()
    obs_cfg = default_obs_cfg()

    builder = obs_core.ObservationBuilder(cl, wl, wr, obs_cfg, device=device)

    track_state = {
        "centerline": cl,
        "w_tr_left_torch": torch.tensor(wl, dtype=torch.float32),
        "w_tr_right_torch": torch.tensor(wr, dtype=torch.float32),
        "track_geom_cache": {},
        "frenet_step_cache": {},
    }

    rng = np.random.default_rng(0)
    max_diff = 0.0
    for _ in range(20):
        b = 4
        # sample positions near the track with random lateral offset
        idx = rng.integers(0, cl.shape[0], size=b)
        base = cl[idx]
        offset = rng.uniform(-1.0, 1.0, size=(b, 2)).astype(np.float32)
        pos_xy = base + offset
        pos = np.concatenate([pos_xy, np.zeros((b, 1), np.float32)], axis=1)
        base_pos = torch.tensor(pos, dtype=torch.float32)

        yaw = rng.uniform(-np.pi, np.pi, size=b).astype(np.float32)
        base_quat = _yaw_quat_wxyz(yaw)

        base_lin_vel = torch.tensor(
            rng.uniform(-1, 6, size=(b, 3)).astype(np.float32)
        )
        base_lin_vel[:, 2] = 0.0
        base_ang_vel = torch.tensor(
            rng.uniform(-1, 1, size=(b, 3)).astype(np.float32)
        )
        base_lin_acc = torch.tensor(
            rng.uniform(-2, 2, size=(b, 3)).astype(np.float32)
        )
        base_lin_acc[:, 2] = 0.0
        last_actions = torch.tensor(
            rng.uniform(-1, 1, size=(b, 2)).astype(np.float32)
        )

        # fresh caches per sample so frenet recomputes
        track_state["track_geom_cache"] = {}
        track_state["frenet_step_cache"] = {}
        episode_steps = torch.zeros(b, dtype=torch.int32)
        step_state = real_utils.build_step_state(
            base_pos=base_pos,
            episode_steps_buf=episode_steps,
            track_state=track_state,
            device=device,
            cache_id="parity",
        )
        step_state["tyre_slip"] = base_lin_vel.new_zeros((b, 8))
        step_state["tyre_load"] = base_lin_vel.new_zeros((b, 4))

        real = real_obs.build_observation(
            num_obs=obs_cfg["num_obs"],
            num_envs=b,
            base_lin_vel=base_lin_vel,
            base_ang_vel=base_ang_vel,
            base_lin_acc=base_lin_acc,
            last_actions=last_actions,
            base_pos=base_pos,
            base_quat=base_quat,
            obs_cfg=obs_cfg,
            step_state=step_state,
            device=device,
        )

        # Gym/ROS deploy has no per-wheel state; parity uses genesis slip values.
        slip = step_state["tyre_slip"]
        load = step_state["tyre_load"]

        mine = builder.build(
            base_lin_vel=base_lin_vel,
            base_ang_vel=base_ang_vel,
            base_lin_acc=base_lin_acc,
            last_actions=last_actions,
            base_pos=base_pos,
            base_quat_wxyz=base_quat,
            tyre_slip=slip,
            tyre_load=load,
        )

        diff = (real - mine).abs().max().item()
        max_diff = max(max_diff, diff)

    assert max_diff < 1e-4, f"obs parity diff too large: {max_diff}"


def test_obs_parity_1v1_opponent_block(real_modules):
    """390-dim parity with a non-trivial opponent-relative block appended."""
    real_utils, real_obs = real_modules
    device = torch.device("cpu")
    cl, wl, wr = _make_track()
    obs_cfg = default_obs_cfg(enable_opponent_obs=True)
    assert obs_cfg["num_obs"] == 390

    builder = obs_core.ObservationBuilder(cl, wl, wr, obs_cfg, device=device)
    track_state = {
        "centerline": cl,
        "w_tr_left_torch": torch.tensor(wl, dtype=torch.float32),
        "w_tr_right_torch": torch.tensor(wr, dtype=torch.float32),
        "track_geom_cache": {},
        "frenet_step_cache": {},
    }

    rng = np.random.default_rng(7)
    b = 2
    idx = rng.integers(0, cl.shape[0], size=b)
    base = cl[idx]
    offset = rng.uniform(-0.5, 0.5, size=(b, 2)).astype(np.float32)
    pos_xy = base + offset
    base_pos = torch.tensor(
        np.concatenate([pos_xy, np.zeros((b, 1), np.float32)], axis=1),
        dtype=torch.float32,
    )
    yaw = rng.uniform(-np.pi, np.pi, size=b).astype(np.float32)
    base_quat = _yaw_quat_wxyz(yaw)
    base_lin_vel = torch.tensor(rng.uniform(-1, 4, size=(b, 3)).astype(np.float32))
    base_lin_vel[:, 2] = 0.0
    base_ang_vel = torch.zeros(b, 3)
    base_lin_acc = torch.zeros(b, 3)
    last_actions = torch.zeros(b, 2)

    track_state["track_geom_cache"] = {}
    track_state["frenet_step_cache"] = {}
    episode_steps = torch.zeros(b, dtype=torch.int32)
    step_state = real_utils.build_step_state(
        base_pos=base_pos,
        episode_steps_buf=episode_steps,
        track_state=track_state,
        device=device,
        cache_id="parity_1v1",
    )
    step_state["tyre_slip"] = base_lin_vel.new_zeros((b, 8))
    step_state["tyre_load"] = base_lin_vel.new_zeros((b, 4))

    # Opponent ~7 m ahead on the synthetic oval (arc-length via index offset).
    mean_seg = float(np.mean(np.linalg.norm(cl[1:] - cl[:-1], axis=1)))
    gap_pts = max(1, int(round(7.0 / max(mean_seg, 1e-6))))
    opp_idx = (idx + gap_pts) % cl.shape[0]
    opp_pos_xy = cl[opp_idx] + rng.uniform(-0.3, 0.3, size=(b, 2)).astype(np.float32)
    opp_pos = torch.tensor(
        np.concatenate([opp_pos_xy, np.zeros((b, 1), np.float32)], axis=1),
        dtype=torch.float32,
    )
    opp_step = real_utils.build_step_state(
        base_pos=opp_pos,
        episode_steps_buf=episode_steps,
        track_state=track_state,
        device=device,
        cache_id="parity_1v1_opp",
    )

    ego_yaw = torch.tensor(yaw, dtype=torch.float32)
    ego_vel = base_lin_vel.clone()
    opp_vel = torch.tensor(rng.uniform(-1, 4, size=(b, 3)).astype(np.float32))

    self_agent = {
        "pos_xy": base_pos[:, :2],
        "yaw": ego_yaw,
        "vel_xy": ego_vel[:, :2],
        "s": step_state["frenet"]["s"],
        "ey": step_state["boundary"]["ey"],
        "L": step_state["frenet"]["L"],
    }
    other_agent = {
        "pos_xy": opp_pos[:, :2],
        "vel_xy": opp_vel[:, :2],
        "s": opp_step["frenet"]["s"],
        "ey": opp_step["boundary"]["ey"],
        "L": opp_step["frenet"]["L"],
    }
    opponent_block = real_obs.obs_opponent(self_agent, other_agent, obs_cfg)

    real = real_obs.build_observation(
        num_obs=obs_cfg["num_obs"],
        num_envs=b,
        base_lin_vel=base_lin_vel,
        base_ang_vel=base_ang_vel,
        base_lin_acc=base_lin_acc,
        last_actions=last_actions,
        base_pos=base_pos,
        base_quat=base_quat,
        obs_cfg=obs_cfg,
        step_state=step_state,
        device=device,
        opponent_block=opponent_block,
    )

    slip = step_state["tyre_slip"]
    load = step_state["tyre_load"]
    mine = builder.build(
        base_lin_vel=base_lin_vel,
        base_ang_vel=base_ang_vel,
        base_lin_acc=base_lin_acc,
        last_actions=last_actions,
        base_pos=base_pos,
        base_quat_wxyz=base_quat,
        tyre_slip=slip,
        tyre_load=load,
        opponent_block=opponent_block,
    )

    diff = (real - mine).abs().max().item()
    assert opponent_block.abs().sum().item() > 0.0
    assert diff < 1e-4, f"1v1 obs parity diff too large: {diff}"
