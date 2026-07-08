"""Default training configuration.

This is the standalone (single-process) trainer's config. Only ``DEFAULT_CONFIG``
is kept here; the legacy distributed ``Config`` class (Redis overrides, Postgres
bootstrap, S3 parameter server) is intentionally not part of this trainer.
"""

DEFAULT_CONFIG = {
    "obs": {
        "num_obs": 380,
        # Observation normalization is now done with empirical running statistics in
        # the trainer (ObsNormalizer), driven by values actually experienced. Keep
        # the env-side fixed scales at 1.0 so near-raw obs reach the normalizer.
        "obs_scales": {
            "lin_vel": 1.0,
            "ang_vel": 1.0,
            "lin_acc": 1.0,
        },
        # Loose guard only: bound a rare spin/contact transient before it can skew
        # the running variance. Real scaling is handled by the trainer normalizer.
        "clip_obs": 50.0,
        # Trainer-side ObsNormalizer parameters.
        "norm_clip": 10.0,
        "norm_eps": 1e-8,
        "contact_margin_m": 0.08,
        # When True, zero obs[372:380] in training so a deploy model trained without
        # tyre-slip sensing matches gym/car (which publish zeros). Default off.
        "zero_tyre_slip_obs": False,
        # 1v1: when enabled, an opponent-relative block of size opponent_obs_dim is
        # appended to the observation (num_obs becomes 380 + opponent_obs_dim). Off
        # by default so the solo (1v0) observation stays 380-dim and unchanged.
        "enable_opponent_obs": False,
        "opponent_obs_dim": 7,
        "future_track_num_points": 60,
        "future_track_horizon_s": 6.0,
        # Floor for the speed-scaled lookahead so the policy still sees the upcoming
        # track while stopped/crawling (speed*horizon -> 0 collapses all samples
        # onto the current point). Only affects speeds below
        # future_track_min_lookahead_m / future_track_horizon_s.
        "future_track_min_lookahead_m": 5.0,
        # Deprecated: corridor edges use per-vertex w_tr_left_m / w_tr_right_m.
        "future_track_width": 2.2,
    },
    "env": {
        "num_actions": 2,
        # Static fallback only. The trainer (standalone_trainer.build_config) derives
        # the real horizon from the track via utils.episode_length_for_track so each
        # track gets ~episode_lap_multiplier laps plus overtaking margin.
        "episode_length": 120.0,
        # Reference pace and lap count used to size the episode horizon per track.
        "expected_lap_speed_mps": 3.5,
        "episode_lap_multiplier": 3.0,
        # IV_2026_SIM centerline loop is ~144 m; at ~3.2 m/s a full lap needs ~43 s.
        # 45 s gives one lap plus margin at 10 Hz control (450 steps) without changing
        # control_dt or episode_length semantics.
        # control_dt = sim_dt * control_interval = 0.005 * 20 = 0.1 s (10 Hz).
        # Halving sim_dt (0.01 -> 0.005) while doubling decimation keeps episode
        # length in seconds and max_episode_steps unchanged (~450 for 45 s).
        "control_interval": 20,
        "sim_dt": 0.005,
        # Tuned via scripts/sweep_physics_integration.py: substep counts 2..10 pass
        # the physics_check stability gate for normal upright driving. Finer substeps
        # (effective 0.625 ms with sim_dt=0.005) help car-car contact stability in
        # 1v1; raised 4->8 after high-speed contacts into a near-stationary self-play
        # opponent produced constraint-force NaNs at high env counts.
        "sim_substeps": 8,
        # Softer constraint solve window; must stay >= 2 * sim_dt (Genesis Newton gate).
        "constraint_timeconst": 0.02,
        "solver_iterations": 50,
        "solver_ls_iterations": 50,
        "show_fps": False,
        # Opt-in torch.compile of the pure-tensor observation math. Off by default
        # (no warmup/recompile risk); enable on the GPU training target after
        # confirming a steady-state throughput gain that outweighs compile warmup.
        "compile_obs": False,
        "clip_actions": 1.0,
        "simulate_action_latency": True,
        "term_oob_margin_m": 0.15,
        # Strict OOB: chicane cuts end the episode quickly so skipping S-bends
        # cannot amortize off-track time against on-track progress.
        "term_oob_max_consecutive": 2,
        "term_speed_threshold": 0.2,
        "term_not_moving_time_s": 2.0,
        "term_not_moving_min_ds": 1e-3,
        "term_heading_error_rad": 3.0,
        "car_spawn_pos": (0.0, 0.0, 0.01),
        "car_spawn_rot": (0.0, 0.0, 0.0),
        "joint_names": [
            "left_rear_wheel_joint",
            "right_rear_wheel_joint",
        ],
        "default_joint_angles": {
            "left_rear_wheel_joint": 0.0,
            "right_rear_wheel_joint": 0.0,
        },
        # Observation/reset throttle scaling only — longitudinal cap comes from power+drag.
        "max_speed": 15.0,
        # Real F1TENTH servo hard-clamps the steering at ~0.33 rad (19 deg): see
        # analysis/analyze_full_lock.py, which measures a ~0.94 m min turning radius
        # from rosbags (servo saturates at 0.85 -> 0.33 rad effective wheel angle).
        # Training at 0.44 rad let the policy assume an unreachable 0.70 m radius and
        # understeer into walls on tight corners; 0.33 rad makes the sim match reality
        # (0.325 / tan(0.33) = 0.95 m min radius).
        "max_steer": 0.33,  # radians (alias for delta_max)
        "delta_max": 0.33,  # radians
        "wheelbase": 0.325,
        "track_width": 0.20,
        "wheel_radius": 0.05,
        "f_drive_max": 23.0,
        "f_brake_max": 23.0,
        "power_max": 255.0,
        "k_drive_front": 0.5,
        "t_delta": 0.1,
        "c_roll": 0.0,
        "dragcoeff": 0.075,
        "tire_friction": 0.65,
        "v_eps": 0.1,
        "enable_aero_drag": True,
        "drive_torque_sign": 1.0,
        # Competition sim track (dfr_f1tenth_gym dev-humble maps/IV_2026_SIM).
        "track": "IV_2026_SIM",
        # --- 1v1 opponent (hard 1v1: exactly one opponent) ---
        # opponent_strategy: None (1v0 / solo), "scripted" (centerline follower),
        # "policy" (frozen-policy self-play opponent), or "mixed" (per-env mix of
        # scripted + policy, the mixed opponent population).
        "opponent_strategy": None,
        # Per-env sampling weights for the "mixed" opponent strategy. On each reset
        # a row is assigned scripted vs policy with these (normalized) probabilities.
        "opponent_mix": {
            "scripted_weight": 0.3,
            "policy_weight": 0.7,
        },
        # Scripted opponent: centerline follower kept below ego pace so an overtake
        # is feasible. Closed-loop P-control holds this setpoint in m/s.
        "opponent_target_speed": 2.5,
        "opponent_spawn_gap_m": 7.0,
        "opponent_kp_ey": 1.0,
        "opponent_kh_heading": 1.0,
        "opponent_kp_speed": 1.0,
        # Collision termination: anisotropic ego-frame box overlap (see
        # terminations.collision_mask). An optional shaped any-collision penalty
        # (collision_k, gated by a "collision" reward scale) adds a dense per-step
        # signal on top of the forfeited progress from episode reset.
        "term_on_collision": True,
        # Only terminate on a collision whose closing speed ||v_ego - v_opp||
        # (world-frame, m/s) exceeds this threshold. Low-speed contacts below it
        # still incur the collision/rear-end penalties and full contact physics but
        # let the episode continue, so the agent learns to recover from light taps
        # instead of resetting on every minor rub. Set to 0.0 to terminate on any
        # overlap (legacy behavior).
        "collision_term_speed_mps": 2.0,
        "car_length": 0.46,
        "car_width": 0.30,
        "collision_margin_m": 0.0,
        # Deprecated: superseded by car_length/car_width/collision_margin_m.
        "collision_dist_m": 0.4,
        # Per-episode domain randomization (disabled by default).
        "domain_randomization": {
            "enabled": False,
            "tire_friction_range": [0.6, 0.85],
            "ground_friction_range": [0.6, 0.85],
            "vehicle_mass_range": [3.2, 4.2],
            "mass_scale_range": [0.9, 1.1],
            "action_latency_steps_range": [0, 2],
            "obs_latency_steps_range": [0, 1],
            "obs_noise_std_range": [0.0, 0.02],
            "action_latency_steps_max": 3,
        },
    },
    "reward": {
        # Reward: course progress (primary), off-course penalty (~time-off x speed^2),
        # tyre-slip penalty, and a small smoothness shaping term. No explicit speed
        # reward: speed is induced purely by progress per step under the gamma=0.9896
        # discount.
        "progress_k_fwd": 5.0,
        "progress_k_back": 5.0,
        "progress_max_lateral_m": 1.0,
        "oob_margin_m": 0.2,
        # Off-course penalty R_soc = -(time off course) * speed^2. The per-step time
        # off course is constant and folds into oob_k; tuned so the penalty at racing
        # speed (~6 m/s) is comparable to the previous shaping while escalating
        # quadratically with speed for fast excursions.
        "oob_k": 0.3,
        "lateral_k": 0.5,
        # 1v1 passing reward gain: per-step reward = passing_k * (ego_ds - opp_ds),
        # i.e. track position gained on the opponent. Only active when a "passing"
        # entry is added to reward_scales (the trainer does this for 1v1), so 1v0 is
        # unaffected.
        "passing_k": 5.0,
        # Any-collision penalty gain: per-step reward = -collision_k on car-to-car
        # overlap (same predicate as collision termination). Only active when a
        # "collision" entry is added to reward_scales (the trainer does this for
        # 1v1), so 1v0 is unaffected.
        "collision_k": 5.0,
        # Rear-end penalty gain Rr: per-step reward = -rear_end_k * ||v_ego - v_opp||^2
        # when the agent collides with an opponent that is ahead on the centerline.
        # Scales with squared closing speed so high-speed rear-ends are punished
        # hardest. Only active when a "rear_end" entry is added to reward_scales.
        "rear_end_k": 5.0,
        # Global downscale applied to the summed reward to keep per-step total and
        # value targets O(1) (progress alone was ~9/step before). Preserves the
        # relative balance between the individual reward terms.
        "global_reward_scale": 0.2,
        "reward_scales": {
            "progress": 5.0,
            "lateral": 1.0,
            "oob_penalty": 0.6,
            "tyre_slip_penalty": 0.05,
            # Mild jerk penalty to curb bang-bang throttle/steer.
            "smoothness": 0.05,
        },
    },
    "model": {
        "hidden_layers": [512, 512, 512],
        "num_quantiles": 32,
        "rew_gamma": 0.9896,
        "n_step": 7,
        "replay_tables": ["1v0"],  # "1v1", "mistake_learning"],
        "minimum_train_samples": 40000,
        "batches_per_epoch": 6000,
        "replay_buffer_limit": 10**7,
        "batch_size": 1024,
        "update_to_data_ratio": 0.01,
    },
    "selfplay": {
        "snapshot_interval": 20_000,
        "refresh_interval": 5_000,
        "pool_size": 5,
        "sample_mode": "mixed",
        "mixed_latest_prob": 0.8,
    },
}
