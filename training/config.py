"""Default training configuration.

This is the standalone (single-process) trainer's config. Only ``DEFAULT_CONFIG``
is kept here; the legacy distributed ``Config`` class (Redis overrides, Postgres
bootstrap, S3 parameter server) is intentionally not part of this trainer.
"""

DEFAULT_CONFIG = {
    "config_version": 2,
    "policy_format_version": 2,
    "obs": {
        "num_obs": 384,
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
        # When True, zero obs[372:380] (tyre slip) in training so a deploy model
        # trained without tyre-slip sensing matches gym/car (which publish zeros).
        # Default off.
        "zero_tyre_slip_obs": False,
        # 1v1: when enabled, an opponent-relative block of size opponent_obs_dim is
        # appended to the observation (num_obs becomes 384 + opponent_obs_dim). Off
        # by default so the solo (1v0) observation stays 384-dim and unchanged.
        "enable_opponent_obs": False,
        "opponent_obs_dim": 6,
        # Range gate for opponent-relative obs masking (matches passing_gate_*).
        "opp_obs_ahead_m": 40.0,
        "opp_obs_behind_m": 20.0,
        "future_track_num_points": 60,
        "future_track_horizon_s": 6.0,
        # Floor for the speed-scaled lookahead so the policy still sees the upcoming
        # track while stopped/crawling (speed*horizon -> 0 collapses all samples
        # onto the current point). Only affects speeds below
        # future_track_min_lookahead_m / future_track_horizon_s.
        "future_track_min_lookahead_m": 5.0,
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
        # control_dt = sim_dt * control_interval = 0.005 * 10 = 0.05 s (20 Hz),
        # matching libs/f1tenth_contract CONTROL_HZ / deploy vehicle_obs.
        "control_interval": 10,
        "sim_dt": 0.005,
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
        "reset_speed_min_mps": 1.0,
        "reset_speed_max_mps": 4.0,
        # Lateral inset (m) from each track edge when sampling a spawn offset, so a
        # car never starts with a wheel on the boundary.
        "reset_spawn_margin_m": 0.2,
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
        # Tyre-slip denominators (modern PhysX form, m/s), scaled down from the
        # full-car PhysX defaults 1.0 / 0.1 / 4.0 for the 1/10 car. Single source
        # for the slip definition: the TorchSim tire model (f1tenth_sim.dynamics)
        # and the on-car C++ builder both normalize by |v_fwd| + one of these
        # offsets (active = drive/brake applied, passive = coasting). The deprecated
        # compute_tyre_slip uses the same offsets.
        "slip_min_lat": 0.2,
        "slip_min_active_long": 0.1,
        "slip_min_passive_long": 0.4,
        # Torch-sim knobs. "model":
        # "dynamic" (Pacejka tires + load transfer + wheel spin) or "kinematic"
        # (Tier 0); "suspension_mode": "quasi_static" or "dynamic";
        # "internal_substeps" subdivides each sim_dt for extra stability.
        # Longitudinal action is always force/brake effort (ADR 0006).
        "torch_sim": {
            "model": "dynamic",
            "suspension_mode": "quasi_static",
            "internal_substeps": 1,
        },
        "longitudinal_mode": "force",
        # Competition sim track (dfr_f1tenth_gym dev-humble maps/IV_2026_SIM).
        "track": "IV_2026_SIM",
        # --- 1v1 opponent (hard 1v1: exactly one opponent) ---
        # opponent_strategy: None (1v0 / solo), "scripted" (centerline follower),
        # "policy" (frozen-policy self-play opponent), or "mixed" (per-env mix of
        # scripted + policy, the mixed opponent population).
        "opponent_strategy": None,
        # Per-env sampling weights for the "mixed" opponent strategy. On each reset
        # a row is assigned scripted vs policy with these (normalized) probabilities.
        # policy_speed_cap_prob: fraction of policy-mode rows given a random speed
        # cap (m/s), so a share of self-play opponents drive their normal line but
        # coast at a slower cruise -- realistically driven yet passable cars, so the
        # ego learns to overtake (not just follow). The cap is sampled skewed toward
        # the high end of policy_speed_cap_range, so most capped opponents are
        # fast-but-passable (near ego pace) with a thin slow tail.
        "opponent_mix": {
            "scripted_weight": 0.25,
            "policy_weight": 0.75,
            "policy_speed_cap_prob": 0.5,
            "policy_speed_cap_range": [3.5, 8.0],
        },
        # Scripted opponent: track follower kept below ego pace so an overtake is
        # feasible. Closed-loop P-control holds this setpoint in m/s;
        # opponent_target_speed_range (if set) samples a per-env cruise instead, and
        # opponent_lateral_offset_m holds a random off-centerline racing line so the
        # scripted cars are a varied slow field rather than one robotic pattern.
        "opponent_target_speed": 2.5,
        "opponent_target_speed_range": [2.0, 4.0],
        "opponent_lateral_offset_m": 0.5,
        "opponent_spawn_gap_min_m": 3.0,
        "opponent_spawn_gap_max_m": 20.0,
        "opponent_spawn_behind_prob": 0.3,
        "opponent_spawn_lateral_independent": True,
        "opponent_reset_speed_min_mps": 1.0,
        "opponent_reset_speed_max_mps": 4.0,
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
        # Body envelope (full length x width, metres) used for the oriented-box
        # collision predicate, car-car contact, and eval rendering. Provisional
        # standard 1/10 Traxxas Slash 4x4 spec (0.568 x 0.296 m); replace with a
        # tape measurement of the assembled car when available.
        "car_length": 0.568,
        "car_width": 0.296,
        "collision_margin_m": 0.0,
        # Per-episode domain randomization. Enabled with narrow bands around the
        # nominal car so the policy does not overfit a razor-edge grip line, without
        # forcing it to hedge against extreme physics. Widen these for a dedicated
        # sim2real robustness phase; the remaining knobs (drive_scale, steer_bias)
        # stay neutral for now.
        "domain_randomization": {
            "enabled": True,
            "tire_friction_range": [0.60, 0.70],
            "vehicle_mass_range": [3.6, 3.9],
            "mass_scale_range": [0.97, 1.03],
            "action_latency_steps_range": [0, 1],
            "obs_latency_steps_range": [0, 1],
            "obs_noise_std_range": [0.0, 0.01],
            "action_latency_steps_max": 3,
            # drive_scale multiplies drive force (motor/gearing spread), steer_bias
            # (rad) is a steering-alignment offset. Widen for sim2real.
            "drive_scale_range": [1.0, 1.0],
            "steer_bias_range": [0.0, 0.0],
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
        "passing_gate_ahead_m": 40.0,
        "passing_gate_behind_m": 20.0,
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
        # Overtake-completed bonus: one-time +overtake_bonus_k when the opponent
        # goes from ahead to behind within overtake_gap_m on the centerline (a
        # genuine pass, not a lap wrap). Only active when an "overtake" entry is
        # added to reward_scales (the trainer does this for 1v1).
        "overtake_bonus_k": 1.0,
        "overtake_gap_m": 5.0,
        # Additive combined-slip penalty shaping. slip_angle_weight balances the
        # (radian) slip-angle term against the (unitless) slip-ratio term; the
        # deadzones (ratio unitless, angle radians) carve out a controlled
        # grip-limit regime that is not penalized. Both default to 0.0 so every bit
        # of slip is penalized (GT Sophy-faithful); raise the deadzones only if the
        # policy is found to be under-driving the tyres.
        "slip_angle_weight": 1.0,
        "slip_deadzone_ratio": 0.0,
        "slip_deadzone_angle": 0.0,
        # Global downscale applied to the summed reward to keep per-step total and
        # value targets O(1) (progress alone was ~9/step before). Preserves the
        # relative balance between the individual reward terms.
        "global_reward_scale": 0.2,
        "reward_scales": {
            "progress": 5.0,
            "lateral": 1.0,
            "oob_penalty": 0.6,
            # Lowered from 0.05: the additive combined-slip penalty is larger in
            # magnitude than the previous slip_ratio * slip_angle product.
            "tyre_slip_penalty": 0.02,
            # Mild jerk penalty to curb bang-bang throttle/steer.
            "smoothness": 0.05,
        },
    },
    "model": {
        "hidden_layers": [512, 512, 512],
        "num_quantiles": 32,
        "rew_gamma": 0.9896,
        "n_step": 7,
        "alpha": 0.01,
        "replay_buffer_limit": 10**7,
        "batch_size": 1024,
        "minimum_train_transitions": 5_000,
        # With the canonical 512 envs and batch size 1024, the former one update
        # per vector tick sampled two replay rows per collected transition.
        "sampled_replay_rows_per_transition": 2.0,
    },
    "schedule": {
        "total_transitions": 256_000_000,
        "log_interval_transitions": 51_200,
        "export_interval_transitions": 5_120_000,
        "eval_interval_transitions": 0,
    },
    "selfplay": {
        "snapshot_interval_transitions": 10_240_000,
        "refresh_interval_transitions": 2_560_000,
        "pool_size": 10,
        "sample_mode": "mixed",
        "mixed_latest_prob": 0.5,
    },
}
