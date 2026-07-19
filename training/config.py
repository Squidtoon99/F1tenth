"""Default training configuration.

This is the standalone (single-process) trainer's config. Only ``DEFAULT_CONFIG``
is kept here; the legacy distributed ``Config`` class (Redis overrides, Postgres
bootstrap, S3 parameter server) is intentionally not part of this trainer.
"""

DEFAULT_CONFIG = {
    "config_version": 3,
    "policy_format_version": 3,
    "simulator": {
        "id": "f1tenth-torch",
        "version": 1,
    },
    "obs": {
        "num_obs": 390,
        # Asymmetric sensor-actor layout (sim-only). Privileged critic stays 390-D.
        "num_actor_obs": 1093,
        "actor_layout_version": 1,
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
        # The policy input is always 390-d. Solo and out-of-range rows carry an
        # exact zero sentinel in the final opponent-relative block.
        "enable_opponent_obs": True,
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
        # Inward normal speed (m/s) at geometric wall contact that ends the episode.
        "wall_impact_term_speed_mps": 4.0,
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
        # Seeded heading jitter (rad) added to the track tangent at spawn, so the
        # policy sees non-tangent starting attitudes (ego + opponent). 0 disables it.
        "reset_spawn_yaw_jitter_rad": 0.1,
        # When true the opponent samples its own off-centerline lateral spawn
        # (bounded by local track width) instead of starting on the centerline.
        "opponent_spawn_lateral_independent": True,
        # Observation/reset throttle scaling only — longitudinal cap comes from power+drag.
        "max_speed": 15.0,
        # Real F1TENTH servo hard-clamps the steering at ~0.33 rad (19 deg): see
        # analysis/analyze_full_lock.py, which measures a ~0.94 m min turning radius
        # from rosbags (servo saturates at 0.85 -> 0.33 rad effective wheel angle).
        # Training at 0.44 rad let the policy assume an unreachable 0.70 m radius and
        # understeer into walls on tight corners; 0.33 rad makes the sim match reality
        # (0.325 / tan(0.33) = 0.95 m min radius).
        "delta_max": 0.33,  # radians
        "wheelbase": 0.325,
        # Wheel-center track from the official 296 mm outer track and 43 mm tyres.
        "track_width": 0.253,
        "wheel_radius": 0.053,
        "f_drive_max": 23.0,
        "f_brake_max": 5.2,
        "power_max": 320.0,
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
        # for the slip definition: the Warp tire model (f1tenth_sim.dynamics)
        # and the on-car C++ builder both normalize by |v_fwd| + one of these
        # offsets (active = drive/brake applied, passive = coasting). The deprecated
        # compute_tyre_slip uses the same offsets.
        "slip_min_lat": 0.2,
        "slip_min_active_long": 0.1,
        "slip_min_passive_long": 0.4,
        # Warp vehicle model parameters.
        "warp_sim": {
            # Stock 68277-4 GTR spring rates: 109 / (109 + 125) = 0.47.
            "roll_stiffness_front": 0.47,
            # 45 A / (200 A/s) = 225 ms from zero to full deployed drive current.
            "longitudinal_slew_rate_per_s": 4.444444444444445,
            # Longitudinal tyre relaxation length (m); 0 disables the force lag
            # (exact pass-through, default). >0 lags contact-force buildup with a
            # speed-dependent time constant tau = tire_relax_len / max(|v|, blend).
            "tire_relax_len": 0.0,
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
            "scripted_weight": 0.10,
            "policy_weight": 0.90,
            "policy_speed_cap_prob": 0.5,
            "policy_speed_cap_range": [5.0, 8.0],
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
        "collision_term_speed_mps": 4.0,
        # Official 68277-4 body envelope (full length x width, metres), used for
        # oriented-box collision, car-car contact, and evaluation rendering.
        "car_length": 0.568,
        "car_width": 0.296,
        "collision_margin_m": 0.0,
        # Per-episode domain randomization. Enabled with narrow bands around the
        # nominal car so the policy does not overfit a razor-edge grip line, without
        # forcing it to hedge against extreme physics. Drive scale covers the
        # unresolved loaded force-per-amp range; steering bias stays neutral.
        "domain_randomization": {
            "enabled": True,
            "tire_friction_range": [0.60, 0.70],
            "vehicle_mass_range": [3.6, 3.9],
            "action_latency_steps_range": [0, 1],
            "obs_latency_steps_range": [0, 1],
            "obs_noise_std_range": [0.0, 0.01],
            "action_latency_steps_max": 3,
            # 35--65 A phase-current uncertainty around the nominal 45 A envelope.
            "drive_scale_range": [0.7777777777777778, 1.4444444444444444],
            "steer_bias_range": [0.0, 0.0],
            # Sensor DR defaults are no-ops so DR-off / unset ranges yield clean
            # LiDAR/IMU; enable by widening a range when training with sensors.
            "lidar_range_noise_std_range": [0.0, 0.0],
            "lidar_far_dropout_prob_range": [0.0, 0.0],
            "lidar_dropout_prob_range": [0.0, 0.0],
            "lidar_angle_bias_range": [0.0, 0.0],
            "lidar_extrinsic_xy_range": [0.0, 0.0],
            "lidar_extrinsic_yaw_range": [0.0, 0.0],
            "imu_accel_bias_range": [0.0, 0.0],
            "imu_gyro_bias_range": [0.0, 0.0],
            "imu_accel_noise_std_range": [0.0, 0.0],
            "imu_gyro_noise_std_range": [0.0, 0.0],
            "imu_axis_misalign_range": [0.0, 0.0],
            # Deploy-shaped VESC proxies on the actor obs (noop-by-default).
            "vesc_speed_bias_range": [0.0, 0.0],
            "vesc_current_bias_range": [0.0, 0.0],
            "vesc_speed_noise_std_range": [0.0, 0.0],
            "vesc_current_noise_std_range": [0.0, 0.0],
        },
    },
    # Hokuyo UST-10LX mock (Warp sensor kernels). Actor layout requires native
    # 1081 beams (beam_decimation must stay 1).
    "sensor": {
        "num_beams": 1081,
        "fov_deg": 270.0,
        "range_min_m": 0.06,
        "range_max_m": 30.0,
        "reliable_range_m": 10.0,
        "beam_decimation": 1,
        # Signed motor-current proxy scale: matches vesc_actuator 10 A drive/brake.
        "vesc_current_scale_a": 10.0,
        # Sphere-trace iteration cap (EDT cell ~2.5 cm; 512 covers >30 m worst case).
        "max_march_steps": 512,
        # Nominal mount in base_link; matches deploy vehicle.yaml opponent_detector.
        "lidar_offset_x": 0.0,
        "lidar_offset_y": 0.0,
        "lidar_offset_yaw": 0.0,
    },
    "reward": {
        # GT Sophy Maggiore reward. Each term is a raw canonical component times one
        # reward_scales coefficient (no duplicate *_k gain). State/event terms carry
        # the 10 Hz cadence factor (control_dt / 0.1 = 0.5 at 20 Hz) and off-course
        # uses the elapsed-time x km/h-squared form; both are applied in the kernel /
        # Torch mirror, so reward_scales holds the paper coefficient directly.
        "progress_max_lateral_m": 1.0,
        "oob_margin_m": 0.2,
        "lateral_k": 0.5,
        "passing_gate_ahead_m": 40.0,
        "passing_gate_behind_m": 20.0,
        # Overtake-completed bonus: one-time +overtake_bonus_k when the opponent
        # goes from ahead to behind within overtake_gap_m on the centerline (a
        # genuine pass, not a lap wrap). Only active when an "overtake" entry is
        # added to reward_scales (the trainer does this for 1v1).
        "overtake_bonus_k": 1.0,
        "overtake_gap_m": 5.0,
        # Retained for backward-compatible reproduction of old additive/deadzone
        # slip configs; the Maggiore parity path uses the product form and ignores
        # these.
        "slip_angle_weight": 1.0,
        "slip_deadzone_ratio": 0.0,
        "slip_deadzone_angle": 0.0,
        # Absolute GT Sophy scale: no extra multiplier in the parity arm. Retained
        # so old configs can still shrink overall magnitude.
        "global_reward_scale": 1.0,
        # Maggiore coefficients. progress (Rcp) and passing (Rps) are distance
        # deltas; collision (Rc), rear_end (Rr) and tyre_slip (Rts) carry the 10 Hz
        # cadence; oob (Rsoc) is the off-course km/h term; wall_penalty (Rw) is the
        # continuous contact term; wall_impact is the one-shot -v_normal^2 hit.
        # lateral/smoothness are disabled (retained for backward compat).
        # passing/collision/rear_end are gated on by the 1v1 trainer.
        "reward_scales": {
            "progress": 1.0,
            "passing": 0.5,
            "collision": 4.0,
            "rear_end": 0.1,
            "tyre_slip_penalty": 0.25,
            "oob_penalty": 0.01,
            "wall_penalty": 0.01,
            "wall_impact": 1.0,
            "lateral": 0.0,
            "smoothness": 0.0,
        },
    },
    "model": {
        # Candidate default for new runs: location-preserving LiDAR CNN actor.
        # flat_mlp remains available for legacy evaluation / explicit baselines.
        "actor_type": "lidar_cnn",
        "actor_hidden_layers": [512, 512, 512],
        "critic_hidden_layers": [1024, 1024, 1024],
        "lidar_pool_bins": 32,
        "num_quantiles": 32,
        "rew_gamma": 0.9896,
        "n_step": 7,
        "alpha": 0.01,
        # Dual float16 actor/critic replay. Request 2M; trainer falls back to 1M
        # only on CUDA OOM or insufficient asymmetric-update headroom.
        "replay_buffer_limit": 2_000_000,
        "batch_size": 1024,
        "minimum_train_transitions": 200_000,
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
        # Immutable incumbent anchor: a fixed policy artifact added to the
        # opponent population and never evicted from the rolling pool. Sampled
        # with anchor_prob on each refresh/selection; None / 0.0 disables it.
        "anchor_ckpt": None,
        "anchor_prob": 0.0,
    },
}
