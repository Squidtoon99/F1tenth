# Reward ratio comparison: GT Sophy, Lee et al. 2025, and F1TENTH training

**Date:** 2026-07-22
**Scope:** Primary paper sources only; repo configs/runs for effective training rewards. No code changes.

---

## 1. Lee et al. 2025 identification (from this repo)

There is **no formal bibliography entry** in `docs/` for the vision/asymmetric paper. The trainer identifies it only via code comments (`training/config.py`, `training/standalone_trainer.py`, `training/qrsac/qrsac.py`, `training/tests/test_steering_reward.py`).

Matching those comments to primary sources yields:

| Field | Value |
| --- | --- |
| **Title** | A Champion-level Vision-based Reinforcement Learning Agent for Competitive Racing in Gran Turismo 7 |
| **Authors** | Hojoon Lee, Takuma Seno, Jun Jet Tai, Kaushik Subramanian, Kenta Kawamoto, Peter Stone, Peter R. Wurman |
| **Venue** | IEEE Robotics and Automation Letters, 2025 |
| **DOI** | [10.1109/LRA.2025.3560873](https://doi.org/10.1109/LRA.2025.3560873) |
| **arXiv** | [2504.09021](https://arxiv.org/abs/2504.09021) |
| **HTML (v2)** | [arxiv.org/html/2504.09021v2](https://arxiv.org/html/2504.09021v2) — Section III-C Reward Function |
| **Author PDF** | [joonleesky.github.io/data/preprint2025visiongt.pdf](https://joonleesky.github.io/data/preprint2025visiongt.pdf) |

GT Sophy baseline paper:

| Field | Value |
| --- | --- |
| **Title** | Outracing champion Gran Turismo drivers with deep reinforcement learning |
| **DOI** | [10.1038/s41586-021-04357-7](https://doi.org/10.1038/s41586-021-04357-7) |
| **Methods (rewards)** | Nature Article, “Methods — Rewards” (approx. PDF p. 8–9) |
| **Extended Data Table 1** | Nature PDF p. 14 (image table; values below transcribed from that figure) |
| **Supplementary PDF** | [41586_2021_4357_MOESM1_ESM.pdf](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-04357-7/MediaObjects/41586_2021_4357_MOESM1_ESM.pdf) (QR-SAC hyperparameters; **no reward table**) |

---

## 2. Timestep / action frequency

| Source | Control rate | Sim / game rate | Notes |
| --- | --- | --- | --- |
| **GT Sophy** | **10 Hz** (0.1 s) | GT at 60 fps; action held 6 frames | Methods; Supplementary rollout pseudocode: `wait_for_obs_at_frame(...+6)` |
| **Lee 2025** | **10 Hz** | GT7 at 60 Hz; ZOH throttle, interpolated steer | Section III-B |
| **F1TENTH `DEFAULT_CONFIG`** | **10 Hz** (`sim_dt=0.005`, `control_interval=20`) | Warp 200 Hz physics | Matches Sophy/Lee discount horizon |
| **`darktoaster_warp_ro_8192`** | **20 Hz** (`control_interval=10`) | Same Warp physics | Same `γ=0.9896` → **~2× shorter real-time return horizon** vs 10 Hz papers |

---

## 3. GT Sophy (Wurman et al., Nature 2022)

### 3.1 Combined reward

Linear sum over transition \((s \to s')\):

\[
R = w_{cp} R_{cp} + w_{soc} R_{soc} + w_{loc} R_{loc} + w_w R_w + w_{ts} R_{ts} + w_{ps} R_{ps} + w_c R_c + w_r R_r + w_{uc} R_{uc}
\]

(Methods names components \(R_{cp}, R_{soc}\) or \(R_{loc}, \ldots\); Extended Data Table 1 uses the same symbols as weights.)

### 3.2 Component definitions (Methods)

| Symbol | Raw term (before table weight) | Sign / mask |
| --- | --- | --- |
| \(R_{cp}\) | \(\Delta l\) (centreline arc-length progress, m) | **Masked to 0 off-course** |
| \(R_{soc}\) | \(-(\Delta s_o)\,(v_{kph})^2\) | Off-course; \(\Delta s_o\) = cumulative off-course time increment |
| \(R_{loc}\) | \(-(\Delta s_o)\,v_{kph}\) | Sarthe variant (linear speed); **doubled** at first/final chicanes |
| \(R_w\) | \(-(\Delta s_w)\,(v_{kph})^2\) | Wall contact time increment |
| \(R_{ts}\) | \(-\sum_i \min(|\text{slip ratio}_i|,1)\,|\text{slip angle}_i|\) | Per tyre |
| \(R_{ps}\) | \(\sum_i (\Delta s_{L,i})\,\max(\mathbb{1}_{b,f}(s_{L,i}),\mathbb{1}_{b,f}(s'_{L,i}))\) | Passing window **\(b=20\) m behind, \(f=40\) m ahead** |
| \(R_c\) | \(-\max_i c_i\) | Any car–car collision indicator |
| \(R_r\) | \(-\sum_i c_i\,\mathbb{1}_{\text{opp ahead}}\,\|v-v_i\|^2\) | Rear-end / closing-speed collision |
| \(R_{uc}\) | \(-\max_i u(s',i)\) | **Sarthe only** — unsporting collision |

**Global scaling:** none stated; weights are per-component multipliers in Extended Data Table 1.

**Termination (game-level, not reward):** 150 s training scenarios; stewards slow penalised cars to 100 km/h in penalty zones (main text). Off-course judged by **game engine** (Methods: “relied on the game engine to determine whether the agent was off course”).

### 3.3 Extended Data Table 1 — numeric weights

Transcribed from Nature PDF **Extended Data Table 1** (page 14 image):

| Course | \(R_{cp}\) | \(R_{soc}\) | \(R_{loc}\) | \(R_w\) | \(R_{ts}\) | \(R_{ps}\) | \(R_c\) | \(R_r\) | \(R_{uc}\) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Seaside | 1 | **0.01** | 0 | 0.01 | 0.25 | **0.5** | 5 | 0.1 | 0 |
| **Maggiore** | 1 | **0.01** | 0 | 0.01 | 0.25 | **0.5** | **4** | **0.1** | 0 |
| Sarthe | 1 | 0 | **5** | 0.01 | 0 | **0.5** | 5 | 0.1 | **5** |

**Not stated in table:** separate one-shot impact terms; steering penalties (Lee-only in GT7 stack).

---

## 4. Lee et al. 2025 (IEEE RA-L)

### 4.1 Combined reward

Section III-C:

\[
r_t = \lambda^p r^p_t + \lambda^o r^o_t + \lambda^b r^b_t + \lambda^v r^v_t + \lambda^c r^c_t + \lambda^s r^s_t + \lambda^t r^t_t + \lambda^h r^h_t
\]

### 4.2 Components and published coefficients

| Term | Equation (paper) | \(\lambda\) or constant |
| --- | --- | --- |
| Progress \(r^p\) | \(p_t - p_{t-1}\) (centreline) | \(\lambda^p = 1.0\) |
| Shortcut \(r^o\) | \(-(s^o_t - s^o_{t-1})\,\|\mathbf{v}_t\|\) | \(\lambda^o = 10.0\) |
| Barrier \(r^b\) | \(-(s^b_t - s^b_{t-1})\,\|\mathbf{v}_t\|\) | \(\lambda^b = 20.0\) |
| Velocity collision \(r^v\) | \(-\|\Delta \mathbf{v}^x_t\|^2\) | \(\lambda^v = 0.5\) |
| Fixed collision \(r^c\) | constant on contact | \(\lambda^c = 6.0\) |
| Overtaking \(r^t\) | gated sum of \((p_t-p^i_t)-(\cdot)\) | \(\lambda^t = 3.0\); **\(c_r=-20\), \(c_f=40\)** m |
| Steering change \(r^s\) | \(-\|\theta^s_t - \theta^s_{t-1}\|\) | \(\lambda^s = 0.5\) |
| Steering history \(r^h\) | \(-m_t\bigl(1+\exp(-c^s(\Delta_t-c^o))\bigr)\) | \(\lambda^h = 5.0\); **\(c^s=182.883569\)**, \(c^o=0.034\), \(c^d=0.014\) |

**Shortcut semantics:** active when **≥3 tyres** outside track limits (Section III-C); **not** a fixed interior margin.

**Missing from Lee reward:** tyre-slip term, separate wall-impact spike, terminal OOB speed hit, global reward scale.

**Training opponents:** fixed GT7 BIAI counts \(\{0,1,2,3,4,7,12,19\}\) with BoP randomisation (Section IV-B) — **not** rolling self-play.

---

## 5. F1TENTH implementation (`training/config.py` + Warp kernel)

Production rewards: **`training/f1tenth_env/kernel.py`** (`compute_reward_and_done`).
Torch mirror: `training/f1tenth_env/rewards.py` (parity tests; **OOB uses Sophy \(v_{kph}^2\) form**, kernel uses **Lee linear-speed OOB** — see §6).

### 5.1 Cadence / scaling helpers

| Mechanism | Effect |
| --- | --- |
| `cadence = control_dt / 0.1` | Scales `collision`, `rear_end`, `tyre_slip` to 10 Hz equivalence |
| `params.oob = scale_oob × control_dt` | OOB increment per step |
| `params.wall = scale_wall × control_dt × 3.6²` | Wall term matches \(-w\,\Delta t\,v_{kph}^2\) |
| `global_reward_scale` | Uniform multiplier on summed terms (default **1.0**) |

### 5.2 `DEFAULT_CONFIG` (“Maggiore parity” comment) — asymmetric / current stack

| Config key | `reward_scales` | Maps to paper |
| --- | ---: | --- |
| `progress` | **1.0** | Sophy \(R_{cp}\) |
| `oob_penalty` | **10.0** | **Lee \(\lambda^o\)**, not Sophy \(R_{soc}=0.01\) |
| `wall_penalty` | **0.01** | Sophy \(R_w\) |
| `wall_impact` | **1.0** | **Not in either paper** (one-shot \(-v_n^2\)) |
| `tyre_slip_penalty` | **0.25** | Sophy \(R_{ts}\) |
| `passing` | **1.0** (1v1) | **2× Sophy \(R_{ps}=0.5\)**; **⅓ Lee \(\lambda^t=3\)** |
| `collision` | **4.0** (1v1) | Sophy Maggiore \(R_c\) |
| `rear_end` | **0.1** (1v1) | Sophy \(R_r\) |
| `steering_change` | **0.25** | **0.5× Lee \(\lambda^s\)** |
| `steering_history` | **0.5** | **0.1× Lee \(\lambda^h\)** |
| `oob_impact` | **0.5** | **Not in either paper** (terminal \(-v\)) |
| `overtake` | gated bonus | **Not in either paper** |
| `global_reward_scale` | **1.0** | — |

**OOB / termination semantics (stricter than papers):** `oob_margin_m=0.15` footprint-aware off-track signal; `term_oob_max_consecutive=3`; `oob_impact` on OOB termination; `collision_term_speed_mps=4.0`.

### 5.3 `darktoaster_warp_ro_8192` (symmetric-era run)

From `training/outputs/runs/darktoaster_warp_ro_8192/config.json` and final `run.log` (2026-07-16, ~4.26B transitions):

| Parameter | Value | vs papers |
| --- | --- | --- |
| `control_interval` | **10 → 20 Hz** | 2× policy rate vs papers |
| `global_reward_scale` | **0.2** | Not in papers |
| `progress` scale + `progress_k_fwd/back` | raw **5.0 × 5.0**; effective **5.0/m** after global scale | **5× per-metre** vs Sophy/Lee \(w_{cp}=1\) |
| `passing` | raw `passing_k` **5 × scale 2.5**; effective **2.5** | **5× Sophy \(R_{ps}\)**; **5/6× Lee \(\lambda^t\)** |
| `collision` | raw `collision_k` **5 × scale 1**; effective **1.0** | **0.25× Sophy Maggiore \(R_c=4\)** |
| `rear_end` | raw `rear_end_k` **5 × scale 0.5**; effective **0.5** | **5× Sophy \(R_r\)**; same coefficient as Lee \(r^v\), with different gates |
| `oob_penalty` | raw `oob_k` **0.3 × scale 0.6**; effective **0.036 × speed²** | Different formula; at 8 m/s, **−2.304/off-track step** |
| `tyre_slip_penalty` | raw **0.02**; effective **0.004** | **0.016× Sophy \(R_{ts}\)** and a different legacy slip form |
| Steering terms | **absent** | — |
| Logged on-track means | `progress≈10.10`, `passing≈0.16`, `total≈2.02`, `speed≈7.94 m/s`, `progress_ds≈0.40` | See §7 |

---

## 6. Weight ratios: papers vs F1TENTH

Legend: **C** = comparable (same units); **NC** = different raw formula/units.

### 6.1 Coefficient ratios on shared symbols (Maggiore / default 1v1)

| Ratio | GT Sophy Table 1 (Maggiore) | Lee 2025 | F1TENTH `DEFAULT_CONFIG` | F1TENTH / Sophy | F1TENTH / Lee |
| --- | ---: | ---: | ---: | ---: | ---: |
| passing / progress | **0.5 / 1 = 0.5** (C on \(\Delta s\)) | **3 / 1 = 3** | **1 / 1 = 1** | **2.0×** | **0.33×** |
| collision / progress | NC (indicator vs m) | NC | NC | — | — |
| rear / collision | NC (\(\|\Delta v\|^2\) vs bool) | NC | NC | — | — |
| tyre slip / progress | NC | **absent** | NC | — | — |
| wall / progress | NC | NC (Lee uses \(\lambda^b=20\) on **linear** barrier term) | NC | — | — |
| steering\_h / steering\_s | absent | **5 / 0.5 = 10** | **0.5 / 0.25 = 2** | — | **0.2× / 0.5×** |

### 6.2 Effective per-step magnitudes (reference point)

**Reference:** on-track, \(\Delta s = 0.40\,\mathrm{m}\) (from `darktoaster` log), \(v = 8\,\mathrm{m/s}\) (\(v_{kph}=28.8\)), full step off-course or wall contact, `control_dt=0.1` (10 Hz).

| Term | Sophy Maggiore effective | Lee 2025 effective | F1TENTH DEFAULT (kernel) |
| --- | ---: | ---: | ---: |
| Progress | \(+0.40\) | \(+0.40\) | \(+0.40\) |
| OOB / shortcut | \(-0.01 \times 0.1 \times 28.8^2 \approx \mathbf{-0.83}\) | \(-10 \times 0.1 \times 8 = \mathbf{-8.0}\) | **Same as Lee:** \(\mathbf{-8.0}\) |
| Wall contact | \(-0.01 \times 0.1 \times 28.8^2 \approx -0.83\) | \(-20 \times 0.1 \times 8 = -16\) | \(-0.01 \times 0.1 \times 28.8^2 \approx -0.83\) (Sophy form) |
| Any collision | \(-4\) | \(-6\) | \(-4\) |
| Rear-end @ \(\|\Delta v\|=4\) m/s | \(-0.1 \times 16 = -1.6\) | \(-0.5 \times 16 = -8\) | \(-1.6\) |
| Tyre slip (unit excess) | \(-0.25\) | — | \(-0.25\) |

**Key findings:**

1. **`oob_penalty: 10.0` matches Lee \(\lambda^o\), not Sophy \(R_{soc}=0.01\).** With the kernel’s **linear-speed** OOB increment, F1TENTH off-course shaping is **~9.6× stronger** than Sophy’s table-weighted **\(v_{kph}^2\)** term at 8 m/s (and uses a **different functional form**).
2. **`passing: 1.0` is 2× Sophy table \(R_{ps}=0.5\)** and **⅓ Lee \(\lambda^t=3\)** on the same \(\Delta s\) increment.
3. **`collision: 4.0`, `rear_end: 0.1`, `tyre_slip: 0.25`, `wall_penalty: 0.01` match Sophy Maggiore table** (given F1TENTH cadence conventions).
4. **Steering penalties are Lee-derived but scaled down** (0.25/0.5 vs 0.5/5.0).
5. **F1TENTH-only terms** (`wall_impact`, `oob_impact`, `overtake`) have **no paper ratio**.

### 6.3 `darktoaster_warp_ro_8192` vs `DEFAULT_CONFIG` (logged effective)

Using log line `progress=10.0964`, `progress_ds=0.4024` gives the logged
pre-global progress gain **\(k \approx 25.1\,\mathrm{m}^{-1}\)**. The kernel then
multiplied the complete sum by `global_reward_scale=0.2`, so the policy's
**effective progress coefficient was approximately \(5.0\,\mathrm{m}^{-1}\)**.

| Quantity | darktoaster (20 Hz, scale 0.2) | DEFAULT (10 Hz, scale 1.0) | Ratio |
| --- | ---: | ---: | ---: |
| Progress \(k\), pre-global log | ~25 / m | ~1 / m | ~25× |
| Progress \(k\), effective in total | **~5 / m** | **~1 / m** | **~5×** |
| Policy cadence | 20 Hz | 10 Hz | 2× |
| `passing` effective coefficient | 2.5 | 1.0 | 2.5× |
| `collision` effective coefficient | 1.0 | 4.0 | 0.25× |
| `rear_end` effective coefficient | 0.5 | 0.1 | 5× |
| Steering penalties | 0 | 0.25 / 0.5 | — |
| Mean total reward / step | ~2.02 | (not logged here) | — |

**Non-comparable confounds:** legacy double `progress_k` × `reward_scales.progress`; 20 Hz with paper `γ=0.9896`; mixed self-play opponent distribution; no steering terms.

---

## 7. Non-comparable terms (explicit)

| Term | Why ratios are invalid |
| --- | --- |
| Sophy \(R_{soc}\) vs Lee \(r^o\) vs F1TENTH kernel OOB | **\(v_{kph}^2\)** vs **\(v\)** vs interior **margin** semantics |
| Sophy \(R_w\) vs Lee \(r^b\) | Same symbol “wall/barrier” but **\(v^2\)** vs **\(v\)**; F1TENTH adds **`wall_impact`** |
| \(R_r\) vs Lee \(r^v\) | Both use \(\|\Delta v\|^2\) but **different gates** (rear-end vs generic collision) |
| F1TENTH `oob_impact`, `overtake` | Not in either paper |
| Sarthe \(R_{loc}\), \(R_{uc}\) | Track-specific; F1TENTH uses single IV_2026 / Austin-style corridor |
| Game steward penalties (100 km/h zones) | **Not in reward sum** |

---

## 8. Summary table (Maggiore / default training intent)

| Component | GT Sophy Table 1 | Lee 2025 \(\lambda\) | F1TENTH DEFAULT scale | Match? |
| --- | ---: | ---: | ---: | --- |
| Progress | 1 | 1.0 | 1.0 | ✅ |
| Off-course / shortcut | **0.01** (\(v_{kph}^2\) raw) | **10.0** (\(\|v\|\) raw) | **10.0** (Lee linear kernel) | ❌ form + weight |
| Wall continuous | 0.01 | 20.0 (barrier, linear) | 0.01 | ⚠ partial (form differs for Lee) |
| Tyre slip | 0.25 | — | 0.25 | ✅ |
| Passing | **0.5** | **3.0** | **1.0** | ❌ |
| Any collision | 4 | 6.0 | 4 | ✅ vs Sophy |
| Rear / velocity collision | 0.1 | 0.5 | 0.1 | ✅ vs Sophy |
| Steering change / history | — | 0.5 / 5.0 | 0.25 / 0.5 | ⚠ Lee constants, reduced \(\lambda\) |

---

## 9. Sources consulted

1. Wurman, P.R. et al. Nature **602**, 223–228 (2022). [doi:10.1038/s41586-021-04357-7](https://doi.org/10.1038/s41586-021-04357-7) — Methods “Rewards”; Extended Data Table 1 (PDF p. 14).
2. Lee, H. et al. IEEE RA-L (2025). [doi:10.1109/LRA.2025.3560873](https://doi.org/10.1109/LRA.2025.3560873) — [arXiv:2504.09021v2](https://arxiv.org/html/2504.09021v2) Section III-B–C.
3. F1TENTH repo: `training/config.py`, `training/f1tenth_env/kernel.py`, `training/f1tenth_env/warp_env.py`, `training/outputs/runs/darktoaster_warp_ro_8192/config.json`, `run.log` (final lines 2026-07-16).

**Not used:** Vasco et al. 2024, wiki, or secondary summaries (per primary-source-only rule).
