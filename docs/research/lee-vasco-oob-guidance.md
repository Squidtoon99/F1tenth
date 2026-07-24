# Lee and Vasco guidance for OOB-dominated training

**Date:** 2026-07-23
**Scope:** Primary papers plus the current repository and local run artifacts. Read-only
analysis; no training or configuration changes.

---

## 1. Sources and evidence labels

1. Hojoon Lee et al., **“A Champion-level Vision-based Reinforcement Learning
   Agent for Competitive Racing in Gran Turismo 7,”** IEEE Robotics and
   Automation Letters (2025), DOI
   [10.1109/LRA.2025.3560873](https://doi.org/10.1109/LRA.2025.3560873);
   [author preprint](https://joonleesky.github.io/data/preprint2025visiongt.pdf).
   Relevant locations: Sections III-B–E, IV-B, V-C; Figure 6. Accessed
   2026-07-23.
2. Miguel Vasco et al., **“A Super-human Vision-based Reinforcement Learning
   Agent for Autonomous Racing in Gran Turismo,”** arXiv:2406.12563 (2024),
   [abstract/versions](https://arxiv.org/abs/2406.12563),
   [full HTML](https://arxiv.org/html/2406.12563). Relevant locations:
   Sections 3.2–3.4, 5.1; Appendices C.1–C.2, I, J; Tables 3–4. Accessed
   2026-07-23.

Claims below are marked as:

- **Paper evidence:** directly stated or reported by Lee or Vasco.
- **F1TENTH adaptation:** a reasoned mapping to this repository, not a result
  reported by either paper.
- **Speculation:** plausible, but not established by the papers or current runs.

The web copies above were checked against the complete local extractions supplied
for both papers.

---

## 2. What the papers actually prescribe

### 2.1 Course-limit rewards

Both papers use the same time-trial atoms, with Lee adding competitive-racing
terms:

\[
r_t^{\text{Vasco}} =
r_t^p + 10r_t^o + 10r_t^w + 3r_t^s + 5r_t^h
\]

\[
r_t^{\text{Lee}} =
1r_t^p + 10r_t^o + 20r_t^b + 0.5r_t^v + 6r_t^c
+ 0.5r_t^s + 3r_t^t + 5r_t^h .
\]

The relevant atoms are:

\[
r_t^p=p_t-p_{t-1}
\]

\[
r_t^o=-(s_t^o-s_{t-1}^o)\lVert v_t\rVert
\]

\[
r_t^w\ \text{or}\ r_t^b
=-(s_t^{w/b}-s_{t-1}^{w/b})\lVert v_t\rVert .
\]

**Paper evidence:**

- “Shortcut” is not synonymous with touching a boundary. It applies while **at
  least three tyres are outside the track limits** (Vasco Section 3.3; Lee
  Section III-C).
- Wall/barrier contact is a distinct event from shortcutting. Vasco weights its
  linear-speed term by 10; Lee weights it by 20.
- Neither paper adds a fixed first-boundary-crossing cost or a terminal OOB
  impact reward.
- Neither paper says to terminate an episode when the car leaves the course.
  Vasco instead states that episodes reset every 150 s and deliberately samples
  starts from in-course and left/right off-course areas (Appendix I).
- Lee uses the same shortcut atom and coefficients while training competitive
  races (Section III-C).

The papers do **not** provide an ablation of shortcut coefficient, OOB
termination, or terminal OOB penalties. Treating coefficient 10 as uniquely
optimal for F1TENTH would therefore be unsupported.

### 2.2 Action and timing

**Paper evidence (Vasco Section 3.2; Lee Section III-B):**

- Two actions: delta steering in \([-3^\circ,3^\circ]\), and one combined
  throttle/brake scalar in \([-1,1]\).
- Control is 10 Hz while GT7 runs at 60 Hz.
- Steering is linearly interpolated between decisions; throttle is held.
- Vasco reports a large degradation from asynchronous training: only 7.45% of
  laps beat the fastest human in Async/Async, versus 69.3% when trained
  synchronously and executed asynchronously (Appendix C.2). This tests
  communication latency, not domain-randomized actuator delay.

### 2.3 Reset, curriculum, and opponents

**Paper evidence:**

- Vasco time trials are solo. The initial position is sampled from in-course and
  left/right off-course areas within 5% of track width, points toward the
  centreline 30 m ahead, and starts at a uniformly sampled 0–104.607 km/h.
  Episodes reset after 150 s (Appendix I).
- Lee samples each episode from solo and
  \(\{1,2,3,4,7,12,19\}\)-opponent scenarios, starts around random track
  positions, and randomizes opponent engine power and body weight by ±25%.
  The opponents are fixed GT7 built-in AI, not rolling self-play (Section IV-B).
- Lee says training directly against 19 opponents hinders learning basic driving
  skills; the scenario mixture is its remedy.

### 2.4 Recurrent learning, regularization, and cadence

**Paper evidence:**

- Lee uses a 512-dimensional GRU actor. Its batch is 512 rows from 16
  trajectories × 32 trained steps, with 16 burn-in steps and stored replay
  hidden state. Zeroing hidden state before warm-up slightly hurts performance;
  hidden size 128 hurts more; no RNN completely fails competitive overtaking
  (Section IV-B and Figure 6).
- Lee uses a 5M replay, 7-step return, 32 quantiles, \(\gamma=0.9896\),
  \(\alpha=0.01\), and \(2.5\times10^{-5}\) learning rate. It reinitializes the
  networks once when replay first fills, at epoch 2,000, while retaining the
  diverse buffer. Random-shift image augmentation uses mirrored padding and a
  maximum shift of four pixels (Sections III-E, IV-B, V-C).
- Lee’s ablations report that both reinitialization and augmentation improve
  stability/final performance; augmentation reduces evaluation variance.
- Vasco is feed-forward, uses batch 512, 2.5M replay, 7-step return, 32
  quantiles, the same \(\gamma,\alpha,\) and learning rates, and critic gradient
  clipping at 10 (Appendix J, Tables 3–4).
- Vasco defines an epoch as 6,000 gradient steps and trains 2,000 or 4,000
  epochs, but does not publish enough rollout accounting to derive a comparable
  update-to-data ratio. Lee likewise does not provide a directly comparable
  sampled-rows-per-environment-transition ratio.

**Paper evidence, stability-related but not OOB-specific:**

- Privileged course points in the critic are important in both papers. Vasco’s
  shorter two-second point horizon is unreliable; four seconds is slightly
  better than the six-second reference on Monza (Appendix C.1).
- Lee’s symmetric critic usually fails to reach first place; removing the RNN
  completely fails overtaking (Figure 6). These do not establish either feature
  as an OOB fix.

---

## 3. Current F1TENTH mapping

The run evidence and table below describe the frozen pre-treatment baseline.
ADR 0013 subsequently selected a deliberate F1TENTH physical-wall adaptation:
terminate when the first projected footprint edge intersects a mapped wall,
mask progress on that transition, and apply only
\(-20\,\Delta t\,\lVert v_{xy}\rVert\). The fixed crossing cost, continuous
quadratic OOB cost, and terminal impact were removed.

This treatment uses Lee's barrier coefficient and linear-speed atom, but its
geometry and immediate termination are repository adaptations. Lee and Vasco
distinguish physical barrier contact from the open-track shortcut condition
(at least three tyres outside), and neither paper reports terminating on
course-limit crossing. The treatment therefore must not be described as the
papers' shortcut rule.

The frozen `DEFAULT_CONFIG`, `standalone_trainer.py`, Warp reward kernel, and the
three requested run snapshots showed:

| Item | Papers | Current fixed-opponent runs | Assessment |
| --- | --- | --- | --- |
| Rate/action | 10 Hz, ±3° delta steer, combined longitudinal | Exact rate and steer cap; normalized combined longitudinal command | Good match |
| Progress | Centreline \(\Delta p\), weight 1 | \(\Delta s\), weight 1; forced to zero whenever any footprint is out | Zeroing is not stated in either paper |
| Shortcut | Linear speed, weight 10, ≥3 tyres out | \(-0.02\,dt\,v_{km/h}^2\), starts when any footprint crosses | Different atom, coefficient, and geometry |
| Boundary/barrier | Separate physical barrier, linear speed | Fixed \(-4\) on first geometric boundary crossing | Repo-only and conflates track limit with barrier |
| OOB termination | Not reported; Vasco trains off-course recovery | Full-car-out reset plus speed-squared terminal cost | Repo-only |
| Reset | Random track location; Vasco includes slight off-course starts | Random in-course lateral point, 0.2 m inset, 1–7 m/s, 10% stationary, ±0.1 rad yaw | No deliberate recovery starts |
| Opponents | Vasco solo; Lee mixes solo through 19 fixed BIAI | Always 1v1; 50/50 scripted/frozen-policy mode | Missing solo/basic-driving curriculum |
| Replay/batch | 2.5M/512 (Vasco), 5M/512 (Lee) | 3M/512 | Within paper range |
| Recurrent batch | Lee 16×32, burn-in 16, stored hidden 512 | 16×32, burn-in 16, stored hidden 512 | Exact structural match |
| Reinit | Once when replay fills | Once at 2,998,272 transitions | Matches principle |
| Augmentation | Four-pixel mirrored image shift | Four-beam reflected LiDAR shift | Same number, different modality; not validated by papers |
| Updates | Not cross-comparable | 2 sampled rows/transition (four 512-row updates per 1,024-env tick) | No paper-supported diagnosis |
| Dynamics randomization | Fixed ego conditions; Lee varies opponents ±25% | Ego friction, mass, drive scale 0.778–1.444, 0–1 step action latency, sensor perturbations | Extra task complexity |

The frozen baseline Warp equations were:

\[
r_{\text{oob}}=-0.02(0.1)(3.6v)^2=-0.02592v^2
\]

while any-footprint progress is zero, first contact adds \(-4\), and full-car-out
adds

\[
r_{\text{terminal}}=-0.02\,T_{\text{skip}}(3.6v)^2.
\]

`standalone_68e197ed` uses \(T_{\text{skip}}=1\) s;
`standalone_db47f38d` uses 10 s. The previous `standalone_642a7a80` used 0.01
for continuous and terminal coefficients and one terminal second.

### 3.1 Quantitative mismatch

At representative F1TENTH speeds:

| Speed | Paper shortcut \(10(-dt\,v)\) | Current continuous OOB | Current/paper magnitude |
| ---: | ---: | ---: | ---: |
| 1 m/s | -1.00 | -0.0259 | 0.026× |
| 2 m/s | -2.00 | -0.1037 | 0.052× |
| 5 m/s | -5.00 | -0.6480 | 0.130× |

This comparison excludes the repo-only \(-4\) crossing event and terminal
penalty. It exposes the central incentive mismatch: paper progress and shortcut
cost are both approximately linear in speed, so their ratio is stable. Current
progress is approximately linear while continuous and terminal OOB costs are
quadratic, so slowing down disproportionately reduces OOB risk. The fixed
crossing cost then makes “avoid trying a fast recovery” attractive.

### 3.2 Requested run evidence

Means below aggregate consecutive 51,200-transition log windows.

| Run/window | Speed | Progress/step | OOB fraction | Lifespan | OOB share of recorded terminations |
| --- | ---: | ---: | ---: | ---: | ---: |
| `68e197ed`, 40–48M, 1 s terminal | 2.80 m/s | 0.228 | 0.142 | 3.05 s | 77.4% |
| `db47f38d`, 40–48M, 10 s terminal | 1.44 m/s | 0.107 | 0.222 | 4.27 s | 77.5% |
| `db47f38d`, 145–155M | 1.79 m/s | 0.157 | 0.099 | 6.63 s | 70.6% |
| `642a7a80`, 145–155M | 5.16 m/s | 0.439 | 0.181 | 120.3 s | 92.2% |
| `642a7a80`, 195–205M | 5.45 m/s | 0.515 | 0.052 | 260.0 s | 41.6% |
| `642a7a80`, 448–458M | 4.79 m/s | 0.433 | 0.100 | 10.48 s | 100.0% |

At 40–48M the ten-second terminal variant’s mean terminal event was \(-10.92\),
versus \(-2.82\) for the one-second variant. It ran at about half the speed and
made about half the progress, without reducing OOB’s share of terminations.
This is direct evidence against increasing the terminal penalty as the first
remedy. It is not a randomized multi-seed result, so it does not prove that the
terminal multiplier alone caused every difference.

The previous run demonstrates that this architecture can reach much longer
episodes and higher speed, then later collapse into almost exclusively OOB
termination. It does **not** isolate self-play, plasticity, or reward shape:
`642a7a80` changed opponents throughout training and differs from the fixed-run
implementation.

---

## 4. Ranked recommendations

### 1. Test the paper’s shortcut atom without repo-only OOB shocks — high confidence

**F1TENTH adaptation:** define shortcut by a tyre/footprint criterion equivalent
to “at least three tyres outside,” use \(-10\,dt\,v\), and remove the fixed
first-crossing and terminal reward from this treatment. Keep barrier collision
separate and activate it only for an actual physical barrier/contact signal.

Why first: this restores the papers’ reward ratio and removes the observed
quadratic low-speed incentive. Both primary papers support the atom; the current
one-second versus ten-second evidence disfavors a stronger terminal shock.

Whether progress should remain zero during shortcutting is **not specified by
these papers**. Hold that choice fixed in the first experiment.

### 2. Train recovery instead of immediately ending it — medium-high confidence

**Paper evidence:** Vasco samples slightly off-course starts and uses a fixed
150-second horizon; neither paper reports OOB termination.

**F1TENTH adaptation:** add a small fraction of recoverable, near-boundary starts
and allow recovery after crossing. If the simulator needs a reset for runaway
cars, reset only after a defined unrecoverable distance/time and do not invent an
extra terminal reward.

This is strongly motivated by Vasco, but the exact F1TENTH reset threshold is not
in either paper.

### 3. Restore a solo/basic-driving curriculum before fixed 1v1 — high confidence

**Paper evidence:** Vasco learns solo; Lee explicitly says direct dense-opponent
training hinders basic driving and mixes solo through 19 fixed-AI scenarios.

**F1TENTH adaptation:** first test a fixed solo/1v1 mixture, with the 1v1
opponent distribution held immutable. Do not use performance-triggered stages in
the first test; that adds another confound.

The exact solo fraction is unsupported. Start with an explicit 50/50 experiment,
not because 50% is in the papers, but because it is a simple identifiable test.

### 4. Run the diagnostic baseline synchronously with reduced randomization — medium confidence

Hold ego dynamics fixed and action latency at zero until course-limit behavior is
stable; then reintroduce one randomization family at a time. Vasco’s latency
ablation supports synchronous training, and neither paper randomizes the ego
vehicle as broadly as this repo.

Applying Vasco’s image-network latency result directly to one-step F1TENTH
actuator randomization is an adaptation, not direct evidence.

### 5. Keep the recurrent/reinit/batch setup unchanged initially — high confidence

The current 512-hidden GRU, 16-step burn-in, 32 trained steps, stored hidden
state, batch 512, replay-full one-shot reinit, and core QR-SAC hyperparameters
closely match Lee. Changing them together with OOB semantics would destroy the
clean comparison. The LiDAR shift augmentation is not validated by Lee, but
there is no current evidence that it causes OOB collapse.

### Unsupported diagnoses to avoid

- “More OOB penalty will solve it”: contradicted by the current A/B direction and
  not ablated in the papers.
- “Self-play caused the collapse”: plausible for `642a7a80`, but unisolated.
- “Replay 50/50 visibility sampling caused it”: changes the effective training
  distribution, but no requested run isolates it.
- “Two sampled rows per transition is unstable”: the papers do not publish a
  comparable update/data ratio.
- “LiDAR shift four equals image shift four”: the modalities and angular scales
  differ.

---

## 5. Controlled experiment sequence

Use cold starts, at least three seeds for promoted comparisons, fixed transition
budgets/checkpoints, and deterministic solo evaluation. Rank by lap completion,
OOB terminations per kilometre, recoveries after course-limit crossing, speed,
and progress—not mean reward alone.

1. **Reproduce:** current one-second fixed-opponent setup, unchanged, to establish
   seed variance.
2. **Reward atom only:** replace current OOB shaping with the paper linear-speed,
   three-tyre-equivalent shortcut atom; remove boundary/terminal reward shocks.
   Hold termination, starts, opponents, randomization, replay, and network fixed.
3. **Termination only:** on the winning reward atom, compare immediate
   full-car-out reset with a recoverable grace region/time and no terminal
   reward.
4. **Recovery resets only:** add Vasco-style near-boundary/off-course starts;
   retain ordinary random in-course starts.
5. **Curriculum only:** compare always-1v1 with a fixed solo/1v1 mixture against
   the same frozen opponent.
6. **Randomization ladder:** zero action latency and narrow/fix ego dynamics,
   then restore latency, friction/mass, drive scale, and sensor perturbations one
   family at a time.
7. **Only after stable completion:** test opponent diversity and stronger racing
   objectives. Do not return to rolling self-play until a fixed-opponent run
   remains stable beyond the previous collapse window.

Promotion gate: require improving OOB terminations per kilometre and lap
completion without a material speed collapse. A lower OOB fraction per step is
insufficient because a slow policy can reduce exposure while still ending nearly
every episode OOB.

---

## 6. Bottom line

The repo matches the papers on action rate, delta steering, recurrent
sequence training, QR-SAC hyperparameters, and replay-full reinitialization. Its
frozen baseline's largest course-limit mismatch was a quadratic any-footprint
OOB cost plus repo-only fixed and terminal shocks. The selected ADR 0013 test
instead treats the mapped edge as a physical wall and uses Lee's linear barrier
atom; it does not claim to implement the papers' open-track shortcut rule. The
observed ten-second terminal A/B moves toward lower speed without fixing OOB
dominance.
