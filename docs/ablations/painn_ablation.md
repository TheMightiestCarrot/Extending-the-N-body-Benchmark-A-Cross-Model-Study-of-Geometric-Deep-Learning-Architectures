# PaiNN Stabilization Ablation (N-body)

This document tracks iterative stabilization experiments on PaiNN for the N-body dataset.

- Dataset: `nbody_small` (pos_dt + vel)
- Dataloader: `painn_nbody`
- Trainer: `trainer_nbody`
- Baseline aggregation: degree-normalized mean (already applied)
- Logging: per-layer stats to `layer_stats.jsonl` in each run dir

Runs:

- R0 (diagnostics only): enable debug stats, no clamps, only grad clip
- R1 (residual damping): residual scales 0.5 (interaction + mixing)
- R2 (full safety net): message tanh + filter gain + message/state clamps + mixing tanh

We'll append each run below with config, early failure (if any), and observations.

---

## Recommended Settings (from R2_full)

- Aggregation: degree-normalized mean (already implemented)
- Interaction: residual_scale_interaction=0.5, tanh_message_scale=5.0, filter_gain=0.5, clip_vector_msg_norm=10.0, clip_scalar_msg_value=10.0
- Mixing: residual_scale_mixing=0.5, tanh_mixing_scale=5.0, clip_mu_norm=20.0, clip_q_value=100.0
- Trainer: clip_gradients_norm=1.0, debug_layer_stats_every=1 during debugging (can disable later)

Rationale: Explosions originate in deeper-layer mixing (L4–L5) with rapidly growing `dq` and vector norms; message bounds plus residual damping and state clamps keep both paths in a safe, equivariant range. R2_full maintained bounded stats over 64 epochs with no NaNs/Infs.

## R0_diag

Run dir: `runs/painn/2025-11-11_19-05-37__R0_diag`
- No NaN/Inf detected within captured steps.
- Top metrics (max value @ step):
  - debug/L5.mix.q_abs_max: 1.03343e+19 @ 0
  - debug/L5.mix.mu_norm_max: 3.61481e+12 @ 0
  - debug/L5.inter.vector_msg_norm_max: 2.16062e+07 @ 0
  - debug/L5.mix.mu_v_norm_max: 7.05219e+06 @ 0
  - debug/L4.mix.q_abs_max: 1.35631e+06 @ 0
  - debug/L5.mix.dq_max: 1.19918e+06 @ 0
- Per-layer highlights:
  - L0: inter.scalar_msg_max=0.079191@20; inter.vector_msg_norm_max=0.119244@20; mix.q_abs_max=27.184@20; mix.mu_norm_max=12.4051@10; mix.dq_max=1.56333@10; mix.dmu_scale_max=1.91312@20
  - L1: inter.scalar_msg_max=0.171136@20; inter.vector_msg_norm_max=0.554681@10; mix.q_abs_max=114.112@10; mix.mu_norm_max=27.3046@20; mix.dq_max=3.33846@20; mix.dmu_scale_max=3.08624@20
  - L2: inter.scalar_msg_max=0.389426@20; inter.vector_msg_norm_max=2.25118@10; mix.q_abs_max=373.54@20; mix.mu_norm_max=54.892@10; mix.dq_max=8.04192@10; mix.dmu_scale_max=7.8219@0
  - L3: inter.scalar_msg_max=1.01199@20; inter.vector_msg_norm_max=10.9571@20; mix.q_abs_max=2621.35@10; mix.mu_norm_max=397.999@0; mix.dq_max=21.8099@20; mix.dmu_scale_max=21.8865@10
  - L4: inter.scalar_msg_max=9.70512@20; inter.vector_msg_norm_max=1617.63@0; mix.q_abs_max=1.35631e+06@0; mix.mu_norm_max=21825.6@20; mix.dq_max=289.79@20; mix.dmu_scale_max=236.152@0
  - L5: inter.scalar_msg_max=4823.02@0; inter.vector_msg_norm_max=2.16062e+07@0; mix.q_abs_max=1.03343e+19@0; mix.mu_norm_max=3.61481e+12@0; mix.dq_max=1.19918e+06@0; mix.dmu_scale_max=1.18497e+06@0


---

## R1_resid0_5

Run dir: `runs/painn/2025-11-11_19-11-58__R1_resid0_5`
- No NaN/Inf detected within captured steps.
- Top metrics (max value @ step):
  - debug/L5.mix.q_abs_max: 2.73284e+124 @ 23
  - debug/L5.mix.mu_norm_max: 1.11644e+83 @ 23
  - debug/L5.inter.vector_msg_norm_max: 2.88266e+42 @ 23
  - debug/L5.mix.dq_max: 7.31931e+41 @ 23
  - debug/L5.mix.mu_v_norm_max: 6.02659e+41 @ 23
  - debug/L5.mix.dmu_scale_max: 5.78667e+41 @ 23
- Per-layer highlights:
  - L0: inter.scalar_msg_max=0.162717@23; inter.vector_msg_norm_max=1.37926@20; mix.q_abs_max=101.412@23; mix.mu_norm_max=28.1116@22; mix.dq_max=4.56249@23; mix.dmu_scale_max=7.65442@22
  - L1: inter.scalar_msg_max=0.263191@22; inter.vector_msg_norm_max=2.00413@23; mix.q_abs_max=186.188@23; mix.mu_norm_max=35.4242@23; mix.dq_max=10.3768@23; mix.dmu_scale_max=11.0991@23
  - L2: inter.scalar_msg_max=0.85701@23; inter.vector_msg_norm_max=13.8796@23; mix.q_abs_max=2449.32@23; mix.mu_norm_max=259.01@23; mix.dq_max=30.8062@23; mix.dmu_scale_max=30.134@23
  - L3: inter.scalar_msg_max=14.5773@23; inter.vector_msg_norm_max=1992.05@23; mix.q_abs_max=1.57369e+07@23; mix.mu_norm_max=40369.6@23; mix.dq_max=431.985@23; mix.dmu_scale_max=417.112@23
  - L4: inter.scalar_msg_max=592413@23; inter.vector_msg_norm_max=7.83754e+09@23; mix.q_abs_max=3.88108e+26@23; mix.mu_norm_max=4.51141e+17@23; mix.dq_max=1.35603e+09@23; mix.dmu_scale_max=1.17029e+09@23
  - L5: inter.scalar_msg_max=5.39488e+25@23; inter.vector_msg_norm_max=2.88266e+42@23; mix.q_abs_max=2.73284e+124@23; mix.mu_norm_max=1.11644e+83@23; mix.dq_max=7.31931e+41@23; mix.dmu_scale_max=5.78667e+41@23


---

## R2_full

Run dir: `runs/painn/2025-11-11_19-14-25__R2_full`
- No NaN/Inf detected within captured steps.
- Top metrics (max value @ step):
  - debug/L5.mix.q_abs_max: 100 @ 4
  - debug/L4.mix.q_abs_max: 70.1741 @ 2
  - debug/L3.mix.q_abs_max: 51.6418 @ 2
  - debug/L2.mix.q_abs_max: 38.0869 @ 4
  - debug/L1.mix.q_abs_max: 27.3535 @ 4
  - debug/L0.mix.q_abs_max: 22.5708 @ 4
- Per-layer highlights:
  - L0: inter.scalar_msg_max=2.25722@62; inter.vector_msg_norm_max=2.6751@52; mix.q_abs_max=22.5708@4; mix.mu_norm_max=10.0958@2; mix.dq_max=1.58871@7; mix.dmu_scale_max=1.77231@1
  - L1: inter.scalar_msg_max=0.918056@51; inter.vector_msg_norm_max=1.55551@25; mix.q_abs_max=27.3535@4; mix.mu_norm_max=10.8051@1; mix.dq_max=1.71738@4; mix.dmu_scale_max=2.00828@7
  - L2: inter.scalar_msg_max=0.935151@51; inter.vector_msg_norm_max=1.20613@51; mix.q_abs_max=38.0869@4; mix.mu_norm_max=11.7914@0; mix.dq_max=2.29579@4; mix.dmu_scale_max=2.72512@7
  - L3: inter.scalar_msg_max=2.07797@46; inter.vector_msg_norm_max=5@61; mix.q_abs_max=51.6418@2; mix.mu_norm_max=12.8477@1; mix.dq_max=2.69514@4; mix.dmu_scale_max=3.13536@2
  - L4: inter.scalar_msg_max=2.1462@53; inter.vector_msg_norm_max=5@39; mix.q_abs_max=70.1741@2; mix.mu_norm_max=13.542@2; mix.dq_max=3.62504@2; mix.dmu_scale_max=3.85857@2
  - L5: inter.scalar_msg_max=1.85079@56; inter.vector_msg_norm_max=5@43; mix.q_abs_max=100@4; mix.mu_norm_max=19.611@4; mix.dq_max=4.02035@7; mix.dmu_scale_max=4.58166@2


---
