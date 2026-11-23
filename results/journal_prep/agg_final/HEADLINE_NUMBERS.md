# HEADLINE NUMBERS — final repaired evidence digest (for narrative rewrites)

> **Regenerated 2026-09-17.** Every value below is copied from the 20260916 final-repair JSON/MD sources listed per section (paths under `/data/lx/tmp/results/`). No legacy-document numbers.
> Training manifest: `/data/lx/tmp/results/_repair_archive/20260916T102154Z/training_final_repair_manifest.json` — sha256 `4376bb14c50e30e12c3a1d47435b17489568f148b6f0abd57592efe5936e1f09`.
> Freeze v7: `4ba90480e09d93d61d442be9569a03c020f6bbed9906af915015b3eb2dded834`.
> Tables regenerated from the same sources: `~/snn/results/journal_prep/{MAIN_TABLE_5M_STATE_FULL,MAIN_TABLE_5M_PIXEL_FULL,DIAG_RELOAD_SUMMARY,SCALE_INVARIANCE,SIGREG_SWEEP,LATENT_ROLLOUT_SUMMARY,UTILITY_MPC_TABLE,FULL_METRIC_MATRIX}.md`, `g2_summary.json`, `pixel_stats_summary.json`.

## E1 — state main line (13 models × 10 splits, seed 0, 1300 cells)
Source: `state_final_corrected_20260916/audit/aggregated_state_cells.json` (experiment=E1; per-cell files `.../E1/<split>/<model>/seed_0/eval_<env>.json`)
1. Pooled per model (100 cells each) — cos / env-SR / LeWM-SR@0.05: STJEWM-trace 0.109 / 0.31 / 0.37; STJEWM-spike 0.208 / 0.25 / 0.16; STJEWM-rate 0.050 / 0.26 / 0.73; STJEWM-no_trace 0.086 / 0.20 / 0.70; STJEWM-leak 0.086 / 0.18 / 0.71; STJEWM-membrane 0.086 / 0.20 / 0.70; ALIF 0.010 / 0.21 / 0.95; SLIF-trace 0.017 / 0.28 / 0.94; SLIF-free 0.034 / 0.25 / 0.85; LeWM 0.203 / 0.15 / 0.23; GRU 0.084 / 0.16 / 0.58; MLP 0.070 / 0.25 / 0.88; LIFTransformer 0.000 / 0.31 / 1.00
2. **LeWM-SR@0.05 falsification ranks**: top of the metric = LIFTransformer 1.00 (a collapsed latent, cos 0.000, that ALSO tops env-SR at 0.31 — env-SR is collapse-insensitive), ALIF 0.95, SLIF-trace 0.94, MLP 0.88 — collapsed/near-constant latents score highest, while STJEWM-trace (env-SR 0.31, tied best) sits near the bottom (0.37). LeWM@.05 ranks do not track behavior; use three-axis diagnostics, not cos<0.05.
3. Shape change vs old gen: oodc_F1F3 now 7 envs (+cheetah_qpos_masked, +delayed_t_maze), oodc_F2F3 8, oodc_F3 2; 100 cells/model (was 96).
4. no_trace and membrane_readout cells are identical value-for-value in E1/E2/E4 (equivalent checkpoints under the final-repair training); both rows are real data.
5. E7 λ variants and E2 seeds live in the same audit file (experiment=E7 / experiment=E2).

## E2 + multiseed (seeds)
Source: `state_final_corrected_20260916/audit/aggregated_state_cells.json` (experiment=E2, training_seed 1/2, 962 cells); `agg_final/b2_multiseed.json` (seeds 0/1/2 × 3 splits × 5 models, 555 cells); `agg_final/g5_multiseed.json` (13 models × seeds 0/1/2 × 3 splits)
1. E2 pooled over seeds 1+2 (74 cells/model): STJEWM-trace cos 0.115 env-SR 0.26; STJEWM-spike 0.207/0.19; LeWM 0.196/0.14; LIFTransformer 0.002/0.26; MLP 0.067/0.26.
2. G5 3-seed cos± (0/1/2): STJEWM-trace 0.118±0.012; STJEWM-spike 0.213±0.016; STJEWM-rate 0.047±0.007; no_trace 0.074±0.013; leak 0.079±0.012; membrane 0.074±0.013; ALIF 0.028±0.021; SLIF-trace 0.017±0.009; SLIF-free 0.030±0.002; LeWM 0.196±0.018; GRU 0.079±0.008; MLP 0.061±0.016; LIFTransformer 0.001±0.002.
3. B2 (5-model subset) agrees with G5 within ±0.003 on cos means (e.g. trace 0.118±0.012 in both).
4. Regime ordering is seed-stable: no seed flips collapse↔non-collapse clusters (CIs exclude cross-cluster values).

## E3 / pixel (130 ckpts × 13 envs = 1690 cells)
Source: `agg_final/pixel_cells.json`; per-cell `5m_pixel_final_20260916/<split>/<model>/seed_0/eval_summary.json` (hashes in `5m_pixel_final_20260916/grid_status.json`)
1. **Pixel pooled per model (ALL 10 splits, 130 cells each — corrected 2026-09-17; the values first published here were oodc_F3-only)** — cos / env-SR / LeWM-SR@0.05: STJEWM-trace 0.115 / 0.21 / 0.29; STJEWM-spike 0.177 / 0.11 / 0.30; STJEWM-rate 0.020 / 0.18 / 0.90; no_trace 0.034 / 0.11 / 0.84; leak 0.029 / 0.10 / 0.87; membrane 0.031 / 0.11 / 0.86; ALIF 0.003 / 0.09 / 0.98; SLIF-trace 0.000 / 0.22 / 1.00; SLIF-free 0.000 / 0.18 / 1.00; LeWM 0.034 / 0.10 / 0.86; GRU 0.044 / 0.14 / 0.86; MLP 0.000 / 0.20 / 1.00; LIFTransformer 0.000 / 0.12 / 1.00.
2. Collapse is graded: exact-zero cos (per-split and pooled) only for Stacked-LIF×2 / MLP / LIF-Transformer; LeWM-SR = 1.00 concentrates exactly on those four families (trace 0.29 / spike 0.30 are the lowest — the proxy metric still anti-correlates with quality).
3. Pixel protocol changed: horizon 5 with replan_every=1 (was horizon 25); env set is now a fixed 13-env grid (no pusht/tworoom/reacher on the pixel side).
4. Paired state-vs-pixel (same env, 1105 pairs): `agg_final/cross_modality.json` → state env-SR 0.252 / cos 0.080 / LeWM-SR 0.679 vs pixel env-SR 0.147 / cos 0.045 / LeWM-SR 0.800.

## E4 — held-out (holdout vs seen, same checkpoints)
Source: `state_final_corrected_20260916/audit/aggregated_state_cells.json` (experiment=E4, 546 cells; eval dirs `.../E4/heldout_*/`)
1. Pooled cos seen→heldout: STJEWM-trace 0.109→0.071; STJEWM-rate 0.050→0.018; SLIF-free 0.034→0.009; LeWM 0.203→0.146; ALIF 0.010→0.028 (only clear increase).
2. env-SR seen→heldout: trace 0.31→0.30; spike 0.25→0.20; LIFTransformer 0.31→0.26; SLIF-trace 0.28→0.29; GRU 0.16→0.16.
3. Headline: heldout cos ≤ seen cos for 12/13 models (goal-conditioned latents stay calibrated under held-out env specs); env-SR drops ≤5pp everywhere.

## E5 — scale axis (G4/G8/G16)
Source: `agg_final/scaling_table.json` (252 cells); raw `generalist_G{4,8,16}_stats/`
1. **Corrected claim (recomputed 2026-09-17): under div relative-range <0.5 AND resp max/min <5, 10 of 12 models fail.** The two that pass are STJEWM-spike (rel range 0.32, resp ratio 1.50 — stable calibration) and LeWM (0.10, 2.07 — stable over-reaction). STJEWM-trace misses marginally on div rel range 0.60 (absolute values 0.0002/0.0002/0.0001).
2. Failure modes now switch across scale: ALIF resp 5.7/9.9 (G4/G8) → 1937.9 (G16, div 0.498); MLP collapsed at G4/G8 (div 0.0) → explodes at G16 (resp 5.9e9, div 1.4e8); SLIF-trace collapsed at G4/G8 (div 0.000, ρ undefined) → non-collapsed at G16 (div 0.022).
3. STJEWM-trace div is scale-flat at 1e-4 (0.0002/0.0002/0.0001) — the near-constant trace readout is itself scale-stable.
4. LeWM has the steadiest div (0.378/0.398/0.357) with resp 55–114; GRU div grows 0.33→0.72.

## E6 — three-axis diagnostics (state + pixel) and the ρ erratum
Source: `5m_stats/` + `5m_stats_fair/` (910 files), `5m_pixel_stats/` (520 files); digests in `journal_prep/DIAG_RELOAD_SUMMARY.md` and `journal_prep/pixel_stats_summary.json`
1. **ρ erratum**: the old narrative "STJEWM readout event-ρ mean 0.948–0.962 vs obs-embedding 0.9984" is dead. Final-repair values (70 cells/model, readout): STJEWM-trace 0.034, spike 0.129, rate 0.004, no_trace 0.377, leak 0.385, membrane 0.377; ALIF 0.221, SLIF-trace 0.071, SLIF-free 0.130; LeWM 0.527, GRU 0.225, MLP 0.138, LIF-Tx −0.011. Same-batch obs-embedding ρ mean 0.814 (range −0.06–1.00) — not a constant ≈1 ceiling.
2. State resp/div magnitudes: STJEWM-trace div 0.0001 / resp 0.26 (near-constant); no_trace/leak/membrane div≈0.21–0.22, resp 74–81; ALIF resp 221.8; GRU 278.4; SLIF-free 700.9; MLP div 1.9e7 (explosive); LIF-Tx div 0.0054 / resp 19.2.
3. Pixel side (40 cells/model): STJEWM-trace resp 0.384 div 0.0024 (small-but-directional) vs ALIF 184.8/0.0527, SLIF-free 175.6/0.0268, GRU 1176.8/0.7837, LeWM 49.8/0.0230, MLP ≈0/0 — baselines' pixel latents stay near-constant, MLP absolutely constant.
4. E8 position probe (`g2_probe/`, 91 cells; digest `journal_prep/g2_summary.json`): LeWM 0.729 > GRU 0.465 > no_trace/membrane 0.257 > leak 0.254 > ALIF 0.242 > spike 0.158 > SLIF-free 0.113 > rate 0.048 > SLIF-trace −0.013 ≈ trace −0.001 ≈ MLP −0.001 ≈ LIF-Tx −0.001. Event-aligned trace is position-free; over-reactive LeWM is position-heavy (event-vs-position dissociation survives the repair).

## E7 — sigreg sweep
Source: `state_final_corrected_20260916/audit/aggregated_state_cells.json` (experiment=E7, 84 cells); checkpoints `5m_sigreg_sweep/`
1. **λ=0 collapse confirmed on new data**: F1 cos 0.005 (λ>0: 0.080–0.132) + LeWM-SR 0.987 (λ>0: 0.373–0.560) — constant-latent false hits. oodc_F2: λ=0 cos 0.063 / LeWM-SR 0.50 vs λ=0.09 cos 0.178 / LeWM-SR 0.067.
2. λ ∈ {0.001, 0.01, 0.09} is flat: F1 cos 0.132 / 0.080 / 0.101, env-SR 0.31 / 0.37 / 0.35.
3. env-SR column added: F1 0.32 (λ=0) vs 0.35 (λ=0.09) — env-SR alone does NOT show the collapse; the LeWM-SR/cos pair does.
4. Rollout-side λ=0 matches: div 0.0001, resp 0.083 (`latent_rollout/.../stjewm_trace_only_sig0.0/`).

## E9 — trace causality ablation
Source: `trace_ablation_final/{cartpole_2d,cheetah}.json` (2 env × 6 modes, 10 eps each, H=5)
1. cartpole_2d: baseline env-SR 0.60 → ablate_all 0.70 (event-window 0.60, non-event 0.70, random 0.70); cos 0.240→0.205.
2. cheetah: baseline env-SR 0.00, all ablations 0.00–0.10 (event 0.00, ablate_all 0.00); cos 0.095→0.075.
3. Corrected claim: zeroing the gated trace inside CEM does NOT hurt success (cartpole 0.60→0.70, cheetah 0.00 stays 0.00) — the trace is not a causal planning lever in the final-repair checkpoints; consistent with the trace readout being near-constant (E6/E12).

## E10 — utility latent-goal MPC
Source: `utility_final/latent_goal_mpc_table.md` (+ per-run `utility_final/latent_goal_mpc/`); copies: `journal_prep/UTILITY_MPC_TABLE.md`, `results/utility/latent_goal_mpc_table.md`
1. Terminal cos at H=20, trace_only: cheetah 0.022, walker 0.034, reacher 0.144, finger 0.046 — calibrated-latent band (~0.02–0.14), improves with horizon.
2. Near-zero terminal cos ≈ collapse signature: no_trace/hidden_leak/membrane/rate_only 0.0001–0.036; SLIF-trace 0.0001–0.13; MLP 0.0000–0.0018.
3. Native env_success stays ≤0.67 except isolated 1.00 cells (no_trace / hidden_leak / membrane / LeWM on finger H=1–5); thresholds: cheetah/walker 0.1, reacher 0.05, finger 0.3 (qpos RMS).
4. LeWM: terminal cos grows with H on reacher/finger (0.0005→0.375, 0.0003→0.194) — over-reactive latents degrade with planning depth.

## E11 — external Spiking-WM (PNAS 2025), within-episode diagnostics
Source: `spiking_wm_final_20260916/<task>.json` (12 tasks); `spiking_wm_final_20260916/grid_status.json`
1. **Strict-load**: `weights_loaded_strict: true` on all 12 tasks; loaded tensors 82 for cartpole_swingup, cheetah_run, cup_catch, pendulum_swingup; 84 reacher_easy/finger_spin/walker_walk, 86 fish_swim, 88 quadruped_walk, 90 humanoid_run, 96 dog_walk; world-model params 7.23–7.48M.
2. Within-episode event-ρ (posterior_categorical_mode / encoder_spike_time_mean): cartpole_swingup 0.121/0.641, cheetah_run 0.344/0.638, cup_catch 0.389/0.775, dog_walk 0.046/0.097, finger_spin 0.594/0.750, fish_swim 0.368/0.749, hopper_hop 0.536/0.771, humanoid_run 0.243/0.528, pendulum_swingup −0.483/0.554, quadruped_walk 0.227/0.261, reacher_easy 0.649/0.835, walker_walk 0.408/0.651.
3. Metric semantics differ from our CEM env-SR/cos (native within-episode correlation diagnostics, 2000 steps) — keep E11 out of cross-model ranking tables.

## E12 — unified-protocol latent rollouts (68 JSON + 68 NPZ)
Source: `latent_rollout/{cross_benchmark_F1,oodc_F2}/<model>/{cartpole_2d,cheetah}.json|.npz`
1. STJEWM-trace (λ=0.09): div 0.0001, resp 0.080, ρ −0.176–0.368 — trace readout near-constant in free rollout.
2. λ=0: div 0.0001, resp 0.083, ρ −0.100–−0.021; λ=0.001/0.01: div 0.0000–0.0004 — all sig checkpoints have near-constant trace readouts here; the λ=0 collapse signature shows in E7 closed-loop (cos→0.005, LeWM-SR→0.99), not in this open-loop stat.
3. MLP: div 24.29, resp 3952.8 (large, not the old 7.5e7); GRU resp 137.7, div 0.641; LeWM div 0.252, resp 21.4, ρ 0.545–0.976.
4. `regime_map_data.json` no longer exists in this generation; rebuild regime maps from the 68 JSONs if needed.
5. ρ per run has SE≈0.07 (n=198): near-zero ρ orderings are noise (same caveat as P12).

## G3 / P11 — energy proxy vs dense
Source: `journal_prep/P11_energy_final/measurements.json` (26 rows); `journal_prep/P11_energy_final/energy_summary.md`
1. STJEWM state: hypothetical proxy 3.10–4.65 MFLOP/step vs dense ledger 9.88–9.95 (trace 3.37 vs 9.95 ≈ 2.95×) — sparsity-weighted proxy only.
2. No-saving baselines (proxy ≈ dense): ALIF 9.70, LIF-Tx 9.57, GRU 10.24, MLP 9.98, LeWM 9.76; SLIF-trace 5.05 / SLIF-free 3.66 are the only other savers.
3. Measured soma sparsity: LIF-Tx 98.4%, ALIF 96.1% — yet their proxies show no saving (weighted partition excludes their dense paths); sparsity ≠ operand savings.
4. P11 states explicitly: not measured hardware energy, speed, or executed FLOPs; pixel frozen ViT excluded from the ledger.

## P12 — synthetic reconstruction (ckpt-generation-independent)
Source: `p12_synthetic_reconstruction/summary.json` (protocol p12_synthetic_reconstruction_20260917)
1. Regime rules reproduce: constant→collapsed (div 3.8e-15), uncorrelated→noise (ρ 0.005), identity k≥0.1→over_reactive at raw scale; div grows monotonically in k (check=true).
2. **Disclosed scale caveat**: div criterion is observation-scale-dependent — div(k·obs)=k·div(obs). The narrative k-crossing (0.3, 0.5) implies obs std 0.10–0.17, incompatible with this repo's raw cartpole scale; at the measured scale the crossing is k∈(0.04, 0.1), scale-free crossing k_per_unit_scale = 0.0435.
3. ρ-vs-σ: documented σ crossing (0.02, 0.05) reproduces (σ* = 0.0405); but "ρ decreases monotonically with σ" is false near the noise floor (SE≈0.07; σ=0.1 vs 0.2 order can invert).
4. Claims-check ledger: 4 of 6 documented claims verified true, 2 false (`documented_claims_check`).

## Pixel stats + probes digests (regenerated JSONs)
Source: `journal_prep/g2_summary.json` (from `g2_probe/`, 91 cells), `journal_prep/pixel_stats_summary.json` (from `5m_pixel_stats/`, 520 files)
1. g2 (position R² mean, n=7): see E6 item 4 — LeWM 0.729 top; trace/MLP/LIF-Tx/SLIF-trace ≈ 0.
2. pixel stats (resp/div/n=40): STJEWM-trace 0.384/0.0024; GRU 1176.8/0.7837; MLP ≈0/0; full table in the JSON.

## E13 — paired control analysis, STJEWM-trace vs LeWM (added 2026-09-17)
Source: `journal_prep/E13_paired_control/paired_control_summary.json` (+ `.md`); script `code/scripts/e13_paired_control.py`; narrative EXP_Report §12.8. Post-hoc paired analysis of E1/E3 final cells — not a pre-registered primary endpoint.
1. Pairing: 1-to-1 on (split, env); episode identity asserted (state `initialization_seed`, pixel `episode_idx`), 0 mismatches. Canonical set = 13 shared DMC qpos envs → state 85 pairs / 425 episodes, pixel 130 pairs / 650 episodes. cos = mean_cos_dist (lower better); cos_adv = LeWM − trace (positive favors trace).
2. **State: advantage real and aligned, no dissociation.** Split-level Δ env-SR +0.167 (exact sign-flip p=0.0078, 10/10 splits same direction incl. 1 zero tie), cos_adv +0.083 (p=0.0039), dissociated splits 0/10, corr(ΔSR, cos_adv) +0.86. Episodes 79 only-trace vs 8 only-LeWM (McNemar p=8.4e-16). Env drivers: cartpole_2d +0.74 (26/0, p=3e-8), fish +0.85 (18/1, p=7.6e-5), finger +0.43 (17/2, p=7e-4).
3. **Pixel: full metric–behavior dissociation.** Split-level Δ env-SR +0.114 (p=0.0020, 10/10 same direction) while cos_adv −0.081 (p=0.0020) — LeWM's latent sits closer to the goal embedding in all 10 splits yet loses env-SR in all 10. Episodes 77:3 (p=1.4e-19). cartpole +0.54 (27/0; 9/10 diss pairs), fish +0.52 (26/0; 9/10), pendulum +0.28 (16/2; 7/10).
4. Difficulty: state medium band (pooled SR 0.1–0.7) carries the advantage (+0.394, 30/5/1 wins/ties/losses); easy saturated (Δ=0), hard both zero (CEM/H ceiling). Low-dim envs (nq<10) ΔSR 0.204/0.137 (state/pixel) vs high-dim 0.161/0.087; corr(nq, ΔSR) −0.29/−0.35.
5. Reading: on pixel, LeWM's near-constant latent trivially matches the goal embedding (cos 0.034) while trace (cos 0.115) keeps state distinctions and wins control — the same collapse-triviality mechanism as E1/§2, now paired per-cell. State shows no dissociation (trace wins both axes).
6. Caveats: post-hoc, no multiple-comparison correction, 5 episodes/cell; phrase as "directionally consistent across all splits (exact p)", never "proof".

## E14 — closed-loop env-SR on the scale axis (added 2026-09-17)
Source: `journal_prep/E14_scale_closed_loop/scale_closed_loop_summary.json` (+ `.md`); raw cells `/data/lx/tmp/results/scale_closed_loop_20260917/` (per-cell sha256 + receipts); script `code/scripts/e14_scale_closed_loop.py`; narrative EXP_Report §12.9.
1. Design: E5 measured only div/resp/ρ ("env_sr_evidence: not evaluated by this diagnostic grid"). E14 adds the closed-loop half on the nested-union in-union core {cartpole_2d, pendulum_2d, cheetah, pusht} × 12 models × G4/G8/G16 = 144 cells, exact E1 protocol (CEM 300/30, H25, budget 50, goal=t+25, 5 eps, pad 128/act 56, pusht 30 iters); repair-generation checkpoints, 0 retrains. Scale mixes env count (4/8/16) and per-env data (2000/2000/10000 windows).
2. **STJEWM-trace pooled env-SR 0.35 → 0.35 → 0.30 (G4/G8/G16)** — in-union control utility survives growing diversity; cartpole 1.00 at all three scales. LeWM 0.10/0.15/0.15 — trace dominates at every scale (+0.25/+0.20/+0.15), and on this core its goal-cos is better too (0.158/0.178/0.101 vs 0.336/0.308/0.325) — no dissociation, consistent with E13-state.
3. **cheetah@G16: env-SR 0.00 for all 12 models** (best 0.40 at G4/G8) — the closed-loop face of E5's "scale = failure-mode amplifier". pendulum (~0.20 everywhere) and pusht (~0) are scale-insensitive ceilings, carrying no scale signal.
4. env-SR remains unrankable alone: collapsed Stacked-LIF-trace tops G4/G8 (0.45/0.45, cos 0.000, LeWM-SR 1.00 trivial); collapsed MLP 0.35/0.25/0.35.
5. Caveats: 5 eps/cell, 4 in-union envs, 3 scale points; post-hoc, not a pre-registered significance test; phrase as "control utility on the shared in-union core is preserved as diversity grows (0.35→0.30), while degradation concentrates in the hard env and is architecture-uniform."
