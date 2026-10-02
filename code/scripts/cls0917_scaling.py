"""E2+E4 (mech0917_b): stability-selectivity-capability scaling across the 8-model pool.

New measurement (Step 1) — paired FD gains on SHARED contexts for all models:
  contexts = 27-frame true windows (cartpole_2d + cheetah, 64 windows/env).
  gain_a_rel: perturb final context action a[-1] +=/-= 0.1*per-dim action std (paired +/-),
              mean_w ||dz_1|| / (||z_1|| + 1e-8) of the one-step closed-loop predict head.
  gain_o_rel: perturb final obs o[-1] +=/-= 0.1*per-dim obs std, re-encode the window,
              same relative displacement.
  CSR       = gain_a_rel(h=1) / (gain_o_rel(h=1) + 1e-8)   (control-selectivity ratio)
  E4 ratio5 = ||dz^a_5|| / (||dz^o_5|| + 1e-8): the same single impulse rolled forward
              5 self-fed steps (predicted latents appended, dataset actions teacher-forced).

Step 2 assembles cited artifacts (NOT recomputed): lambda_state/c20 (S1 rollout),
gain20 (S2 gain), action_sens_rel (S3 part_a pooled), E1 seed-0 pooled env-SR (100 cells).
Step 3: Spearman rho(env-SR, metric) over 8 models (descriptive, n=8), scaling plane
figure (x=CSR log, y=env-SR, size=c20), bivariate verdicts.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/lx/snn")
# torch/numpy may pull a module named `code` into sys.modules before the project
# package is importable; evict any non-package binding so code.* resolves to /home/lx/snn/code.
if "code" in sys.modules and not hasattr(sys.modules["code"], "__path__"):
    del sys.modules["code"]

import numpy as np
import torch

from code.scripts import mech0917_multistep as ms
from code.scripts.event_align import build_model

# ---------------------------------------------------------------- constants
OUT_DIR = Path("/home/lx/snn/results/mech_0917_b/scaling")
CTX = ms.CTX                      # 27
H_E4 = 5                          # causal-rollout horizon for ratio5
REL = 0.1                         # perturbation = REL * per-dim std
EPS = 1e-8
SEED_WIN = 0
N_WIN = 64

# (short, build hint, checkpoint, family)
MODELS = [
    ("trace",      "stjewm_trace_only",
     "/data/lx/tmp/results/5m_5mpar/oodc_F1/stjewm_trace_only/seed_0/final.pt", "stjewm"),
    ("spike",      "stjewm_spike_only",
     "/data/lx/tmp/results/5m_5mpar/oodc_F1/stjewm_spike_only/seed_0/final.pt", "stjewm"),
    ("membrane",   "stjewm_membrane_readout",
     "/data/lx/tmp/results/5m_5mpar/oodc_F1/stjewm_membrane_readout/seed_0/final.pt", "stjewm"),
    ("hidden_leak", "stjewm_hidden_leak",
     "/data/lx/tmp/results/5m_5mpar/oodc_F1/stjewm_hidden_leak/seed_0/final.pt", "stjewm"),
    ("lewm",       "lewm_baseline_v2",
     "/data/lx/tmp/results/5m/oodc_F1/lewm_baseline_v2/seed_0/final.pt", "baseline"),
    ("gru",        "gru_baseline",
     "/data/lx/tmp/results/5m/oodc_F1/gru_baseline/seed_0/final.pt", "baseline"),
    ("mlp",        "mlp_baseline",
     "/data/lx/tmp/results/5m/oodc_F1/mlp_baseline/seed_0/final.pt", "baseline"),
    ("slif_trace", "stacked_lif_trace",
     "/data/lx/tmp/results/5m/oodc_F1/stacked_lif_trace/seed_0/final.pt", "baseline"),
]
SHORT = {hint: short for short, hint, _, _ in MODELS}

ENVS = ms.ENVS  # cartpole_2d, cheetah (npz + probe pos)

# cited artifacts
P_ROLLOUT = Path("/home/lx/snn/results/mech_0917_b/rollout/rollout_summary.json")
P_GAIN = Path("/home/lx/snn/results/mech_0917_b/gain/gain_summary.json")
P_AEMA = Path("/home/lx/snn/results/mech_0917_b/action_ema/action_ema_summary.json")
P_E1_AGG = Path("/data/lx/tmp/results/state_final_corrected_20260916/audit/aggregated_state_cells.json")
P_MAIN_TABLE = Path("/home/lx/snn/results/journal_prep/MAIN_TABLE_5M_STATE_FULL.md")
P_PAIRED = Path("/home/lx/snn/results/journal_prep/E13_paired_control/paired_control_summary.json")

# E1 seed-0 pooled env-SR per model, 100 cells each (MAIN_TABLE_5M_STATE_FULL.md
# "E1 每模型汇总"; used as fallback + cross-check for the full-precision aggregate read)
ENV_SR_MD = {
    "stjewm_trace_only": 0.31, "stjewm_spike_only": 0.25,
    "stjewm_membrane_readout": 0.20, "stjewm_hidden_leak": 0.18,
    "lewm_baseline_v2": 0.15, "gru_baseline": 0.16,
    "mlp_baseline": 0.25, "stacked_lif_trace": 0.28,
}


# ---------------------------------------------------------------- Step 1: paired FD gains
def roll_n_steps(model, ctx: torch.Tensor, act_win: torch.Tensor,
                 future_acts: torch.Tensor, h: int) -> torch.Tensor:
    """Self-fed closed-loop rollout for h steps; returns (B,h,D) predicted latents.

    Identical convention to mech0917_multistep.rollout: predict(ctx[-CTX:], act[-CTX:])[:,-1]
    per-position next-latent head, predicted latent appended, dataset action a_{t+h} fed.
    """
    buf_e, buf_a = ctx, act_win
    preds = []
    for k in range(1, h + 1):
        p = model.predict(buf_e[:, -CTX:], buf_a[:, -CTX:])[:, -1]
        preds.append(p)
        if k < h:
            buf_e = torch.cat([buf_e, p.unsqueeze(1)], dim=1)
            buf_a = torch.cat([buf_a, future_acts[:, k - 1].unsqueeze(1)], dim=1)
    return torch.stack(preds, dim=1)


def fd_env(model, env_name: str, obs: np.ndarray, act: np.ndarray, episodes,
           device: str, n_win: int) -> dict:
    rng = np.random.default_rng(SEED_WIN)
    starts, _ends, ts = ms.sample_windows(episodes, n_win, rng)
    n = len(ts)

    obs_std = obs.std(axis=0)      # per-dim, stored dims
    act_std = act.std(axis=0)
    d_obs, d_act = obs.shape[1], act.shape[1]

    o_seqs = [obs[t - CTX + 1: t + 1] for t in ts]
    a_seqs = [act[t - CTX + 1: t + 1] for t in ts]
    fut = np.stack([act[t + 1: t + H_E4 + 1] for t in ts])
    fut = np.pad(fut, ((0, 0), (0, 0), (0, ms.PAD_ACT - d_act)))
    fut = torch.from_numpy(fut).to(device)

    results = {"rel": REL, "n_windows": int(n), "obs_dim": d_obs, "act_dim": d_act,
               "obs_std_mean": float(obs_std.mean()), "act_std_mean": float(act_std.mean())}
    # accumulators: dz norms and base norms per step, action/obs channels
    acc = {ch: np.zeros((n, H_E4)) for ch in ("dza", "dzo")}
    acc["zb"] = np.zeros((n, H_E4))

    for i in range(0, n, 16):
        sl = slice(i, min(i + 16, n))
        with torch.inference_mode():
            a_win = ms.pad_batch(a_seqs[sl], ms.PAD_ACT).to(device)          # (B,CTX,56)
            ctx_b = ms.forward_windows(model, o_seqs[sl], a_seqs[sl], device)  # (B,27,D)

            # action channel: perturb a[-1] +- delta_a on a copy of the padded window
            da = torch.zeros_like(a_win)
            da[:, -1, :d_act] = (REL * torch.from_numpy(act_std).float().to(device))
            za_p = roll_n_steps(model, ctx_b, a_win + da, fut[sl], H_E4)
            za_m = roll_n_steps(model, ctx_b, a_win - da, fut[sl], H_E4)

            # obs channel: perturb o[-1] +- delta_o, re-encode the 27-frame window
            do = (REL * torch.from_numpy(obs_std).float().to(device))
            ctx_op, ctx_om = [], []
            for o_seq in o_seqs[sl]:
                o_p = o_seq.copy(); o_p[-1, :d_obs] = o_p[-1, :d_obs] + do.cpu().numpy()
                o_m = o_seq.copy(); o_m[-1, :d_obs] = o_m[-1, :d_obs] - do.cpu().numpy()
                ctx_op.append(o_p); ctx_om.append(o_m)
            zo_p = roll_n_steps(model, ms.forward_windows(model, ctx_op, a_seqs[sl], device),
                                a_win, fut[sl], H_E4)
            zo_m = roll_n_steps(model, ms.forward_windows(model, ctx_om, a_seqs[sl], device),
                                a_win, fut[sl], H_E4)
            zb = roll_n_steps(model, ctx_b, a_win, fut[sl], H_E4)

        acc["dza"][sl] = (za_p - za_m).norm(dim=-1).cpu().numpy()
        acc["dzo"][sl] = (zo_p - zo_m).norm(dim=-1).cpu().numpy()
        acc["zb"][sl] = zb.norm(dim=-1).cpu().numpy()

    dza, dzo, zb = acc["dza"], acc["dzo"], acc["zb"]
    gain_a = (dza / (zb + EPS)).mean(axis=0)   # per-step h=1..5
    gain_o = (dzo / (zb + EPS)).mean(axis=0)
    ratio_pw = dza[:, H_E4 - 1] / (dzo[:, H_E4 - 1] + EPS)  # per-window ratio5
    results.update(
        gain_a_rel={"1": float(gain_a[0]), "5": float(gain_a[H_E4 - 1])},
        gain_o_rel={"1": float(gain_o[0]), "5": float(gain_o[H_E4 - 1])},
        csr=float(gain_a[0] / (gain_o[0] + EPS)),
        ratio5=float(ratio_pw.mean()),
        ratio5_median=float(np.median(ratio_pw)),
        ratio5_normmeans=float(dza[:, H_E4 - 1].mean() / (dzo[:, H_E4 - 1].mean() + EPS)),
        dza_mean={"1": float(dza[:, 0].mean()), "5": float(dza[:, H_E4 - 1].mean())},
        dzo_mean={"1": float(dzo[:, 0].mean()), "5": float(dzo[:, H_E4 - 1].mean())},
        zb_mean={"1": float(zb[:, 0].mean()), "5": float(zb[:, H_E4 - 1].mean())},
        dza_decay5_over_1=float(dza[:, H_E4 - 1].mean() / (dza[:, 0].mean() + EPS)),
        dzo_decay5_over_1=float(dzo[:, H_E4 - 1].mean() / (dzo[:, 0].mean() + EPS)),
    )
    return results


def run_measurement(device: str, n_win: int) -> dict:
    data = {}
    for env_name, cfg in ENVS.items():
        obs, act, episodes = ms.load_env_data(cfg["npz"], min_len=CTX + H_E4)
        data[env_name] = (obs, act, episodes)
        print(f"[data] {env_name}: obs{obs.shape} act{act.shape} "
              f"{len(episodes)} eps >= {CTX + H_E4}", flush=True)

    out = {}
    for short, hint, ckpt, family in MODELS:
        t0 = time.time()
        model = ms.load_model(hint, ckpt, device)
        per_env = {}
        for env_name in ENVS:
            obs, act, episodes = data[env_name]
            per_env[env_name] = fd_env(model, env_name, obs, act, episodes, device, n_win)
        # equal-weight pool over the two envs
        pooled = {
            "gain_a_rel_1": float(np.mean([per_env[e]["gain_a_rel"]["1"] for e in ENVS])),
            "gain_o_rel_1": float(np.mean([per_env[e]["gain_o_rel"]["1"] for e in ENVS])),
            "gain_a_rel_5": float(np.mean([per_env[e]["gain_a_rel"]["5"] for e in ENVS])),
            "gain_o_rel_5": float(np.mean([per_env[e]["gain_o_rel"]["5"] for e in ENVS])),
            "csr": float(np.mean([per_env[e]["csr"] for e in ENVS])),
            "ratio5": float(np.mean([per_env[e]["ratio5"] for e in ENVS])),
        }
        degenerate = (pooled["gain_a_rel_1"] < 1e-6 and pooled["gain_o_rel_1"] < 1e-6)
        out[short] = {"family": family, "pooled": pooled, "per_env": per_env,
                      "degenerate": bool(degenerate)}
        print(f"[fd] {short:12s} csr={pooled['csr']:.4g} gain_a1={pooled['gain_a_rel_1']:.3g} "
              f"gain_o1={pooled['gain_o_rel_1']:.3g} ratio5={pooled['ratio5']:.4g} "
              f"deg={degenerate} ({time.time()-t0:.1f}s)", flush=True)
        del model
        torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------- Step 2: assemble
def load_e1_env_sr() -> tuple[dict, str]:
    """E1 seed-0 pooled env-SR per model (100 cells). Full precision from the
    aggregated audit file; MAIN_TABLE md values as cross-check/fallback."""
    if P_E1_AGG.exists():
        d = json.loads(P_E1_AGG.read_text())
        pool: dict[str, list[float]] = {}
        for e in d["evals"]:
            if e.get("experiment") != "E1" or e.get("training_seed") != 0:
                continue
            sr = e.get("metrics", {}).get("success_rate_env")
            if sr is not None:
                pool.setdefault(e["model"], []).append(float(sr))
        got = {m: float(np.mean(v)) for m, v in pool.items() if m in ENV_SR_MD}
        if set(got) == set(ENV_SR_MD):
            for m, v in got.items():
                assert abs(v - ENV_SR_MD[m]) < 0.01, f"env-SR drift {m}: {v} vs {ENV_SR_MD[m]}"
            return got, str(P_E1_AGG)
    return dict(ENV_SR_MD), f"{P_MAIN_TABLE} (fallback constants)"


def assemble(fd: dict) -> tuple[dict, dict]:
    roll = json.loads(P_ROLLOUT.read_text())["pooled"]
    gain = json.loads(P_GAIN.read_text())["pooled"]
    aema = json.loads(P_AEMA.read_text())["part_a_pooled"]
    env_sr, env_sr_src = load_e1_env_sr()

    table = {}
    for short, hint, ckpt, family in MODELS:
        fdm = fd[short]["pooled"]
        table[short] = {
            "family": family,
            "env_sr": env_sr[hint],
            "lambda_state": roll[short]["lambda_state"],
            "c20": roll[short]["c20"],
            "gain20": gain[hint]["gain20"]["mean"],
            "action_sens_rel": aema[hint]["s1step_rel"],
            "gain_a_rel": fdm["gain_a_rel_1"],
            "gain_o_rel": fdm["gain_o_rel_1"],
            "csr": fdm["csr"],
            "ratio5": fdm["ratio5"],
            "dza_decay5_over_1": float(np.mean(
                [fd[short]["per_env"][e]["dza_decay5_over_1"] for e in ENVS])),
            "dzo_decay5_over_1": float(np.mean(
                [fd[short]["per_env"][e]["dzo_decay5_over_1"] for e in ENVS])),
            "degenerate": fd[short]["degenerate"],
        }
    prov = {
        "lambda_state_c20": str(P_ROLLOUT),
        "gain20": str(P_GAIN),
        "action_sens_rel": f"{P_AEMA} (part_a_pooled.s1step_rel)",
        "env_sr": env_sr_src + " | cross-check " + str(P_PAIRED),
    }
    return table, prov


# ---------------------------------------------------------------- Step 3: analysis
CORR_METRICS = ["lambda_state", "c20", "gain20", "gain_a_rel", "gain_o_rel", "csr",
                "action_sens_rel", "ratio5"]


def spearman_block(table: dict, models: list[str]) -> dict:
    from scipy.stats import spearmanr
    y = np.array([table[m]["env_sr"] for m in models])
    out = {}
    for k in CORR_METRICS:
        x = np.array([table[m][k] for m in models])
        rho, p = spearmanr(x, y)
        out[k] = {"rho": float(rho), "p": float(p)}
    return out


def verdicts(table: dict, rho8: dict, rho6: dict) -> dict:
    models = list(table)
    best8 = max(rho8, key=lambda k: abs(rho8[k]["rho"]))
    best6 = max(rho6, key=lambda k: abs(rho6[k]["rho"]))

    def rank(vals: list[float]) -> list[int]:
        order = np.argsort(np.argsort(vals))
        return [int(r) for r in order]

    # contractive cluster: LeWM / GRU / trace are all strongly contractive (c20 << 1)
    cluster = ["trace", "lewm", "gru"]
    c20_cl = {m: table[m]["c20"] for m in cluster}
    sr_cl = {m: table[m]["env_sr"] for m in cluster}
    csr_cl = {m: table[m]["csr"] for m in cluster}
    # stability alone separates capability iff env-SR ordering follows |c20| ordering
    # within the cluster (all contractive; more contraction = smaller c20)
    sr_rank = rank([sr_cl[m] for m in cluster])
    c20_rank = rank([c20_cl[m] for m in cluster])
    csr_rank = rank([csr_cl[m] for m in cluster])
    stability_alone = sr_rank == c20_rank
    csr_separates = sr_rank == csr_rank

    # necessary-not-sufficient: is (contractive AND highest CSR in cluster) unique to trace?
    trace_cond = (table["trace"]["c20"] < 1.0
                  and all(table["trace"]["csr"] > table[m]["csr"] for m in ("lewm", "gru")))
    joint_separates = bool(trace_cond and not stability_alone)
    return {
        "best_single_metric_n8": {"metric": best8, **rho8[best8]},
        "best_single_metric_n6": {"metric": best6, **rho6[best6]},
        "contractive_cluster": {
            "members": cluster,
            "c20": c20_cl, "env_sr": sr_cl, "csr": csr_cl,
            "env_sr_rank": {m: int(r) for m, r in zip(cluster, sr_rank)},
            "c20_rank": {m: int(r) for m, r in zip(cluster, c20_rank)},
            "csr_rank": {m: int(r) for m, r in zip(cluster, csr_rank)},
            "stability_alone_separates_capability": bool(stability_alone),
            "csr_rank_matches_envsr_rank": bool(csr_separates),
        },
        "necessary_not_sufficient": {
            "trace_contractive": bool(table["trace"]["c20"] < 1.0),
            "trace_csr_exceeds_lewm_gru": trace_cond,
            "stability_necessary_not_sufficient": joint_separates,
            "statement": (
                "stability is necessary-but-not-sufficient iff trace is contractive and "
                "out-CSRs lewm/gru while stability alone does not order env-SR in the cluster"
            ),
        },
    }


# ---------------------------------------------------------------- outputs
def fmt(x, sig: int = 3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    if isinstance(x, float) and x != 0 and (abs(x) < 1e-3 or abs(x) >= 1e4):
        return f"{x:.{sig}g}"
    return f"{x:.{sig}f}" if abs(x) >= 0.01 or x == 0 else f"{x:.{sig}g}"


def plane_figure(table: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    color = {"stjewm": "#d62728", "baseline": "#1f77b4"}
    offset = {"trace": (-38, -12), "lewm": (7, -4), "gru": (7, -4), "slif_trace": (-64, 12)}
    for short, row in table.items():
        x, y = row["csr"], row["env_sr"]
        size = 30 + 900 / (1 + np.log10(max(row["c20"], 1e-9) + 1.0))  # log-compressed
        if row["degenerate"]:
            ax.scatter(x, y, s=size, marker="x", color="0.4", zorder=3)
        else:
            ax.scatter(x, y, s=size, marker="o", alpha=0.85,
                       color=color[row["family"]], edgecolor="k", linewidth=0.5, zorder=3)
        ax.annotate(f"{short}\n(c20={fmt(row['c20'])})" if row["degenerate"] else short,
                    (x, y), textcoords="offset points", xytext=offset.get(short, (6, 10)),
                    fontsize=8)
    ax.set_xscale("log")
    ax.axvline(1.0, color="0.6", linestyle="--", linewidth=0.8)
    ax.text(1.02, ax.get_ylim()[0] + 0.005, "action=obs", fontsize=7, color="0.4",
            rotation=90, va="bottom")
    ax.set_xlabel("CSR = gain_a_rel / gain_o_rel  (log scale; >1 action-dominated)")
    ax.set_ylabel("E1 pooled env-SR (seed 0, 100 cells)")
    ax.set_title("Stability-selectivity plane (marker size = c20 stability, log-compressed)")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color="#d62728", label="STJEWM readout"),
               plt.Line2D([], [], marker="o", linestyle="", color="#1f77b4", label="baseline"),
               plt.Line2D([], [], marker="x", linestyle="", color="0.4", label="degenerate (mlp)")]
    ax.legend(handles=handles, fontsize=8, loc="best")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_md(path: Path, table: dict, rho8: dict, rho6: dict, verd: dict,
             prov: dict, n_win: int) -> None:
    L: list[str] = []
    L.append("# E2+E4 — stability-selectivity-capability scaling (cls0917_scaling)\n")
    L.append("## Protocol (new measurement, shared contexts)\n")
    L.append(f"- Paired one-step closed-loop FD gains on **shared true 27-frame windows** "
             f"({', '.join(ENVS)}; {n_win} windows/env, seed {SEED_WIN}).")
    L.append(f"- gain_a_rel: a[-1] ± {REL}·per-dim action std (paired +/−), "
             f"mean_w ||dz_1||/(||z_1||+1e-8) through the per-position predict head.")
    L.append(f"- gain_o_rel: o[-1] ± {REL}·per-dim obs std, window re-encoded, same formula.")
    L.append("- CSR = gain_a_rel/gain_o_rel (h=1); **>1 = action-dominated, <1 = observation-dominated**.")
    L.append(f"- E4 ratio5: same single impulse rolled {H_E4} self-fed steps "
             f"(predicted latents appended, dataset actions teacher-forced); "
             f"ratio5 = ||dz^a_5||/(||dz^o_5||+1e-8).")
    L.append("- npz field check: cartpole {observations,next_observations,actions,rewards,dones} "
             "— **no velocity/qvel field**; obs are qpos-based states (cartpole D=2/A=1, "
             "cheetah D=9/A=6). No velocity probes are part of E2; nothing derived here.\n")
    L.append("## Scope mismatch (explicit)\n")
    L.append("- The FD gains (CSR, ratio5) are measured on **2 envs** (cartpole_2d, cheetah) "
             "with shared contexts, while env-SR is the **10-split pooled 13-env state suite** "
             "(100 cells/model). Correlations mix these scopes — interpret descriptively.\n")
    L.append("## Master table (8 models)\n")
    hdr = ("| model | env-SR | λ_state | c20 | gain20 | act_sens | gain_a_rel | gain_o_rel "
           "| CSR | ratio5 | flag |")
    L.append(hdr)
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for short, r in table.items():
        flag = "degenerate (all gains ~0)" if r["degenerate"] else \
               ("contractive" if r["c20"] < 1 else "expansive")
        L.append(f"| {short} | {fmt(r['env_sr'],3)} | {fmt(r['lambda_state'])} | {fmt(r['c20'])} "
                 f"| {fmt(r['gain20'])} | {fmt(r['action_sens_rel'])} | {fmt(r['gain_a_rel'])} "
                 f"| {fmt(r['gain_o_rel'])} | {fmt(r['csr'])} | {fmt(r['ratio5'])} | {flag} |")
    L.append("")
    L.append("Cited artifacts (not recomputed): " + "; ".join(f"{k} ← `{v}`" for k, v in prov.items()) + "\n")
    L.append("## Spearman ρ(env-SR, metric) — n=8 descriptive only\n")
    L.append("| metric | ρ (n=8) | p | ρ (n=6, no mlp/slif_trace) | p |")
    L.append("|---|---:|---:|---:|---:|")
    for k in CORR_METRICS:
        L.append(f"| {k} | {rho8[k]['rho']:+.3f} | {rho8[k]['p']:.3f} "
                 f"| {rho6[k]['rho']:+.3f} | {rho6[k]['p']:.3f} |")
    L.append("\n**n=8: every ρ is descriptive only** (critical value for p<0.05 two-sided "
             "is ρ≳0.74; with n=6 it is ρ≳0.89). No inferential claim is made.\n")
    b8, b6 = verd["best_single_metric_n8"], verd["best_single_metric_n6"]
    L.append(f"Best single rank correlate of env-SR: **{b8['metric']}** (ρ={b8['rho']:+.3f}, n=8); "
             f"excluding mlp+slif_trace: **{b6['metric']}** (ρ={b6['rho']:+.3f}, n=6).\n")
    cl = verd["contractive_cluster"]
    L.append("## Bivariate verdict\n")
    L.append(f"- Contractive cluster c20: " + ", ".join(f"{m}={fmt(v)}" for m, v in cl["c20"].items())
             + " — all strongly contractive.")
    L.append(f"- env-SR in cluster: " + ", ".join(f"{m}={fmt(v)}" for m, v in cl["env_sr"].items()) + ".")
    L.append(f"- **Stability alone separates capability in-cluster: "
             f"{'YES' if cl['stability_alone_separates_capability'] else 'NO'}** "
             f"(c20 rank {'=' if cl['stability_alone_separates_capability'] else '≠'} env-SR rank).")
    L.append(f"- CSR rank matches env-SR rank in-cluster: "
             f"**{'YES' if cl['csr_rank_matches_envsr_rank'] else 'NO'}** "
             "(CSR: " + ", ".join(f"{m}={fmt(v)}" for m, v in cl["csr"].items()) + ").")
    ns = verd["necessary_not_sufficient"]
    L.append(f"- **(stability AND CSR) jointly separate trace from lewm/gru: "
             f"{'YES' if ns['stability_necessary_not_sufficient'] else 'NO'}** "
             f"(trace contractive={ns['trace_contractive']}, "
             f"trace out-CSRs lewm/gru={ns['trace_csr_exceeds_lewm_gru']}) — "
             "the stability-is-necessary-not-sufficient question.\n")
    L.append("## E4 causal ratio (h=5, self-fed)\n")
    L.append("| model | ratio5 | ||dz^a_5||/||dz^a_1|| | ||dz^o_5||/||dz^o_1|| |")
    L.append("|---|---:|---:|---:|")
    for short, r in table.items():
        L.append(f"| {short} | {fmt(r['ratio5'])} | {fmt(r['dza_decay5_over_1'])} "
                 f"| {fmt(r['dzo_decay5_over_1'])} |")
    L.append("\nReading: ratio5 << 1 = the action impulse decays relative to the observation "
             "impulse by step 5 (observation-dominated dynamics); ratio5 ≈ 1 = action and "
             "observation perturbations persist equally.\n")
    L.append("Figure: `scaling_plane.png` (x=CSR log, y=env-SR, size=c20).\n")
    path.write_text("\n".join(L))


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="E2+E4 scaling: stability x selectivity x capability")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-win", type=int, default=N_WIN)
    ap.add_argument("--fig-only", action="store_true",
                    help="re-render md+png from the existing full scaling_summary.json")
    args = ap.parse_args()
    if args.smoke:
        args.n_win = min(args.n_win, 6)
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = "_smoke" if args.smoke else ""

    if args.fig_only:
        prev = json.loads((OUT_DIR / "scaling_summary.json").read_text())
        table, prov = prev["table"], prev["provenance"]
        rho8, rho6, verd = prev["spearman_n8"], prev["spearman_n6_no_mlp_slif"], prev["verdicts"]
        n_win = prev["protocol"]["n_windows_per_env"]
    else:
        fd = run_measurement(device, args.n_win)
        table, prov = assemble(fd)
        rho8 = spearman_block(table, list(table))
        rho6 = spearman_block(table, [m for m in table if m not in ("mlp", "slif_trace")])
        verd = verdicts(table, rho8, rho6)
        n_win = args.n_win

        summary = {
            "experiment": "E2+E4 stability-selectivity-capability scaling (cls0917_scaling)",
            "protocol": {
                "contexts": "27-frame true windows (shared across models)",
                "envs": {k: v["npz"] for k, v in ENVS.items()},
                "n_windows_per_env": args.n_win,
                "window_seed": SEED_WIN,
                "rel_perturbation": REL,
                "paired_fd": "dz = z(+) - z(-); gain_rel = mean_w ||dz_h||/(||z_h_base||+1e-8)",
                "csr": "gain_a_rel(h=1)/(gain_o_rel(h=1)+1e-8)",
                "ratio5": f"mean_w ||dz^a_{H_E4}||/(||dz^o_{H_E4}||+1e-8), impulse self-fed {H_E4} steps",
                "device": device,
                "smoke": bool(args.smoke),
                "npz_fields": "observations,next_observations,actions,rewards,dones (no velocity field)",
            },
            "table": table,
            "spearman_n8": rho8,
            "spearman_n6_no_mlp_slif": rho6,
            "caveat": "n=8 (n=6 restricted): descriptive only, no inferential claims",
            "verdicts": verd,
            "provenance": prov,
        }
        (OUT_DIR / f"scaling_summary{tag}.json").write_text(json.dumps(summary, indent=2))
    write_md(OUT_DIR / f"scaling_summary{tag}.md", table, rho8, rho6, verd, prov, n_win)
    plane_figure(table, OUT_DIR / f"scaling_plane{tag}.png")
    print(f"[done] wrote {OUT_DIR}/scaling_summary{tag}.{{json,md}} scaling_plane{tag}.png",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
