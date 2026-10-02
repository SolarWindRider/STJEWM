#!/usr/bin/env python
"""E13 — paired STJEWM-trace vs LeWM control analysis.

Question: does STJEWM-trace's pooled env-SR advantage over LeWM survive a
per-environment, per-split, paired analysis, and does the advantage dissociate
from terminal latent-goal cosine similarity ("more informative != more useful
for control")?

Inputs (final-repair generation, sha-bound):
  state: /data/lx/tmp/results/state_final_corrected_20260916/audit/aggregated_state_cells.json
         trace cells /data/lx/tmp/results/5m_5mpar/<split>/stjewm_trace_only/seed_0/
         LeWM  cells /data/lx/tmp/results/5m/<split>/lewm_baseline_v2/seed_0/
  pixel: /data/lx/tmp/results/agg_final/pixel_cells.json + per-cell
         /data/lx/tmp/results/5m_pixel_final_20260916/<split>/{stjewm,lewm_baseline}/seed_0/eval_summary.json

Pairing rules:
  - state: one-to-one on (split, env); episode pairing verified by identical
    initialization_seed lists in both per_episode arrays (asserted at runtime).
  - pixel: one-to-one on (split, env); episode pairing verified by identical
    episode_idx lists (asserted at runtime).
  - Canonical analysis restricts to the 13 DMC qpos envs shared by state and
    pixel suites (pusht / tworoom / reacher / delayed_t_maze /
    cheetah_qpos_masked excluded from cross-modality claims).
  - Split-level statistic: mean per-split equal-weight env delta, tested with an
    exact sign-flip randomization over the 10 splits (2^10 outcomes).
  - Episode-level statistic: exact McNemar on discordant paired episodes.

Outputs (protected artifacts, never silently overwritten):
  results/journal_prep/E13_paired_control/paired_control_summary.json
  results/journal_prep/E13_paired_control/paired_control_table.md

Run:  python -m code.scripts.e13_paired_control
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/lx/snn")
from code.scripts.audited_results import write_new_json  # noqa: E402

STATE_AGG = Path("/data/lx/tmp/results/state_final_corrected_20260916/audit/aggregated_state_cells.json")
PIXEL_AGG = Path("/data/lx/tmp/results/agg_final/pixel_cells.json")
PIXEL_ROOT = Path("/data/lx/tmp/results/5m_pixel_final_20260916")
TRACE_MODEL_STATE = "stjewm_trace_only"
LEWM_MODEL_STATE = "lewm_baseline_v2"
CANONICAL_ENVS = [
    "cartpole_2d", "pendulum_2d", "finger", "ball_in_cup", "cheetah", "walker",
    "hopper", "quadruped", "humanoid", "humanoid_CMU", "dog", "fish", "stacker",
]
CANONICAL_PIXEL_ENVS = [e.replace("_2d", "") for e in CANONICAL_ENVS]
ENV_NQ = {"cartpole_2d": 2, "pendulum_2d": 2, "finger": 3, "ball_in_cup": 4,
          "cheetah": 9, "walker": 9, "hopper": 7, "quadruped": 30,
          "humanoid": 28, "humanoid_CMU": 63, "dog": 87, "fish": 14, "stacker": 20}
PIXEL_ENV_NQ = {e.replace("_2d", ""): n for e, n in ENV_NQ.items()}


def exact_signflip_p(values) -> float | None:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    n = len(vals)
    if n == 0:
        return None
    obs = abs(float(vals.mean()))
    extreme = 0
    for mask in range(1 << n):
        s = sum(v if (mask >> i) & 1 else -v for i, v in enumerate(vals))
        if abs(s / n) >= obs - 1e-15:
            extreme += 1
    return extreme / float(1 << n)


def mcnemar_exact(b: int, c: int) -> float | None:
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2.0
    return min(p, 1.0)


def _episode_maps_state(path: Path) -> dict:
    payload = json.loads(path.read_text())
    return {e["initialization_seed"]: bool(e["env_success"]) for e in payload["per_episode"]}


def _episode_maps_pixel(path: Path, env: str) -> dict:
    payload = json.loads(path.read_text())
    return {e["episode_idx"]: bool(e["env_success"]) for e in payload["results_per_env"][env]["per_episode"]}


def _write_new(path: Path, text: str) -> None:
    """Refuse to silently overwrite an existing canonical artifact."""
    path = Path(path)
    if path.exists():
        raise SystemExit(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main() -> int:
    state = json.loads(STATE_AGG.read_text())
    pixel = json.loads(PIXEL_AGG.read_text())
    out_dir = Path("/home/lx/snn/results/journal_prep/E13_paired_control")

    # ---- state cells ----------------------------------------------------
    state_rows = {}
    for r in state["evals"]:
        if r.get("experiment") != "E1" or r["model"] not in (TRACE_MODEL_STATE, LEWM_MODEL_STATE):
            continue
        m = r["metrics"]
        state_rows[(r["split"], r["env"], r["model"])] = {
            "env_sr": float(m["success_rate_env"]),
            "cos": float(m["mean_cos_dist"]),
            "source": Path(r["output"]),
        }
    state_pairs, state_ep = [], []
    for (split, env, model), a in state_rows.items():
        if model != TRACE_MODEL_STATE:
            continue
        b = state_rows.get((split, env, LEWM_MODEL_STATE))
        if b is None:
            continue
        ta = _episode_maps_state(a["source"])
        tb = _episode_maps_state(b["source"])
        if sorted(ta) != sorted(tb):
            raise SystemExit(f"state episode identity mismatch at {split}/{env}")
        state_pairs.append({
            "split": split, "env": env,
            "trace_env_sr": a["env_sr"], "lewm_env_sr": b["env_sr"],
            "delta_env_sr": a["env_sr"] - b["env_sr"],
            "trace_cos": a["cos"], "lewm_cos": b["cos"],
            "cos_advantage": b["cos"] - a["cos"],  # positive: trace farther from collapse on the lower-is-better metric
        })
        for s in sorted(ta):
            state_ep.append({"split": split, "env": env, "t": ta[s], "l": tb[s]})

    # ---- pixel cells ----------------------------------------------------
    pixel_pairs, pixel_ep = [], []
    for split_dir in sorted(PIXEL_ROOT.iterdir()):
        tp = split_dir / "stjewm" / "seed_0" / "eval_summary.json"
        lp = split_dir / "lewm_baseline" / "seed_0" / "eval_summary.json"
        if not tp.exists() or not lp.exists():
            continue
        t = json.loads(tp.read_text())
        l = json.loads(lp.read_text())
        for env in t["results_per_env"]:
            tm = t["results_per_env"][env]
            lm = l["results_per_env"].get(env)
            if lm is None:
                continue
            ta = _episode_maps_pixel(tp, env)
            tb = _episode_maps_pixel(lp, env)
            if sorted(ta) != sorted(tb):
                raise SystemExit(f"pixel episode identity mismatch at {split_dir.name}/{env}")
            pixel_pairs.append({
                "split": split_dir.name, "env": env,
                "trace_env_sr": float(tm["success_rate_env"]),
                "lewm_env_sr": float(lm["success_rate_env"]),
                "delta_env_sr": float(tm["success_rate_env"]) - float(lm["success_rate_env"]),
                "trace_cos": float(tm["mean_cos_dist"]),
                "lewm_cos": float(lm["mean_cos_dist"]),
                "cos_advantage": float(lm["mean_cos_dist"]) - float(tm["mean_cos_dist"]),
            })
            for s in sorted(ta):
                pixel_ep.append({"split": split_dir.name, "env": env, "t": ta[s], "l": tb[s]})

    # ---- canonical summaries -------------------------------------------
    def split_summary(pairs, canon_envs):
        bys = defaultdict(list)
        for x in pairs:
            if x["env"] in canon_envs:
                bys[x["split"]].append(x)
        rows = []
        for s in sorted(bys):
            v = bys[s]
            rows.append({
                "split": s, "n_env": len(v),
                "trace_env_sr": float(np.mean([x["trace_env_sr"] for x in v])),
                "lewm_env_sr": float(np.mean([x["lewm_env_sr"] for x in v])),
                "delta_env_sr": float(np.mean([x["delta_env_sr"] for x in v])),
                "trace_cos": float(np.mean([x["trace_cos"] for x in v])),
                "lewm_cos": float(np.mean([x["lewm_cos"] for x in v])),
                "cos_advantage": float(np.mean([x["cos_advantage"] for x in v])),
            })
        return rows

    def env_summary(pairs, canon_envs, ep, nq_map):
        by = defaultdict(list)
        for x in pairs:
            if x["env"] in canon_envs:
                by[x["env"]].append(x)
        out = {}
        for env in canon_envs:
            v = by.get(env)
            if not v:
                continue
            d = np.array([x["delta_env_sr"] for x in v])
            ca = np.array([x["cos_advantage"] for x in v])
            ev = [e for e in ep if e["env"] == env]
            ot = sum(1 for e in ev if e["t"] and not e["l"])
            ol = sum(1 for e in ev if e["l"] and not e["t"])
            out[env] = {
                "n_pairs": len(v), "nq_dim": nq_map.get(env),
                "trace_env_sr": float(np.mean([x["trace_env_sr"] for x in v])),
                "lewm_env_sr": float(np.mean([x["lewm_env_sr"] for x in v])),
                "delta_env_sr": float(d.mean()),
                "wins": int((d > 0).sum()), "ties": int((d == 0).sum()), "losses": int((d < 0).sum()),
                "p_signflip_dsr": exact_signflip_p(d),
                "trace_cos": float(np.mean([x["trace_cos"] for x in v])),
                "lewm_cos": float(np.mean([x["lewm_cos"] for x in v])),
                "cos_advantage": float(ca.mean()),
                "p_signflip_cos": exact_signflip_p(ca),
                "dissociated_pairs": int(((d > 0) & (ca < 0)).sum()),
                "episode_mcnemar": {"only_trace": ot, "only_lewm": ol, "p_exact": mcnemar_exact(ot, ol)},
            }
        return out

    def block(pairs, canon_envs, ep, nq_map):
        canon_pairs = [x for x in pairs if x["env"] in canon_envs]
        ss = split_summary(canon_pairs, canon_envs)
        ds = np.array([x["delta_env_sr"] for x in ss])
        ca = np.array([x["cos_advantage"] for x in ss])
        ev = [e for e in ep if e["env"] in canon_envs]
        ot = sum(1 for e in ev if e["t"] and not e["l"])
        ol = sum(1 for e in ev if e["l"] and not e["t"])
        return {
            "canonical_envs": canon_envs,
            "n_pairs": len(canon_pairs),
            "splits": ss,
            "split_level": {
                "delta_env_sr": float(ds.mean()),
                "p_signflip_dsr": exact_signflip_p(ds),
                "cos_advantage": float(ca.mean()),
                "p_signflip_cos": exact_signflip_p(ca),
                "dissociated_splits": int(((ds > 0) & (ca < 0)).sum()),
                "same_positive_splits": int(((ds > 0) & (ca > 0)).sum()),
                "split_delta_sr_vs_cos_adv_corr": float(np.corrcoef(ds, ca)[0, 1]),
            },
            "envs": env_summary(pairs, canon_envs, ep, nq_map),
            "episode_mcnemar": {
                "n_episodes": len(ev),
                "both": sum(1 for e in ev if e["t"] and e["l"]),
                "only_trace": ot, "only_lewm": ol,
                "neither": sum(1 for e in ev if not e["t"] and not e["l"]),
                "p_exact": mcnemar_exact(ot, ol),
            },
        }

    result = {
        "status": "completed",
        "analysis": "E13_paired_control_v1",
        "question": "Does the pooled STJEWM-trace env-SR advantage survive paired per-env/per-split analysis, and does it dissociate from terminal latent-goal cosine?",
        "state": block(state_pairs, CANONICAL_ENVS, state_ep, ENV_NQ),
        "pixel": block(pixel_pairs, CANONICAL_PIXEL_ENVS, pixel_ep, PIXEL_ENV_NQ),
        "difficulty_strata_state": {},
        "coverage": {
            "state_pairs_total": len(state_pairs),
            "pixel_pairs_total": len(pixel_pairs),
            "episode_identity_asserted": True,
            "state_aggregate": str(STATE_AGG),
            "state_aggregate_sha_note": "see aggregated_state_cells.json header",
            "pixel_aggregate": str(PIXEL_AGG),
            "training_manifest_sha256": state["training_manifest_sha256"],
            "pixel_grid_status_sha256": pixel["grid_status_sha256"],
            "trace_state_root": "/data/lx/tmp/results/5m_5mpar",
            "lewm_state_root": "/data/lx/tmp/results/5m",
            "pixel_root": str(PIXEL_ROOT),
        },
        "interpretation_notes": [
            "cos is mean_cos_dist (lower is better); cos_advantage is LeWM minus trace, so positive favors trace.",
            "env-SR advantage favors trace when delta_env_sr is positive.",
            "A control-dissociation cell requires delta_env_sr > 0 with cos_advantage < 0.",
            "Episode counts are 5 per cell; McNemar p-values are exact but per-env power is limited (n=35 state / n=50 pixel per env).",
            "Split-level sign-flip p-values are exact over 10 split means; they test the paired advantage, not a pre-registered primary endpoint.",
        ],
    }

    # difficulty strata (state only, by pooled model difficulty)
    cp = [x for x in state_pairs if x["env"] in CANONICAL_ENVS]
    for name, cond in (("easy_gt0.7", lambda m: m > 0.7),
                       ("medium_0.1to0.7", lambda m: 0.1 <= m <= 0.7),
                       ("hard_lt0.1", lambda m: m < 0.1)):
        sel = [x for x in cp if cond((x["trace_env_sr"] + x["lewm_env_sr"]) / 2.0)]
        if not sel:
            continue
        d = np.array([x["delta_env_sr"] for x in sel])
        ca = np.array([x["cos_advantage"] for x in sel])
        result["difficulty_strata_state"][name] = {
            "n_pairs": len(sel),
            "mean_trace_env_sr": float(np.mean([x["trace_env_sr"] for x in sel])),
            "mean_lewm_env_sr": float(np.mean([x["lewm_env_sr"] for x in sel])),
            "delta_env_sr": float(d.mean()),
            "wins": int((d > 0).sum()), "ties": int((d == 0).sum()), "losses": int((d < 0).sum()),
            "cos_advantage": float(ca.mean()),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_new_json(out_dir / "paired_control_summary.json", result)

    # markdown table
    lines = [
        "# E13 — Paired STJEWM-trace vs LeWM control analysis (final-repair generation)",
        "",
        "Pairing: 1-to-1 on (split, env), episode identity asserted. cos = mean_cos_dist (lower is better);",
        "`cos_adv` = LeWM − trace (positive favors trace). Split p: exact sign-flip over 10 split means.",
        "Episode p: exact McNemar on discordant paired episodes.",
        "",
    ]
    for label in ("state", "pixel"):
        b = result[label]
        lines += [f"## {label} — 13 canonical DMC envs, 10 splits (n_pairs={b['n_pairs']})", "",
                  "| split | n_env | trace env-SR | LeWM env-SR | Δ env-SR | trace cos | LeWM cos | cos_adv |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for s in b["splits"]:
            lines.append(f"| {s['split']} | {s['n_env']} | {s['trace_env_sr']:.3f} | {s['lewm_env_sr']:.3f} | "
                         f"{s['delta_env_sr']:+.3f} | {s['trace_cos']:.3f} | {s['lewm_cos']:.3f} | {s['cos_advantage']:+.3f} |")
        sl = b["split_level"]
        em = b["episode_mcnemar"]
        lines += ["",
                  f"Split level: Δ env-SR {sl['delta_env_sr']:+.3f} (sign-flip p={sl['p_signflip_dsr']}); "
                  f"cos_adv {sl['cos_advantage']:+.3f} (p={sl['p_signflip_cos']}); "
                  f"dissociated splits {sl['dissociated_splits']}/10; corr(ΔSR, cos_adv) {sl['split_delta_sr_vs_cos_adv_corr']:+.2f}.",
                  "",
                  f"Episodes: n={em['n_episodes']}, only-trace {em['only_trace']}, only-LeWM {em['only_lewm']}, "
                  f"McNemar p={em['p_exact']:.3g}.",
                  "",
                  "| env | nq | trace SR | LeWM SR | ΔSR (w/t/l, p) | cos_adv (p) | diss pairs | only-t/l (p) |",
                  "|---|---:|---:|---:|---|---|---:|---|"]
        for env, v in b["envs"].items():
            p_env = v["episode_mcnemar"]["p_exact"]
            p_env_txt = "n/a" if p_env is None else f"{p_env:.3g}"
            p_sr_txt = "n/a" if v["p_signflip_dsr"] is None else f"{v['p_signflip_dsr']:.3g}"
            p_cos_txt = "n/a" if v["p_signflip_cos"] is None else f"{v['p_signflip_cos']:.3g}"
            lines.append(
                f"| {env} | {v['nq_dim']} | {v['trace_env_sr']:.2f} | {v['lewm_env_sr']:.2f} | "
                f"{v['delta_env_sr']:+.2f} ({v['wins']}/{v['ties']}/{v['losses']}, p={p_sr_txt}) | "
                f"{v['cos_advantage']:+.3f} (p={p_cos_txt}) | {v['dissociated_pairs']}/{v['n_pairs']} | "
                f"{v['episode_mcnemar']['only_trace']}/{v['episode_mcnemar']['only_lewm']} (p={p_env_txt}) |")
        lines.append("")
    if result["difficulty_strata_state"]:
        lines += ["## State difficulty strata (pooled difficulty = mean of both models' env-SR)", "",
                  "| stratum | n_pairs | trace SR | LeWM SR | ΔSR (w/t/l) | cos_adv |", "|---|---:|---:|---:|---|---:|"]
        for k, v in result["difficulty_strata_state"].items():
            lines.append(f"| {k} | {v['n_pairs']} | {v['mean_trace_env_sr']:.3f} | {v['mean_lewm_env_sr']:.3f} | "
                         f"{v['delta_env_sr']:+.3f} ({v['wins']}/{v['ties']}/{v['losses']}) | {v['cos_advantage']:+.3f} |")
        lines.append("")
    _write_new(out_dir / "paired_control_table.md", "\n".join(lines))
    print(f"[e13_paired_control] wrote {out_dir}/paired_control_summary.json and paired_control_table.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
