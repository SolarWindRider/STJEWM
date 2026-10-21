"""Build the third-round-audit verification bundle (extraction only, no new training).

Produces /home/lx/snn/verification_bundle/ per the 2026-10-21 audit section 7C:
CSVs (state/E2/E4/E7/E8/E9 cells, c20, gain, M4 timeseries, params),
provenance manifest, INDEX.md with claim->file->panel mapping and Q1-Q5 answers,
extraction scripts and logs. Run: python -m code.scripts.build_verification_bundle
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, "/home/lx/snn")

ROOT = Path("/home/lx/snn")
OUT = ROOT / "verification_bundle"
AGG = Path("/data/lx/tmp/results/state_final_corrected_20260916/audit/aggregated_state_cells.json")

HEAD = (ROOT / ".git/HEAD").resolve()
GIT_COMMIT = "see provenance_manifest.csv (recorded at build time)"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"[csv] {path.name}: {len(rows)} rows")


def main() -> int:
    OUT.mkdir(exist_ok=True)
    (OUT / "scripts").mkdir(exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    commit = torch.__version__  # placeholder replaced below
    import subprocess
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()

    agg = json.load(open(AGG))
    evals = agg["evals"]
    proto_cols = ("cem_samples", "cem_elites", "cem_iters", "horizon",
                  "eval_budget", "goal_offset", "history_size")

    # ---------------- E1/E2/E4/E7 per-cell CSVs from the audited aggregate
    exp_file = {"E1": "state_eval_cells.csv", "E2": "e2_seed_paired.csv",
                "E4": "e4_heldout_cells.csv", "E7": "e7_sigreg_cells.csv"}
    header = ["experiment", "id", "split", "model", "env_id", "seed",
              "n_episodes", "n_seeds", *proto_cols,
              "success_rate_env", "success_rate_env_std",
              "mean_cos_dist", "mean_cos_dist_std",
              "success_rate_lewm_005", "success_rate_lewm_001",
              "checkpoint", "output", "command"]
    for exp, fname in exp_file.items():
        rows = []
        for e in evals:
            if e.get("experiment") != exp:
                continue
            m = e.get("metrics", {})
            rows.append([exp, e.get("id"), e.get("split"), e.get("model"),
                         e.get("env_id"), e.get("seed"),
                         m.get("n_episodes"), m.get("n_seeds"),
                         *[m.get(c) for c in proto_cols],
                         m.get("success_rate_env"), m.get("success_rate_env_std"),
                         m.get("mean_cos_dist"), m.get("mean_cos_dist_std"),
                         m.get("success_rate_lewm_005"),
                         m.get("success_rate_lewm_001"),
                         e.get("checkpoint"), e.get("output"), e.get("command")])
        write_csv(OUT / fname, header, rows)

    # ---------------- E8 probe cells (g2_summary.json)
    g2 = json.load(open(ROOT / "results/journal_prep/g2_summary.json"))
    rows = []
    for model, payload in g2.items():
        if model.startswith("_"):
            continue
        rows.append([model, payload.get("n"), payload.get("mean")])
    write_csv(OUT / "e8_probe_cells.csv",
              ["model", "n_envs", "mean_position_r2_raw_unclipped"], rows)

    # ---------------- E9 ablation paired (trace_ablation_final)
    rows = []
    for env_json in sorted(glob_jt()):
        d = json.load(open(env_json))
        env = d["env"]
        res = d.get("results", {})
        drops = d.get("drops_pp", {})
        for mode, payload in res.items():
            sr = payload.get("success_rate_env") if isinstance(payload, dict) else None
            cos = payload.get("mean_cos_dist") if isinstance(payload, dict) else None
            rows.append([env, d.get("model"), mode, sr, cos,
                         drops.get(mode), d.get("protocol", {}).get("horizon"),
                         d.get("protocol", {}).get("n_episodes"),
                         d.get("intervention_definition"),
                         d.get("causal_scope"), env_json])
    write_csv(OUT / "e9_ablation_paired.csv",
              ["env", "model", "mode", "success_rate_env", "mean_cos_dist",
               "env_sr_drop_pp", "horizon", "n_episodes",
               "intervention_definition", "causal_scope", "source_file"], rows)

    # ---------------- c20 per model (mech_0917_b rollout, per model x env)
    roll = json.load(open(ROOT / "results/mech_0917_b/rollout/rollout_summary.json"))
    rows = []
    for model, envs in roll["results"].items():
        for env, payload in envs.items():
            c_curve = payload.get("c", {})
            st20 = payload.get("state", {}).get("20", {})
            rows.append([model, env, payload.get("lambda_lat"),
                         payload.get("lambda_state"), payload.get("lambda_emp"),
                         payload.get("contractive"),
                         c_curve.get("20"),
                         ";".join(f"{h}:{c_curve[h]}" for h in sorted(c_curve, key=int)),
                         st20.get("rmse_true"), st20.get("rmse_pred"),
                         st20.get("r2_true"), st20.get("r2_pred"),
                         payload.get("n_windows"), payload.get("n_windows_test")])
    write_csv(OUT / "c20_per_model.csv",
              ["model", "env", "lambda_lat", "lambda_state", "lambda_emp",
               "contractive", "c_20", "c_curve_h:value", "state20_rmse_true",
               "state20_rmse_pred", "state20_r2_true", "state20_r2_pred",
               "n_windows", "n_windows_test"], rows)

    # ---------------- gain per model (absolute g_obs / g_a at h=20)
    gain = json.load(open(ROOT / "results/mech_0917_b/gain/gain_summary.json"))
    rows = []
    for model, envs in gain.get("results", {}).items():
        for env, payload in envs.items():
            g20 = payload.get("gain_h", {}).get("20", {})
            rows.append([model, env, payload.get("emb_dim"),
                         payload.get("n_contexts"), payload.get("n_dirs"),
                         payload.get("eps_mean"),
                         payload.get("base_pred_norm", {}).get("20"),
                         g20.get("mean"), g20.get("max"), g20.get("median")])
    write_csv(OUT / "gain_per_model.csv",
              ["model", "env", "emb_dim", "n_contexts", "n_dirs", "eps_mean",
               "base_pred_norm_20", "gain20_mean", "gain20_max",
               "gain20_median"], rows)

    # ---------------- M4 response timeseries
    m4 = json.load(open(ROOT / "results/mech_0917/transient/transient_summary.json"))
    rows = []
    for env, payload in m4["envs"].items():
        for model, mp in payload.get("models", {}).items():
            for m, mm in mp.get("per_m", {}).items():
                taus = mm.get("curve_tau_rel", [])
                l2 = mm.get("curve_l2_full", [])
                co = mm.get("curve_cos_full", [])
                for tau, dl2, dco in zip(taus, l2, co):
                    rows.append([env, model, m, tau, dl2, dco,
                                 mm.get("d_peak_l2_mean"),
                                 mm.get("residual_l2_50_mean"),
                                 mm.get("frac_not_recovered"),
                                 mm.get("persistent", {}).get("d_l2_at_50_mean")])
    write_csv(OUT / "m4_response_timeseries.csv",
              ["env", "model", "m", "tau_rel", "d_l2", "d_cos",
               "d_peak_l2_mean", "residual_l2_50_mean", "frac_not_recovered",
               "persistent_d_l2_at_50_mean"], rows)

    # ---------------- params_all_models.csv (+ runtime encoder-usage check)
    sys.path.insert(0, "/home/lx/snn")
    from code.scripts.event_align import build_model

    seen = {}
    for e in evals:
        if e.get("experiment") != "E1":
            continue
        mk_, ck_ = e["model"], e.get("checkpoint")
        if mk_ not in seen and ck_:
            seen[mk_] = ck_
    rows = []
    log_lines = []
    for model, ckpt in sorted(seen.items()):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = ck["model"]
        groups = defaultdict(int)
        for k, v in sd.items():
            groups[k.split(".")[0]] += v.numel()
        total = sum(groups.values())
        frozen = sum(n for g_, n in groups.items() if g_ == "encoder")
        trainable = total - frozen
        encoder_called = ""
        if any(g_ == "encoder" for g_ in groups):
            try:
                hint = model if model.startswith(("stjewm", "alif", "stacked")) else model
                m = build_model(hint, 128, 56, ck_args=ck["args"],
                                state_dict=sd)
                m.load_state_dict(sd, strict=True)
                m.eval()
                calls = {"n": 0}
                if hasattr(m, "encoder"):
                    m.encoder.register_forward_hook(
                        lambda mod, i, o: calls.__setitem__("n", calls["n"] + 1))
                with torch.no_grad():
                    s = torch.zeros(1, 1, 128)
                    a = torch.zeros(1, 1, 56)
                    m(s, a)
                    m.predict(torch.zeros(1, 27, 192), torch.zeros(1, 27, 56))
                encoder_called = calls["n"]
                log_lines.append(
                    f"{model}: encoder forward calls during state encode+predict = {calls['n']}")
            except Exception as exc:  # noqa: BLE001
                encoder_called = f"check_failed: {exc}"
                log_lines.append(f"{model}: encoder check failed: {exc}")
        else:
            encoder_called = 0
            log_lines.append(f"{model}: no encoder module (baseline)")
        rows.append([model, f"{total}", f"{frozen}", f"{trainable}",
                     "; ".join(f"{g_}:{n}" for g_, n in
                               sorted(groups.items(), key=lambda x: -x[1])),
                     encoder_called, ckpt])
        log_lines.append(
            f"{model}: total={total} frozen_encoder={frozen} "
            f"trainable={trainable} groups={dict(groups)}")
    write_csv(OUT / "params_all_models.csv",
              ["model", "total_params", "frozen_encoder_params",
               "trainable_params", "per_group", "encoder_forward_calls_state_mode",
               "checkpoint"], rows)
    (OUT / "logs/param_count.log").write_text("\n".join(log_lines) + "\n")

    # ---------------- provenance manifest
    rows = []
    for f in sorted(OUT.rglob("*")):
        if f.is_file() and f.name != "provenance_manifest.csv":
            rows.append([str(f.relative_to(OUT)), f.stat().st_size, sha256(f),
                         commit, datetime.now(timezone.utc).isoformat()])
    for src in [AGG, ROOT / "results/journal_prep/EXP_Report.md",
                ROOT / "results/mech_0917_b/rollout/rollout_summary.json",
                ROOT / "results/mech_0917_b/gain/gain_summary.json",
                ROOT / "results/mech_0917_b/scaling/scaling_summary.json",
                ROOT / "results/mech_0917/transient/transient_summary.json",
                ROOT / "results/journal_prep/g2_summary.json",
                Path("/data/lx/tmp/results/agg_final/g5_multiseed.json")]:
        if src.exists():
            rows.append([f"EXTERNAL::{src}", src.stat().st_size, sha256(src),
                         commit, datetime.now(timezone.utc).isoformat()])
    write_csv(OUT / "provenance_manifest.csv",
              ["file", "bytes", "sha256", "repo_commit", "built_utc"], rows)

    # ---------------- copy this script into the bundle
    (OUT / "scripts/build_verification_bundle.py").write_text(
        Path(__file__).read_text())
    print("[bundle] done ->", OUT)
    return 0


def glob_jt():
    import glob
    return sorted(glob.glob("/data/lx/tmp/results/trace_ablation_final/*.json"))


if __name__ == "__main__":
    sys.exit(main())
