"""Re-aggregate ood1_table.md from results/oodc/<split>/<split>/<model>/seed_0/*.json."""
import sys, json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, "/home/lx/snn")


def is_probe_file(stem: str) -> bool:
    return (stem.endswith("_position") or stem.endswith("_velocity")
            or stem.endswith("_future_k") or stem == "loss_log"
            or stem.startswith("eval_"))


def collect_cells(root: Path):
    cells = []
    for split_dir in sorted(root.iterdir()):
        if not split_dir.is_dir(): continue
        nested = split_dir / split_dir.name
        if not nested.exists() or not nested.is_dir(): continue
        for model_dir in sorted(nested.iterdir()):
            if not model_dir.is_dir(): continue
            for seed_dir in sorted(model_dir.iterdir()):
                if not seed_dir.is_dir(): continue
                for env_json in sorted(seed_dir.glob("*.json")):
                    if is_probe_file(env_json.stem): continue
                    try:
                        d = json.loads(env_json.read_text())
                    except: continue
                    cells.append({
                        "split": split_dir.name,
                        "model": model_dir.name,
                        "env_id": env_json.stem,
                        "div": d.get("divergence", float("nan")),
                        "resp": d.get("responsiveness", float("nan")),
                        "rho": d.get("rho", float("nan")),
                        "env_sr": d.get("env_sr", float("nan")),
                    })
    return cells


def f1(x):
    if isinstance(x, float):
        if x != x: return "nan"
        return f"{x:.4f}"
    return str(x) if x is not None else "nan"


def f2(x):
    if isinstance(x, float):
        if x != x: return "nan"
        return f"{x:.3f}"
    return str(x) if x is not None else "nan"


def avg(lst):
    if not lst: return "nan"
    return f"{sum(lst)/len(lst):.4f}"


def main():
    out_root = Path("results/oodc")
    cells = collect_cells(out_root)
    print(f"Total cells: {len(cells)}")

    out_md = Path("results/utility/ood1_table.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_md.open("w") as f:
        f.write("# OOD Path-C: 3-family DMC cross-sub-family transfer (v0.7.10b)\n\n")
        f.write("3 DMC sub-families: F1 classic control, F2 locomotion, F3 sparse-POMDP.\n")
        f.write("6 splits (3 OOD1: F1, F2, F3 trained; 3 OOD2: F1F2, F1F3, F2F3 trained).\n")
        f.write("12 ckpts per split, 1 seed, 2K windows/env, 3 episodes per held-out env.\n\n")
        f.write("Per-cell metric: `div` (latent per-dim std), `resp` (mean |delta-lat|/|delta-obs|), ")
        f.write("`rho` (corr ||delta-obs|| vs ||delta-lat||), `env_sr` (closed-loop success rate).\n\n")
        f.write(f"Total: {len(cells)} ckpt x env cells.\n\n")
        f.write("## Per-cell\n\n")
        f.write("| split | model | env | div | resp | rho | env_sr |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for c in cells:
            f.write(f"| {c['split']} | {c['model']} | {c['env_id']} | "
                    f"{f1(c['div'])} | {f2(c['resp'])} | {f2(c['rho'])} | {f2(c['env_sr'])} |\n")
        agg = defaultdict(list)
        for c in cells:
            agg[(c["split"], c["model"])].append((c["div"], c["resp"], c["rho"], c["env_sr"]))
        f.write("\n## Mean per (split, model)\n\n")
        f.write("| split | model | n_envs | mean_div | mean_resp | mean_rho | mean_env_sr |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for k, vs in sorted(agg.items()):
            ds = [v[0] for v in vs if isinstance(v[0], float) and v[0] == v[0]]
            rs = [v[1] for v in vs if isinstance(v[1], float) and v[1] == v[1]]
            hos = [v[2] for v in vs if isinstance(v[2], float) and v[2] == v[2]]
            srs = [v[3] for v in vs if isinstance(v[3], float) and v[3] == v[3]]
            f.write(f"| {k[0]} | {k[1]} | {len(vs)} | {avg(ds)} | {avg(rs)} | {avg(hos)} | {avg(srs)} |\n")
        family_assign = {
            "stjewm_trace_only": "STJEWM",
            "stjewm_spike_only": "STJEWM",
            "stjewm_rate_only": "STJEWM",
            "stjewm_no_trace": "STJEWM",
            "stjewm_hidden_leak": "STJEWM",
            "stjewm_membrane_readout": "STJEWM",
            "alif_timecell_baseline": "SNN-baselines",
            "stacked_lif_trace": "SNN-baselines",
            "stacked_lif_free": "SNN-baselines",
            "mlp_baseline": "non-SNN",
            "gru_baseline": "non-SNN",
            "lewm_baseline_v2": "non-SNN",
        }
        fam_agg = defaultdict(list)
        for c in cells:
            fam = family_assign.get(c["model"], "?")
            fam_agg[(c["split"], fam)].append((c["div"], c["resp"], c["rho"], c["env_sr"]))
        f.write("\n## Per-split, per-family mean\n\n")
        f.write("STJEWM = trace, spike, rate, no_trace, hidden_leak, membrane_readout.\n")
        f.write("SNN-baselines = alif_timecell, stacked_lif_trace, stacked_lif_free.\n")
        f.write("non-SNN baselines = mlp, gru, lewm.\n\n")
        f.write("| split | family | n_cells | mean_div | mean_resp | mean_rho | mean_env_sr |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for k, vs in sorted(fam_agg.items()):
            ds = [v[0] for v in vs if isinstance(v[0], float) and v[0] == v[0]]
            rs = [v[1] for v in vs if isinstance(v[1], float) and v[1] == v[1]]
            hos = [v[2] for v in vs if isinstance(v[2], float) and v[2] == v[2]]
            srs = [v[3] for v in vs if isinstance(v[3], float) and v[3] == v[3]]
            f.write(f"| {k[0]} | {k[1]} | {len(vs)} | {avg(ds)} | {avg(rs)} | {avg(hos)} | {avg(srs)} |\n")
    print(f"Wrote {out_md} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
