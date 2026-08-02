#!/usr/bin/env python
"""Aggregate completed B4 per-cell JSONs into the decisive journal-prep summary."""
import json
from pathlib import Path

BASE = Path("/home/lx/snn/results/journal_prep/B4_ablation")
OLD = Path("/home/lx/snn/results/aggregate/event_window_ablation")
cells = [json.loads(p.read_text()) for p in sorted(BASE.glob("*.json")) if p.name != "smoke.json"]

lines = [
    "# B4 — Event-window and CEM-rollout trace ablation",
    "",
    "## Protocol",
    "",
    "Each cell uses a 99-step random-policy probe. Event steps satisfy `||Δobs|| > median + 1×MAD`; event windows are ±2 steps around at most 12 peaks. Matched low-change and random windows contain the same number of steps. Evaluation uses 300 CEM samples, 30 elites, 10 iterations, horizon 5, budget 50, history size 1, and 10 episodes × 1 seed.",
    "",
    "The restored five modes ablate trace only in the post-step history-update `predict` call. The sixth mode, `cem_rollout_ablation`, zeros trace in every internal CEM candidate-rollout `model.predict` call.",
    "",
    "## Per-cell results",
    "",
    "| modality | env | mode | env-SR | env-SR drop | mean cos dist | cos increase |",
    "|---|---|---|---:|---:|---:|---:|",
]
for cell in cells:
    base = cell["results"]["baseline"]
    for name, result in cell["results"].items():
        if name == "baseline":
            env_drop = 0.0
            cos_inc = 0.0
        else:
            env_drop = cell["drops"][name]["env_sr_drop_pp"]
            cos_inc = cell["drops"][name]["cos_dist_increase"]
        lines.append(f"| {cell['modality']} | {cell['env']} | {name} | {result['success_rate_env']:.3f} | {env_drop:+.1f} pp | {result['mean_cos_dist']:.6f} | {cos_inc:+.6f} |")

lines += ["", "## Old versus 5M-aligned verdict", ""]
old_trace = json.loads((OLD / "cartpole_2d_stjewm_trace_only.json").read_text())
new_cart = next(c for c in cells if c["env"] == "cartpole_2d" and c["modality"] == "state")
lines += [
    f"The historical cartpole/trace-only checkpoint had baseline env-SR {old_trace['results']['baseline']['success_rate_env']:.1f}; its event-window drop was {old_trace['drops']['event_window']['env_sr_drop_pp']:+.1f} pp and the old report rejected event-specific causal use. The 5M-aligned generalist checkpoint has baseline env-SR {new_cart['results']['baseline']['success_rate_env']:.1f}. This is qualitatively worse rather than a reproduction of the old 0.5 success level, so the baseline sanity check does **not** reproduce the old checkpoint's absolute behavior; checkpoint replacement materially changed planning performance.",
    "",
    "The 5M state cells have zero env-SR in all modes, so only the continuous latent metric discriminates them. Event-window ablation does not uniquely hurt: cartpole event-window cos increase is zero, and cheetah event-window slightly improves cos distance. Thus the old negative event-specific verdict remains unchanged.",
    "",
    "## Decisive CEM-rollout comparison",
    "",
]

comparisons = []
for c in cells:
    history = c["drops"]["ablate_all"]
    cem = c["drops"]["cem_rollout_ablation"]
    comparisons.append((c, history, cem))
    lines.append(f"- **{c['modality']} / {c['env']}**: history-path `ablate_all` drop = {history['env_sr_drop_pp']:+.1f} pp, CEM-rollout drop = {cem['env_sr_drop_pp']:+.1f} pp; cos increases {history['cos_dist_increase']:+.6f} vs {cem['cos_dist_increase']:+.6f}.")

strict_env_bigger = sum(cem["env_sr_drop_pp"] > hist["env_sr_drop_pp"] for _, hist, cem in comparisons)
lines += [
    "",
    f"**Verdict: NO.** CEM-internal trace ablation does not show a bigger env-SR drop than all-step history-path ablation in any of the {len(comparisons)} required cells ({strict_env_bigger}/{len(comparisons)}). On the pixel cheetah cell both reduce env-SR from 0.3 to 0.2 (10 pp), while on both state cells env-SR stays at zero. The continuous cos metric is mixed: CEM ablation is slightly worse than history ablation for cartpole state, but improves cheetah state and is smaller for cheetah pixel. Therefore this experiment finds **no consistent causal evidence that the planner relies more strongly on trace inside CEM rollouts**.",
    "",
    "Because there is one seed and ten episodes, a 10 pp pixel difference is one episode and should not be treated as statistically resolved.",
    "",
    "## Artifacts",
    "",
]
for c in cells:
    lines.append(f"- `{c['env']}_{c['model']}_{c['modality']}.json` — `{c['ckpt']}`")

(BASE / "summary.md").write_text("\n".join(lines) + "\n")
print(BASE / "summary.md")
