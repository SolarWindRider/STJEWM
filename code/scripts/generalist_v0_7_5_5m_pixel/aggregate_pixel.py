#!/usr/bin/env python3
"""Aggregate v0.7.15 pixel ckpts into a single per-(model, split) table."""
import json
from pathlib import Path

ROOT = Path("/home/lx/snn")
BASE = ROOT / "results" / "5m_pixel"
OUT = ROOT / "results" / "aggregate" / "generalist_5m_pixel_table.md"
OUT.parent.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("stjewm_trace_only", "STJEWM-trace"),
    ("stjewm_hidden_leak", "STJEWM-leak"),
    ("stjewm_spike_only", "STJEWM-spike"),
    ("stjewm_rate_only", "STJEWM-rate"),
    ("stjewm_no_trace", "STJEWM-no-trace"),
    ("stjewm_membrane_readout", "STJEWM-membrane"),
    ("cubifae_baseline", "CubifAE"),
    ("slt_lif_mpc_trace", "SLT-trace"),
    ("slt_lif_mpc_free", "SLT-free"),
    ("gru_baseline", "GRU"),
    ("lewm_baseline_v2", "LeWM-v2"),
    ("spikedreamer_baseline", "SpikeDreamer"),
    ("mlp_baseline", "MLP"),
]

SPLITS = [
    "cross_benchmark_F1", "cross_benchmark_F2", "cross_benchmark_F3",
    "oodc_F1", "oodc_F1F2", "oodc_F1F3", "oodc_F2", "oodc_F2F3", "oodc_F3",
    "generalist_16env",
]


def collect_env_sr_lewm(split, model, seed=0):
    ckpt_dir = BASE / split / model / f"seed_{seed}"
    if not ckpt_dir.exists():
        return None
    eval_files = list(ckpt_dir.glob("eval_*.json"))
    if not eval_files:
        return None
    env_srs = []
    lewm_srs = []
    for f in eval_files:
        try:
            d = json.load(open(f))
            if "success_rate_env" in d:
                env_srs.append(d["success_rate_env"])
            if "success_rate_lewm_005" in d:
                lewm_srs.append(d["success_rate_lewm_005"])
        except Exception:
            pass
    if not env_srs and not lewm_srs:
        return None
    return {
        "env_sr": sum(env_srs) / len(env_srs) if env_srs else None,
        "lewm_sr": sum(lewm_srs) / len(lewm_srs) if lewm_srs else None,
        "n_envs": len(env_srs),
    }


def main():
    lines = []
    lines.append("# v0.7.15 - 5M-aligned Pixel Re-Training Table")
    lines.append("")
    lines.append("All ckpts trained with **frozen ViT-Tiny pixel encoder** (5.5M frozen)")
    lines.append("replacing the state_projector. **Trainable params: 4.97-5.13M (5M-aligned)**.")
    lines.append("")
    lines.append("Setup: image_size 84 (faster than 224, same architecture).")
    lines.append("Other settings: 1 epoch, batch 32, AdamW lr=3e-4, 1 seed.")
    lines.append("")
    lines.append("**Status: in progress (v0.7.15, 2026-07-31).**")
    lines.append("")
    lines.append("## Per-(model, split) env-SR / LeWM-SR")
    lines.append("")
    lines.append("| Model | " + " | ".join(SPLITS) + " |")
    lines.append("|" + "---|" * (len(SPLITS) + 1))

    for model_code, model_name in MODELS:
        row = "| " + model_name + " |"
        for split in SPLITS:
            stats = collect_env_sr_lewm(split, model_code)
            if stats is None:
                row += " - |"
            else:
                env = f"{stats['env_sr']:.2f}" if stats['env_sr'] is not None else "-"
                lewm = f"{stats['lewm_sr']:.2f}" if stats['lewm_sr'] is not None else "-"
                row += f" {env} / {lewm} |"
        lines.append(row)
    lines.append("")
    lines.append("**Cross-modality comparison (state vs pixel):** see cross_modality_table.md.")
    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
