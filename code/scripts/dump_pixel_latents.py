"""Dump representative pixel trajectories using the repaired diagnostic protocol."""
import argparse
import json
from pathlib import Path

from code.scripts.measure_pixel_stats import measure_pixel

MODELS = ["stjewm", "lewm_baseline", "gru_baseline", "mlp_baseline"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="cross_benchmark_F1")
    p.add_argument("--envs", nargs="+", default=["cartpole", "cheetah"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--results", type=Path, default=Path("/data/lx/tmp/results/5m_pixel"))
    p.add_argument("--out", type=Path, default=Path("/data/lx/tmp/results/pixel_latent_traj"))
    args = p.parse_args()
    for model in MODELS:
        ckpt = args.results / args.split / model / "seed_0" / "final.pt"
        for env in args.envs:
            path = args.out / args.split / model / f"{env}_latents.npz"
            if path.exists():
                raise RuntimeError(f"Archive old trajectory before rerunning: {path}")
            result = measure_pixel(str(ckpt), env, 84, 200, args.device, out_npz=path)
            path.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False))
            print(f"[dump] {model}/{env}: div={result['divergence']:.6g}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
