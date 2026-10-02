"""Rerun the matched E6/E12 representation grid without mixing interfaces."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from code.eval.closed_loop import make_env
from code.scripts.event_align import ENV_KIND_MAP, build_model
from code.scripts.generalist_v0_7_5_5m.measure_latent_stats_5m import MODELS
from code.scripts.latent_rollout import PROTOCOL_VERSION, collect_state_rollout


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path("/data/lx/tmp/results"))
    p.add_argument("--splits", nargs="+", default=["cross_benchmark_F1", "oodc_F2"])
    p.add_argument("--envs", nargs="+", default=["cartpole_2d", "cheetah"])
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--include-sigreg", action="store_true")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    torch.set_num_threads(1)
    entries = [(model, "5m_5mpar" if model.startswith("stjewm") else "5m")
               for model in args.models]
    if args.include_sigreg:
        entries += [(f"stjewm_trace_only_sig{weight}", "5m_sigreg_sweep")
                    for weight in ["0.09", "0.01", "0.001", "0.0"]]
    completed = 0
    for split in args.splits:
        for model_name, family in entries:
            checkpoint = args.results / family / split / model_name / "seed_0/final.pt"
            ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
            saved = ck.get("args", {})
            state_dim = saved.get("pad_obs_to") or 128
            action_dim = saved.get("action_dim") or 56
            model = build_model(model_name, state_dim, action_dim, saved, state_dict=ck["model"])
            model.load_state_dict(ck["model"], strict=True)
            model.to(args.device).eval()
            for env_name in args.envs:
                path = args.results / "latent_rollout" / split / model_name / f"{env_name}.json"
                if path.exists():
                    old = json.loads(path.read_text())
                    if old.get("protocol_version") != PROTOCOL_VERSION:
                        raise RuntimeError(f"Archive obsolete output first: {path}")
                    if not path.with_suffix(".npz").exists():
                        raise RuntimeError(f"Missing trajectory for {path}")
                    continue
                env = make_env(ENV_KIND_MAP[env_name], data_path=None)
                try:
                    result, arrays = collect_state_rollout(
                        model, env, state_dim, action_dim, n_steps=200,
                        n_resets=2, seed=0, device=args.device,
                    )
                finally:
                    env.close()
                result.update(env=env_name, model=model_name, split=split,
                              ckpt=str(checkpoint), weights_loaded_strict=True)
                payload = json.dumps(result, indent=2, allow_nan=False)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(payload)
                np.savez(path.with_suffix(".npz"), **arrays)
                if family != "5m_sigreg_sweep":
                    g1 = args.results / "g1" / family / model_name / f"{env_name}_{split}.json"
                    g1.parent.mkdir(parents=True, exist_ok=True)
                    g1.write_text(payload)
                completed += 1
                print(f"[rollout {completed}] {split}/{model_name}/{env_name} "
                      f"readout_div={result['divergence']:.6g}", flush=True)
    print(f"Completed {completed} matched representation rollouts", flush=True)


if __name__ == "__main__":
    main()
