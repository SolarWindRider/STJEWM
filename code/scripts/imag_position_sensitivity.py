"""Diagnostic: how much can CEM's imagination see an action at each window
position?

Tests the hypothesis that the imagination action-window structure
(candidates at positions 0..4, zero-pad to 27, readout at position 26) makes
planning effectively blind / "1-step". For each position p, predict with a
single ±1.0 action at p (zeros elsewhere) and measure the relative response
of the predicted final latent vs the all-zero-action baseline.

Run: python -m code.scripts.imag_position_sensitivity --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/lx/snn")
if "code" in sys.modules and not hasattr(sys.modules["code"], "__path__"):
    del sys.modules["code"]

from code.scripts.event_align import build_model
from code.core.encode import encode_obs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="/home/lx/snn/results/actionctx_ablation/"
                                     "imag_position_sensitivity.json")
    args = ap.parse_args()

    CKPTS = {
        "trace": ("/data/lx/tmp/results/5m_5mpar/oodc_F1/stjewm_trace_only/seed_0/final.pt",
                  "stjewm_trace_only"),
        "lewm": ("/data/lx/tmp/results/5m/oodc_F1/lewm_baseline_v2/seed_0/final.pt",
                 "lewm_baseline_v2"),
    }
    data = np.load("/home/lx/snn/data/dm_control/cartpole_250k.npz")
    obs = np.asarray(data["observations"], dtype=np.float32)[:200]
    acts = np.asarray(data["actions"], dtype=np.float32)[:200]
    obs = np.stack([np.pad(o.reshape(-1), (0, 128 - o.reshape(-1).shape[-1]))
                    for o in obs])  # eval protocol (model_state convention)
    pos_list = [0, 2, 4, 8, 12, 16, 20, 24, 25, 26]

    results = {}
    for key, (ckpt, hint) in CKPTS.items():
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = build_model(hint, 128, 56, ck_args=ck["args"],
                            state_dict=ck["model"])
        model.load_state_dict(ck["model"], strict=True)
        model = model.to(args.device).eval()
        dev = args.device
        with torch.no_grad():
            ctx = torch.stack([
                encode_obs(model, torch.from_numpy(o), 56, dev)
                for o in obs[:27]
            ]).unsqueeze(0).to(dev)  # (1, 27, D)
            z0 = model.predict(ctx, torch.zeros(1, 27, 56, device=dev))[0, -1]
            rows = []
            mag = float(np.clip(np.abs(acts[0]).mean(), 0.5, 1.0))
            for p in pos_list:
                resp = []
                for sign in (+1.0, -1.0):
                    a = torch.zeros(1, 27, 56, device=dev)
                    a[0, p] = sign * mag
                    z = model.predict(ctx, a)[0, -1]
                    resp.append(float(
                        (z - z0).norm() / z0.norm().clamp_min(1e-8)))
                rows.append({"pos": p,
                             "rel_resp_+": resp[0], "rel_resp_-": resp[1]})
        results[key] = {
            "baseline_norm": float(z0.norm()),
            "per_position": rows,
            "action_used": mag,
        }
        print(f"[{key}] predict last-step relative response to a single "
              f"action at position p:")
        for r in rows:
            print(f"  pos {r['pos']:2d}: +{r['rel_resp_+']:.5f}  "
                  f"-{r['rel_resp_-']:.5f}")
        del model
        torch.cuda.empty_cache()

    from pathlib import Path
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print("saved", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
