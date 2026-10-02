#!/usr/bin/env python
"""R1 — trace_proj repair: readout-only refit, same JEPA objective.

M3/S1 found the deployed readout emb = trace_proj(trace) (nn.Linear(192,192,
bias=False)) ill-conditioned: linear decode of qpos from emb is far worse than
from the raw gated trace, so S1's "stable AND informative" claim fails at the
readout. This script tests the cheapest clean repair: refit ONLY trace_proj
with the exact 3-term JEPA objective (pred + 0.09*sigreg + 0.5*goal), the exact
training data windows (load_dmc, window=27, max_windows=2000/env, action padded
to 56) and optimizer family (AdamW lr 3e-4, wd 1e-3), everything else frozen.
If the pathology was an optimization artifact of the 1-epoch joint training,
converging the readout alone recovers it; if not, it is intrinsic to the
objective and we report that.

Output (never overwrites): /data/lx/tmp/results/traceproj_repair_20260917/
  oodc_F1/stjewm_trace_only/seed_0/final.pt   (repaired checkpoint, args annotated)
  oodc_F1/stjewm_trace_only/seed_0/repair_receipt.json

Run: python -m code.scripts.repair_traceproj --device cuda:0
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/lx/snn")

from code.data.loaders import load_dmc  # noqa: E402
from code.native_losses import stjewm_loss  # noqa: E402
from code.sigreg import SIGReg  # noqa: E402
from code.scripts.event_align import build_model  # noqa: E402

BASE_CKPT = "/data/lx/tmp/results/5m_5mpar/oodc_F1/stjewm_trace_only/seed_0/final.pt"
CONFIG = "/home/lx/snn/configs/oodc_5m/oodc_F1.json"
OUT_DIR = Path("/data/lx/tmp/results/traceproj_repair_20260917/oodc_F1/stjewm_trace_only/seed_0")
H, GOAL_OFFSET, PAD_OBS, PAD_ACT = 1, 25, 128, 56


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def loss_batch(model, sigreg, state, action, lambda_sigreg=0.09, lambda_goal=0.5):
    """Exact train.py 3-term JEPA composition (H=1, T_pred=1, goal_offset=25)."""
    out = model(state, action)
    emb, emb_pre = out["emb"], out["emb_pre_cell"]
    pred_emb = model.predict(emb[:, :H], action[:, :H])
    tgt_emb = emb[:, H:H + 1]
    sigreg_loss = sigreg(emb_pre.transpose(0, 1))
    with torch.no_grad():
        goal_state = state[:, H + GOAL_OFFSET:H + GOAL_OFFSET + 1]
        zero_act = torch.zeros(goal_state.shape[0], 1, action.shape[-1],
                               device=state.device, dtype=action.dtype)
        goal_emb_target = model(goal_state, zero_act)["emb"][:, 0]
    out_full = model(state[:, :H + GOAL_OFFSET], action[:, :H + GOAL_OFFSET])
    goal_pred = out_full["emb"][:, -1]
    return stjewm_loss(pred_emb, tgt_emb, emb_pre, sigreg,
                       goal_pred, goal_emb_target,
                       lambda_sigreg, lambda_goal)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-frac", type=int, choices=[9], help="unused; val = every 10th window")
    args = p.parse_args()
    device = args.device

    torch.manual_seed(0)
    np_rng = np.random.default_rng(0)

    ck = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    model = build_model("stjewm_trace_only", state_dim=PAD_OBS, action_dim=PAD_ACT,
                        ck_args=ck["args"], state_dict=ck["model"])
    model.load_state_dict(ck["model"], strict=True)
    model.to(device)
    model.train()

    # ---- data: exact training windows (oodc_F1 train envs)
    specs = json.loads(Path(CONFIG).read_text())["specs"]
    dsets = [load_dmc(s["path"], history_size=1, goal_offset=s["goal_offset"],
                      max_windows=s["max_windows"], pad_obs_to=PAD_OBS,
                      env_id=s["env_id"]) for s in specs]
    lengths = [len(d) for d in dsets]
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    all_idx = np.arange(int(offsets[-1]))
    val_mask = (all_idx % 10) == 9
    train_idx = all_idx[~val_mask]
    val_idx = all_idx[val_mask]

    def get_window(global_idx: int):
        d = dsets[int(np.searchsorted(offsets, global_idx, side="right") - 1)]
        item = d[int(global_idx - offsets[int(np.searchsorted(offsets, global_idx, side="right") - 1)])]
        a = item["action"]
        if a.shape[-1] < PAD_ACT:  # multi-env factory convention: native dims + zero pad
            a = torch.cat([a, torch.zeros(*a.shape[:-1], PAD_ACT - a.shape[-1])], dim=-1)
        return item["state"], a

    def run_val(tag: str):
        model.eval()
        tot = n = 0
        with torch.inference_mode():
            for i in range(0, len(val_idx), 64):
                idx = val_idx[i:i + 64]
                s = torch.stack([get_window(j)[0] for j in idx]).to(device)
                a = torch.stack([get_window(j)[1] for j in idx]).to(device)
                _, parts = loss_batch(model, sigreg, s, a)
                tot += float(parts["total"]) * len(idx)
                n += len(idx)
        model.train()
        val = tot / max(n, 1)
        print(f"[repair] val {tag}: total={val:.6f}", flush=True)
        return val

    sigreg = SIGReg(knots=17, num_proj=1024).to(device)
    W = model.trace_proj.weight
    for p_ in model.parameters():
        p_.requires_grad_(False)
    W.requires_grad_(True)
    _tr = {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}
    print(f"[repair] trainable params: {_tr} (sum={sum(_tr.values())}, W.numel={W.numel()})",
          flush=True)
    assert sum(p_.numel() for p_ in model.parameters() if p_.requires_grad) == W.numel()

    sv0 = torch.linalg.svdvals(W.detach().float().cpu())
    opt = torch.optim.AdamW([W], lr=args.lr, weight_decay=1e-3)

    val0 = run_val("before")
    print(f"[repair] W before: cond={float(sv0[0] / max(sv0[-1], 1e-12)):.3g} "
          f"sv_max={float(sv0[0]):.4g} sv_min={float(sv0[-1]):.4g}", flush=True)

    curve = []
    for step in range(1, args.steps + 1):
        idx = train_idx[np_rng.choice(len(train_idx), size=args.batch, replace=False)]
        s = torch.stack([get_window(j)[0] for j in idx]).to(device)
        a = torch.stack([get_window(j)[1] for j in idx]).to(device)
        opt.zero_grad()
        loss, parts = loss_batch(model, sigreg, s, a)
        loss.backward()
        opt.step()
        if step % 250 == 0 or step == 1:
            curve.append({"step": step, "train_total": float(loss.item())})
            print(f"[repair] step {step}: total={float(loss.item()):.6f}", flush=True)

    val1 = run_val("after")
    sv1 = torch.linalg.svdvals(W.detach().float().cpu())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_ckpt = OUT_DIR / "final.pt"
    new_args = dict(ck["args"])
    new_args["traceproj_repair"] = {
        "base_checkpoint": BASE_CKPT, "base_sha256": sha256(BASE_CKPT),
        "steps": args.steps, "batch": args.batch, "lr": args.lr,
        "frozen": "all params except trace_proj.weight",
        "objective": "identical 3-term JEPA (pred + 0.09 sigreg + 0.5 goal), oodc_F1 train windows",
        "val_total_before": val0, "val_total_after": val1,
    }
    torch.save({"model": model.state_dict(), "args": new_args}, out_ckpt)
    receipt = {
        "base_checkpoint": BASE_CKPT, "base_sha256": sha256(BASE_CKPT),
        "out_checkpoint": str(out_ckpt), "out_sha256": sha256(str(out_ckpt)),
        "steps": args.steps, "loss_curve": curve,
        "val_total_before": val0, "val_total_after": val1,
        "W_singular_values_before": [float(x) for x in sv0[:5]] + [float(sv0[-1])],
        "W_singular_values_after": [float(x) for x in sv1[:5]] + [float(sv1[-1])],
        "W_cond_before": float(sv0[0] / max(sv0[-1], 1e-12)),
        "W_cond_after": float(sv1[0] / max(sv1[-1], 1e-12)),
    }
    (OUT_DIR / "repair_receipt.json").write_text(json.dumps(receipt, indent=2))
    print(f"[repair] wrote {out_ckpt}; val {val0:.6f} -> {val1:.6f}; "
          f"cond {receipt['W_cond_before']:.3g} -> {receipt['W_cond_after']:.3g}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
