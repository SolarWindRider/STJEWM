#!/usr/bin/env python
"""R2 — controlled accessibility training: L = L_JEPA + lambda * L_decode.

Hypothesis under test: ST-JEWM's contractive-but-ill-conditioned readout is not
a necessity — a small auxiliary state-accessibility loss (linear decoder from
emb to the standardized current state) can restore decodability while keeping
the contractive dynamics, giving a dose-response over lambda in
{0.1, 1.0, 10.0}. lambda = 0 is the existing final checkpoint (identical
objective without the decode term).

Protocol parity with the original training: oodc_F1 train envs, load_dmc
windows (27 frames, max_windows 2000/env, action padded to 56), batch 32,
AdamW lr 3e-4 wd 1e-3, exactly 1 epoch = len(windows)//32 steps (same budget
as the deployed generation), seed 0. The decoder (Linear 192->128) predicts the
standardized padded state at every window position (causality holds).

Saved artifacts (never overwrite): /data/lx/tmp/results/accessibility_20260917/
  oodc_F1/stjewm_trace_only_lam<l>/seed_0/final.pt     # model state_dict WITHOUT decoder keys (drop-in for eval)
  oodc_F1/stjewm_trace_only_lam<l>/seed_0/decoder.pt   # decoder weights + target standardization
  oodc_F1/stjewm_trace_only_lam<l>/seed_0/receipt.json # losses, decode R2, provenance

Run: python -m code.scripts.train_accessibility --lambda-decode 1.0 --device cuda:0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/lx/snn")

from code.data.loaders import load_dmc  # noqa: E402
from code.sigreg import SIGReg  # noqa: E402
from code.scripts.event_align import build_model  # noqa: E402
from code.native_losses import stjewm_loss  # noqa: E402

BASE_CKPT = "/data/lx/tmp/results/5m_5mpar/oodc_F1/stjewm_trace_only/seed_0/final.pt"
CONFIG = "/home/lx/snn/configs/oodc_5m/oodc_F1.json"
OUT_ROOT = Path("/data/lx/tmp/results/accessibility_20260917")
H, GOAL_OFFSET, PAD_OBS, PAD_ACT = 1, 25, 128, 56


def build_data():
    specs = json.loads(Path(CONFIG).read_text())["specs"]
    dsets, stds = [], []
    for s in specs:
        dsets.append(load_dmc(s["path"], history_size=1, goal_offset=s["goal_offset"],
                              max_windows=s["max_windows"], pad_obs_to=PAD_OBS,
                              env_id=s["env_id"]))
        d = np.load(s["path"])
        obs = d["observations"][:, 0, :].astype(np.float64)
        stds.append(obs.std(axis=0))
    std = np.zeros(PAD_OBS, dtype=np.float32)
    for i, s in enumerate(stds):
        std[: len(s)] = np.maximum(std[: len(s)], s.astype(np.float32))
    std = np.maximum(std, 1e-6)
    return dsets, torch.from_numpy(std)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--lambda-decode", type=float, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    a = p.parse_args()
    device = a.device
    tag = ("lam%s" % a.lambda_decode).replace(".", "")

    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    ck = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    model = build_model("stjewm_trace_only", state_dim=PAD_OBS, action_dim=PAD_ACT,
                        ck_args=ck["args"], state_dict=ck["model"])
    model.load_state_dict(ck["model"], strict=True)
    model.to(device)
    model.train()
    decoder = nn.Linear(192, PAD_OBS).to(device)
    sigreg = SIGReg(knots=17, num_proj=1024).to(device)

    dsets, state_std = build_data()
    state_std = state_std.to(device)
    lengths = [len(d) for d in dsets]
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    n_total = int(offsets[-1])
    steps_per_epoch = n_total // a.batch
    total_steps = steps_per_epoch * a.epochs

    params = [p_ for p_ in model.parameters()] + list(decoder.parameters())
    for p_ in params:
        p_.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-3)

    def get_window(gi: int):
        di = int(np.searchsorted(offsets, gi, side="right") - 1)
        item = dsets[di][int(gi - offsets[di])]
        act = item["action"]
        if act.shape[-1] < PAD_ACT:
            act = torch.cat([act, torch.zeros(*act.shape[:-1], PAD_ACT - act.shape[-1])], dim=-1)
        return item["state"], act

    def jepa_and_decode(state, action):
        out = model(state, action)
        emb, emb_pre = out["emb"], out["emb_pre_cell"]
        pred_emb = model.predict(emb[:, :H], action[:, :H])
        tgt_emb = emb[:, H:H + 1]
        sigreg_loss = sigreg(emb_pre.transpose(0, 1))
        with torch.no_grad():
            goal_state = state[:, H + GOAL_OFFSET:H + GOAL_OFFSET + 1]
            zero_act = torch.zeros(goal_state.shape[0], 1, action.shape[-1],
                                   device=device, dtype=action.dtype)
            goal_emb_target = model(goal_state, zero_act)["emb"][:, 0]
        out_full = model(state[:, :H + GOAL_OFFSET], action[:, :H + GOAL_OFFSET])
        goal_pred = out_full["emb"][:, -1]
        loss_jepa, parts = stjewm_loss(pred_emb, tgt_emb, emb_pre, sigreg,
                                       goal_pred, goal_emb_target,
                                       0.09, 0.5)
        s_std = state / state_std
        s_hat = decoder(emb)
        loss_decode = torch.mean((s_hat - s_std) ** 2)
        return loss_jepa, parts, loss_decode

    print(f"[access/{tag}] windows={n_total} steps/epoch={steps_per_epoch} "
          f"total_steps={total_steps} lambda={a.lambda_decode}", flush=True)
    curve, step = [], 0
    for _ in range(a.epochs):
        order = rng.permutation(n_total)
        for bstart in range(0, len(order) - a.batch + 1, a.batch):
            idx = order[bstart:bstart + a.batch]
            s = torch.stack([get_window(int(g))[0] for g in idx]).to(device)
            ac = torch.stack([get_window(int(g))[1] for g in idx]).to(device)
            opt.zero_grad()
            loss_jepa, parts, loss_decode = jepa_and_decode(s, ac)
            loss = loss_jepa + a.lambda_decode * loss_decode
            loss.backward()
            opt.step()
            step += 1
            if step % 50 == 0 or step == 1:
                curve.append({"step": step, "jepa": float(loss_jepa.item()),
                              "decode": float(loss_decode.item())})
                print(f"[access/{tag}] step {step}: jepa={float(loss_jepa.item()):.4f} "
                      f"decode={float(loss_decode.item()):.4f}", flush=True)

    # held-out decode R2 (last 5% windows, unseen in shuffle? deterministic tail split)
    val_idx = np.arange(int(n_total * 0.95), n_total)
    with torch.inference_mode():
        se = n = 0
        sbar = 0.0
        for i in range(0, len(val_idx), 64):
            idx = val_idx[i:i + 64]
            s = torch.stack([get_window(int(g))[0] for g in idx]).to(device)
            ac = torch.stack([get_window(int(g))[1] for g in idx]).to(device)
            emb = model(s, ac)["emb"]
            s_std = s / state_std
            s_hat = decoder(emb)
            se += float(((s_hat - s_std) ** 2).sum())
            sbar += float((s_std ** 2).sum())
            n += s.numel()
        decode_r2 = 1.0 - se / sbar

    out_dir = OUT_ROOT / "oodc_F1" / f"stjewm_trace_only_{tag}" / "seed_0"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_sd = {k: v for k, v in model.state_dict().items() if not k.startswith("state_decoder")}
    new_args = dict(ck["args"])
    new_args["accessibility_training"] = {
        "lambda_decode": a.lambda_decode, "epochs": a.epochs, "batch": a.batch,
        "lr": a.lr, "base_checkpoint": BASE_CKPT,
        "base_sha256": hashlib.sha256(Path(BASE_CKPT).read_bytes()).hexdigest(),
        "decoder": "Linear(192,128) on standardized padded state, all positions",
        "val_decode_r2": decode_r2,
    }
    torch.save({"model": base_sd, "args": new_args}, out_dir / "final.pt")
    torch.save({"decoder": decoder.state_dict(), "state_std": state_std.cpu()},
               out_dir / "decoder.pt")
    receipt = {"lambda_decode": a.lambda_decode, "steps": step,
               "val_decode_r2": decode_r2, "curve": curve,
               "base_sha256": hashlib.sha256(Path(BASE_CKPT).read_bytes()).hexdigest(),
               "final_sha256": hashlib.sha256((out_dir / "final.pt").read_bytes()).hexdigest()}
    (out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(f"[access/{tag}] done: val_decode_r2={decode_r2:.4f} -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
