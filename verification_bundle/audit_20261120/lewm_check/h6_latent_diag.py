#!/usr/bin/env python
"""H6: why does latent planning fail? Diagnose the generalist_16env checkpoints on PushT data.

For each model (our harness conventions: zero-action encoding, padded 128-d state / 56-d action,
history 1, our CEM with native N(0,1) proposals):
  1-step   teacher-forced latent prediction R2 (vs persistence)
  25-step  open-loop latent rollout with the RECORDED actions: error vs true latents
  spread   how much 25-step predictions depend on the action sequence (64 random plans)
  costs    ||z25 - z_goal||^2 for: stay-still plan, recorded (expert) plan, CEM plan
"""
import json
import sys

sys.path.insert(0, "/home/lx/snn")
import numpy as np
import torch

from code.data import load_dataset
from code.core.cem import CEM, make_native_action_predict_hook
from code.scripts.event_align import build_model

H5 = "/home/lx/LeWM/data/pusht_expert_train.h5"
R = "/data/lx/tmp/results"
DEV = "cuda"
MODELS = [("stjewm_trace_only", "5m_5mpar"), ("stjewm_spike_only", "5m_5mpar"), ("stjewm_rate_only", "5m_5mpar"),
          ("stjewm_membrane_readout", "5m_5mpar"), ("stjewm_no_trace", "5m_5mpar"), ("stjewm_hidden_leak", "5m_5mpar"),
          ("alif_timecell_baseline", "5m"), ("lif_transformer_baseline", "5m"), ("stacked_lif_trace", "5m"),
          ("stacked_lif_free", "5m"), ("gru_baseline", "5m"), ("mlp_baseline", "5m"), ("lewm_baseline_v2", "5m")]


def load(name, root):
    ck = torch.load(f"{R}/{root}/generalist_16env/{name}/seed_0/final.pt", map_location="cpu", weights_only=False)
    ck_args = ck.get("args", {})
    sd = {k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}
    state_dim = next((int(sd[k].shape[1]) for k in ("state_encoder.proj.0.weight", "state_projector.proj.0.weight",
                                                    "state_projector.0.weight", "state_proj.0.weight") if k in sd),
                     ck_args.get("pad_obs_to"))
    action_dim = next((int(sd[k].shape[1]) for k in ("action_encoder.proj.0.weight", "action_encoder.0.weight",
                                                     "action_proj.0.weight") if k in sd), ck_args.get("action_dim"))
    if ck_args["model"] == "mlp_baseline":
        action_dim = int(sd["net.0.weight"].shape[1] - sd["state_proj.2.weight"].shape[0])
    model = build_model(ck_args["model"], state_dim, action_dim, ck_args, state_dict=sd)
    model.load_state_dict(sd, strict=True)
    return model.to(DEV).eval(), state_dim, action_dim, ck


@torch.no_grad()
def enc(model, s7, S_DIM, A_DIM):
    s = torch.from_numpy(s7).float().reshape(-1, 1, 7).to(DEV)
    s = torch.nn.functional.pad(s, (0, S_DIM - 7))
    a = torch.zeros(s.shape[0], 1, A_DIM, device=DEV)
    return model(s, a)["emb"][:, 0]


@torch.no_grad()
def rollout(model, z0, acts, hook):
    """z0 (N,D); acts (N,T,2) native -> list of latents per step."""
    N, T, _ = acts.shape
    h = z0.unsqueeze(1)
    out = []
    A = hook
    for t in range(T):
        a_native = acts[:, t]
        low = torch.tensor([-1.0, -1.0], device=DEV); high = torch.tensor([1.0, 1.0], device=DEV)
        a = torch.clamp(a_native, low, high)
        a = torch.nn.functional.pad(a, (0, 56 - 2))
        nxt = model.predict(h, a.unsqueeze(1))[:, -1]
        h = nxt.unsqueeze(1)
        out.append(nxt)
    return torch.stack(out, 1)          # (N,T,D)


def main():
    ds = load_dataset("pusht", path=H5, history_size=1, goal_offset=25, pad_obs_to=None, env_id="swm/PushT-v1")
    rng = np.random.default_rng(0)
    pick = rng.choice(len(ds), size=256, replace=False)
    items = [ds[int(i)] for i in pick]
    S = np.stack([np.asarray(it["state"], np.float32) for it in items])          # (N,25,7) frames t..t+24
    G = np.stack([np.asarray(it["goal_state"], np.float32) for it in items])     # (N,7) frame t+25
    Arec = np.stack([np.asarray(it["action"], np.float32) for it in items])      # (N,24,2) actions a_t..a_{t+23}
    N = len(pick)
    print(f"windows: {N} | recorded action std {Arec.std():.3f}, mean |a| {np.abs(Arec).mean():.3f}", flush=True)

    results = {}
    for name, root in MODELS:
        try:
            model, S_DIM, A_DIM, ck = load(name, root)
        except Exception as ex:
            print(f"SKIP {name}: {type(ex).__name__}: {str(ex)[:120]}", flush=True)
            continue
        z_all = torch.stack([enc(model, S[:, t], S_DIM, A_DIM) for t in range(25)], 1)   # (N,25,D) true latents
        z0 = z_all[:, 0]
        z_goal = enc(model, G, S_DIM, A_DIM)
        D = z0.shape[-1]

        # 1-step teacher-forced
        a1 = torch.nn.functional.pad(torch.from_numpy(Arec[:, 0]).to(DEV), (0, 56 - 2))
        p1 = model.predict(z_all[:, :-1].reshape(-1, 1, D), a1.unsqueeze(1).repeat_interleave(24, 0).reshape(-1, 1, 56)[:, 0].reshape(-1, 1, 56)) if False else None
        pred1 = []
        for t in range(24):
            at = torch.nn.functional.pad(torch.from_numpy(Arec[:, t]).to(DEV), (0, 56 - 2))
            pred1.append(model.predict(z_all[:, t].unsqueeze(1), at.unsqueeze(1))[:, -1])
        pred1 = torch.stack(pred1, 1)
        err1 = ((pred1 - z_all[:, 1:]) ** 2).sum(-1)
        pers1 = ((z_all[:, :-1] - z_all[:, 1:]) ** 2).sum(-1)
        r2_1step = float(1 - err1.sum() / pers1.sum())

        # 25-step open-loop with recorded actions (a_0..a_23, then a_24 = a_23)
        A25 = Arec[:, :25]                                  # recorded actions a_t .. a_{t+24}
        zhat = rollout(model, z0, torch.from_numpy(np.ascontiguousarray(A25)).to(DEV), None)   # (N,25,D)
        true_err = ((zhat[:, -1] - z_goal) ** 2).sum(-1)
        c_init = ((z0 - z_goal) ** 2).sum(-1)
        track = ((zhat[:, :24] - z_all[:, 1:]) ** 2).sum(-1).mean()

        # stay-still plan
        zhat0 = rollout(model, z0, torch.zeros(N, 25, 2, device=DEV), None)
        c_stay = ((zhat0[:, -1] - z_goal) ** 2).sum(-1)

        # action sensitivity: 64 random plans with dataset-like scale
        plans = torch.from_numpy(rng.normal(0, 0.21, size=(64, 25, 2)).astype(np.float32)).to(DEV)
        zs = torch.stack([rollout(model, z0[:32], plans[i:i + 1].expand(32, 25, 2).contiguous(), None)[:, -1]
                          for i in range(64)], 0)            # (64,N/32,D)
        spread = zs.std(0).pow(2).sum(-1).sqrt()                # per-window std over plans
        goal_scale = (z_goal[:32] - z0[:32]).norm(dim=-1)
        sens = float((spread.mean() / goal_scale.mean()))

        # CEM plan cost vs recorded-action cost
        cem = CEM(model, action_dim=2, horizon=25, n_samples=300, n_elites=30, n_iters=30, history_size=1,
                  device=DEV, predict_hook=make_native_action_predict_hook(56, [-1.0, -1.0], [1.0, 1.0], DEV))
        c_cem, mag = [], []
        for i in range(16):
            seq = cem.plan(z0[i:i + 1], z_goal[i])
            zc = rollout(model, z0[i:i + 1], seq.unsqueeze(0), None)[:, -1]
            c_cem.append(float(((zc[0] - z_goal[i]) ** 2).sum()))
            mag.append(float(seq.abs().mean()))
        res = dict(latent_dim=D, latent_norm=float(z0.norm(dim=-1).mean()),
                   r2_1step=round(r2_1step, 4),
                   openloop25_rel_err=round(float(true_err.mean() / c_init.mean()), 4),   # <1: better than staying put
                   rollout_track_err=round(float(track), 5), step_err=round(float(err1.mean()), 5),
                   c_init=round(float(c_init.mean()), 4), c_stay=round(float(c_stay.mean()), 4),
                   c_recorded=round(float(true_err.mean()), 4), c_cem=round(float(np.mean(c_cem)), 4),
                   cem_plan_mean_abs_action=round(float(np.mean(mag)), 3),
                   action_sensitivity=round(sens, 4), steps=ck.get("step"))
        results[name] = res
        print(f"{name:26s} {json.dumps(res)}", flush=True)
    json.dump(results, open("/tmp/lewm_check/h6_pusht_latent_diag.json", "w"), indent=1)


if __name__ == "__main__":
    main()
