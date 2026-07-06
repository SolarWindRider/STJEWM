"""Readout mode tests for code/stjewm.STJEWM._readout.

Stage 0 verification for v0.7.4: each ReadoutMode returns a tensor of the
right shape, and RATE_ONLY reads the SPIKE TRAIN (not h) per the
membrane-forbidden protocol.

Run as:
    /home/lx/miniconda3/envs/snn/bin/python code/tests/test_readout_mode.py
Exits 0 on pass, 1 on fail. No pytest required.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Repo root on path so `import code.stjewm` works regardless of CWD.
_REPO = Path("/home/lx/snn")
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "code"))

import torch

from code.stjewm import STJEWM, ReadoutMode  # noqa: E402


def _build_tiny_model(readout_mode: str) -> STJEWM:
    """Small state-input model — no ViT encoder needed."""
    return STJEWM(
        d_hid=32,
        embed_dim=32,
        action_dim=4,
        action_emb_dim=32,
        state_dim=8,
        cell_n_layers=1,
        n_d=2,
        trace_beta=0.9,
        freeze_encoder=True,
        readout_mode=readout_mode,
    )


def _inputs(B: int = 2, T: int = 10, D: int = 32):
    torch.manual_seed(0)
    h = torch.randn(B, T, D)                 # continuous hidden
    spike = (torch.randn(B, T, D) > 0.5).float()   # {0, 1} spike train
    trace = torch.randn(B, T, D)
    return h, spike, trace


def _check(name: str, ok: bool, msg: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}{(': ' + msg) if msg else ''}")
    return ok


def test_rate_only_is_spike_rate() -> bool:
    """RATE_ONLY must depend on spike and NOT on h."""
    model = _build_tiny_model("rate_only")
    h1, spike1, trace = _inputs()
    h2, spike2, trace2 = _inputs()
    out1 = model._readout(h1, spike1, trace)
    # Reuse h2's spike but vary h to prove RATE_ONLY ignores h.
    out_h2_with_spike1 = model._readout(h2, spike1, trace)
    if not _check("RATE_ONLY shape (B,T,D)", out1.shape == (2, 10, 32),
                  f"got {tuple(out1.shape)}"):
        return False
    if not _check("RATE_ONLY invariant to h",
                  torch.allclose(out1, out_h2_with_spike1, atol=1e-6),
                  "RATE_ONLY leaked h into output"):
        return False
    # Flip spike -> output should change
    out_flipped = model._readout(h1, 1.0 - spike1, trace)
    if not _check("RATE_ONLY reacts to spike", not torch.allclose(out1, out_flipped, atol=1e-6),
                  "RATE_ONLY unchanged when spike flipped"):
        return False
    return True


def test_each_mode_shape() -> bool:
    ok = True
    h, spike, trace = _inputs()
    for mode in ["hidden_leak", "trace_only", "membrane_readout",
                 "spike_only", "rate_only", "no_trace"]:
        model = _build_tiny_model(mode)
        out = model._readout(h, spike, trace)
        if not _check(f"{mode} shape", out.shape == h.shape,
                      f"got {tuple(out.shape)}"):
            ok = False
    return ok


def test_no_trace_returns_h() -> bool:
    model = _build_tiny_model("no_trace")
    h, spike, trace = _inputs()
    out = model._readout(h, spike, trace)
    return _check("NO_TRACE returns h", torch.equal(out, h))


def test_membrane_returns_h_detached() -> bool:
    model = _build_tiny_model("membrane_readout")
    h, spike, trace = _inputs()
    h.requires_grad_(True)
    out = model._readout(h, spike, trace)
    return _check("MEMBRANE_READOUT detached",
                  not out.requires_grad and torch.equal(out, h.detach()))


def test_hidden_leak_uses_trace() -> bool:
    model = _build_tiny_model("hidden_leak")
    h, spike, trace1 = _inputs()
    trace2 = torch.randn_like(trace1)
    out1 = model._readout(h, spike, trace1)
    out2 = model._readout(h, spike, trace2)
    return _check("HIDDEN_LEAK uses trace",
                  not torch.allclose(out1, out2, atol=1e-6))


def main() -> int:
    results = [
        test_each_mode_shape(),
        test_no_trace_returns_h(),
        test_membrane_returns_h_detached(),
        test_hidden_leak_uses_trace(),
        test_rate_only_is_spike_rate(),
    ]
    n_pass = sum(results)
    n_total = len(results)
    print(f"\n{n_pass}/{n_total} test groups passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())