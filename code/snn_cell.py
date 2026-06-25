"""SNN cells: LIF + MultiCompartment with V2-ST-JEWM spike trace interface."""
import math
import torch
import torch.nn as nn


class _ATanSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_minus_thresh, alpha):
        ctx.save_for_backward(v_minus_thresh)
        ctx.alpha = alpha
        return (v_minus_thresh > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        a = ctx.alpha
        denom = 1.0 + (math.pi * a * x / 2.0) ** 2
        grad = (a / 2.0) / denom
        return grad * grad_output, None


def atan_spike(v, v_thresh, alpha=2.0):
    return _ATanSurrogate.apply(v - v_thresh, alpha)


def _build_trace(spk_seq, beta):
    spk_stack = torch.stack(spk_seq, dim=1)
    if beta <= 0:
        return spk_stack
    T = spk_stack.shape[1]
    r = torch.zeros_like(spk_stack)
    r[:, 0] = spk_stack[:, 0]
    for t in range(1, T):
        r[:, t] = beta * r[:, t - 1] + spk_stack[:, t]
    return r


class LIFCell(nn.Module):
    def __init__(self, d_in, d_hid, v_thresh=0.3, v_reset=0.0, tau_m=20.0, t_ref=2.0,
                 dt=1.0, alpha_surr=2.0, init_scale=3.0, trace_beta=0.9):
        super().__init__()
        self.d_in, self.d_hid = d_in, d_hid
        self.v_thresh, self.v_reset = v_thresh, v_reset
        self.tau_m, self.t_ref, self.dt = tau_m, t_ref, dt
        self.alpha_surr = alpha_surr
        self.trace_beta = float(trace_beta)
        self.w_in = nn.Linear(d_in, d_hid, bias=True)
        with torch.no_grad():
            self.w_in.weight.mul_(init_scale)
            self.w_in.bias.zero_()
        self.register_buffer("decay", torch.exp(torch.tensor(-dt / tau_m)))

    def forward(self, x):
        B, T, _ = x.shape
        v = torch.zeros(B, self.d_hid, device=x.device)
        refr = torch.zeros_like(v)
        v_seq, spk_seq = [], []
        for t in range(T):
            I = self.w_in(x[:, t])
            v = self.decay * v + (1.0 - self.decay) * I
            v = torch.where(refr > 0, torch.full_like(v, self.v_reset), v)
            spk = atan_spike(v, self.v_thresh, self.alpha_surr)
            v = torch.where(spk > 0, torch.full_like(v, self.v_reset), v)
            refr = torch.where(
                spk > 0,
                torch.full_like(refr, self.t_ref / self.dt),
                (refr - 1.0).clamp(min=0.0),
            )
            v_seq.append(v)
            spk_seq.append(spk)
        return {
            "v": torch.stack(v_seq, dim=1),
            "spike": torch.stack(spk_seq, dim=1),
            "trace": _build_trace(spk_seq, self.trace_beta),
        }


class MultiCompartmentCell(nn.Module):
    def __init__(self, d_in, d_hid, N_d=3, tau_d_range=(10.0, 50.0),
                 tau_s_range=(2.0, 10.0), v_thresh=0.3, v_reset=0.0, t_ref=2.0,
                 dt=1.0, alpha_surr=2.0, init_scale=3.0,
                 decouple_d_to_s=False, decouple_s_to_d=False,
                 trace_beta=0.9):
        super().__init__()
        self.decouple_d_to_s = decouple_d_to_s
        self.decouple_s_to_d = decouple_s_to_d
        self.trace_beta = float(trace_beta)
        self.d_in, self.d_hid, self.N_d = d_in, d_hid, N_d
        self.v_thresh, self.v_reset = v_thresh, v_reset
        self.t_ref, self.dt, self.alpha_surr = t_ref, dt, alpha_surr
        tau_d = torch.linspace(tau_d_range[0], tau_d_range[1], N_d)
        tau_s = torch.tensor([(tau_s_range[0] + tau_s_range[1]) / 2.0])
        self.register_buffer("tau_d", tau_d)
        self.register_buffer("tau_s", tau_s)
        self.register_buffer("decay_d", torch.exp(-dt / tau_d))
        self.register_buffer("decay_s", torch.exp(-dt / tau_s))
        self.w_in_d = nn.Linear(d_in, N_d * d_hid, bias=True)
        # s -> d: per-compartment, output N_d * d_hid
        self.w_s_to_d = nn.Linear(d_hid, N_d * d_hid, bias=False)
        self.w_d_to_s = nn.Linear(d_hid, d_hid, bias=False)
        self.w_in_s = nn.Linear(d_in, d_hid, bias=False)
        with torch.no_grad():
            self.w_in_d.weight.mul_(init_scale * (N_d ** 0.5))
            self.w_in_d.bias.zero_()
            self.w_in_s.weight.mul_(init_scale)
            self.w_d_to_s.weight.mul_(0.1)
            self.w_s_to_d.weight.mul_(0.1 * (N_d ** 0.5))

    def forward(self, x):
        B, T, _ = x.shape
        device = x.device
        V_d = torch.zeros(B, self.N_d, self.d_hid, device=device)
        V_s = torch.zeros(B, self.d_hid, device=device)
        refr = torch.zeros_like(V_s)
        V_d_seq, V_s_seq, spk_seq = [], [], []
        for t in range(T):
            I_d_raw = self.w_in_d(x[:, t]).view(B, self.N_d, self.d_hid)
            I_d_fb = self.w_s_to_d(V_s).view(B, self.N_d, self.d_hid)
            if self.decouple_s_to_d:
                I_d_fb = torch.zeros_like(I_d_fb)
            decay_d = self.decay_d.view(1, -1, 1)
            V_d = decay_d * V_d + (1.0 - decay_d) * (I_d_raw + I_d_fb)
            d_agg = V_d.mean(dim=1)
            d_to_s = self.w_d_to_s(d_agg)
            if self.decouple_d_to_s:
                d_to_s = torch.zeros_like(d_to_s)
            I_s = d_to_s + self.w_in_s(x[:, t])
            V_s = self.decay_s * V_s + (1.0 - self.decay_s) * I_s
            V_s = torch.where(refr > 0, torch.full_like(V_s, self.v_reset), V_s)
            spk = atan_spike(V_s, self.v_thresh, self.alpha_surr)
            V_s = torch.where(spk > 0, torch.full_like(V_s, self.v_reset), V_s)
            refr = torch.where(
                spk > 0,
                torch.full_like(refr, self.t_ref / self.dt),
                (refr - 1.0).clamp(min=0.0),
            )
            V_d_seq.append(V_d)
            V_s_seq.append(V_s)
            spk_seq.append(spk)
        return {
            "V_d": torch.stack(V_d_seq, dim=1),
            "V_s": torch.stack(V_s_seq, dim=1),
            "spike": torch.stack(spk_seq, dim=1),
            "trace": _build_trace(spk_seq, self.trace_beta),
        }
