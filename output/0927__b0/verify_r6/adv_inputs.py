"""Adversarial inputs for the r6 speculative-softmax verification. CPU-generated (seeded), bf16.

Logit structure is injected through coordinates 0/1 of the head dim: with c = 128**0.25,
q[...,0] = c*A_i and k[...,0] = c*B_j add exactly A_i*B_j to the scaled logit (c^2 * 1/sqrt(128) = 1),
up to bf16 rounding of the injected values. Kernel and reference see the same bf16 tensors.
"""
import math

import torch

SHAPES = {
    #            b  sq    skv   hq  hkv  d
    "toy":     (1, 256, 256, 2, 1, 128),
    "short_q": (1, 128, 512, 4, 1, 128),
    "gqa4":    (2, 256, 256, 8, 2, 128),
    "proxy":   (1, 4096, 4096, 32, 8, 128),
    "prod":    (4, 8192, 8192, 32, 8, 128),
}
C = 128 ** 0.25
TILE = 64

KINDS = [
    "randn", "x4", "x16", "x64",
    "stair9", "stair30", "ramp02", "ramp01", "ramp1",
    "late_jump", "periodic_jump30",
    "jump88", "jump89", "jump1e3", "stair100", "stair1e4",
    "huge_off1e4", "huge_off1e6", "neg1e4", "neg1e6", "neg_late", "neg_early",
    "dom_first", "dom_mid", "dom_last", "diag_pos", "diag_neg",
    "bnd_sum", "bnd_one7", "bnd_one8",
    "ext_pos1e30", "ext_neg1e32",
]
# undefined for every fp32 kernel with a finite -1e30 running-max seed / fp32 logit spacing: reported, not scored
INFORMATIONAL = {"ext_pos1e30", "ext_neg1e32"}


def make(shape, kind, seed=1234):
    b, sq, skv, hq, hkv, d = SHAPES[shape]
    g = torch.Generator().manual_seed(seed + 7919 * KINDS.index(kind))
    rn = lambda *s: torch.randn(*s, generator=g, dtype=torch.float32)
    q, k, v = rn(b, sq, hq, d), rn(b, skv, hkv, d), rn(b, skv, hkv, d)
    j = torch.arange(skv, dtype=torch.float32)
    t = (j // TILE)
    i = torch.arange(sq, dtype=torch.float32)
    shift = skv - sq

    def inject(A, B, coord=0):
        # A: [sq] or scalar per query position, B: [skv] per key position
        A = torch.as_tensor(A, dtype=torch.float32).expand(sq) if not torch.is_tensor(A) or A.dim() == 0 else A
        q[:, :, :, coord] = (C * A).view(1, sq, 1)
        k[:, :, :, coord] = (C * B).view(1, skv, 1)

    if kind == "randn":
        pass
    elif kind in ("x4", "x16", "x64"):
        f = float(kind[1:]); q *= f; k *= f
    elif kind == "stair9":          # tile level +9 per tile: stale max overshoots every tile
        inject(1.0, 9.0 * t)
    elif kind == "stair30":
        inject(1.0, 30.0 * t)
    elif kind == "ramp02":           # monotone ramp 12.8 logits / tile
        inject(1.0, 0.2 * j)
    elif kind == "ramp01":           # 6.4 / tile: r4 keeps deferring, r6 half-sums cross e^7
        inject(1.0, 0.1 * j)
    elif kind == "ramp1":            # 64 / tile
        inject(1.0, 1.0 * j)
    elif kind == "late_jump":        # max jumps late in the row
        B = torch.zeros(skv); B[skv - 1] = 40.0; B[skv - 70] = 20.0; inject(1.0, B)
    elif kind == "periodic_jump30":
        B = torch.zeros(skv); B[100::256] = 30.0; B[(skv * 3) // 4] = 60.0; inject(1.0, B)
    elif kind in ("jump88", "jump89", "jump1e3"):   # one jump mid row; stale exp -> e^88 / inf
        L = {"jump88": 88.0, "jump89": 89.5, "jump1e3": 1e3}[kind]
        inject(1.0, L * (t >= (t.max() + 1) // 2).float())
    elif kind == "stair100":         # every tile overflows exp2 on the stale path
        inject(1.0, 100.0 * t)
    elif kind == "stair1e4":
        inject(1.0, 1e4 * t)
    elif kind == "huge_off1e4":      # all logits around +1e4
        inject(1.0, torch.full((skv,), 1e4))
    elif kind == "huge_off1e6":
        inject(1.0, torch.full((skv,), 1e6))
    elif kind == "neg1e4":           # very negative rows
        inject(1.0, torch.full((skv,), -1e4))
    elif kind == "neg1e6":
        inject(1.0, torch.full((skv,), -1e6))
    elif kind == "neg_late":         # tile 0 normal, later tiles underflow to 0
        inject(1.0, torch.where(t >= 1, torch.tensor(-1e4), torch.tensor(0.0)))
    elif kind == "neg_early":        # tile 0 at -1e4, later tiles jump +1e4 (stale exp overflows)
        inject(1.0, torch.where(t == 0, torch.tensor(-1e4), torch.tensor(0.0)))
    elif kind in ("dom_first", "dom_mid", "dom_last"):
        j0 = {"dom_first": 0, "dom_mid": skv // 2 + 17, "dom_last": skv - 1}[kind]
        B = torch.zeros(skv); B[j0] = 30.0; inject(1.0, B)
    elif kind in ("diag_pos", "diag_neg"):   # q_i ~ +-k_{i+shift}: the diagonal key dominates / is lowest
        gam = 2.65 if kind == "diag_pos" else -2.65
        gq = hq // hkv
        kk = k[:, shift:shift + sq].repeat_interleave(gq, dim=2)
        q = gam * kk + 0.1 * q
    elif kind == "bnd_sum":           # 32-key half-sums straddle e^7: level delta in [3.40, 3.70]
        q[..., 1:] = 0.0
        G = min(sq // 16, 31)
        A = 3.40 + 0.30 * ((i // 16) % G) / (G - 1)
        inject(A, (t >= 1).float())
    elif kind in ("bnd_one7", "bnd_one8"):   # one key per tile at delta straddling 7 (trigger) / 8 (r4 rescale)
        q[..., 2:] = 0.0
        lo = 6.80 if kind == "bnd_one7" else 7.80
        G = min(sq // 16, 31)
        A = lo + 0.40 * ((i // 16) % G) / (G - 1)
        spec = ((j % TILE) == 5) & (t >= 1)
        inject(A, spec.float(), coord=0)
        inject(1.0, torch.where((t >= 1) & ~spec, torch.tensor(-30.0), torch.tensor(0.0)), coord=1)
    elif kind == "ext_pos1e30":
        inject(1.0, torch.full((skv,), 1e30)); q[..., 1:] *= 0; k[..., 1:] *= 1
    elif kind == "ext_neg1e32":       # below the kernel's BIG_NEG=-1e30 seed
        inject(1.0, torch.full((skv,), -1e32))
    else:
        raise KeyError(kind)
    bf = lambda x: x.to(torch.bfloat16).contiguous()
    return bf(q), bf(k), bf(v)


def sample_spec(shape):
    """(batches, q heads, q rows) whose reference is computed. None = all."""
    b, sq, skv, hq, hkv, d = SHAPES[shape]
    if shape == "prod":
        rows = list(range(0, 128)) + list(range(4096 - 64, 4096 + 64)) + list(range(sq - 128, sq))
        return [0, 3], [0, 13, 31], rows
    if shape == "proxy":
        return [0], [0, 9, 22, 31], list(range(sq))
    return list(range(b)), list(range(hq)), list(range(sq))
