###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""fp32 reference, inputs and metrics shared by hkl.test and hkl.ladder.

Problems (a ladder file's ``problem``): ``gqa`` (Llama: q [B, S, Hq, D], k / v [B, S, Hkv, D], BSHD) and
``mla`` (DeepSeek-V3: q / k [S, B, H, 192], v [S, B, H, 128], SBHD, MHA). Causal is bottom-right: query
p sees kv <= p + Skv - Sq. LSE is the natural log of the scaled row sums, [B, Hq, Sq] fp32.

The reference runs in fp32 on the GPU, one (batch, head chunk) at a time so the score matrix stays
below ~1 GiB, from the same bf16 inputs the kernels get.
"""

import math

import torch

__all__ = ["Shape", "make_inputs", "ref_fwd", "ref_bwd", "snr_db", "useful_flops", "poison_allocator"]


class Shape:
    def __init__(self, kind, b, sq, skv, hq, hkv, d, dv, causal=True, layout="bshd"):
        self.kind, self.b, self.sq, self.skv, self.hq, self.hkv = kind, b, sq, skv, hq, hkv
        self.d, self.dv, self.causal, self.layout = d, dv, bool(causal), layout

    @classmethod
    def from_problem(cls, p, **over):
        q = dict(p)
        q.update(over)
        return cls(q["kind"], q["b"], q["sq"], q["skv"], q["hq"], q["hkv"], q["d"], q.get("dv", q["d"]),
                   q.get("causal", True), q.get("layout", "bshd"))  # fmt: skip

    def __repr__(self):
        c = "causal" if self.causal else "full"
        return f"b{self.b} sq{self.sq} skv{self.skv} hq{self.hq} hkv{self.hkv} d{self.d}/{self.dv} {c} {self.layout}"

    def tensor(self, s, h, d, device, dtype=torch.bfloat16):
        shape = (self.b, s, h, d) if self.layout == "bshd" else (s, self.b, h, d)
        return torch.empty(shape, device=device, dtype=dtype)


def _bshd(t, layout):
    """[B, S, H, D] view of a kernel tensor."""
    return t if layout == "bshd" else t.permute(1, 0, 2, 3)


def make_inputs(sh, device, seed=0, scale=1.0):
    """q, k, v, do (bf16, the problem's layout, contiguous), N(0, scale^2)."""
    g = torch.Generator(device=device).manual_seed(seed)

    def rnd(s, h, d):
        t = sh.tensor(s, h, d, device)
        t.copy_(torch.randn(t.shape, generator=g, device=device, dtype=torch.float32) * scale)
        return t

    return {"q": rnd(sh.sq, sh.hq, sh.d), "k": rnd(sh.skv, sh.hkv, sh.d), "v": rnd(sh.skv, sh.hkv, sh.dv),
            "do": rnd(sh.sq, sh.hq, sh.dv)}  # fmt: skip


def _chunk(sh):
    per_head = 4 * sh.sq * sh.skv * 3  # S, P, and a temporary
    return max(1, min(sh.hq, (1 << 30) // max(per_head, 1)))


def _mask(sh, device):
    if not sh.causal:
        return None
    p = torch.arange(sh.sq, device=device)[:, None]
    kv = torch.arange(sh.skv, device=device)[None, :]
    return kv > p + (sh.skv - sh.sq)  # True = masked


def ref_fwd(sh, q, k, v, scale=None):
    """(o fp32 in the problem's layout, lse fp32 [B, Hq, Sq])."""
    scale = sh.d ** -0.5 if scale is None else scale
    qb, kb, vb = _bshd(q, sh.layout), _bshd(k, sh.layout), _bshd(v, sh.layout)
    g = sh.hq // sh.hkv
    o = torch.empty((sh.b, sh.sq, sh.hq, sh.dv), device=q.device, dtype=torch.float32)
    lse = torch.empty((sh.b, sh.hq, sh.sq), device=q.device, dtype=torch.float32)
    m = _mask(sh, q.device)
    hc = _chunk(sh)
    for b in range(sh.b):
        for h0 in range(0, sh.hq, hc):
            hs = list(range(h0, min(sh.hq, h0 + hc)))
            qq = qb[b, :, hs].float().permute(1, 0, 2)  # [h, sq, d]
            kk = kb[b, :, [h // g for h in hs]].float().permute(1, 0, 2)
            vv = vb[b, :, [h // g for h in hs]].float().permute(1, 0, 2)
            s = torch.matmul(qq, kk.transpose(1, 2)) * scale
            if m is not None:
                s.masked_fill_(m, float("-inf"))
            l = torch.logsumexp(s, dim=-1)  # [h, sq]
            p = torch.exp(s - l[..., None])
            o[b, :, hs] = torch.matmul(p, vv).permute(1, 0, 2)
            lse[b, hs] = l
            del s, p
    return (o if sh.layout == "bshd" else o.permute(1, 0, 2, 3).contiguous()), lse


def ref_bwd(sh, q, k, v, do, scale=None):
    """(dq, dk, dv) fp32 in the problem's layout (dk / dv with Hkv heads)."""
    scale = sh.d ** -0.5 if scale is None else scale
    qb, kb, vb, dob = (_bshd(t, sh.layout) for t in (q, k, v, do))
    g = sh.hq // sh.hkv
    dq = torch.zeros((sh.b, sh.sq, sh.hq, sh.d), device=q.device, dtype=torch.float32)
    dk = torch.zeros((sh.b, sh.skv, sh.hkv, sh.d), device=q.device, dtype=torch.float32)
    dv = torch.zeros((sh.b, sh.skv, sh.hkv, sh.dv), device=q.device, dtype=torch.float32)
    m = _mask(sh, q.device)
    hc = _chunk(sh)
    for b in range(sh.b):
        for h0 in range(0, sh.hq, hc):
            hs = list(range(h0, min(sh.hq, h0 + hc)))
            kvh = [h // g for h in hs]
            qq = qb[b, :, hs].float().permute(1, 0, 2).requires_grad_(True)
            kk = kb[b, :, kvh].float().permute(1, 0, 2).requires_grad_(True)
            vv = vb[b, :, kvh].float().permute(1, 0, 2).requires_grad_(True)
            s = torch.matmul(qq, kk.transpose(1, 2)) * scale
            if m is not None:
                s = s.masked_fill(m, float("-inf"))
            o = torch.matmul(torch.softmax(s, dim=-1), vv)
            gq, gk, gv = torch.autograd.grad(o, (qq, kk, vv), dob[b, :, hs].float().permute(1, 0, 2))
            dq[b, :, hs] = gq.permute(1, 0, 2)
            for j, kh in enumerate(kvh):  # GQA: sum the q heads of a kv head
                dk[b, :, kh] += gk[j]
                dv[b, :, kh] += gv[j]
            del s, o, gq, gk, gv
    if sh.layout != "bshd":
        dq, dk, dv = (t.permute(1, 0, 2, 3).contiguous() for t in (dq, dk, dv))
    return dq, dk, dv


def snr_db(x, ref):
    """10 log10(|ref|^2 / |x - ref|^2) in fp64 (inf when equal); NaN / inf in x gives -inf."""
    x, ref = x.double(), ref.double()
    if not torch.isfinite(x).all():
        return float("-inf")
    err = (x - ref).pow(2).sum().item()
    sig = ref.pow(2).sum().item()
    return float("inf") if err == 0 else 10 * math.log10(sig / err)


def visible_pairs(sh):
    """(query, kv) pairs a causal / full attention computes, per (batch, q head)."""
    if not sh.causal:
        return sh.sq * sh.skv
    c = sh.skv - sh.sq
    return sh.sq * (c + 1) + sh.sq * (sh.sq - 1) // 2


def useful_flops(sh, direction):
    """Useful FLOPs: fwd 2 GEMMs (QK^T: 2 d, PV: 2 dv per pair), bwd 5 (S, dP, dV, dK, dQ)."""
    pairs = visible_pairs(sh) * sh.b * sh.hq
    if direction == "fwd":
        return pairs * (2 * sh.d + 2 * sh.dv)
    return pairs * (2 * sh.d + 2 * sh.dv + 2 * sh.dv + 2 * sh.d + 2 * sh.d)


def poison_allocator(device, nbytes=None):
    """NaN-fill free memory of torch's caching allocator, so an output a kernel leaves unwritten reads as NaN
    (outputs are torch.empty inside the launch modules)."""
    free, _ = torch.cuda.mem_get_info(device)
    n = int((nbytes or free * 0.6) // 4)
    t = torch.empty(n, dtype=torch.int32, device=device)
    t.fill_(0x7FC07FC0)  # NaN as fp32 AND as each bf16 half (0x7FC0): an fp32 NaN would read as bf16 0 in one half
    del t
