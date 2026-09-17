"""Shapes, inputs, the forward that feeds the backward, and path-based impl loading."""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch

# Every shape in op.shape, plus the edge cases the config implies. `spec` marks the three
# that op.shape lists: those are the ones benchmark/validation score.
SHAPES = {
    #                     b  sq    skv   hq  hkv  d
    "fast":              (1, 1024, 1024, 8,  2,   128),   # op.shape[0], fast-iteration
    "proxy":             (1, 4096, 4096, 32, 8,   128),   # op.shape[1], quarter-FLOP proxy
    "prod":              (4, 8192, 8192, 32, 8,   128),   # op.shape[2], production
    # edge cases implied by op.config, not scored
    "toy":               (1, 64,   64,   2,  1,   128),   # smallest legal, mha
    "gqa4_small":        (2, 128,  128,  8,  2,   128),   # GQA=4, batch>1
    "mha":               (1, 256,  256,  4,  4,   128),   # heads_q == heads_kv, GQA=1
    "unequal_seqlen":    (1, 512,  1024, 8,  2,   128),   # sq != skv -> bottom-right bites
    "unequal_seqlen_2":  (2, 1024, 2048, 4,  1,   128),   # again, GQA=4, batch>1
    "sq_gt_skv":         (2, 1024, 512,  4,  1,   128),   # sq > skv: NON-CAUSAL ONLY
}
SPEC_SHAPES = ("fast", "proxy", "prod")

# Under BOTTOM-RIGHT causal a query i attends keys j <= i + (Skv - Sq). With Sq > Skv the
# first Sq-Skv queries have an EMPTY window, so their lse is -inf and p = exp(-inf - -inf)
# is NaN in any implementation, the fp32 reference included. That is an undefined input,
# not a kernel defect, so `sq_gt_skv` is exercised non-causal only.
# The two large op.shape entries are checked causal only: op.config.causal is
# bottom-right for the whole job, and the non-causal path is already covered at six
# smaller shapes. The fp32 reference at (4, 8192, 8192, 32) takes minutes per pass.
CAUSAL_MODES = {"sq_gt_skv": (False,), "proxy": (True,), "prod": (True,)}


def causal_modes(shape):
    return CAUSAL_MODES.get(shape, (True, False))


def make_inputs(name, seed=0, device="cuda"):
    b, sq, skv, hq, hkv, d = SHAPES[name]
    g = torch.Generator(device=device).manual_seed(seed)

    def rnd(*shape):
        return torch.randn(*shape, generator=g, device=device, dtype=torch.float32).to(
            torch.bfloat16)

    return rnd(b, sq, hq, d), rnd(b, skv, hkv, d), rnd(b, skv, hkv, d), rnd(b, sq, hq, d)


def forward_reference(q, k, v, causal=True, softmax_scale=None, q_chunk=1024):
    """fp32 forward producing o (bf16) and lse (fp32, NATURAL log), bottom-right causal.

    Written here rather than taken from a library: the backward under test consumes o and
    lse, so a library forward would put a library in the reference chain.
    """
    b, sq, hq, d = q.shape
    skv, hkv = k.shape[1], k.shape[2]
    g = hq // hkv
    scale = float(softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(d))
    shift = skv - sq
    o = torch.empty((b, sq, hq, d), device=q.device, dtype=torch.bfloat16)
    lse = torch.empty((b, hq, sq), device=q.device, dtype=torch.float32)
    for bi in range(b):
        for h in range(hq):
            hk = h // g
            kb = k[bi, :, hk, :].float()
            vb = v[bi, :, hk, :].float()
            for q0 in range(0, sq, q_chunk):
                q1 = min(q0 + q_chunk, sq)
                s = (q[bi, q0:q1, h, :].float() @ kb.T) * scale
                if causal:
                    qi = torch.arange(q0, q1, device=s.device).unsqueeze(1)
                    kj = torch.arange(skv, device=s.device).unsqueeze(0)
                    s = s.masked_fill(kj > qi + shift, float("-inf"))
                l = torch.logsumexp(s, dim=-1)
                p = torch.exp(s - l.unsqueeze(1))
                o[bi, q0:q1, h, :] = (p @ vb).to(torch.bfloat16)
                lse[bi, h, q0:q1] = l
    return o, lse


def load_impl(impl_dir):
    """Load an implementation BY PATH and return its `attn_bwd` callable.

    The directory is the identity. It is never turned into a name and looked up: a round's
    code lives at rounds/<n>/op/, whose basename is `op`, and resolving that name against
    anything would measure the wrong code while reporting the right one.
    """
    impl_dir = Path(impl_dir).resolve()
    src = impl_dir / "impl.py"
    if not src.is_file():
        raise FileNotFoundError(f"no impl.py under {impl_dir}")
    # NOT added to sys.path: impl.py loads its own siblings by path, and putting two
    # implementation directories on sys.path is how one arm ends up running the other's
    # module.
    name = "op_impl_" + str(abs(hash(str(impl_dir))))
    spec = importlib.util.spec_from_file_location(name, src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    fn = getattr(mod, "attn_bwd", None)
    if fn is None:
        raise AttributeError(f"{src} defines no `attn_bwd`")
    return fn


def poison_allocator(device="cuda"):
    """Fill the caching allocator's free blocks with NaN.

    An implementation allocates its own outputs, so the test cannot prefill them. Freeing
    NaN-filled blocks of the sizes about to be requested makes `torch.empty` hand back NaN
    instead of zeros, so an element the kernel never writes shows up as non-finite rather
    than as a plausible zero. Best-effort by construction -- the isfinite coverage assert
    below is the gate, this only makes it bite.
    """
    blocks = [torch.full((n,), float("nan"), device=device, dtype=torch.float32)
              for n in (1 << 24, 1 << 22, 1 << 20, 1 << 18, 1 << 16)]
    del blocks
    torch.cuda.synchronize()


def sqnr_db(ref, got):
    ref = ref.float()
    got = got.float()
    num = ref.pow(2).mean()
    den = (ref - got).pow(2).mean().clamp_min(1e-30)
    return float(10.0 * torch.log10(num / den))
