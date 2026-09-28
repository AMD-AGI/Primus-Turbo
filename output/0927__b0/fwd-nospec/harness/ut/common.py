"""Shapes, inputs, SQNR and path-based impl loading. Hashed into refcache provenance
(together with eager/impl.py): keep gate logic OUT of this file -- it lives in op/gates.py.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

# Every shape in op.shape, plus the edge cases the config implies. SPEC_SHAPES are scored.
SHAPES = {
    #                   b  sq    skv   hq  hkv  d
    "fast":            (1, 1024, 1024, 8,  2,   128),   # op.shape[0], fast-iteration
    "proxy":           (1, 4096, 4096, 32, 8,   128),   # op.shape[1], quarter-FLOP proxy
    "prod":            (4, 8192, 8192, 32, 8,   128),   # op.shape[2], production
    # edge cases implied by op.config, not timed
    "toy":             (1, 256,  256,  2,  1,   128),   # smallest smoke shape, GQA=2
    "short_q":         (1, 128,  512,  4,  1,   128),   # sq < BLOCK_M (256) and sq != skv
    "gqa4_batch2":     (2, 256,  256,  8,  2,   128),   # the job's GQA=4, batch>1
    "mha":             (1, 256,  256,  4,  4,   128),   # heads_q == heads_kv
    "unequal_seqlen":  (1, 512,  1024, 8,  2,   128),   # sq != skv -> bottom-right bites
    "unequal_seqlen2": (2, 1024, 2048, 4,  1,   128),   # again, GQA=4, batch>1
    "sq_gt_skv":       (2, 1024, 512,  4,  1,   128),   # sq > skv: NON-CAUSAL ONLY
}
SPEC_SHAPES = ("fast", "proxy", "prod")
EDGE_SHAPES = tuple(s for s in SHAPES if s not in SPEC_SHAPES)

# Bottom-right causal with sq > skv leaves the first sq-skv queries with an EMPTY window:
# lse = -inf and o = NaN in every implementation, the fp32 reference included. Undefined
# input, not a defect, so sq_gt_skv is checked non-causal only. The spec shapes are checked
# causal only (op.config.causal is bottom-right for the whole job; non-causal is covered by
# the edge shapes).
CAUSAL_MODES = {"sq_gt_skv": (False,), "fast": (True,), "proxy": (True,), "prod": (True,)}


def causal_modes(shape):
    return CAUSAL_MODES.get(shape, (True, False))


def make_inputs(name, seed=0, device="cuda"):
    """q, k, v in bf16, from a seeded generator ON THE DEVICE (cpu and cuda differ)."""
    b, sq, skv, hq, hkv, d = SHAPES[name]
    g = torch.Generator(device=device).manual_seed(seed)

    def rnd(*shape):
        return torch.randn(*shape, generator=g, device=device, dtype=torch.float32).to(
            torch.bfloat16)

    return rnd(b, sq, hq, d), rnd(b, skv, hkv, d), rnd(b, skv, hkv, d)


def load_impl(impl_dir):
    """Load an implementation BY PATH and return its `attn_fwd` callable.

    The directory is the identity. It is never turned into a name and looked up: a round's
    code lives at rounds/<n>/op/, whose basename is `op`.
    """
    impl_dir = Path(impl_dir).resolve()
    src = impl_dir / "impl.py"
    if not src.is_file():
        raise FileNotFoundError(f"no impl.py under {impl_dir}")
    # NOT added to sys.path: impl.py loads its own siblings by path.
    name = "op_impl_" + str(abs(hash(str(impl_dir))))
    if name in sys.modules:
        return getattr(sys.modules[name], "attn_fwd")
    spec = importlib.util.spec_from_file_location(name, src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    fn = getattr(mod, "attn_fwd", None)
    if fn is None:
        raise AttributeError(f"{src} defines no `attn_fwd`")
    return fn


def sqnr_db(ref, got):
    ref = ref.float()
    got = got.float()
    num = ref.pow(2).mean()
    den = (ref - got).pow(2).mean().clamp_min(1e-30)
    return float(10.0 * torch.log10(num / den))
