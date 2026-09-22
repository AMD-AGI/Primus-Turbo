#!/usr/bin/env python3
"""Precompute op/eager's fp32 reference tensors once, so validation.py stops recomputing them.

WHY. The precision gate's own reference is the most reliable way this job has found to fault
the card. `op/eager/impl.py` and `common.forward_reference` are fp32 GEMMs, and on gfx1250 in
this image they dispatch to a Tensile kernel (`Cijk_Ailk_Bljk_SB_MT128x64x8_..._WG16_16_1`,
group_seg_size 6144, workgroup 256) that intermittently raises
HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION. It aborted round 3's gate, round 8's gate, and on
2026-09-22 it escalated into an unrecoverable MES state that cost a power cycle. The kernels
under test are not implicated -- none of them is a 256-thread 6144-byte-LDS dispatch.

The reference is deterministic: make_inputs seeds a torch.Generator, and both reference
functions are fixed fp32 arithmetic. So it need only be computed ONCE. Every later round then
compares against stored tensors and issues no reference GEMM at all, which removes that fault
from the per-round risk budget entirely.

    python3 build_refcache.py <job_context/op> [--ref-device cpu] <shape> [<shape> ...]

`--ref-device cpu` computes the reference on the CPU. The INPUTS are still generated on the
GPU whatever this is set to, because `make_inputs` seeds a `torch.Generator(device=...)` and
the same seed yields different numbers on cpu and cuda -- a CPU-seeded reference would be a
reference for inputs the gate never uses. Input generation is pure `randn`: no GEMM, no risk.
Use it for `prod`, whose reference faults the card reliably ("Memory access fault ... Reason:
Page not present", a GCVM_L2 no-retry page fault, 2026-09-22). With 255 cores the CPU path
costs minutes, once.

Writes <op>/refcache/<shape>.pt holding o, lse, dq, dk, dv plus a provenance block. The
provenance records the shape tuple, the seed, and the SHA-256 of eager/impl.py and
ut/common.py; validation.py must refuse a cache whose provenance does not match, because a
stale reference would weaken the gate silently rather than fail it loudly.

One shape per invocation is deliberate: a fault at prod must not cost the fast and proxy
caches that already succeeded.
"""
import hashlib
import sys
from pathlib import Path

argv = sys.argv[1:]
OP = Path(argv.pop(0)).resolve()
REF_DEVICE = "cuda"
if argv and argv[0] == "--ref-device":
    argv.pop(0)
    REF_DEVICE = argv.pop(0)
SHAPES = argv
if not SHAPES:
    raise SystemExit("usage: build_refcache.py <job_context/op> [--ref-device cpu] <shape> [...]")

sys.path.insert(0, str(OP / "ut"))
import torch  # noqa: E402
from common import SHAPES as SHAPE_TABLE, forward_reference, load_impl, make_inputs  # noqa: E402

eager_attn_bwd = load_impl(OP / "eager")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


# The cache holds fp32 arithmetic performed on a particular stack, not a platform-independent
# truth. Record enough that a later run can tell whether it is still looking at its own
# reference: a different card or torch can change the last bits while both SHAs still match.
PROV = {
    "seed": 0,
    "eager_sha": sha(OP / "eager" / "impl.py"),
    "common_sha": sha(OP / "ut" / "common.py"),
    # str(), not the TorchVersion object: torch.load defaults to weights_only=True since
    # PyTorch 2.6 and refuses a pickle containing it, which would make the cache unloadable
    # by the very gate it exists to protect.
    "torch": str(torch.__version__),
    "device": torch.cuda.get_device_properties(0).gcnArchName,
    "ref_device": REF_DEVICE,
}

out_dir = OP / "refcache"
out_dir.mkdir(exist_ok=True)

for name in SHAPES:
    dst = out_dir / f"{name}.pt"
    # Inputs on the GPU always -- see the --ref-device note in the docstring.
    q, k, v, do = make_inputs(name, seed=PROV["seed"], device="cuda")
    if REF_DEVICE != "cuda":
        q, k, v, do = (t.to(REF_DEVICE) for t in (q, k, v, do))
    o, lse = forward_reference(q, k, v, causal=True)
    dq, dk, dv = eager_attn_bwd(do, q, k, v, o, lse, causal=True)
    for tag, t in (("o", o), ("lse", lse), ("dq", dq), ("dk", dk), ("dv", dv)):
        if not bool(torch.isfinite(t).all()):
            raise SystemExit(f"{name}: {tag} is not all finite -- refusing to cache it")
    torch.save({"provenance": dict(PROV, shape=name, dims=tuple(SHAPE_TABLE[name])),
                "o": o.cpu(), "lse": lse.cpu(),
                "dq": dq.cpu(), "dk": dk.cpu(), "dv": dv.cpu()}, dst)
    mb = dst.stat().st_size / 2**20
    print(f"  {name:6s} cached {mb:8.1f} MB  dq{tuple(dq.shape)} dk{tuple(dk.shape)}", flush=True)
    del q, k, v, do, o, lse, dq, dk, dv
    torch.cuda.empty_cache()
print("provenance:", PROV)
