"""Op-level check of ONE e2e attention arm at ONE shape, in its own process.

    opcheck.py --arm {turbo,asm,fly} --shape {fast,prod} [--iters 51] [--json out.json]

Inputs are the bwd job's `make_inputs(shape, seed=0)` (copied to arms/ref/common.py) and the
reference is the bwd job's cached fp32 eager reference (refcache/<shape>.pt: o, lse, dq, dk, dv),
read-only. No fp32 GEMM runs on the card here: rebuilding that reference on gfx1250 is the one
thing in this campaign that has reliably faulted the card (refcache_util.py docstring).

What is measured, per arm:
  correctness
    chain  E2EAttention module (the exact object the converter installs) -> o, then
           torch.autograd.grad(o, (q, k, v), do)  -> SQNR of o, dq, dk, dv vs the fp32 reference.
           This is the e2e path, adapter included.
    kbwd   (asm/fly) the bare backward kernel fed the REFERENCE o/lse, i.e. the bwd job's gate.
  timing (CUDA events, median over --iters after warm-up; one arm per process)
    k_fwd / k_bwd   the bare kernel entry points (asm: fmha_fwd_with_sink_asm / asm_backward
                    WITHOUT the GQA sum; fly: attn_fwd / attn_bwd; turbo: Triton dense_forward /
                    dense_backward, i.e. the product kernels without flash_attn_func)
    m_fwd / m_bwd   through the module + autograd, exactly as e2e calls it
    parts           adapter pieces timed alone (asm: dq_acc.zero_, GQA sum)
    overhead        m_* - k_*  (what the adapter / product wrapper costs on top of the kernel)
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent                  # e2e/attn_backends
E2E = HERE.parent
JOB_OP = Path("/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/"
              "gfx1250-flydsl-attn-bwd-20260917-115934/job_context/op")

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True, choices=["turbo", "asm", "fly"])
ap.add_argument("--shape", default="prod", choices=["fast", "proxy", "prod"])
ap.add_argument("--iters", type=int, default=51)
ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--json")
args = ap.parse_args()

os.environ["E2E_ATTN"] = args.arm            # the ONE switch; the shim reads it at import
sys.path.insert(0, str(HERE))
if args.arm != "turbo":
    sys.path.insert(0, "/home/lihuzhan/.local/flydsl0341")

import torch  # noqa: E402

t_start = time.time()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


common = load("e2e_ref_common", E2E / "arms" / "ref" / "common.py")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]  # noqa: E731
assert sha(E2E / "arms/ref/common.py") == sha(JOB_OP / "ut/common.py")
assert sha(E2E / "arms/ref/eager_impl.py") == sha(JOB_OP / "eager/impl.py")

res = {"arm": args.arm, "shape": args.shape, "dims": common.SHAPES[args.shape],
       "iters": args.iters, "env": {k: os.environ.get(k) for k in (
           "TORCH_BLAS_PREFER_HIPBLASLT", "HIPBLASLT_TENSILE_LIBPATH", "PYTHONPATH")}}
print("ARM", args.arm, "SHAPE", args.shape, common.SHAPES[args.shape], flush=True)
print("device", torch.cuda.get_device_properties(0).gcnArchName, flush=True)

q, k, v, do = common.make_inputs(args.shape, seed=0)
blob = torch.load(JOB_OP / "refcache" / f"{args.shape}.pt", map_location="cpu")
prov = blob["provenance"]
assert prov["seed"] == 0 and tuple(prov["dims"]) == tuple(common.SHAPES[args.shape])
assert prov["eager_sha"] == sha(JOB_OP / "eager/impl.py"), prov
# The cache was built against common.py 988c14caed5d9a80 (0917 copy); the job's current file
# (d441e55aba5c0b89, 09-25) differs ONLY in the "toy" SHAPES row (64 -> 128), so make_inputs
# and every other shape are unchanged and the cached reference still describes these inputs.
assert prov["common_sha"] in (sha(JOB_OP / "ut/common.py"), "988c14caed5d9a80"), prov
res["ref_provenance"] = {k: str(v_) for k, v_ in prov.items()}

import e2e_attn  # noqa: E402
from e2e_attn import E2EAttention  # noqa: E402

if args.arm == "turbo":
    import primus_turbo  # noqa: E402  -- the shim, redirected to wt-bakeoff
    from primus_turbo.pytorch.kernels.attention.attention_triton_impl import (  # noqa: E402
        dense_backward, dense_forward)
    import flydsl
    res["versions"] = f"primus_turbo {primus_turbo.__file__}; flydsl {flydsl.__version__}"
    kfwd = lambda q, k, v, s: dense_forward(q, k, v, s, True)  # noqa: E731
    kbwd = lambda do, q, k, v, o, lse, s: dense_backward(do, q, k, v, o, lse, s, True)[:3]  # noqa
else:
    from e2e_attn import arms  # noqa: E402
    kfwd = arms.get_fwd(args.arm)
    if args.arm == "asm":
        _ab = arms.asm_bwd()

        def kbwd(do, q, k, v, o, lse, s):     # kernel only: no GQA sum
            return _ab.mod.asm_backward(q, k, v, o, do, lse, softmax_scale=float(s),
                                        hip=_ab.hip, dkdv_heads="q", causal=True,
                                        scratch=_ab.get_scratch(q))
        full_bwd = _ab
    else:
        kbwd = arms.get_bwd(args.arm)
        full_bwd = kbwd
    res["versions"] = arms.check_versions()
    assert not [m for m in sys.modules if m.startswith("primus_turbo")], "real primus_turbo loaded"
print("versions:", res["versions"], flush=True)
print("BLAS env:", res["env"]["TORCH_BLAS_PREFER_HIPBLASLT"], res["env"]["HIPBLASLT_TENSILE_LIBPATH"],
      flush=True)

b, sq, hq, d = q.shape
scale = 1.0 / d ** 0.5


def sqnr(ref, got):
    fin = int(torch.isfinite(got).sum())
    if fin != got.numel():
        return f"UNCOVERED {fin}/{got.numel()}"
    r = ref.to(got.device) if ref.device != got.device else ref
    return round(common.sqnr_db(r, got), 2)


def poison():
    blocks = [torch.full((n,), float("nan"), device="cuda", dtype=torch.float32)
              for n in (1 << 28, 1 << 26, 1 << 24, 1 << 22, 1 << 20)]
    del blocks
    torch.cuda.synchronize()


# ------------------------------------------------------------------ correctness: chain
mod = E2EAttention(causal=True)
poison()
qg, kg, vg = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
o = mod(qg, kg, vg)
dq, dk, dv = torch.autograd.grad(o, (qg, kg, vg), do)
torch.cuda.synchronize()
chain = {"o": sqnr(blob["o"], o), "dq": sqnr(blob["dq"], dq),
         "dk": sqnr(blob["dk"], dk), "dv": sqnr(blob["dv"], dv)}
res["sqnr_chain_db"] = chain
res["adapter_copies"] = dict(e2e_attn.STATS)
print("SQNR chain (module fwd+autograd bwd vs fp32 ref):", chain, "adapter copies:",
      e2e_attn.STATS, flush=True)
del qg, kg, vg, o, dq, dk, dv

# ------------------------------------------------------------------ correctness: kernel bwd
if args.arm != "turbo":
    o_ref = blob["o"].to("cuda").to(torch.bfloat16).contiguous()
    lse_ref = blob["lse"].to("cuda").float().contiguous()
    o_k, lse_k = kfwd(q, k, v, scale)
    res["sqnr_kfwd_db"] = {"o": sqnr(blob["o"], o_k), "lse": sqnr(blob["lse"], lse_k),
                           "lse_shape": list(lse_k.shape)}
    poison()
    gq, gk, gv = full_bwd(do, q, k, v, o_ref, lse_ref, scale)
    torch.cuda.synchronize()
    res["sqnr_kbwd_refolse_db"] = {"dq": sqnr(blob["dq"], gq), "dk": sqnr(blob["dk"], gk),
                                   "dv": sqnr(blob["dv"], gv)}
    print("SQNR kernel fwd:", res["sqnr_kfwd_db"], flush=True)
    print("SQNR kernel bwd on reference o/lse (job gate form):", res["sqnr_kbwd_refolse_db"],
          flush=True)
    del o_ref, lse_ref, o_k, lse_k, gq, gk, gv
del blob
torch.cuda.empty_cache()

# ------------------------------------------------------------------ timing
# Everything is timed INTERLEAVED in one loop, each piece between its own event pair, and the
# order within an iteration alternates (kernel-first / module-first). A first version timed
# k_fwd as its own back-to-back block and got ASM fwd 1.438 ms vs 1.138 through the module:
# a sustained burst of one kernel on this VR-throttled card runs at a lower clock than the
# fwd/bwd mix, so separate blocks measure duty cycle, not adapter cost.
o_k, lse_k = kfwd(q, k, v, scale)
qg, kg, vg = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
if args.arm == "asm":
    s = _ab.get_scratch(q)
    hkv = k.shape[2]
    res["asm_resident_scratch_mib"] = sum(t.numel() * t.element_size() for t in s.values()) >> 20


def ev():
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    return e


def run_kernel(acc):
    e0 = ev(); kfwd(q, k, v, scale); e1 = ev()
    kbwd(do, q, k, v, o_k, lse_k, scale); e2 = ev()
    acc.append(("k_fwd", e0, e1)); acc.append(("k_bwd", e1, e2))


def run_module(acc):
    e0 = ev(); o = mod(qg, kg, vg); e1 = ev()
    g = torch.autograd.grad(o, (qg, kg, vg), do); e2 = ev()
    acc.append(("m_fwd", e0, e1)); acc.append(("m_bwd", e1, e2))
    del o, g


def run_parts(acc):
    if args.arm != "asm":
        return
    e0 = ev(); s["dq_acc"].zero_(); e1 = ev()
    s["dk"].view(b, sq, hkv, hq // hkv, d).sum(dim=3).to(k.dtype)
    s["dv"].view(b, sq, hkv, hq // hkv, d).sum(dim=3).to(v.dtype); e2 = ev()
    acc.append(("part_dq_acc_zero", e0, e1)); acc.append(("part_gqa_sum_dk_dv", e1, e2))


samples = {}
for i in range(args.warmup + args.iters):
    acc = []
    order = (run_kernel, run_module, run_parts) if i % 2 == 0 else (run_module, run_kernel, run_parts)
    for f in order:
        f(acc)
    torch.cuda.synchronize()
    if i >= args.warmup:
        for key, a, bb in acc:
            samples.setdefault(key, []).append(a.elapsed_time(bb))
T = {}
for key, ts in samples.items():
    ts.sort()
    T[key] = {"med": round(ts[len(ts) // 2], 4), "p10": round(ts[len(ts) // 10], 4),
              "p90": round(ts[(9 * len(ts)) // 10], 4)}

T["overhead_fwd_ms"] = round(T["m_fwd"]["med"] - T["k_fwd"]["med"], 4)
T["overhead_bwd_ms"] = round(T["m_bwd"]["med"] - T["k_bwd"]["med"], 4)
res["timing_ms"] = T
res["max_mem_alloc_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
res["wall_s"] = round(time.time() - t_start, 1)
for key, val in T.items():
    print(f"  {key:22s} {val}", flush=True)
print("adapter copies after timing:", e2e_attn.STATS, flush=True)
if args.json:
    Path(args.json).write_text(json.dumps(res, indent=1))
print("DONE", json.dumps({k: res[k] for k in ("arm", "shape", "sqnr_chain_db")}), flush=True)
