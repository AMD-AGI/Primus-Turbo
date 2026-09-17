import os, sys, time
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
TRACE = "/tmp/toy_trace.txt"
def mark(s):
    with open(TRACE, "a") as f:
        f.write(f"{time.time():.3f} {s}\n"); f.flush(); os.fsync(f.fileno())
open(TRACE, "w").close()
mark("start")
sys.path.insert(0, "/home/lihuzhan/code/aiter-src")
sys.path.insert(0, "/tmp")
import flydsl_024_shim as _shim
_shim.install_expr_shims()
import torch; mark("torch imported")
from aiter.ops.flydsl.fmha_kernels import flydsl_flash_attn_batch_func as F; mark("aiter imported")
import aiter.ops.flydsl.kernels.fmha_gfx1250.fmha_fwd_prefill_a16w16_m32x8  # noqa: F401
_shim.patch_create_llvm_ptr(); mark("shims: " + " | ".join(_shim.installed))

b, s, hq, hkv, d = 1, 256, 2, 1, 128
torch.manual_seed(0)
q = torch.randn(b, s, hq, d, device="cuda", dtype=torch.bfloat16)
k = torch.randn(b, s, hkv, d, device="cuda", dtype=torch.bfloat16)
v = torch.randn(b, s, hkv, d, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize(); mark("tensors allocated")
print("arch:", torch.cuda.get_device_properties(0).gcnArchName, flush=True)

out = torch.full((b, s, hq, d), float("nan"), device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize(); mark("nan-prefilled out")

t0 = time.time()
mark("CALLING flydsl_flash_attn_batch_func (JIT build + first launch)")
r = F(q, k, v, softmax_scale=d**-0.5, causal=True, return_lse=True, out=out)
mark("call returned")
torch.cuda.synchronize(); mark("synchronized")
print(f"first call (build+launch): {time.time()-t0:.2f}s", flush=True)

if r is None:
    print("RESULT: gate declined this configuration (returned None)"); mark("declined"); sys.exit(3)
o, lse = r
print("out:", tuple(o.shape), o.dtype, " lse:", tuple(lse.shape), lse.dtype, flush=True)
fin_o = int(torch.isfinite(o).sum()); fin_l = int(torch.isfinite(lse).sum())
print(f"isfinite coverage  out {fin_o}/{o.numel()}   lse {fin_l}/{lse.numel()}", flush=True)
mark("coverage checked")

# reference
qf, kf, vf = q.float(), k.float(), v.float()
kr = kf.repeat_interleave(hq // hkv, dim=2); vr = vf.repeat_interleave(hq // hkv, dim=2)
sc = (qf.transpose(1,2) @ kr.transpose(1,2).transpose(-1,-2)) * (d**-0.5)
m = torch.triu(torch.ones(s, s, device=sc.device, dtype=torch.bool), 1)
sc = sc.masked_fill(m, float("-inf"))
p = sc.softmax(-1)
ref = (p @ vr.transpose(1,2)).transpose(1,2)
ref_lse = torch.logsumexp(sc, -1)          # [b, hq, s], NATURAL log
def db(a, bb):
    a, bb = a.float(), bb.float()
    return float(10*torch.log10(a.pow(2).mean()/(a-bb).pow(2).mean().clamp_min(1e-30)))
print(f"SQNR out      {db(ref, o):.2f} dB", flush=True)
print(f"SQNR lse (ln) {db(ref_lse, lse):.2f} dB", flush=True)
r_ratio = (lse.float()/ref_lse.clamp_min(1e-6)).mean().item()
print(f"lse ratio got/ln_ref = {r_ratio:.4f}   (1.4427 would mean log2)", flush=True)
mark("done")
