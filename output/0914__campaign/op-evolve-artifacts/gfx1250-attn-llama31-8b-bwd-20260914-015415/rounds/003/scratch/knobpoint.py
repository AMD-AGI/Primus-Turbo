"""One knob setting, one process, one fresh Triton cache. Census + time.

WHY ONE PROCESS PER POINT: get_cache_invalidating_env_vars() returns only
AMDGCN_USE_BUFFER_OPS / TRITON_HIP_USE_{ASYNC_COPY,BLOCK_PINGPONG,IN_THREAD_TRANSPOSE}.
AMDGCN_SCALARIZE_PACKED_FOPS and AMDGCN_ANALYZE_SMALL_TENSOR_RANGE are NOT in the
key, so sweeping them inside one process serves a stale binary and reads as
"this knob does nothing".
"""
import os, sys, glob, re, json, statistics
JC = "/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-attn-llama31-8b-bwd-20260914-015415/job_context"
sys.path.insert(0, os.path.join(JC, "op")); sys.path.insert(0, os.path.join(JC, "op", "ut"))
import torch, benchmark
from shapes import ALL_SHAPES

LABEL = os.environ["LABEL"]
OPDIR = os.environ["OPDIR"]
DOTIME = os.environ.get("DOTIME", "1") == "1"
shape = [s for s in ALL_SHAPES if s["name"] == "b4_s8192_hq32_hkv8_d128"][0]
mod = benchmark.load_impl(OPDIR)
q, k, v, do = benchmark.make_inputs(shape)
out = mod.attention(q, k, v, causal=True)
g = torch.autograd.grad(out, (q, k, v), do, retain_graph=True)
torch.cuda.synchronize()

cache = os.environ["TRITON_CACHE_DIR"]
cen = {}
for f in sorted(glob.glob(os.path.join(cache, "*", "bwd_kernel_causal.amdgcn"))):
    src = open(f).read()
    gg = lambda p: (re.search(p, src) or [None, "?"])[1]
    cen = dict(vgpr=gg(r"\.vgpr_count:\s*(\d+)"), spill=gg(r"\.vgpr_spill_count:\s*(\d+)"),
               sspill=gg(r"\.sgpr_spill_count:\s*(\d+)"), sgpr=gg(r"\.sgpr_count:\s*(\d+)"),
               scratchB=gg(r"\.private_segment_fixed_size:\s*(\d+)"),
               lds=gg(r"\.group_segment_fixed_size:\s*(\d+)"),
               wmma=len(re.findall("v_wmma", src)), msb=len(re.findall("s_set_vgpr_msb", src)),
               sload=len(re.findall("scratch_load", src)), sstore=len(re.findall("scratch_store", src)),
               ds=len(re.findall(r"\bds_", src)), ninstr=len(re.findall(r"^\s+[a-z]", src, re.M)))
ms = None
if DOTIME:
    def bwd(): torch.autograd.grad(out, (q, k, v), do, retain_graph=True)
    benchmark._warm(bwd, 3.0)
    t = benchmark._time_continuous(bwd, 30)
    ms = statistics.median(t)
# reference gradients for a cross-point sanity check
ck = [float(x.double().float().abs().sum()) for x in g]
print("RESULT " + json.dumps(dict(label=LABEL, ms=ms, census=cen, absum=ck)), flush=True)
