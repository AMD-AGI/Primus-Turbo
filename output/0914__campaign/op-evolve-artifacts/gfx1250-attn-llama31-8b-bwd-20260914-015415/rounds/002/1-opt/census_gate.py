"""Route row 1 -- r2.i3.g10. The spill gate.

    python census_gate.py <impl_dir> [TILE_N1]

Compiles the candidate once (no timing loop, no palindromic session, no idle
device needed) and prints what the .amdgcn says about register pressure. On
this kernel at this tile, time tracks `vgpr_spill_count` and nothing else, so
this is a prediction the timing run can falsify -- for free, in seconds.

Verdict is against the incumbent's 266:
    >= 266  ->  DISCARD, do not spend a timing run on it
    <  266  ->  MEASURE
"""
import os, sys, glob, re, json

INCUMBENT_SPILLS = 266
JC = "/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-attn-llama31-8b-bwd-20260914-015415/job_context"
sys.path.insert(0, os.path.join(JC, "op")); sys.path.insert(0, os.path.join(JC, "op", "ut"))
import torch, benchmark
from shapes import ALL_SHAPES

impl_dir = sys.argv[1]
cache = os.environ["TRITON_CACHE_DIR"]
shape = [s for s in ALL_SHAPES if s["name"] == "b4_s8192_hq32_hkv8_d128"][0]
mod = benchmark.load_impl(impl_dir)

if len(sys.argv) > 2:                      # optional: force the dk/dv tile
    n1 = int(sys.argv[2])
    import importlib
    FB = sys.modules[[m for m in sys.modules if m.endswith("attention_fused_bwd_impl")][0]]
    FB.fused_backward_tile = lambda _s, _n1=n1: _n1
    KM = sys.modules[[m for m in sys.modules if m.endswith("fused_mha_bwd_kernel")][0]]
    KM._check_block_invariant = lambda cfg: None
    print(f"[gate] forcing BLOCK_N1 = BLOCK_M2 = {n1}")

q, k, v, do = benchmark.make_inputs(shape)
out = mod.attention(q, k, v, causal=True)
torch.autograd.grad(out, (q, k, v), do, retain_graph=True)
torch.cuda.synchronize()

for f in sorted(glob.glob(os.path.join(cache, "*", "bwd_kernel_causal.amdgcn"))):
    src = open(f).read()
    g = lambda p: (re.search(p, src) or [None, "?"])[1]
    spills = int(g(r"\.vgpr_spill_count:\s*(\d+)"))
    print(f"--- {os.path.basename(os.path.dirname(f))}")
    print(f"    vgpr_count={g(r'\.vgpr_count:\s*(\d+)')} "
          f"vgpr_spill_count={spills} sgpr_spill_count={g(r'\.sgpr_spill_count:\s*(\d+)')} "
          f"scratch_B_per_lane={g(r'\.private_segment_fixed_size:\s*(\d+)')} "
          f"lds={g(r'\.group_segment_fixed_size:\s*(\d+)')}")
    for ins in ("v_wmma", "s_set_vgpr_msb", "scratch_load", "scratch_store",
                r"\bds_", "v_pk_mul_f32", r"v_add_nc_u32|v_add3_u32|v_dual_add_nc_u32"):
        print(f"      {ins:<44} {len(re.findall(ins, src))}")
    print(f"    VERDICT vs incumbent {INCUMBENT_SPILLS}: "
          f"{'DISCARD (>= incumbent, already known not faster)' if spills >= INCUMBENT_SPILLS else 'MEASURE (a falsifiable prediction)'}")
