#!/usr/bin/env python3
"""The fused body's register allocation under _probe_bwdcfg's arms.

BLOCK_KV is the one lever both the deterministic fold and an atomic dQ share (both scale
with S^2/(2*BLOCK_KV)), and what shuts it is the dK/dV accumulator: BLOCK_KV*D*2/256 dwords,
128 at bkv=128 and 256 at bkv=256, against a 512-dword pool the body already fills to 462.
This dumps vgpr/agpr/spill/scratch per arm so donors can be screened without a bench.

usage: _isa_body.py <arm>[+<arm>...]
"""
import glob, os, re, sys

ARM = sys.argv[1] if len(sys.argv) > 1 else "base"
DUMP = "/tmp/isa_body_" + re.sub(r"[^A-Za-z0-9]+", "_", ARM)
os.environ.update(FLYDSL_DUMP_IR="1", FLYDSL_DUMP_DIR=DUMP, FLYDSL_RUNTIME_ENABLE_CACHE="0")
os.system(f"rm -rf {DUMP}")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import _probe_bwdcfg as P
import primus_turbo.flydsl.attention.flash_attn_bwd as M

# Every arm prints the geometry the builder actually RESOLVED, not the one asked for.
# Two whole classes of wrong reading come from skipping this: a `kw:` override the host
# already passes is INERT and reads as a tested donor, and an arm that quietly moves the
# kernel onto another path reads as a huge win (bkv256+bq16 came back at 90 vgpr).
_seen = {}
_b = M.build_flash_attn_bwd_dkdv_module
def _spy(**kw):
    _seen.update({k: kw.get(k) for k in
                  ("block_kv", "block_q", "kv_halves", "head_dim", "num_heads_q",
                   "num_kv_heads", "q_split", "g3_kreg", "k_reg", "q_pref", "g3_defer",
                   "flat_wg", "waves_per_eu")})
    return _b(**kw)
M.build_flash_attn_bwd_dkdv_module = _spy

P.patch_all(ARM)
from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    flash_attn_sbhd_flydsl_backward_impl as fb, flash_attn_sbhd_flydsl_forward_impl as ff)

B, S, HQ, HKV, D = 2, 8192, 64, 8, 128
mk = lambda H: torch.randn(S, B, H, D, device="cuda", dtype=torch.bfloat16)
q, k, v = mk(HQ), mk(HKV), mk(HKV)
do = torch.randn_like(q)
o, lse = ff(q, k, v, causal=True, window_size=(-1, -1), return_lse=True)
fb(do, q, k, v, o, lse.view(B, S, HQ).permute(0, 2, 1), causal=True, window_size=(-1, -1))
torch.cuda.synchronize()
for p in sorted(glob.glob(os.path.join(DUMP, "**", "21_final_isa.s"), recursive=True)):
    kn = os.path.basename(os.path.dirname(p))
    if "dkdv" not in kn:
        continue
    b = open(p).read()
    g = lambda n: int(re.search(rf"{n}: *(\d+)", b).group(1)) if re.search(rf"{n}: *(\d+)", b) else 0
    print("   resolved:", {k: v for k, v in _seen.items() if v is not None})
    print("ARM %-34s vgpr=%3d agpr=%3d unified=%3d spill=%4dB scratch=%4d lds=%6d instr=%d"
          % (ARM, g("vgpr_count"), g("agpr_count"), g("vgpr_count"),
             g("private_segment_fixed_size"), b.count("scratch_"),
             g(r"\.amdhsa_group_segment_fixed_size"), len(re.findall(r"^\s+[a-z]", b, re.M))))
