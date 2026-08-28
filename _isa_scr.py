#!/usr/bin/env python3
"""ISA screen for one a16-body arm: registers, spill, and an OPCODE-PRESENCE check.

`_isa_body.py` answers "does it allocate"; it does not answer "does it still compute".
r19's `block_q=16` built at spill 0 and timed -23% with two of the five GEMMs silently
dead-code-eliminated behind a `range_constexpr` bound of 0. `v_exp_f32` and the MFMA count
are the cheapest tell, so every geometry screen prints them here.

usage: _isa_scr.py <arm>            arm = `_probe_bwdcfg` arm string, e.g. kw:k_reg=1+kw:mfma_tie=3
"""
import glob
import os
import re
import sys

ARM = sys.argv[1] if len(sys.argv) > 1 else "base"
# The dump dir must carry every env knob too: two arms with the same NAME but different
# PT_*/BQ/BKV env share one directory and a parallel batch then reads its neighbour's dump.
_tag = ARM + "".join(f"_{k}{v}" for k, v in sorted(os.environ.items())
                     if k.startswith(("PT_", "BQ", "BKV", "QSP", "KW")))
DUMP = "/tmp/isa_scr_" + re.sub(r"[^A-Za-z0-9]+", "_", _tag)
os.environ.update(FLYDSL_DUMP_IR="1", FLYDSL_DUMP_DIR=DUMP, FLYDSL_RUNTIME_ENABLE_CACHE="0")
os.system(f"rm -rf {DUMP}")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

import _probe_bwdcfg as P  # noqa: E402
import primus_turbo.flydsl.attention.flash_attn_bwd as M  # noqa: E402

_seen = {}
_b = M.build_flash_attn_bwd_dkdv_module


def _spy(**kw):
    _seen.update(kw)
    return _b(**kw)


M.build_flash_attn_bwd_dkdv_module = _spy
for _k, _v in (("BKV", "_A16_BLOCK_KV"), ("BQ", "_A16_BLOCK_Q"), ("QSP", "_A16_Q_SPLIT"), ("QSP8", "_A16_Q_SPLIT_G8")):
    if os.environ.get(_k):
        setattr(M, _v, int(os.environ[_k]))
P.patch_all(ARM)
from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (  # noqa: E402
    flash_attn_sbhd_flydsl_backward_impl as fb,
)
from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (  # noqa: E402
    flash_attn_sbhd_flydsl_forward_impl as ff,
)

B, S, HQ, HKV, D = 1, 8192, 64, 8, 128
mk = lambda H: torch.randn(S, B, H, D, device="cuda", dtype=torch.bfloat16)
q, k, v = mk(HQ), mk(HKV), mk(HKV)
do = torch.randn_like(q)
o, lse = ff(q, k, v, causal=True, window_size=(-1, -1), return_lse=True)
fb(do, q, k, v, o, lse.view(B, S, HQ).permute(0, 2, 1), causal=True, window_size=(-1, -1))
torch.cuda.synchronize()
KEYS = ("block_kv", "block_q", "kv_halves", "q_split", "g3_kreg", "k_reg", "q_pref", "g3s_pack",
        "g3_defer", "flat_wg", "mfma_tie", "mfma_tie_cons", "g3_dbat", "g3_st_n")
for p in sorted(glob.glob(os.path.join(DUMP, "**", "21_final_isa.s"), recursive=True)):
    if "dkdv" not in os.path.basename(os.path.dirname(p)):
        continue
    b = open(p).read()
    g = lambda n: int(re.search(rf"{n}: *(\d+)", b).group(1)) if re.search(rf"{n}: *(\d+)", b) else 0
    print("  resolved:", {k: _seen.get(k) for k in KEYS})
    print("ARM %-40s vgpr=%3d agpr=%3d spill=%5dB scratch=%4d lds=%6d instr=%5d "
          "mfma=%4d exp=%4d bar=%4d dsrd=%4d accmov=%4d accrd=%4d accwr=%4d"
          % (ARM, g("vgpr_count"), g("agpr_count"), g("private_segment_fixed_size"),
             b.count("scratch_"), g(r"\.amdhsa_group_segment_fixed_size"),
             len(re.findall(r"^\s+[a-z]", b, re.M)),
             b.count("v_mfma_f32_16x16x32_bf16"), b.count("v_exp_f32"),
             b.count("s_barrier"), len(re.findall(r"\bds_read", b)),
             b.count("v_accvgpr_mov_b32"), b.count("v_accvgpr_read"), b.count("v_accvgpr_write")))
