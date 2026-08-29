#!/usr/bin/env python3
"""ISA anatomy of the D=64 a16 dkdv kernel (the D64 twin of _isa_body.py, which is D128-only).

usage: _isa64.py <tag> [kw=v,...]
Dumps flash_attn_bwd_dkdv_kernel_*/21_final_isa.s and prints md5 + register/instruction
counts, so an arm can be proven non-inert before it is timed.
"""
import glob, hashlib, os, re, sys

TAG = sys.argv[1] if len(sys.argv) > 1 else "base"
KW = {}
if len(sys.argv) > 2 and sys.argv[2]:
    for p in sys.argv[2].split(","):
        k, v = p.split("=")
        KW[k] = None if v == "None" else (True if v == "True" else (False if v == "False" else int(v)))
DUMP = "/tmp/isa64_" + re.sub(r"[^A-Za-z0-9]+", "_", TAG)
os.environ.update(FLYDSL_DUMP_IR="1", FLYDSL_DUMP_DIR=DUMP, FLYDSL_RUNTIME_ENABLE_CACHE="0")
os.system(f"rm -rf {DUMP}")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import primus_turbo.flydsl.attention.flash_attn_bwd as M

_seen = {}
_b = M.build_flash_attn_bwd_dkdv_module


def _spy(**kw):
    kw.update(KW)
    _seen.update({k: kw.get(k) for k in
                  ("block_kv", "block_q", "kv_halves", "head_dim", "q_split", "g3_kreg",
                   "k_reg", "q_pref", "g3_defer", "mfma_tie", "mfma_tie_cons", "agpr",
                   "g3d", "wsq_a16")})
    return _b(**kw)


M.build_flash_attn_bwd_dkdv_module = _spy

from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    flash_attn_sbhd_flydsl_backward_impl as fb, flash_attn_sbhd_flydsl_forward_impl as ff)

B, S, HQ, HKV, D = 1, 8192, 64, 8, 64
mk = lambda H: torch.randn(S, B, H, D, device="cuda", dtype=torch.bfloat16)
q, k, v = mk(HQ), mk(HKV), mk(HKV)
do = torch.randn_like(q)
o, lse = ff(q, k, v, causal=True, window_size=(-1, -1), return_lse=True)
fb(do, q, k, v, o, lse.view(B, S, HQ).permute(0, 2, 1), causal=True, window_size=(-1, -1))
torch.cuda.synchronize()

FAM = ("v_mfma", "v_accvgpr_read", "v_accvgpr_write", "v_accvgpr_mov", "v_cvt_pk_bf16_f32",
       "ds_read", "ds_write", "s_waitcnt", "buffer_load", "buffer_store", "buffer_atomic",
       "v_exp_f32", "s_nop", "s_barrier")
for p in sorted(glob.glob(os.path.join(DUMP, "**", "21_final_isa.s"), recursive=True)):
    kn = os.path.basename(os.path.dirname(p))
    if "dkdv" not in kn:
        continue
    b = open(p).read()
    g = lambda n: int(re.search(rf"{n}: *(\d+)", b).group(1)) if re.search(rf"{n}: *(\d+)", b) else 0
    print("resolved:", {k: v for k, v in _seen.items() if v is not None})
    print("%s %s md5=%s vgpr=%d agpr=%d spill=%dB lds=%d instr=%d"
          % (TAG, kn, hashlib.md5(b.encode()).hexdigest()[:12], g("vgpr_count"), g("agpr_count"),
             g("private_segment_fixed_size"), g(r"\.amdhsa_group_segment_fixed_size"),
             len(re.findall(r"^\s+[a-z]", b, re.M))))
    print("   " + "  ".join("%s=%d" % (f, len(re.findall(r"\b" + f, b))) for f in FAM))
