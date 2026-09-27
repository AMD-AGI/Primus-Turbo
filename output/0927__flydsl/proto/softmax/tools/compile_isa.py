"""Compile-only the gfx1250 forward kernel and dump IR/ISA (no GPU work).

Usage: python3 compile_isa.py <impl_dir> <dump_dir> <prod|fast|win_sink> [FLAG=0|1 ...]
  prod     b4 s8192 hq32 hkv8 d128 bf16 bshd causal lse
  fast     b1 s1024 hq8  hkv2 d128 bf16 bshd causal lse
  win_sink prod shape, window(128,0), lse, sink (mask_left + sink paths)
FLAG=val overrides a module global of fmha_fwd_prefill_a16w16_m32x8 before the build
(e.g. SOFTMAX_PK_EXP=0 ENABLE_DEFER_RESCALE=1). Derived from
output/0925__flydsl/api-audit/compile_fwd_isa.py.
"""
import importlib
import importlib.util as ilu
import os
import pathlib
import sys

assert os.environ.get("COMPILE_ONLY") == "1", "refusing to run without COMPILE_ONLY=1"
assert os.environ.get("HIP_VISIBLE_DEVICES", "x") == "", "hide the GPU"
sys.path.insert(0, "/home/lihuzhan/.local/flydsl0341")

impl_dir = pathlib.Path(sys.argv[1]).resolve()
os.environ["FLYDSL_DUMP_DIR"] = sys.argv[2]
variant = sys.argv[3]
overrides = dict(a.split("=", 1) for a in sys.argv[4:])

import torch  # noqa: E402
import flydsl  # noqa: E402
import flydsl.compiler as flyc  # noqa: E402

assert flydsl.__version__ == "0.3.4.1", flydsl.__file__

name = f"flydsl_fwd__proto_{abs(hash(str(impl_dir)))}"
d = impl_dir / "flydsl_fwd"
spec = ilu.spec_from_file_location(name, d / "__init__.py", submodule_search_locations=[str(d)])
pkg = ilu.module_from_spec(spec)
sys.modules[name] = pkg
spec.loader.exec_module(pkg)
kern = importlib.import_module(f"{name}.fmha_fwd_prefill_a16w16_m32x8")
print("kernel module:", kern.__file__)
for k, v in overrides.items():
    assert hasattr(kern, k), k
    setattr(kern, k, bool(int(v)))
print("flags:", {k: getattr(kern, k, None) for k in (
    "SOFTMAX_PK_EXP", "SOFTMAX_LANE_ROWSUM", "SOFTMAX_BRANCHFREE_RESCALE", "SOFTMAX_PK_ROWSUM", "ENABLE_DEFER_RESCALE")})

if variant == "fast":
    B, S, HQ, HKV, D = 1, 1024, 8, 2, 128
else:
    B, S, HQ, HKV, D = 4, 8192, 32, 8, 128
gqa = HQ // HKV
mask_left = variant == "win_sink"
mask_right, ret_lse = True, True
has_sink = variant == "win_sink"
kern._ensure_bshd_kernel(mask_left, mask_right, ret_lse, has_sink, gqa, qk_hdim=D, dtype_str="bf16")
fn = kern._launch_fns[("bshd", mask_left, mask_right, ret_lse, has_sink, gqa, D, "bf16")]

ph = torch.empty(64, dtype=torch.bfloat16)
lse_ph = torch.empty(64, dtype=torch.float32)
s_seq, s_head = HQ * D, D
k_seq, k_head = HKV * D, D
lse_seq, lse_head, lse_batch = 1, S, HQ * S
win_left = 128 if mask_left else 0
args = (ph, ph, ph, ph, lse_ph, ph, 1.0 / D ** 0.5,
        s_seq, k_seq, k_seq, s_seq, s_head, k_head, k_head, s_head,
        lse_seq, lse_head, lse_batch, win_left, 0, S, S, HKV, B, None)
flyc.compile(fn, *args)
print("COMPILE_OK")
