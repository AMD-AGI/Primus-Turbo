"""Compile-only the gfx1250 forward kernel for the prod shape and dump IR/ISA.

Usage: python3 compile_fwd_isa.py <impl_dir> <dump_dir> [prod|thd|win_sink]
  prod     bshd causal lse           (default; the production call)
  thd      thd  causal lse           (varlen entry + kv_len==0 zero-fill path)
  win_sink bshd window(128,0) lse sink (mask_left + sink paths)
Prod shape: b4 s8192 hq32 hkv8 d128 causal bf16, return_lse=True (what impl.py calls).
No GPU work: COMPILE_ONLY=1 must be set; tensors are tiny CPU placeholders (only their
pointers enter the kernel ABI), strides/lengths are the prod-shape integers.
"""
import importlib
import importlib.util as ilu
import os
import pathlib
import sys

assert os.environ.get("COMPILE_ONLY") == "1", "refusing to run without COMPILE_ONLY=1"
sys.path.insert(0, "/home/lihuzhan/.local/flydsl0341")

impl_dir = pathlib.Path(sys.argv[1]).resolve()
os.environ["FLYDSL_DUMP_DIR"] = sys.argv[2]

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

variant = sys.argv[3] if len(sys.argv) > 3 else "prod"
SHAPES = {"prod": (4, 8192, 32, 8, 128), "fast": (1, 1024, 8, 2, 128), "d192": (4, 8192, 32, 8, 192), "d256": (4, 8192, 32, 8, 256)}
shape = sys.argv[4] if len(sys.argv) > 4 else "prod"
B, S, HQ, HKV, D = SHAPES[shape]
print("shape", shape, B, S, HQ, HKV, D)
gqa = HQ // HKV
layout = "thd" if variant == "thd" else "bshd"
mask_left = variant == "win_sink"
mask_right, ret_lse = (variant != "noncausal"), True
has_sink = variant == "win_sink"
ensure = kern._ensure_thd_kernel if layout == "thd" else kern._ensure_bshd_kernel
ensure(mask_left, mask_right, ret_lse, has_sink, gqa, qk_hdim=D, dtype_str="bf16")
fn = kern._launch_fns[(layout, mask_left, mask_right, ret_lse, has_sink, gqa, D, "bf16")]

ph = torch.empty(64, dtype=torch.bfloat16)  # pointer placeholder (CPU)
lse_ph = torch.empty(64, dtype=torch.float32)
# contiguous BSHD strides in elements
s_seq, s_head = HQ * D, D
k_seq, k_head = HKV * D, D
lse_seq, lse_head, lse_batch = 1, S, HQ * S
win_left = 128 if mask_left else 0
if layout == "bshd":
    args = (ph, ph, ph, ph, lse_ph, ph, 1.0 / D ** 0.5,
            s_seq, k_seq, k_seq, s_seq, s_head, k_head, k_head, s_head,
            lse_seq, lse_head, lse_batch, win_left, 0, S, S, HKV, B, None)
else:
    cu = torch.zeros(B + 1, dtype=torch.int32)
    args = (ph, ph, ph, ph, lse_ph, ph, cu, cu, 1.0 / D ** 0.5,
            s_seq, k_seq, k_seq, s_seq, s_head, k_head, k_head, s_head,
            HQ, 1, win_left, 0, S, S, HKV, B, None)
flyc.compile(fn, *args)
print("COMPILE_OK")
