"""Compile-only the gfx1250 forward kernel and dump IR/ISA (adapted from api-audit/compile_fwd_isa.py).

Usage: python3 compile_isa.py <impl_dir> <dump_dir> <prod|fast> [KNOB=VAL ...]
KNOB=VAL overrides module constants of fmha_fwd_prefill_a16w16_m32x8 before tracing
(e.g. KV_UNROLL=1 N_KV_PP=2). Never launches: COMPILE_ONLY=1, CPU placeholder tensors.
prod: b4 s8192 hq32 hkv8 d128 bf16 causal lse. fast: b1 s1024 hq8 hkv2 (same gqa=4 ->
same compile-time specialization; only runtime ints differ).
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
os.environ["FLYDSL_DUMP_IR"] = "1"
os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"
shape = sys.argv[3]
knobs = dict(a.split("=", 1) for a in sys.argv[4:])

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
for k, v in knobs.items():
    assert hasattr(kern, k), k
    setattr(kern, k, type(getattr(kern, k))(int(v)) if not isinstance(getattr(kern, k), bool) else v == "1")
print("kernel module:", kern.__file__)
print("knobs:", {k: getattr(kern, k) for k in ("N_KV_PP", "MIN_KV_BLK_BYTES", "KV_UNROLL") if hasattr(kern, k)})

B, S, HQ, HKV = {"prod": (4, 8192, 32, 8), "fast": (1, 1024, 8, 2)}[shape]
D = 128
gqa = HQ // HKV
mask_left, mask_right, ret_lse, has_sink = False, True, True, False
kern._ensure_bshd_kernel(mask_left, mask_right, ret_lse, has_sink, gqa, qk_hdim=D, dtype_str="bf16")
fn = kern._launch_fns[("bshd", mask_left, mask_right, ret_lse, has_sink, gqa, D, "bf16")]

ph = torch.empty(64, dtype=torch.bfloat16)
lse_ph = torch.empty(64, dtype=torch.float32)
s_seq, s_head = HQ * D, D
k_seq, k_head = HKV * D, D
lse_seq, lse_head, lse_batch = 1, S, HQ * S
args = (ph, ph, ph, ph, lse_ph, ph, 1.0 / D ** 0.5,
        s_seq, k_seq, k_seq, s_seq, s_head, k_head, k_head, s_head,
        lse_seq, lse_head, lse_batch, 0, 0, S, S, HKV, B, None)
flyc.compile(fn, *args)
print("COMPILE_OK")
