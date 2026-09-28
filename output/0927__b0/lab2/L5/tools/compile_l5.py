"""COMPILE_ONLY build of the fwd kernel for one config; dumps IR/ISA. No GPU work.

Usage: python3 compile_l5.py <impl_dir> <dump_dir> <shape> <cfg> [l5]
  shape in fast|proxy|prod|gqa1|gqa2  (gqa1 = b1 s4096 hq32 hkv32; gqa2 = toy b1 s256 hq2 hkv1)
  cfg   in causal|nc
  l5    optional: build the bshd kernel with q_origin_align=True via the gated key (gate arm)
Adapted from output/0927__flydsl/proto/pairing/compile_fwd_isa.py.
"""
import importlib, importlib.util as ilu, os, pathlib, sys

assert os.environ.get("COMPILE_ONLY") == "1", "refusing to run without COMPILE_ONLY=1"
assert os.environ.get("FLYDSL_GPU_ARCH") == "gfx1250" and os.environ.get("ARCH") == "gfx1250"
sys.path.insert(0, "/home/lihuzhan/.local/flydsl0341")
impl_dir = pathlib.Path(sys.argv[1]).resolve()
os.environ["FLYDSL_DUMP_DIR"] = sys.argv[2]
os.environ["FLYDSL_DUMP_IR"] = "1"
import torch  # noqa: E402
import flydsl  # noqa: E402
import flydsl.compiler as flyc  # noqa: E402
assert flydsl.__version__ == "0.3.4.1", flydsl.__file__

name = f"flydsl_fwd__l5_{abs(hash(str(impl_dir)))}"
d = impl_dir / "flydsl_fwd"
spec = ilu.spec_from_file_location(name, d / "__init__.py", submodule_search_locations=[str(d)])
pkg = ilu.module_from_spec(spec); sys.modules[name] = pkg; spec.loader.exec_module(pkg)
kern = importlib.import_module(f"{name}.fmha_fwd_prefill_a16w16_m32x8")
print("kernel module:", kern.__file__)

shape, cfg = sys.argv[3], sys.argv[4]
l5 = len(sys.argv) > 5 and sys.argv[5] == "l5"
B, S, HQ, HKV, D = {"prod": (4, 8192, 32, 8, 128), "proxy": (1, 4096, 32, 8, 128),
                    "fast": (1, 1024, 8, 2, 128), "gqa1": (1, 4096, 32, 32, 128), "gqa2": (1, 256, 2, 1, 128)}[shape]
gqa = HQ // HKV
mask_right = cfg == "causal"
print("shape", shape, (B, S, HQ, HKV, D), "cfg", cfg, "gqa", gqa, "l5", l5,
      "Q_ORIGIN_ALIGN", getattr(kern, "Q_ORIGIN_ALIGN", None))
key = ("bshd", False, mask_right, True, False, gqa, D, "bf16")
if l5:
    kern._ensure_bshd_kernel(False, mask_right, True, False, gqa, qk_hdim=D, dtype_str="bf16",
                             q_origin_align=True)
    key = key + ("l5",)
else:
    kern._ensure_bshd_kernel(False, mask_right, True, False, gqa, qk_hdim=D, dtype_str="bf16")
fn = kern._launch_fns[key]
ph = torch.empty(64, dtype=torch.bfloat16); lse_ph = torch.empty(64, dtype=torch.float32)
s_seq, s_head, k_seq, k_head = HQ * D, D, HKV * D, D
args = (ph, ph, ph, ph, lse_ph, ph, 1.0 / D ** 0.5,
        s_seq, k_seq, k_seq, s_seq, s_head, k_head, k_head, s_head,
        1, S, HQ * S, 0, 0, S, S, HKV, B, None)
flyc.compile(fn, *args)
print("COMPILE_OK")
