"""COMPILE_ONLY build of one fwd kernel variant; dumps ISA. No GPU work.
Usage: python3 compile_arm.py <impl_dir> <dump_dir> <module m32x8|m32x2> <gqa> <causal|nc>
Adapted from ../lab2/L5/tools/compile_l5.py."""
import importlib, importlib.util as ilu, os, pathlib, sys
assert os.environ.get("COMPILE_ONLY") == "1", "refusing to run without COMPILE_ONLY=1"
assert os.environ.get("FLYDSL_GPU_ARCH") == "gfx1250" and os.environ.get("ARCH") == "gfx1250"
sys.path.insert(0, "/home/lihuzhan/.local/flydsl0341")
impl_dir = pathlib.Path(sys.argv[1]).resolve()
os.environ["FLYDSL_DUMP_DIR"] = sys.argv[2]; os.environ["FLYDSL_DUMP_IR"] = "1"
import torch, flydsl, flydsl.compiler as flyc  # noqa
assert flydsl.__version__ == "0.3.4.1", flydsl.__file__
name = f"flydsl_fwd__ns_{abs(hash(str(impl_dir)))}"
d = impl_dir / "flydsl_fwd"
spec = ilu.spec_from_file_location(name, d / "__init__.py", submodule_search_locations=[str(d)])
pkg = ilu.module_from_spec(spec); sys.modules[name] = pkg; spec.loader.exec_module(pkg)
kern = importlib.import_module(f"{name}.fmha_fwd_prefill_a16w16_{sys.argv[3]}")
gqa = int(sys.argv[4]); mask_right = sys.argv[5] == "causal"; D = 128
print("kernel", kern.__file__, "SPEC_STALE_MAX", kern.SPEC_STALE_MAX, "gqa", gqa, "causal", mask_right)
kern._ensure_bshd_kernel(False, mask_right, True, False, gqa, qk_hdim=D, dtype_str="bf16")
fn = kern._launch_fns[("bshd", False, mask_right, True, False, gqa, D, "bf16")]
B, S, HKV = 1, 4096, 8; HQ = HKV * gqa
ph = torch.empty(64, dtype=torch.bfloat16); lse_ph = torch.empty(64, dtype=torch.float32)
s_seq, s_head, k_seq, k_head = HQ * D, D, HKV * D, D
args = (ph, ph, ph, ph, lse_ph, ph, 1.0 / D ** 0.5, s_seq, k_seq, k_seq, s_seq, s_head, k_head, k_head, s_head,
        1, S, HQ * S, 0, 0, S, S, HKV, B, None)
flyc.compile(fn, *args)
print("COMPILE_OK")
