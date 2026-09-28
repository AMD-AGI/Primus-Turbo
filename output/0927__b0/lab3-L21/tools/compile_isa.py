"""Compile-only one L24 arm for one config and dump IR/ISA. NO GPU work.
usage: compile_isa.py <arm_dir> <config>
configs (bshd, return_lse=True, no sink, no window -- what impl.py calls):
  prod  (4,8192,32,8)  causal gqa4     proxy (1,4096,32,8) causal gqa4   fast (1,1024,8,2) causal gqa4
  nc_g4 (4,8192,32,8)  non-causal gqa4 c_g1  (1,1024,8,8)  causal gqa1   nc_g1 (1,1024,8,8) non-causal gqa1
Shape integers are runtime kernel args; (causal, gqa) select the binary, so prod/proxy/fast share one.
"""
import importlib, importlib.util as ilu, os, pathlib, sys
assert os.environ.get("COMPILE_ONLY") == "1", "refusing to run without COMPILE_ONLY=1"
assert os.environ.get("FLYDSL_GPU_ARCH") == "gfx1250" and os.environ.get("ARCH") == "gfx1250"
sys.path.insert(0, "/home/lihuzhan/.local/flydsl0341")
import torch  # noqa: E402
import flydsl  # noqa: E402
import flydsl.compiler as flyc  # noqa: E402
assert flydsl.__version__ == "0.3.4.1" and "flydsl0341" in flydsl.__file__, flydsl.__file__

impl_dir = pathlib.Path(sys.argv[1]).resolve()
cfg = sys.argv[2]
CFG = {"prod": ((4, 8192, 32, 8), True), "proxy": ((1, 4096, 32, 8), True), "fast": ((1, 1024, 8, 2), True),
       "nc_g4": ((4, 8192, 32, 8), False), "c_g1": ((1, 1024, 8, 8), True), "nc_g1": ((1, 1024, 8, 8), False)}
(B, S, HQ, HKV), causal = CFG[cfg]
D = 128
name = f"flydsl_fwd__L21_{abs(hash(str(impl_dir)))}"
d = impl_dir / "flydsl_fwd"
spec = ilu.spec_from_file_location(name, d / "__init__.py", submodule_search_locations=[str(d)])
pkg = ilu.module_from_spec(spec); sys.modules[name] = pkg; spec.loader.exec_module(pkg)
kern = importlib.import_module(f"{name}.fmha_fwd_prefill_a16w16_m32x8")
print("kernel module:", kern.__file__)
gqa = HQ // HKV
ml, mr, lse, sink = False, causal, True, False
kern._ensure_bshd_kernel(ml, mr, lse, sink, gqa, qk_hdim=D, dtype_str="bf16")
fn = kern._launch_fns[("bshd", ml, mr, lse, sink, gqa, D, "bf16")]
print("cfg", cfg, B, S, HQ, HKV, D, "causal", causal, "gqa", gqa, "hints", fn.compile_hints)
ph = torch.empty(64, dtype=torch.bfloat16); lse_ph = torch.empty(64, dtype=torch.float32)
s_seq, s_head, k_seq, k_head = HQ * D, D, HKV * D, D
args = (ph, ph, ph, ph, lse_ph, ph, 1.0 / D ** 0.5,
        s_seq, k_seq, k_seq, s_seq, s_head, k_head, k_head, s_head,
        1, S, HQ * S, 0, 0, S, S, HKV, B, None)
flyc.compile(fn, *args)
print("COMPILE_OK")
