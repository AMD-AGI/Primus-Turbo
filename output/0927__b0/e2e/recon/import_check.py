"""Import-only check for the combined asm+fly e2e process (no kernel launch, no torch.cuda call).

Verifies one process can hold: flydsl 0.3.4.1, the fwd r6 champion tree, the bwd r20 tree
re-pinned to 0.3.4.1 (bwd341/op0341), aiter.ops.mha (ASM fwd), the ASM bwd launcher loaded
by file path, and torchtitan's llama3 + Primus' Attention subclass -- WITHOUT primus_turbo.
"""
import importlib.util as ilu
import os
import sys
import time

t0 = time.time()
sys.path.insert(0, "/home/lihuzhan/.local/flydsl0341")
sys.path.insert(1, "/home/lihuzhan/code/aiter-src")
print("ENV", os.environ.get("TORCH_BLAS_PREFER_HIPBLASLT"), os.environ.get("HIPBLASLT_TENSILE_LIBPATH"))

import torch  # noqa: E402
import flydsl  # noqa: E402
print("flydsl", flydsl.__version__, flydsl.__file__)


def load(name, path):
    spec = ilu.spec_from_file_location(name, path)
    m = ilu.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


FWD = "/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/fwd/champion_r6"
BWD = "/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0925__flydsl/bwd341/op0341"
KA = "/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/primus_turbo/pytorch/kernels/attention/_asm_bwd_kernargs.py"
fwd = load("e2e_fly_fwd", FWD + "/impl.py")
print("fwd ok", fwd.attn_fwd)
bwd = load("e2e_fly_bwd", BWD + "/impl.py")
print("bwd ok", bwd.attn_bwd)
from aiter.ops.mha import fmha_fwd_with_sink_asm  # noqa: E402
print("aiter asm fwd ok", fmha_fwd_with_sink_asm)
ka = load("e2e_asm_bwd_kernargs", KA)
print("asm bwd kernargs ok", ka.ASM_DIR, [p.name for p in sorted(ka.ASM_DIR.glob("bwd_hd128*.co"))][:6])
print("flydsl still", flydsl.__version__, sys.modules["flydsl"].__file__)
import torchtitan.models.llama3.model.model as ttm  # noqa: E402
sys.path.insert(0, "/home/lihuzhan/code/2026_0828__primus/Primus")
from primus.backends.torchtitan.models.llama3.model.model import Attention  # noqa: E402
print("primus llama3 Attention ok", Attention)
print("primus_turbo loaded:", [m for m in sys.modules if m.startswith("primus_turbo")][:5])
print("cuda initialized:", torch.cuda.is_initialized())
print("elapsed %.1fs" % (time.time() - t0))
