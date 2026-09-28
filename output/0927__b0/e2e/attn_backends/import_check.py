"""Import-only check of the e2e injection (no kernel launch): does the converter's import path
resolve to the right TurboAttention under the given E2E_ATTN, and which flydsl is loaded?"""
import os, sys, time
t0 = time.time()
from primus_turbo.pytorch.modules import TurboAttention
from primus_turbo.pytorch.core.low_precision import Float8QuantConfig, ScalingGranularity
import primus_turbo
print("E2E_ATTN", os.environ.get("E2E_ATTN"), "TurboAttention", TurboAttention.__module__, TurboAttention.__qualname__)
print("primus_turbo.__file__", primus_turbo.__file__, "path", list(primus_turbo.__path__))
m = TurboAttention(causal=True, fp8_config=None)
print("module layer", m.layer, "sched", m.sched)
fl = sys.modules.get("flydsl")
print("flydsl loaded:", fl.__version__ if fl else None, fl.__file__ if fl else None)
print("primus_turbo modules:", len([k for k in sys.modules if k.startswith("primus_turbo")]))
sys.path.insert(0, "/home/lihuzhan/code/2026_0828__primus/Primus")
from primus.backends.torchtitan.primus_turbo_extensions import primus_turbo_converter as pc
print("converter import ok", pc.PrimusTubroConverter)
print("elapsed %.1fs" % (time.time() - t0))
