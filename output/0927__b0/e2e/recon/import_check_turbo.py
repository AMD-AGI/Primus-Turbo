"""Can wt-bakeoff's primus_turbo (Triton stock attention) share a process with flydsl 0.3.4.1?
Import-only; reports which flydsl-backed pieces of primus_turbo failed to import."""
import sys, time, traceback
t0 = time.time()
sys.path.insert(0, "/home/lihuzhan/.local/flydsl0341")
sys.path.insert(1, "/home/lihuzhan/code/aiter-src")
sys.path.insert(2, "/home/lihuzhan/code/2026_0903__turbo/wt-bakeoff")
import flydsl
print("flydsl", flydsl.__version__, flydsl.__file__)
try:
    import primus_turbo.pytorch as pt
    print("primus_turbo", pt.__file__)
    from primus_turbo.pytorch.modules import TurboAttention
    from primus_turbo.pytorch.core.low_precision import Float8QuantConfig, ScalingGranularity
    print("TurboAttention ok", TurboAttention)
    import primus_turbo.pytorch.kernels.attention.attention_flydsl_impl as fi
    print("FLYDSL_AVAILABLE", fi.FLYDSL_AVAILABLE, repr(getattr(fi, "_FLYDSL_IMPORT_ERROR", None))[:200])
    from primus_turbo.pytorch.ops.attention.flash_attn_interface import flash_attn_func
    print("flash_attn_func ok")
except Exception:
    traceback.print_exc()
print("flydsl still", sys.modules["flydsl"].__version__, sys.modules["flydsl"].__file__)
print("flydsl-importing primus_turbo modules:", sorted(m for m in sys.modules if m.startswith("primus_turbo.flydsl"))[:10])
print("elapsed %.1fs" % (time.time() - t0))
