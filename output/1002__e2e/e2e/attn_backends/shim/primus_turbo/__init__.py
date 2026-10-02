"""SHIM `primus_turbo` for the e2e runs -- A0 copy of 2026-10-02 (output/1002__e2e/e2e).
Only change vs output/0927__b0/e2e/attn_backends/shim: nkfix_b0 is imported from the absolute
output/0927__b0/gemm path (this copy lives elsewhere). Never the real package in P2.

Primus' converter does `from primus_turbo.pytorch.modules import TurboAttention` and
`from primus_turbo.pytorch.core.low_precision import Float8QuantConfig, ScalingGranularity`.
This package sits first on PYTHONPATH and decides, from E2E_ATTN, what those names are:

  E2E_ATTN=turbo  (P1): this package's __path__ is re-pointed at the REAL primus_turbo
      (wt-bakeoff) and the real __init__ is executed, so every submodule is the product code.
      A meta-path hook then swaps `primus_turbo.pytorch.modules.TurboAttention` for
      e2e_attn.E2EAttention right after the real module finishes importing; E2EAttention
      calls the real flash_attn_func, i.e. the product Triton path, only adding profiler
      ranges.
  anything else   (P2): only the two names above exist. Any other primus_turbo submodule
      raises ImportError naming itself -- the real package cannot load under flydsl 0.3.4.1
      (sparse_mla_bwd.py imports flydsl.expr.buffer_ops, removed in 0.3.2).
"""
import importlib.abc
import importlib.machinery
import os
import sys
from pathlib import Path

_E2E_BACKENDS = str(Path(__file__).resolve().parents[2])        # .../e2e/attn_backends
if _E2E_BACKENDS not in sys.path:
    sys.path.insert(0, _E2E_BACKENDS)

IS_E2E_SHIM = True

# Opt-in GEMM layout workaround (output/0927__b0/gemm/nkfix_b0.py). Installed here because this
# package is imported exactly once, by the training worker, after torch is loaded and before
# the first step -- in both P1 and P2. Off unless E2E_NKFIX=1.
if os.environ.get("E2E_NKFIX", "0") not in ("", "0"):
    _GEMM = "/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/gemm"
    if _GEMM not in sys.path:
        sys.path.insert(0, _GEMM)
    import nkfix_b0
    nkfix_b0.install()
_MODE = "turbo" if os.environ.get("E2E_ATTN", "").strip() == "turbo" else "shim"
_REAL = os.environ.get("E2E_REAL_PRIMUS_TURBO",
                       "/home/lihuzhan/code/2026_0903__turbo/wt-bakeoff/primus_turbo")


class _PatchTurboAttention(importlib.abc.MetaPathFinder):
    TARGET = "primus_turbo.pytorch.modules"

    def find_spec(self, name, path=None, target=None):
        if name != self.TARGET:
            return None
        for f in sys.meta_path:
            if f is self or not hasattr(f, "find_spec"):
                continue
            spec = f.find_spec(name, path, target)
            if spec is not None:
                break
        else:
            return None
        real_exec = spec.loader.exec_module

        def exec_module(module):
            real_exec(module)
            import e2e_attn
            module._RealTurboAttention = module.TurboAttention
            module.TurboAttention = e2e_attn.E2EAttention
            print(f"[e2e_attn] P1: {name}.TurboAttention -> e2e_attn.E2EAttention "
                  f"(real class from {module.__file__})", file=sys.stderr, flush=True)
        spec.loader.exec_module = exec_module
        return spec


if _MODE == "turbo":
    _real_init = Path(_REAL) / "__init__.py"
    if not _real_init.is_file():
        raise ImportError(f"E2E_ATTN=turbo but the real primus_turbo is missing: {_REAL}")
    __path__[:] = [_REAL]                       # noqa: F821 -- every submodule = product code
    __file__ = str(_real_init)
    sys.meta_path.insert(0, _PatchTurboAttention())
    exec(compile(_real_init.read_text(), __file__, "exec"), globals())
    print(f"[e2e_attn] P1: primus_turbo -> {_REAL}", file=sys.stderr, flush=True)
else:
    print(f"[e2e_attn] P2: primus_turbo is the e2e shim ({__file__}); E2E_ATTN="
          f"{os.environ.get('E2E_ATTN')!r}", file=sys.stderr, flush=True)
