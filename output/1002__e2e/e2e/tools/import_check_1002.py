"""CPU-only import check of the 1002 e2e adapter (NO kernel, NO device query, NO arm load).

Run inside fa-repro with HIP_VISIBLE_DEVICES=-1 (preflight.sh does):
  PYTHONPATH=<kit>/attn_backends/shim:<kit>/attn_backends E2E_ATTN=<spec> E2E_FLY_TREES=<json> \
  E2E_EXPECT_BLAS_LIB=<lib> TORCH_BLAS_PREFER_HIPBLASLT=1 HIPBLASLT_TENSILE_LIBPATH=<lib> python3 import_check_1002.py
Checks what the converter will see: `primus_turbo.pytorch.modules.TurboAttention` is this kit's
E2EAttention, the schedule parses, every scheduled arm is registered, step 1 is asm, the BLAS guard
sees the expected library, flydsl/aiter/primus_turbo-real are NOT imported (arms load lazily on
the card, never here), and torch.cuda was never initialised.
"""
import json
import os
import sys

import torch

from primus_turbo.pytorch.modules import TurboAttention
from primus_turbo.pytorch.core.low_precision import Float8QuantConfig, ScalingGranularity  # noqa: F401
import primus_turbo
import e2e_attn
from e2e_attn import arms

kit = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
assert TurboAttention is e2e_attn.E2EAttention, TurboAttention
assert primus_turbo.__file__.startswith(kit), primus_turbo.__file__
assert e2e_attn.__file__.startswith(kit), e2e_attn.__file__
m = TurboAttention(causal=True, fp8_config=None)
warm, cyc = m.sched
first = (warm + cyc)[0]
assert first == ("asm", "asm"), f"step 1 must be ASM (09-28 lesson), got {first}"
names = sorted({a for t in warm + cyc for a in t})
assert all(arms.known(a) for a in names), names
assert not arms.BLAS_EVENTS, arms.BLAS_EVENTS
for mod in ("flydsl", "aiter"):
    assert mod not in sys.modules, f"{mod} imported at model build -- arms must load lazily"
assert not [k for k in sys.modules if k.startswith("primus_turbo.pytorch.ops")], "real primus_turbo loaded"
assert not torch.cuda.is_initialized(), "torch.cuda initialised by an import"
print("IMPORT_CHECK_OK", json.dumps({
    "shim": primus_turbo.__file__, "e2e_attn": e2e_attn.__file__, "arms": names,
    "warm": warm, "cycle": cyc, "fly_trees": arms.FLY_TREES,
    "blas_env": {k: os.environ.get(k) for k in ("TORCH_BLAS_PREFER_HIPBLASLT", "HIPBLASLT_TENSILE_LIBPATH",
                                                 "E2E_EXPECT_BLAS_LIB")},
    "timer": None if e2e_attn.TIMER is None else e2e_attn.TIMER.path}))
