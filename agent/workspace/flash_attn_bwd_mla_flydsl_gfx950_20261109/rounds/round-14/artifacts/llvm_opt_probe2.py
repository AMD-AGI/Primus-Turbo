"""Round-14 candidates: which LLVM cl::opts exist in FlyDSL's bundled LLVM (python llvm_opt_probe2.py)."""
from flydsl._mlir._mlir_libs import _mlirDialectsFly as F

CANDS = [
    ("consider-local-interval-cost", "bool", True),
    ("misched-regpressure", "bool", False),
    ("amdgpu-premra-rematerialize", "bool", True),
    ("amdgpu-mfma-vgpr-form", "bool", False),
    ("amdgpu-spill-vgpr-to-agpr", "bool", False),
    ("misched-cluster", "bool", False),
    ("misched-cyclicpath", "bool", False),
    ("misched-fusion", "bool", False),
    ("enable-misched", "bool", False),
    ("misched-postra", "bool", True),
    ("amdgpu-igrouplp-exact-solver", "bool", True),
    ("amdgpu-enable-rewrite-partial-reg-uses", "bool", True),
    ("amdgpu-disable-rematerialization-sched-stage", "bool", True),
    ("amdgpu-schedule-optimize-ilp", "bool", True),
    ("misched-limit", "int", 256),
    ("dag-maps-huge-region", "int", 1000),
    ("regalloc-eviction-max-interference-cutoff", "int", 10),
    ("split-threshold-for-reg-with-hint", "int", 75),
    ("amdgpu-sched-strategy", "str", "iterative-ilp"),
]
for name, kind, val in CANDS:
    fn = {"bool": F.set_llvm_option_bool, "int": F.set_llvm_option_int, "str": F.set_llvm_option_str}[kind]
    try:
        old = fn(name, val)
        fn(name, old)
        print(f"OK   {name} ({kind}) default={old!r}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"MISS {name}: {type(e).__name__}: {str(e)[:100]}", flush=True)
