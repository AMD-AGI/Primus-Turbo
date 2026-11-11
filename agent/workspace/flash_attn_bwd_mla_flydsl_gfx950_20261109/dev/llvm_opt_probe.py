"""Check which LLVM cl::opts exist in FlyDSL's bundled LLVM: python llvm_opt_probe.py"""
from flydsl._mlir._mlir_libs import _mlirDialectsFly as F

CANDS = [
    ("exhaustive-register-search", "bool", True),
    ("split-spill-mode", "str", "speed"),
    ("amdgpu-spill-vgpr-to-agpr", "bool", False),
    ("amdgpu-use-amdgpu-trackers", "bool", True),
    ("amdgpu-disable-unclustered-high-rp-reschedule", "bool", True),
    ("amdgpu-disable-clustered-low-occupancy-reschedule", "bool", True),
    ("amdgpu-schedule-metric-bias", "int", 0),
    ("amdgpu-schedule-relaxed-occupancy", "bool", True),
    ("greedy-reverse-local-assignment", "bool", True),
    ("enable-local-reassign", "bool", True),
    ("consider-local-interval-cost", "bool", True),
    ("misched-regpressure", "bool", False),
    ("amdgpu-sched-strategy", "str", "max-ilp"),
    ("amdgpu-premra-rematerialize", "bool", True),
    ("amdgpu-disable-rewrite-mfma-form-sched-stage", "bool", True),
    ("greedy-regclass-priority-trumps-globalness", "bool", True),
    ("regalloc-csr-first-time-cost", "int", 0),
    ("amdgpu-mfma-vgpr-form", "bool", False),
]
for name, kind, val in CANDS:
    fn = {"bool": F.set_llvm_option_bool, "int": F.set_llvm_option_int, "str": F.set_llvm_option_str}[kind]
    try:
        old = fn(name, val)
        fn(name, old)
        print(f"OK   {name} ({kind}) default={old!r}")
    except Exception as e:  # noqa: BLE001
        print(f"MISS {name}: {type(e).__name__}: {str(e)[:100]}")
