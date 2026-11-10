_U = ("amdgpu-disable-unclustered-high-rp-reschedule", True)
_G = ("greedy-regclass-priority-trumps-globalness", True)
_O = [
    ("misched-regpressure", False),
    ("amdgpu-mfma-vgpr-form", False),
    ("amdgpu-spill-vgpr-to-agpr", False),
    ("misched-cluster", False),
    ("misched-cyclicpath", False),
    ("misched-fusion", False),
    ("enable-misched", False),
    ("misched-postra", True),
    ("amdgpu-igrouplp-exact-solver", True),
    ("amdgpu-enable-rewrite-partial-reg-uses", False),
    ("misched-limit", 1024),
    ("dag-maps-huge-region", 4000),
    ("regalloc-eviction-max-interference-cutoff", 100),
    ("split-threshold-for-reg-with-hint", 0),
    ("amdgpu-sched-strategy", "iterative-ilp"),
    ("amdgpu-sched-strategy", "iterative-minreg"),
    ("amdgpu-sched-strategy", "iterative-maxocc"),
]
CONFIGS = [("r13 (U+G)", {})] + [("U+G+%s=%s" % o, {"llvm_opts": (_U, _G, o)}) for o in _O]
