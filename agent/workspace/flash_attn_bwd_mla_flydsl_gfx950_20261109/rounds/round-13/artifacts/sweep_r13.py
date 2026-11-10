_O = [
    ("exhaustive-register-search", True),
    ("amdgpu-use-amdgpu-trackers", True),
    ("amdgpu-disable-unclustered-high-rp-reschedule", True),
    ("amdgpu-disable-clustered-low-occupancy-reschedule", True),
    ("amdgpu-schedule-relaxed-occupancy", True),
    ("greedy-reverse-local-assignment", True),
    ("enable-local-reassign", True),
    ("greedy-regclass-priority-trumps-globalness", True),
    ("amdgpu-disable-rewrite-mfma-form-sched-stage", False),
    ("amdgpu-schedule-metric-bias", 0),
    ("amdgpu-schedule-metric-bias", 100),
    ("amdgpu-sched-strategy", "max-memory-clause"),
    ("amdgpu-sched-strategy", "max-ilp"),
]
CONFIGS = [("r9 default", {})] + [("%s=%s" % o, {"llvm_opts": (o,)}) for o in _O]
