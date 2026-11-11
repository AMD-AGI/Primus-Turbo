_U = ("amdgpu-disable-unclustered-high-rp-reschedule", True)
_G = ("greedy-regclass-priority-trumps-globalness", True)
_O = [
    ("misched-cluster", False),
    ("misched-fusion", False),
    ("enable-misched", False),
    ("amdgpu-enable-rewrite-partial-reg-uses", False),
]
CONFIGS = [("r13 (U+G)", {})] + [("+%s=%s" % o, {"llvm_opts": (_U, _G, o)}) for o in _O]
