_U = ("amdgpu-disable-unclustered-high-rp-reschedule", True)
_G = ("greedy-regclass-priority-trumps-globalness", True)
CONFIGS = [
    ("r9 default", {}),
    ("no-unclustered-resched", {"llvm_opts": (_U,)}),
    ("regclass-priority", {"llvm_opts": (_G,)}),
    ("no-unclustered+regclass", {"llvm_opts": (_U, _G)}),
]
