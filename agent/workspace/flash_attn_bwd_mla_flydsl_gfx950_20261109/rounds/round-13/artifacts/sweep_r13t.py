_T = ("amdgpu-use-amdgpu-trackers", True)
_U = ("amdgpu-disable-unclustered-high-rp-reschedule", True)
_G = ("greedy-regclass-priority-trumps-globalness", True)
CONFIGS = [
    ("r9 default", {}),
    ("trackers", {"llvm_opts": (_T,)}),
    ("no-unclustered-resched", {"llvm_opts": (_U,)}),
    ("regclass-priority", {"llvm_opts": (_G,)}),
    ("trackers+no-unclustered", {"llvm_opts": (_T, _U)}),
    ("trackers+regclass", {"llvm_opts": (_T, _G)}),
]
