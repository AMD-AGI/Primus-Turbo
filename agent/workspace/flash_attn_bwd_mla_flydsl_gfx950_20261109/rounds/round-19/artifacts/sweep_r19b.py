_L1 = ("amdgpu-disable-unclustered-high-rp-reschedule", True)
_L2 = ("greedy-regclass-priority-trumps-globalness", True)
CONFIGS = [
    ("g2d=2", {"g2d": 2}),
    ("g2d=3", {"g2d": 3}),
    ("g2d=2 g3d=2", {"g2d": 2, "g3d": 2}),
    ("g2d=2 g3d=3", {"g2d": 2, "g3d": 3}),
    ("g2d=2 g3d=5", {"g2d": 2, "g3d": 5}),
    ("g2d=2 g3d=6", {"g2d": 2, "g3d": 6}),
    ("g2d=2 llvm_opts=none", {"g2d": 2, "llvm_opts": ()}),
    ("g2d=2 llvm_opts=L1 only", {"g2d": 2, "llvm_opts": (_L1,)}),
]
