_L1 = ("amdgpu-disable-unclustered-high-rp-reschedule", True)
CONFIGS = [
    ("r18", {}),
    ("g2d=2", {"g2d": 2}),
    ("g2d=2 g3d=2", {"g2d": 2, "g3d": 2}),
    ("g2d=2 g3d=3", {"g2d": 2, "g3d": 3}),
    ("g2d=2 llvm_opts=L1 only", {"g2d": 2, "llvm_opts": (_L1,)}),
]
