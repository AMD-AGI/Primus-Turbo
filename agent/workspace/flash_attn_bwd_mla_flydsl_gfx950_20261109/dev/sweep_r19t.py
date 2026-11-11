_C = "/perf_apps/xiaoming/Primus-Turbo/agent/workspace/flash_attn_bwd_mla_flydsl_gfx950_20261109"
_L2 = ("greedy-regclass-priority-trumps-globalness", True)
CONFIGS = [
    ("r18", {}),
    ("g2d=2", {"g2d": 2}),
    ("llvm_opts=L2 only", {"llvm_opts": (_L2,)}),
    ("llvm_opts=none", {"llvm_opts": ()}),
    ("g2d=2 + L2 only", {"g2d": 2, "llvm_opts": (_L2,)}),
    ("r16 (racy, reference only)", {"__pkg__": _C + "/rounds/round-16/pkg"}),
]
