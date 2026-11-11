C = "/perf_apps/xiaoming/Primus-Turbo/agent/workspace/flash_attn_bwd_mla_flydsl_gfx950_20261109"
CONFIGS = [
    ("r16", {"__pkg__": C + "/rounds/round-16/pkg"}),
    ("r18 (DS_PAR)", {"__pkg__": "/perf_apps/xiaoming/megaattention_overlap_bench"}),
]
