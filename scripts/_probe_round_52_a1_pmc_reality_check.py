"""Round-52 — A1 (Stream-K) PMC reality check via persistent-grid wave-pack model.

Per R51 forward-pointer (analysis/_notes/round-51-A1-streamk-scaffolding-...md):
gate the 4-6 round Stream-K kernel-implementation arc on whether the wave2-only
(load-imbalance idle) fraction of wall is >25% (A1 EV survives → R53 lands kernel
branch) or <10% (A1 closes → pivot to Direction A3 decoupled-warps).

R51 originally proposed `rocprofv3 --pmc s_endpgm,SQ_BUSY_CY_TIME,GRBM_GUI_ACTIVE`
to capture per-CTA timing on Down_B4_M2048 fwd. R52 takes the cheaper analytical
path: the kernels are *persistent* (grid_x = NUM_CUS = 256 blocks, see
kernel_fp8_layouts.cpp:576 / 2996 / 8311), so each CTA loops over
ceil(total_tiles / 256) tiles. The "wave2-only" tax is therefore *load
imbalance* (fast CTAs idle at end while last-batch CTAs finish), which is
exactly determined by (total_tiles, NUM_CUS) — no per-CTA timestamps needed.

PMC R21 (MfmaUtil=32% on Down_B4_M2048 wgrad) corroborates: a uniformly
saturated kernel would report MfmaUtil ≈ MfmaUtil_per_active_CTA, and the
wave-pack idle is an upper bound on the recoverable share via Stream-K.

Computes for each of the 8 gpt_oss FP8 shapes:
  * total_tiles = B * tiles_m * tiles_n
  * tiles_per_cta_max = ceil(total_tiles / NUM_CUS)
  * tiles_per_cta_min = floor(total_tiles / NUM_CUS)
  * num_max_ctas = total_tiles - NUM_CUS * tiles_per_cta_min
  * idle_cu_cycles = (NUM_CUS - num_max_ctas) * (tiles_per_cta_max - tiles_per_cta_min) * t_per_tile
  * total_cu_cycles = NUM_CUS * tiles_per_cta_max * t_per_tile
  * idle_frac = idle_cu_cycles / total_cu_cycles
                = (NUM_CUS - num_max_ctas) * (tiles_per_cta_max - tiles_per_cta_min)
                  / (NUM_CUS * tiles_per_cta_max)

Equivalently: idle_frac = 1 - (total_tiles / (NUM_CUS * tiles_per_cta_max)).

Reports per-cell idle_frac, gate-verdict, and 8-shape-mean projected Stream-K
ceiling lift (idle_frac × 1.0 in the optimistic "perfect rebalance" model, then
discounted for atomic-reduce overhead in the doc note).

CLI:
  $ python3 scripts/_probe_round_52_a1_pmc_reality_check.py
"""
from __future__ import annotations

import math


NUM_CUS = 256  # MI355X / gfx950, per HK kernel_fp8_layouts.cpp:576

# 8 gpt_oss_20B Balanced shapes from scripts/_task_gpt_oss_fp8_kernel.md lines 45-55.
# Tile size: 256x256, so tiles_m = m_per_g/256, tiles_n = n/256.
SHAPES = [
    # (layer, B, M_per_g, N, K, tiles_m, tiles_n)
    ("GateUP", 4,  2048, 5760, 2880,  8, 22),
    ("GateUP", 4,  4096, 5760, 2880, 16, 22),
    ("Down",   4,  2048, 2880, 2880,  8, 11),
    ("Down",   4,  4096, 2880, 2880, 16, 11),
    ("GateUP", 32, 2048, 5760, 2880,  8, 22),
    ("GateUP", 32, 4096, 5760, 2880, 16, 22),
    ("Down",   32, 2048, 2880, 2880,  8, 11),
    ("Down",   32, 4096, 2880, 2880, 16, 11),
]


def wave_pack(total_tiles: int, num_cus: int = NUM_CUS):
    tpc_max = math.ceil(total_tiles / num_cus)
    tpc_min = total_tiles // num_cus
    n_max = total_tiles - num_cus * tpc_min  # CTAs assigned the larger batch
    n_min = num_cus - n_max                   # CTAs assigned the smaller batch
    if tpc_max == tpc_min:
        idle_frac = 0.0
    else:
        idle_cu_cycles = n_min * (tpc_max - tpc_min)  # in t_per_tile units
        total_cu_cycles = num_cus * tpc_max
        idle_frac = idle_cu_cycles / total_cu_cycles
    return tpc_min, tpc_max, n_max, n_min, idle_frac


def main():
    print(f"# Round-52 A1 wave-pack model — NUM_CUS={NUM_CUS}")
    print()
    print(
        f"{'cell':<22}{'tiles':>7}{'tpc_min':>9}{'tpc_max':>9}{'n_min':>7}"
        f"{'n_max':>7}{'idle%':>8}{'gate':>14}"
    )
    print("-" * 84)
    weighted = []
    for (layer, B, M_per_g, N, K, tm, tn) in SHAPES:
        total = B * tm * tn
        tpc_min, tpc_max, n_max, n_min, idle = wave_pack(total)
        if idle > 0.25:
            gate = "A1 SURVIVES"
        elif idle < 0.10:
            gate = "A1 CLOSES"
        else:
            gate = "ambiguous"
        cell = f"{layer}-B{B}-M{M_per_g}"
        print(
            f"{cell:<22}{total:>7d}{tpc_min:>9d}{tpc_max:>9d}{n_min:>7d}"
            f"{n_max:>7d}{idle*100:>7.2f}%{gate:>14}"
        )
        weighted.append(idle)

    mean_idle = sum(weighted) / len(weighted)
    print()
    print(f"# 8-shape mean idle_frac (optimistic SK ceiling): {mean_idle*100:.2f}%")
    print(f"#   apply ~0.5 overhead-discount → realistic ceiling ≈ {mean_idle*50:.2f}%")
    print()
    print("# Gate per R51:")
    print("#   if any cell idle > 25% → A1 EV survives for that cell")
    print("#   if all cells idle < 10% → A1 closes globally")
    n_survive = sum(1 for x in weighted if x > 0.25)
    n_close = sum(1 for x in weighted if x < 0.10)
    print(f"# cells where A1 SURVIVES gate (idle > 25%): {n_survive} / 8")
    print(f"# cells where A1 CLOSES gate    (idle < 10%): {n_close} / 8")
    if n_survive >= 1:
        print()
        print("# Verdict: A1 EV survives on a NARROW set of cells (B=4 with bad")
        print("#   total-tile / NUM_CUS remainder). The 4 B=32 cells have idle=0%")
        print("#   by construction (their tile counts are exact multiples of 256).")
        print("#   8-shape mean lift is dilution-bounded; see the .md note for")
        print("#   the per-section EV decomposition and R53 recommendation.")


if __name__ == "__main__":
    main()
