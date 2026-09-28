"""Build L12 4-wave arms from base/ (a read-only copy of the fwd champion r4).

Every arm = full copy of base/ with textual edits (each edit asserted to hit exactly once):
  managers (all arms):  the four `num_waves != 8` raises (Q/K/V V2, O V3) become `not in (4, 8)`;
                        `_tdm_load_views` gets a `num_warps` argument (was hard-wired
                        num_warps=_DEFAULT_NUM_WAVES) and K/V V2 pass self.num_waves.
  kernel (per arm):     NUM_WAVES, WMMA_ROW_PER_WAVE, waves_per_eu (both launch sites),
                        MIN_KV_BLK_BYTES, LDS allocation size, LO/HI role split,
                        R-generic packed row-sum (pairs of rows), zero-fill LSE loop.
"""
import json, pathlib, re, shutil, sys

L = pathlib.Path(__file__).resolve().parent.parent
BASE = L / "base"

ARMS = {
    # name: dict(nw, r, wpe, min_kv, lds, roles)
    # 1 wave per SIMD, each wave owns 64 rows (4 WMMA tiles); same BLOCK_M=256, same LDS map.
    "w4r4":       dict(nw=4, r=4, wpe=1, min_kv=64 * 1024, lds=320 * 1024, roles="split"),
    "w4r4_lo":    dict(nw=4, r=4, wpe=1, min_kv=64 * 1024, lds=320 * 1024, roles="alllo"),
    "w4r4_hi":    dict(nw=4, r=4, wpe=1, min_kv=64 * 1024, lds=320 * 1024, roles="allhi"),
    # halve the WG: 4 waves x 32 rows, BLOCK_M=128, full LDS -> occupancy 1 WG/CU (4 waves/CU)
    "w4r2":       dict(nw=4, r=2, wpe=1, min_kv=64 * 1024, lds=320 * 1024, roles="split"),
    # halve the WG and shrink LDS so 2 WGs fit a CU (8 waves/CU again, barrier scope 4 waves)
    "w4r2_occ2":  dict(nw=4, r=2, wpe=2, min_kv=0, lds=160 * 1024, roles="split"),
    # control: 8 waves through the SAME edited code path (must be ISA-identical to base)
    "w8r2_ctrl":  dict(nw=8, r=2, wpe=2, min_kv=64 * 1024, lds=320 * 1024, roles="split"),
    # retunes of the 1-wave-per-SIMD build (VGPR 758 leaves ~260 headroom)
    "w4r4_compact": dict(nw=4, r=4, wpe=1, min_kv=0, lds=320 * 1024, roles="split"),
    "w4r4_nodefer": dict(nw=4, r=4, wpe=1, min_kv=64 * 1024, lds=320 * 1024, roles="split", defer=False),
    "w4r4_r3":      dict(nw=4, r=3, wpe=1, min_kv=64 * 1024, lds=320 * 1024, roles="split"),
    # retunes of the winning half-WG / 2-WG-per-CU build
    "occ2_lo":      dict(nw=4, r=2, wpe=2, min_kv=0, lds=160 * 1024, roles="alllo"),
    "occ2_hi":      dict(nw=4, r=2, wpe=2, min_kv=0, lds=160 * 1024, roles="allhi"),
    "occ2_kv32":    dict(nw=4, r=2, wpe=2, min_kv=32 * 1024, lds=160 * 1024, roles="split"),
    # diagnostic: BLOCK_M=128 with 8 waves x 16 rows, 1 WG/CU (separates grid granularity from WG decoupling)
    "w8r1":         dict(nw=8, r=1, wpe=2, min_kv=64 * 1024, lds=320 * 1024, roles="split"),
    # the job's champion moved to round 6 (speculative stale-max softmax) at 14:19:31 during this lab;
    # the same L12 edits ported onto a snapshot of it (base_r6/ == rounds/006/op)
    "r6_occ2":      dict(nw=4, r=2, wpe=2, min_kv=0, lds=160 * 1024, roles="split", base="base_r6"),
    "r6_occ2_lo":   dict(nw=4, r=2, wpe=2, min_kv=0, lds=160 * 1024, roles="alllo", base="base_r6"),
    "r6_occ2_hi":   dict(nw=4, r=2, wpe=2, min_kv=0, lds=160 * 1024, roles="allhi", base="base_r6"),
    "r6_occ2_lo_kv32": dict(nw=4, r=2, wpe=2, min_kv=32 * 1024, lds=160 * 1024, roles="alllo", base="base_r6"),
    "r6_ctrl":      dict(nw=8, r=2, wpe=2, min_kv=64 * 1024, lds=320 * 1024, roles="split", base="base_r6"),
}


def sub1(txt, old, new, what):
    n = txt.count(old)
    assert n == 1, f"{what}: expected 1 hit, got {n}"
    return txt.replace(old, new)


def patch_managers(txt):
    n = txt.count("        if num_waves != _DEFAULT_NUM_WAVES:\n")
    assert n == 5, n  # Q/K/V V2, O V2, O V3
    txt = txt.replace("        if num_waves != _DEFAULT_NUM_WAVES:\n",
                      "        if num_waves not in (4, _DEFAULT_NUM_WAVES):  # L12: 4-wave legal\n")
    txt = sub1(txt, "    lds_base,\n    elem_dtype,\n):\n    \"\"\"Build a LIST of ``(atom, g_view, lds_view)`` TDM",
               "    lds_base,\n    elem_dtype,\n    num_warps=_DEFAULT_NUM_WAVES,\n):\n    \"\"\"Build a LIST of ``(atom, g_view, lds_view)`` TDM",
               "tdm sig")
    txt = sub1(txt, "            num_warps=_DEFAULT_NUM_WAVES,\n            pad_interval=w,",
               "            num_warps=num_warps,\n            pad_interval=w,", "tdm num_warps")
    n = txt.count("            lds_base=ptr_lds,\n            elem_dtype=self.elem_dtype,\n        )\n")
    assert n == 2, n
    txt = txt.replace("            lds_base=ptr_lds,\n            elem_dtype=self.elem_dtype,\n        )\n",
                      "            lds_base=ptr_lds,\n            elem_dtype=self.elem_dtype,\n"
                      "            num_warps=self.num_waves,\n        )\n")
    return txt


def patch_kernel(txt, c):
    txt = sub1(txt, "NUM_WAVES = 8  #", f"NUM_WAVES = {c['nw']}  # L12 (was 8) --", "NUM_WAVES")
    txt = sub1(txt, "WMMA_ROW_PER_WAVE = 2  #", f"WMMA_ROW_PER_WAVE = {c['r']}  # L12 (was 2) --", "R")
    txt = sub1(txt, "MIN_KV_BLK_BYTES = 64 * 1024\n", f"MIN_KV_BLK_BYTES = {c['min_kv']}  # L12\n", "min_kv")
    n = txt.count('_launch.compile_hints["waves_per_eu"] = 2\n')
    assert n == 2, n
    txt = txt.replace('_launch.compile_hints["waves_per_eu"] = 2\n',
                      f'_launch.compile_hints["waves_per_eu"] = {c["wpe"]}  # L12\n')
    txt = sub1(txt, 'smem = fx.SharedAllocator().allocate(get_lds_capacity_bytes("gfx1250"))',
               f'smem = fx.SharedAllocator().allocate(min({c["lds"]}, get_lds_capacity_bytes("gfx1250")))  # L12',
               "lds alloc")
    # after slot_bytes is known, assert the 2-slot ring fits the allocation (compile-time)
    txt = sub1(txt, "    slot_bytes = max(k_blk_bytes + v_blk_bytes, q_mgr.get_lds_size_in_byte())\n",
               "    slot_bytes = max(k_blk_bytes + v_blk_bytes, q_mgr.get_lds_size_in_byte())\n"
               f"    assert N_KV_PP * slot_bytes <= min({c['lds']}, get_lds_capacity_bytes('gfx1250')), (\n"
               "        f'L12: LDS ring {N_KV_PP}x{slot_bytes} exceeds allocation')\n",
               "slot assert")
    if c.get("defer") is False:
        txt = sub1(txt, "ENABLE_DEFER_RESCALE = True\n", "ENABLE_DEFER_RESCALE = False  # L12\n", "defer")
    # role split
    cond_old = "if warp_idx // fx.Int32(NUM_WAVES // 2) == fx.Int32(0):"
    n = txt.count(cond_old)
    assert n == 2, n
    if c["roles"] == "alllo":
        txt = txt.replace(cond_old, "if warp_idx >= fx.Int32(0):  # L12 all waves LO role")
    elif c["roles"] == "allhi":
        txt = txt.replace(cond_old, "if warp_idx < fx.Int32(0):  # L12 all waves HI role")
    # R-generic packed row sum: pair rows (2i, 2i+1); per-row association unchanged
    _rs_end = "        local_sum_list = _tree_reduce_multi(p_flat_list, add3, fadd_t)\n"
    _rs0 = txt.index("    if R == 2:\n        # r4 g10 (L16)")
    old_rs = txt[_rs0:txt.index(_rs_end, _rs0) + len(_rs_end)]
    new_rs = '''    if R % 2 == 0:
        # r4 g10 (L16), L12 generalised to any even R: rows (2i, 2i+1) run in lockstep as one
        # v2 tree -> v_pk_add_f32; per-row association unchanged (bitwise).
        def vadd_t(a, b):
            return fx.Vector(arith.addf(_ir(a), _ir(b), fastmath=_no_reassoc))

        vadd3 = lambda a, b, c: vadd_t(vadd_t(a, b), c)
        pair_leaves = [
            [
                fx.Vector.from_elements(
                    [p_flat_list[2 * pi][i], p_flat_list[2 * pi + 1][i]], fx.Float32
                )
                for i in range(len(p_flat_list[0]))
            ]
            for pi in range(R // 2)
        ]
        tots = _tree_reduce_multi(pair_leaves, vadd3, vadd_t)
        local_sum_list = []
        for tot in tots:
            local_sum_list += [fx.Float32(tot[0]), fx.Float32(tot[1])]
    else:
        local_sum_list = _tree_reduce_multi(p_flat_list, add3, fadd_t)
'''
    txt = sub1(txt, old_rs, new_rs, "row-sum")
    # zero-fill (THD only): one LSE per packed row needs BLOCK_M/BLOCK_SIZE rounds when BLOCK_M > BLOCK_SIZE
    old_z = txt[txt.index("        prow = row0 + tid  # one LSE per packed row"):txt.index("\n\n\n# ============================================================================\n# Builder")]
    body = old_z.replace("        prow = row0 + tid  # one LSE per packed row (BLOCK_SIZE threads == BLOCK_M)\n", "")
    body = "\n".join(("    " + ln if ln.strip() else ln) for ln in body.split("\n"))
    new_z = ("        assert BLOCK_M % BLOCK_SIZE == 0\n"
             "        for zr in range(BLOCK_M // BLOCK_SIZE):  # L12: BLOCK_M may exceed BLOCK_SIZE\n"
             "            prow = row0 + fx.Int32(zr * BLOCK_SIZE) + tid\n" + body)
    txt = txt.replace(old_z, new_z)
    assert "BLOCK_M * cpr // BLOCK_SIZE" in txt
    return txt


def make(name, c):
    d = L / "arms" / name
    if d.exists():  # never rebuild an arm in place (measured arms are evidence); delete by hand first
        print("exists, skipped", name); return
    shutil.copytree(L / c.get("base", "base"), d, ignore=shutil.ignore_patterns("__pycache__"))
    m = d / "flydsl_fwd" / "fmha_b16_buffer_managers.py"
    m.write_text(patch_managers(m.read_text()))
    k = d / "flydsl_fwd" / "fmha_fwd_prefill_a16w16_m32x8.py"
    k.write_text(patch_kernel(k.read_text(), c))
    (d / "L12_ARM.json").write_text(json.dumps(dict(name=name, **c), indent=1))
    print("built", name, c)


if __name__ == "__main__":
    names = sys.argv[1:]
    for n in names:
        make(n, ARMS[n])
