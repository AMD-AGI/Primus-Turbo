"""Apply the L4 XCD (b, kv_head) remap to an arm copy of the round-4 champion.
usage: apply_l4.py <arm_dir> <off|bmajor|spread>"""
import pathlib, sys
arm, mode = pathlib.Path(sys.argv[1]), sys.argv[2]
assert mode in ("off", "bmajor", "spread")
f = arm / "flydsl_fwd" / "fmha_fwd_prefill_a16w16_m32x8.py"
s = f.read_text()
old_tail = '''    gyz = gy * gz
    rank = lin // gyz
    rem = lin - rank * gyz
    if axis == "x":
        return gx - fx.Int32(1) - rank
    if axis == "y":
        return rem % gy
    return rem // gy
'''
new_tail = '''    gyz = gy * gz
    rank = lin // gyz
    rem = lin - rank * gyz
    if axis == "x":
        return gx - fx.Int32(1) - rank
    grp = _xcd_group(rem, rank, gyz)
    if axis == "y":
        return grp % gy
    return grp // gy
'''
helper = '''# L4 (lab2, 2026-09-27): XCD-major (b, kv_head) remap on top of longest-first.
# Model: hardware sends linear block id ``lin`` to XCD ``lin % NUM_XCD`` (kyle-learnings,
# num_xcc = 8 per KFD). _lpt_block_id keeps x = gx-1-rank untouched (so every WG's causal
# work, and hence the dispatch balance, is byte-for-byte the champion's) and only permutes
# which (y, z) group a slot ``rem`` in [0, gy*gz) of one rank step gets.
#   off    : grp = rem (champion). With gyz % 8 == 0, XCD = rem % 8, so each (b, kvh) group
#            already lives on ONE XCD; prod (gy=8): XCD k = kv_head k of all 4 batches.
#   bmajor : when gyz % 8 == 0, XCD k owns the CONTIGUOUS groups [k*q, (k+1)*q), q = gyz/8
#            (y fastest): prod XCD k = kv_heads 4(k%2)..+3 of batch k//2. Other grids: off.
#   spread : anti-locality control, grp = (rem + rank) % gyz: a group moves XCD every rank.
# All three are bijections of [0, gyz) for each rank (tools/bijection_check.py).
XCD_REMAP = "__MODE__"
NUM_XCD = 8


def _xcd_group(rem, rank, gyz):
    if XCD_REMAP == "off":
        return rem
    if XCD_REMAP == "spread":
        return (rem + rank) % gyz
    assert XCD_REMAP == "bmajor", XCD_REMAP
    nx = fx.Int32(NUM_XCD)
    q = gyz // nx
    ok = fx.Int32(1) - fx.min(gyz % nx, fx.Int32(1))  # 1 iff gyz % 8 == 0 (branch-free gate)
    mapped = (rem % nx) * q + rem // nx
    return rem + ok * (mapped - rem)


def _lpt_block_id(axis):'''.replace("__MODE__", mode)
assert s.count(old_tail) == 1 and s.count("def _lpt_block_id(axis):") == 1
s = s.replace(old_tail, new_tail).replace("def _lpt_block_id(axis):", helper)
f.write_text(s)
print("patched", f, mode)
