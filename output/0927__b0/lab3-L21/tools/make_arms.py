#!/usr/bin/env python3
"""Build the L21 (fixed running max) arms from a verbatim copy of the fwd champion (round 4).

Every arm is a full copy of arms/champ_copy with ONE kernel-source patch (the same for all
arms) plus a different value of the module constant L21_MODE:

  off     champion code path (patched source, mode off)  -> must be ISA-identical to champion
  fmax0   plain fixed max: m = L21_FMAX0 for every tile, no max tree, no rescale, no guard
  ft1     first-K-tile max: tile start_tile runs the champion softmax (sets m = row max),
          every later tile uses that m fixed, no guard
  fmax0g  m seeded to L21_FMAX0, every tile fixed-m fast path + per-tile guard
          (tile_sum > 2^64 or d_new < 2^-60, ballot) -> wave-uniform slow path = champion
          softmax (+ O *= corr); m_seed = BIG_NEG where d_prev == 0
  bnegg   same as fmax0g with m seeded BIG_NEG: tile 0 always takes the slow path, so m is
          the first-tile row max, later tiles fast path + guard (no peeled loop)
  ft1g    ft1 (peeled first tile) + the same guard on later tiles

No index / address expression changes; the peeled loop only changes the [lo, hi) bounds
of the existing tile loops (bounds proof in tools/bounds_proof.py).
"""
import json, pathlib, re, shutil, sys

L = pathlib.Path(__file__).resolve().parent.parent
SRC = L / "arms" / "champ_copy"
KF = "flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py"


def sub1(txt, old, new):
    n = txt.count(old)
    assert n == 1, (n, old[:80])
    return txt.replace(old, new)


def patch(txt):
    # 1. constants
    txt = sub1(txt, "BIG_NEG = -1.0e30\n", '''BIG_NEG = -1.0e30

# L21 (lab3-L21): fixed running max. "off" = champion. See output/0927__b0/lab3-L21/RESULT.md.
L21_MODE = "off"
L21_FMAX0 = 0.0          # natural-log units (softmax_scale already folded into Q)
L21_GUARD_HI = 2.0**64   # per-tile row-sum above this -> slow path (catches exp2 overflow = inf)
L21_GUARD_LO = 2.0**-60  # running denom below this -> slow path (catches all-underflow rows)
''')

    # 2. fixed-max softmax (fast path) -- appended right before _pv_gemm
    txt = sub1(txt, "def _pv_gemm(", '''def _softmax_fixed(
    *,
    s_list,
    m_list,
    d_prev_list,
    lane_idx,
    n_block,
    kv_pos_base=None,
    q_max_list=None,
    q_min_list=None,
    kv_len=None,
    elem_dtype,
    guard,
):
    """L21 fast path: p = exp(S - m) with m FIXED (no row max, no corr, no O rescale).

    Masking, exp and row-sum are the champion's ``_softmax`` code verbatim; only the max tree,
    the deferred-rescale ballot and corr are gone. Returns ``(p, d_new, need)``: need is a
    wave-uniform i1 (guard=True) that is set when any lane's per-tile row sum exceeds
    L21_GUARD_HI (exp2 overflow shows up as inf here) or the running denom is still below
    L21_GUARD_LO (every p underflowed / row empty so far); None when guard=False.
    """
    NKV = n_block // WMMA_N
    f32 = T.f32
    fast = arith.FastMathFlags.fast
    neg_inf = fx.Float32(float("-inf"))
    zero = fx.Float32(0.0)
    log2e = fx.Float32(LOG2E)

    def fadd(a, b):
        return fx.Float32(arith.addf(_raw(a), _raw(b), fastmath=fast))

    _FF = arith.FastMathFlags
    _no_reassoc = _FF.nnan | _FF.ninf | _FF.nsz | _FF.arcp | _FF.contract | _FF.afn

    def fadd_t(a, b):
        return fx.Float32(arith.addf(_raw(a), _raw(b), fastmath=_no_reassoc))

    def fsub(a, b):
        return fx.Float32(arith.subf(_raw(a), _raw(b), fastmath=fast))

    def fmul(a, b):
        return fx.Float32(arith.mulf(_raw(a), _raw(b), fastmath=fast))

    def exp2(x):
        return fx.Float32(rocdl.exp2(f32, _raw(x)))

    sel_lo, sel_hi = _raw(fx.Int32(0x76543210)), _raw(fx.Int32(0xFEDCBA98))

    def peer(v):
        return fx.Float32(
            rocdl.permlanex16(
                f32, _raw(v), _raw(v), sel_lo, sel_hi, fi=False, bound_control=False
            )
        )

    khalf = lane_idx // fx.Int32(WMMA_M)
    R = len(s_list)
    q_max_list = q_max_list if q_max_list is not None else [None] * R
    q_min_list = q_min_list if q_min_list is not None else [None] * R

    s_masked_list = []
    for r in range(R):
        s = s_list[r]
        q_max, q_min = q_max_list[r], q_min_list[r]
        s_masked = []
        for kvt in range(NKV):
            svec = fx.Vector(_ir(s[kvt]))
            for i in range(8):
                sval = fx.Float32(svec[i])
                if q_max is not None or q_min is not None or kv_len is not None:
                    kv_pos = (
                        kv_pos_base + khalf * fx.Int32(8) + fx.Int32(kvt * WMMA_N + i)
                    )
                    if q_max is not None:
                        ubound = (
                            q_max
                            if kv_len is None
                            else fx.min(q_max, kv_len - fx.Int32(1))
                        )
                        sval = (kv_pos > ubound).select(neg_inf, sval)
                    if q_min is not None:
                        sval = (kv_pos < q_min).select(neg_inf, sval)
                    if kv_len is not None and q_max is None:
                        sval = (kv_pos >= kv_len).select(neg_inf, sval)
                s_masked.append(sval)
        s_masked_list.append(s_masked)

    neg_m_list = [fsub(zero, fmul(m_list[r], log2e)) for r in range(R)]

    p_list, p_flat_list = [], []
    for r in range(R):
        neg_m, s_masked = neg_m_list[r], s_masked_list[r]
        p, p_flat, idx = [], [], 0
        for kvt in range(NKV):
            pe = []
            l2 = fx.Vector.from_elements([log2e], fx.Float32).broadcast_to(2)
            n2 = fx.Vector.from_elements([neg_m], fx.Float32).broadcast_to(2)
            for i in range(0, 8, 2):
                sv = fx.Vector.from_elements(
                    [s_masked[idx], s_masked[idx + 1]], fx.Float32
                )
                av = fx.Vector(fmath.fma(_ir(sv), _ir(l2), _ir(n2)))
                for e in range(2):
                    pj = exp2(fx.Float32(av[e]))
                    pe.append(pj)
                    p_flat.append(pj)
                idx += 2
            p.append(fx.Vector.from_elements(pe, fx.Float32).to(elem_dtype))
        p_list.append(p)
        p_flat_list.append(p_flat)

    add3 = lambda a, b, c: fadd_t(fadd_t(a, b), c)
    if R == 2:
        def vadd_t(a, b):
            return fx.Vector(arith.addf(_ir(a), _ir(b), fastmath=_no_reassoc))

        vadd3 = lambda a, b, c: vadd_t(vadd_t(a, b), c)
        leaves = [
            fx.Vector.from_elements([p_flat_list[0][i], p_flat_list[1][i]], fx.Float32)
            for i in range(len(p_flat_list[0]))
        ]
        (tot,) = _tree_reduce_multi([leaves], vadd3, vadd_t)
        local_sum_list = [fx.Float32(tot[0]), fx.Float32(tot[1])]
    else:
        local_sum_list = _tree_reduce_multi(p_flat_list, add3, fadd_t)

    d_new_list, tsum_list = [], []
    for r in range(R):
        tsum = fadd(local_sum_list[r], peer(local_sum_list[r]))
        tsum_list.append(tsum)
        d_new_list.append(fadd(d_prev_list[r], tsum))

    need = None
    if guard:
        lane_need = None
        for r in range(R):
            nr = (tsum_list[r] > fx.Float32(L21_GUARD_HI)) | (
                d_new_list[r] < fx.Float32(L21_GUARD_LO)
            )
            lane_need = nr if lane_need is None else (lane_need | nr)
        mask = rocdl.ballot(fx.Int32.ir_type, lane_need)
        need = fx.Int32(mask) != fx.Int32(0)
    return p_list, d_new_list, need


def _pv_gemm(''')

    # 3. main_loop gets a compile-time softmax mode
    txt = sub1(txt, "    def main_loop(t, state, *, mask_left, mask_right, kv_len):\n",
               "    def main_loop(t, state, *, mask_left, mask_right, kv_len, smode=\"full\"):\n")

    old_sm_start = "        p_list, m_new_list, d_new_list, corr_list, do_rescale_list = _softmax(\n"
    i0 = txt.index(old_sm_start)
    i1 = txt.index("        o_new_list = _pv_gemm(", i0)
    champ_block = txt[i0:i1]
    # indent champion block under `if smode == "full":`
    champ_ind = "".join(("    " + ln if ln.strip() else ln) for ln in champ_block.splitlines(True))
    new_block = '''        if smode == "full":
''' + champ_ind + '''        else:
            p_list, d_new_list, do_slow = _softmax_fixed(
                s_list=s_list,
                m_list=m_prev,
                d_prev_list=d_prev,
                lane_idx=lane_idx,
                n_block=n_block,
                kv_pos_base=kv_tile_start,
                q_max_list=q_max_list,
                q_min_list=q_min_list,
                kv_len=kv_len,
                elem_dtype=elem_dtype,
                guard=(smode == "guard"),
            )
            m_new_list = list(m_prev)
            o_resc_list = [
                [fx.Vector(_ir(o_acc[qt][dt])) for dt in range(d_tiles)]
                for qt in range(R)
            ]
            if do_slow is not None:
                # Guard fired somewhere in the wave: redo this tile with the champion's
                # online-softmax update from the ENTRY state (m_prev, d_prev, o_acc).
                # A row with nothing accumulated yet (d_prev == 0, O == 0) restarts
                # from BIG_NEG so m may move DOWN to its real row max (underflow case).
                def _l21_slow():
                    m_seed = [
                        (d_prev[qt] > fx.Float32(0.0)).select(
                            m_prev[qt], fx.Float32(BIG_NEG)
                        )
                        for qt in range(R)
                    ]
                    p2, m2, d2, corr2, _unused = _softmax(
                        s_list=s_list,
                        m_prev_list=m_seed,
                        d_prev_list=d_prev,
                        lane_idx=lane_idx,
                        n_block=n_block,
                        kv_pos_base=kv_tile_start,
                        q_max_list=q_max_list,
                        q_min_list=q_min_list,
                        kv_len=kv_len,
                        elem_dtype=elem_dtype,
                    )
                    o2 = []
                    for qt in range(R):
                        cv = fx.Vector.from_elements(
                            [corr2[qt]], fx.Float32
                        ).broadcast_to(8)
                        o2.append(
                            [fx.Vector(_ir(o_acc[qt][dt])) * cv for dt in range(d_tiles)]
                        )
                    return p2, list(m2), list(d2), o2

                @flyc.jit
                def _l21_guarded(p_r, m_r, d_r, o_r, do_slow):
                    if do_slow:
                        p_r, m_r, d_r, o_r = _l21_slow()
                    return p_r, m_r, d_r, o_r

                p_list, m_new_list, d_new_list, o_resc_list = _l21_guarded(
                    p_list, m_new_list, d_new_list, o_resc_list, do_slow
                )

'''
    txt = txt[:i0] + new_block + txt[i1:]

    # 4. _run_tiles threads smode through
    txt = sub1(txt, "    def _run_tiles(state, lo_i32, hi_i32, *, mask_left, mask_right, kv_len):\n",
               "    def _run_tiles(state, lo_i32, hi_i32, *, mask_left, mask_right, kv_len, smode=\"full\"):\n")
    txt = sub1(txt, """                mask_right=mask_right,
                kv_len=kv_len,
            )
            final_state = yield next_state""", """                mask_right=mask_right,
                kv_len=kv_len,
                smode=smode,
            )
            final_state = yield next_state""")

    # 5. m seed
    txt = sub1(txt, "        m_init = [fx.Float32(BIG_NEG) for _ in range(R)]\n",
               "        m_init = [fx.Float32(L21_FMAX0 if L21_MODE in (\"fmax0\", \"fmax0g\") else BIG_NEG)\n"
               "                  for _ in range(R)]\n")

    # 6. the three tile loops -> mode-aware (peeled first tile for ft1/ft1g)
    a = txt.index("    state = _init\n    if mask_left:\n        state = _run_tiles(")
    b = txt.index("    final = state\n", a)
    old_loops = txt[a:b]
    new_loops = '''    state = _init
    if L21_MODE == "off":
''' + "".join(("    " + ln if ln.strip() else ln) for ln in old_loops[len("    state = _init\n"):].splitlines(True)) + '''    else:
        _SM = {"fmax0": "fixed", "ft1": "fixed", "fmax0g": "guard", "bnegg": "guard",
               "ft1g": "guard"}[L21_MODE]
        s_lo = start_tile
        if L21_MODE in ("ft1", "ft1g"):
            # Peel tile start_tile (n_tiles >= start_tile + 1 always, see bounds_proof.py)
            # with the champion softmax and the most general masking (every mask is
            # exact on any tile, so running it on an interior tile is a no-op mask).
            state = _run_tiles(
                state,
                start_tile,
                start_tile + fx.Int32(1),
                mask_left=mask_left,
                mask_right=mask_right,
                kv_len=kv_len,
                smode="full",
            )
            s_lo = start_tile + fx.Int32(1)
        c_lo = fx.max(clean_lo, s_lo)
        c_hi = fx.max(clean_hi, s_lo)
        if mask_left:
            state = _run_tiles(
                state,
                s_lo,
                c_lo,
                mask_left=mask_left,
                mask_right=mask_right,
                kv_len=None,
                smode=_SM,
            )
        state = _run_tiles(
            state, c_lo, c_hi, mask_left=None, mask_right=None, kv_len=None, smode=_SM
        )
        state = _run_tiles(
            state,
            c_hi,
            fx.Int32(n_tiles),
            mask_left=mask_left,
            mask_right=mask_right,
            kv_len=kv_len,
            smode=_SM,
        )
'''
    txt = txt[:a] + new_loops + txt[b:]
    return txt


def main():
    modes = sys.argv[1:] or ["off", "fmax0", "ft1", "fmax0g", "bnegg", "ft1g"]
    base = (SRC / KF).read_text()
    patched = patch(base)
    for m in modes:
        d = L / "arms" / m
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(SRC, d, ignore=shutil.ignore_patterns("__pycache__"))
        t = sub1(patched, 'L21_MODE = "off"\n', f'L21_MODE = "{m}"\n')
        (d / KF).write_text(t)
        (d / "L21_ARM.json").write_text(json.dumps({"mode": m}))
        print("built", d)


if __name__ == "__main__":
    main()
