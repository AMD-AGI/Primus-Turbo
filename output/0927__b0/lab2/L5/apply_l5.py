"""Apply lever L5 (causal-aligned q-tile origin) to a copy of the champion tree.

Usage: python3 apply_l5.py <arm_dir> <mode>
  mode = rot   : L5 always on (bshd + thd), module constant Q_ORIGIN_ALIGN = True
  mode = gate  : L5 is a compile-time kernel key; the bshd host path enables it only when the
                 origin shift pad_w != 0 (never at fast/proxy/prod), thd always on.
  mode = off   : the plumbing with Q_ORIGIN_ALIGN = False (ISA must equal the champion's).
Pure text edits with exact-match asserts; run on a FRESH copy of op/current.
"""
import pathlib, sys

arm = pathlib.Path(sys.argv[1]); mode = sys.argv[2]
K = arm / "flydsl_fwd" / "fmha_fwd_prefill_a16w16_m32x8.py"
M = arm / "flydsl_fwd" / "fmha_b16_buffer_managers.py"
k = K.read_text(); m = M.read_text()

def sub(s, old, new, count=1):
    n = s.count(old)
    assert n == count, (old[:80], n)
    return s.replace(old, new)

# ---------------------------------------------------------------- kernel: constant + helper
k = sub(k, 'O_VARIANT = "v3"\n', 'O_VARIANT = "v3"\n'
 '# L5 causal-aligned q-tile origin. The grid pads the packed (seq, head) row space up to\n'
 '# grid_x*BLOCK_M; the champion puts that padding at the END (the last q tile = the one with the\n'
 '# longest causal KV range). With Q_ORIGIN_ALIGN the packed rows are ROTATED by\n'
 '# pad_w = (pad mod BLOCK_M) rounded down to whole waves, so the padding moves into tile 0 (the\n'
 '# shortest tile) and every tile ends pad_w rows earlier. Rotation is a bijection on\n'
 '# [0, grid_x*BLOCK_M); the wrapped rows are exactly padding rows (>= q_len*gqa), so every\n'
 '# existing seq>=q_len mask handles them. pad == 0 at every scored shape (sq*gqa % 256 == 0).\n'
 f'Q_ORIGIN_ALIGN = {mode != "off"}\n')

k = sub(k, '''def _packed_tile_indices(gqa_ratio, warp_idx, lane_idx):''',
'''def _q_row_origin(gqa_ratio, q_len, warp_idx, align):
    """L5: this wave's first packed row plus the WG's valid packed-row bounds [lo, hi].

    align=False reproduces the champion exactly: warp_row0 = bx*BLOCK_M + w*RPW,
    lo = bx*BLOCK_M, hi = bx*BLOCK_M + BLOCK_M - 1.
    align=True rotates by pad_w (a multiple of RPW, 0 <= pad_w <= pad, pad_w < BLOCK_M), so a
    wave never straddles the wrap point and warp_row0 stays in [0, gx*BLOCK_M).
    """
    rpw = WMMA_ROW_PER_WAVE * WMMA_M
    base = _lpt_block_id("x") * fx.Int32(BLOCK_M)
    if not align:
        return base + warp_idx * fx.Int32(rpw), base, base + fx.Int32(BLOCK_M - 1)
    tot = fx.Int32(fx.grid_dim.x) * fx.Int32(BLOCK_M)
    pad = fx.max(tot - q_len * fx.Int32(gqa_ratio), fx.Int32(0))
    pad_w = ((pad // fx.Int32(rpw)) % fx.Int32(BLOCK_M // rpw)) * fx.Int32(rpw)
    r = base + warp_idx * fx.Int32(rpw) - pad_w
    r = (r < fx.Int32(0)).select(r + tot, r)
    lo = fx.max(base - pad_w, fx.Int32(0))
    hi = base + fx.Int32(BLOCK_M - 1) - pad_w
    return r, lo, hi


def _packed_tile_indices(gqa_ratio, warp_idx, lane_idx, warp_row0=None):''')

k = sub(k, '''    warp_row0 = _lpt_block_id("x") * BLOCK_M + warp_idx * (
        WMMA_ROW_PER_WAVE * WMMA_M
    )
    q_head_idx = []''', '''    if warp_row0 is None:
        warp_row0 = _lpt_block_id("x") * BLOCK_M + warp_idx * (
            WMMA_ROW_PER_WAVE * WMMA_M
        )
    q_head_idx = []''')

# _core_attention signature + use
k = sub(k, '''    elem_dtype,  # compile-time fx.BFloat16 / fx.Float16 for Q/K/V/P/O fragments
):
    """Layout-agnostic m32x8 compute''', '''    elem_dtype,  # compile-time fx.BFloat16 / fx.Float16 for Q/K/V/P/O fragments
    q_origin_align=False,  # compile-time L5 (see Q_ORIGIN_ALIGN)
):
    """Layout-agnostic m32x8 compute''')
k = sub(k, '''    kv_head, q_head_idx, seq_idx = _packed_tile_indices(gqa_ratio, warp_idx, lane_idx)

    # K/V staging''', '''    if q_origin_align:
        assert USE_TDM_LOADER and O_VARIANT == "v3", "L5 is wired for Q V2 + O V3 only"
        warp_row0, wg_row_lo, wg_row_hi = _q_row_origin(
            gqa_ratio, q_len, warp_idx, True
        )
        kv_head, q_head_idx, seq_idx = _packed_tile_indices(
            gqa_ratio, warp_idx, lane_idx, warp_row0=warp_row0
        )
    else:
        # champion op order, emitted verbatim (keeps the L5-off IR byte-identical)
        kv_head, q_head_idx, seq_idx = _packed_tile_indices(gqa_ratio, warp_idx, lane_idx)

    # K/V staging''')
k = sub(k, '''        block_x=_lpt_block_id("x"),
        warp_idx=warp_idx,
        lane_idx=lane_idx,
        ptr_lds=q_lds_base,
    )''', '''        block_x=_lpt_block_id("x"),
        warp_idx=warp_idx,
        lane_idx=lane_idx,
        ptr_lds=q_lds_base,
        packed_row0=warp_row0 if q_origin_align else None,
    )''')
k = sub(k, '''        wg_max_seq = (block_x * fx.Int32(BLOCK_M) + fx.Int32(BLOCK_M - 1)) // fx.Int32(
            gqa_ratio
        )''', '''        wg_max_seq = (
            wg_row_hi
            if q_origin_align
            else block_x * fx.Int32(BLOCK_M) + fx.Int32(BLOCK_M - 1)
        ) // fx.Int32(gqa_ratio)''')
k = sub(k, '''        wg_min_seq = (block_x * fx.Int32(BLOCK_M)) // fx.Int32(gqa_ratio)''',
           '''        wg_min_seq = (
            wg_row_lo if q_origin_align else block_x * fx.Int32(BLOCK_M)
        ) // fx.Int32(gqa_ratio)''', count=2)
k = sub(k, '''        wg_max_seq = fx.min(
            (block_x * fx.Int32(BLOCK_M) + fx.Int32(BLOCK_M - 1))
            // fx.Int32(gqa_ratio),
            q_len - fx.Int32(1),
        )''', '''        wg_max_seq = fx.min(
            (
                wg_row_hi
                if q_origin_align
                else block_x * fx.Int32(BLOCK_M) + fx.Int32(BLOCK_M - 1)
            )
            // fx.Int32(gqa_ratio),
            q_len - fx.Int32(1),
        )''')
k = sub(k, '''            ptr_lds=o_lds_base,
            o_frags=o_norm,
            qtile=qt,
        )''', '''            ptr_lds=o_lds_base,
            o_frags=o_norm,
            qtile=qt,
            **({"warp_base": warp_row0} if q_origin_align else {}),
        )''')

# build(): compile-time flag, threaded into both entries' _ca_kw
k = sub(k, '''    gqa_ratio: int = 1,
):
    """Build the m32x8 device kernel''', '''    gqa_ratio: int = 1,
    q_origin_align: bool = None,
):
    """Build the m32x8 device kernel''')
k = sub(k, '''    GQA_RATIO = int(gqa_ratio)
''', '''    GQA_RATIO = int(gqa_ratio)
    # L5 only pays under a pure causal right edge: the CPU sweep (tools/l5_proof.out) shows 0 KV
    # visits saved non-causal and MORE visits under a finite left window, so both keep the champion.
    Q_ORIGIN = bool(Q_ORIGIN_ALIGN if q_origin_align is None else q_origin_align) and (
        MASK_RIGHT and not MASK_LEFT
    )
''')
k = sub(k, '''                    "elem_dtype": ELEM_DTYPE,
                }''', '''                    "elem_dtype": ELEM_DTYPE,
                    "q_origin_align": Q_ORIGIN,
                }''')
k = sub(k, '''            "elem_dtype": ELEM_DTYPE,
        }''', '''            "elem_dtype": ELEM_DTYPE,
            "q_origin_align": Q_ORIGIN,
        }''')

if mode == "gate":
    # bshd kernel key gains the L5 flag; host enables it only when the rotation is non-zero.
    k = sub(k, '''    qk_hdim: int = DEFAULT_QK_HDIM,
    dtype_str: str = DEFAULT_DTYPE,
):
    key = (
        "bshd",''', '''    qk_hdim: int = DEFAULT_QK_HDIM,
    dtype_str: str = DEFAULT_DTYPE,
    q_origin_align: bool = False,
):
    key = (
        "bshd",''')
    k = sub(k, '''        int(qk_hdim),
        str(dtype_str),
    )
    if key in _launch_fns:
        return
    kernel = build_fmha_fwd_prefill_a16w16_m32x8(
        layout="bshd",''', '''        int(qk_hdim),
        str(dtype_str),
    ) + (("l5",) if q_origin_align else ())
    if key in _launch_fns:
        return
    kernel = build_fmha_fwd_prefill_a16w16_m32x8(
        layout="bshd",
        q_origin_align=bool(q_origin_align),''')
    k = sub(k, '''    _ensure_bshd_kernel(
        mask_left,
        mask_right,
        bool(return_lse),
        has_sink,
        gqa,
        qk_hdim=qk_hdim,
        dtype_str=dtype_str,
    )
''', '''    # L5 gate: rotate the q origin only when there is padding to move (pad_w != 0). At every
    # scored shape seq_len_q*gqa % BLOCK_M == 0, so the champion kernel runs unchanged.
    _rpw = WMMA_ROW_PER_WAVE * WMMA_M
    _pad = -(-(seq_len_q * gqa) // BLOCK_M) * BLOCK_M - seq_len_q * gqa
    l5 = ((_pad // _rpw) % (BLOCK_M // _rpw)) != 0 and mask_right and not mask_left
    _ensure_bshd_kernel(
        mask_left,
        mask_right,
        bool(return_lse),
        has_sink,
        gqa,
        qk_hdim=qk_hdim,
        dtype_str=dtype_str,
        q_origin_align=l5,
    )
''')
    k = sub(k, '''                qk_hdim,
                dtype_str,
            )
        ],
        out,
        q,
        k,
        v,
        lse_ptr,
        sink_ptr,
        softmax_scale,
        stride_q_seq,''', '''                qk_hdim,
                dtype_str,
            )
            + (("l5",) if l5 else ())
        ],
        out,
        q,
        k,
        v,
        lse_ptr,
        sink_ptr,
        softmax_scale,
        stride_q_seq,''')

# ---------------------------------------------------------------- managers: row overrides
m = sub(m, '''        block_x,
        warp_idx,
        lane_idx,
        ptr_lds,
    ):
        """Issue this wave's per-warp TDM copy''', '''        block_x,
        warp_idx,
        lane_idx,
        ptr_lds,
        packed_row0=None,  # L5: rotated first packed row of this wave (overrides block_x)
    ):
        """Issue this wave's per-warp TDM copy''')
m = sub(m, '''        packed_row0 = block_x * fx.Int32(self.block_m) + warp_idx * fx.Int32(
            self.rows_per_warp
        )
        seq0 = packed_row0 // fx.Int32(gqa)''', '''        if packed_row0 is None:
            packed_row0 = block_x * fx.Int32(self.block_m) + warp_idx * fx.Int32(
                self.rows_per_warp
            )
        seq0 = packed_row0 // fx.Int32(gqa)''')
# OManager16bV3 only (the 3rd store_o_to_vram); add warp_base override
i3 = m.index("class OManager16bV3")
head, tail = m[:i3], m[i3:]
tail = sub(tail, '''        o_frags,
        qtile=0,
    ):''', '''        o_frags,
        qtile=0,
        warp_base=None,  # L5: rotated first packed row of this wave (overrides block_x)
    ):''')
tail = sub(tail, '''            self._warp_base = block_x * fx.Int32(self.block_m) + warp_idx * fx.Int32(
                self.rows_per_warp
            )''', '''            self._warp_base = (
                warp_base
                if warp_base is not None
                else block_x * fx.Int32(self.block_m)
                + warp_idx * fx.Int32(self.rows_per_warp)
            )''')
m = head + tail
K.write_text(k); M.write_text(m)
print("applied", mode, "to", arm)
