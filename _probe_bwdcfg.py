#!/usr/bin/env python3
"""Backward config A/B that never touches the kernel tree.

Patches the HOST-side dispatch plan / gates of flash_attn_bwd (and, for ``kw:`` arms, the
dkdv builder's kwargs), so an arm can be priced without a source edit and a pricing probe
can never be mistaken for a candidate. One shape per process, min of ``reps``.

usage: _probe_bwdcfg.py <B> <Hq> <Hkv> <S> <D> <arm> [reps] [W]
arms:  base | equal | merge | fills4 | nored | qsp<N> | slices<N> | qint | qintd | tslice
       | hbr | wdiv<N> | bkv<N>
       | bq<N> | halfband | notailred | ldcm<N> | ilv<N> | pad<N> | plan:<qs>:<n,...>
       | kw:name=val[,name=val...]
       | red:name=val[,name=val...]  (the fold kernel's builder, e.g. red:block=128)
Several arms combine with ``+``, e.g. ``wdiv2+nored``.

``hbr`` and ``wdiv<N>`` price the two ENDS of the dQ split-K round trip separately, which is
what the band-pair question needs and what a single combined arm cannot give: making the
bands share slots collapses the READ too (a q row's repeated reads of one slot hit L2), so
the write side has to be priced with the fold already off (wdiv<N> + nored).

``halfband`` is the two ends together: the whole dQ partial stream at HALF its bytes, which
is the upper bound on any scheme that gives one work-group two kv bands (n_bands 64 -> 32).
It halves the BYTES rather than picking alternate bands -- the read side widens the fold's
band unit and the write side drops the far half of every slot at the buffer bound -- because
bytes are what the stream costs (the whole _WSQ_BAND_ILV address-pattern axis measures flat).

``kw:q_split=`` is NOT the way to price the q-loop split: the host plans the chunk dispatch
off _qsplit_for and passes qsp_lo/n_qsp against ITS count, so overriding only the builder's
copy trips the builder's own range assert. Use the ``qsp<N>`` arm, which moves both.

★ ``kw:block_q=`` is the SAME trap and it is worse, because it does not assert -- it returns
``det=false``. _BWD_BLOCK_Q is a module constant that the odo builder, the fold builder,
_wsq_ring_for, _qsp_absolute, _qsp_cuttable and _fuse_qsplit_for all capture as a DEFAULT
ARGUMENT (frozen at def time), and no caller passes it. Overriding the dkdv builder's copy
alone leaves the fold, the odo kernel, the partial ring and the q_split subset semantics
planning at 64 against a body running at 128 -- which is round 3's documented "two dispatches
of one q_split subset stamp on each other's slots" failure, wearing a register costume.
Use the ``bq<N>`` arm, which moves the constant AND rebinds every frozen default.

★ A ``kw:`` override is INERT wherever the host already passes the value you are asking for,
and it fails SILENTLY -- same binary, same wall, and it reads like a tested donor. The live
case is ``bkv256`` on D128: ``_fuse_halves`` is already 2 there, so its ``_pair`` branch
already sets g3d=2, q_pref=0, g3_defer=0, g3_kreg=0 and k_reg=0. ``bkv256+kw:kv_halves=2``,
``bkv256+kw:g3d=2`` and both together dump byte-identical ISA (576 B spill, 880 scratch ops,
23907 instructions). Print the builder kwargs, or diff the ISA against the arm without the
override, before recording a ``kw:`` reading as a donor screen.

★ Two more inert families, both already recorded as measured verdicts by mistake. On the
DEFAULT D128 path the host passes ``q_pref=True`` and ``g3_kreg=True``, so ``kw:q_pref=1``
and ``kw:g3_kreg=1`` are the same binary as base -- their "straddles base" readings are the
signature of a no-op arm, not of a knob without a lever. And a positive ``kw:g3d=`` cannot
reach the ISA at all wherever G3_KREG holds the whole band's K^T (the D128 fused path does):
the depth sizes a read ring that no longer exists, so 4/8/10 compile byte-identical (the
builder's own G3D comment records this for D64). Live knobs to override are the ones whose
host value is None or False.

★ Live and worth naming: ``kw:wsq_a16=1`` / ``=2`` are the packed-bf16 atomic dQ arms (dQ is
wrong under both, like wsq_atomic); pair either with ``nored``, since the scheme they stand
for deletes the fold and comparing against a base that still runs one prices the wrong thing.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import primus_turbo.flydsl.attention.flash_attn_bwd as bwd
from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    flash_attn_sbhd_flydsl_backward_impl,
    flash_attn_sbhd_flydsl_forward_impl,
)

DEV, DT = "cuda", torch.bfloat16


def _rebind_block_q(n):
    """Move _BWD_BLOCK_Q everywhere it is read, including defaults frozen at def time."""
    bwd._BWD_BLOCK_Q = n
    for fn in vars(bwd).values():
        d = getattr(fn, "__defaults__", None)
        if not d:
            continue
        names = fn.__code__.co_varnames[: fn.__code__.co_argcount]
        tail = names[len(names) - len(d) :]
        if "block_q" in tail:
            i = tail.index("block_q")
            fn.__defaults__ = d[:i] + (n,) + d[i + 1 :]


def _bat_cut(p, B, n, sbhd, how):
    """Split SBHD chunks into ``n`` batch pieces; how = 1 first only, -1 all but last, 0 all."""
    if not sbhd or n < 2 or B % n or any(c[0] is not None for c in p):
        return p
    per, out = B // n, []
    for i, c in enumerate(p):
        b, lo, nq = c[:3]
        cut = how == 0 or (how == 1 and i == 0) or (how == -1 and i < len(p) - 1)
        out += [(b, lo, nq, j * per, per) for j in range(n)] if cut else [tuple(c)]
    return out


def patch(arm):
    plan = bwd._pipe_chunks
    if arm == "equal":  # one q_split subset per chunk, every band width
        bwd._pipe_chunks = lambda B, qs, bkv, sq, hd=64, sbhd=False, **_: (
            [(None, s, 1) for s in range(qs)] if sbhd else plan(B, qs, bkv, sq, hd, sbhd)
        )
    elif arm == "merge":  # the pair merge at every band width, i.e. [1,2,1] at q_split=4
        bwd._pipe_chunks = lambda B, qs, bkv, sq, hd=64, sbhd=False, **_: (
            [(None, 0, 1), (None, 1, 2)] + [(None, s, 1) for s in range(3, qs)]
            if sbhd and qs >= 4
            else plan(B, qs, bkv, sq, hd, sbhd)
        )
    elif arm == "fills4":  # the flat pipeline-width gate, before it scaled with the band
        bwd._dq_pipe_fills = lambda block_kv: 4
    elif arm.startswith("pfills"):  # the pipeline-width gate at N fills, whatever the band
        # The gate is checked on a ONE-SPLIT chunk, so a shape whose single-split chunk falls
        # under it takes the unpiped path and exposes its WHOLE fold. Lowering the gate here
        # prices what a WIDER chunk would buy on such a shape; pair it with plan: to set the
        # chunk widths the lowered gate is supposed to stand for.
        n = int(arm[6:])
        bwd._dq_pipe_fills = lambda block_kv: n
    elif arm.startswith("bplan"):  # SBHD pipeline cut on the BATCH axis, N batches a chunk
        # The split axis is not the only one an SBHD chunk can cut: the body takes its batch
        # range as a compile-time base plus the grid's count (bat_lo), so no tensor has to be
        # sliced. A batch chunk holds n_bands*Hkv*(N)*q_split work-groups against a one-subset
        # split chunk's n_bands*Hkv*B, so it is the WIDER cut wherever q_split > B/N -- which
        # is what a shape under the _DQ_PIPE_FILLS gate needs. Pair with pfills to open it.
        per = int(arm[5:]) if arm[5:] else 1
        bwd._pipe_chunks = lambda B, qs, bkv, sq, hd=64, sbhd=False, **_: (
            [(None, 0, qs, b, per) for b in range(0, B, per)]
            if sbhd
            else plan(B, qs, bkv, sq, hd, sbhd)
        )
    elif arm.startswith("bxs"):  # every chunk a (batch sub-range x one split subset), N a batch
        # bplan's body is FASTER than the split plan's (one kv-head per XCD) but its folds are
        # a whole batch wide and cost more than they hide; this keeps the subset-wide fold
        # cadence and takes the batch range with it.
        per = int(arm[3:]) if arm[3:] else 1
        bwd._pipe_chunks = lambda B, qs, bkv, sq, hd=64, sbhd=False, **_: (
            [(None, s, 1, b, per) for b in range(0, B, per) for s in range(qs)]
            if sbhd
            else plan(B, qs, bkv, sq, hd, sbhd)
        )
    elif arm == "nogrid":  # the uniform batch cut off, i.e. the tail-only cut it replaced
        bwd._dq_grid_cut = lambda *a, **k: 1
    elif arm.startswith("hcut"):  # cut the FIRST chunk into N batch pieces
        # The tail cut asks when the LAST fold can stop being exposed; this asks the opposite
        # end -- when the FIRST fold can start. A hidden fold runs on the side queue and is
        # free to spill past the body it was issued under, so what bounds the pipeline is not
        # any one window but the total body the side queue has to spread over. Cutting the head
        # hands it one more window at the front without narrowing the tail's.
        n = int(arm[4:])
        bwd._pipe_chunks = lambda B, qs, bkv, sq, hd=64, sbhd=False, **kw: _bat_cut(
            plan(B, qs, bkv, sq, hd, sbhd, **kw), B, n, sbhd, 1
        )
    elif arm.startswith("bcut"):  # cut EVERY chunk but the last into N batch pieces
        n = int(arm[4:])
        bwd._pipe_chunks = lambda B, qs, bkv, sq, hd=64, sbhd=False, **kw: _bat_cut(
            plan(B, qs, bkv, sq, hd, sbhd, **kw), B, n, sbhd, -1
        )
    elif arm.startswith("acut"):  # cut EVERY chunk into N batch pieces (head and tail)
        n = int(arm[4:])
        bwd._pipe_chunks = lambda B, qs, bkv, sq, hd=64, sbhd=False, **kw: _bat_cut(
            plan(B, qs, bkv, sq, hd, sbhd, **kw), B, n, sbhd, 0
        )
    elif arm.startswith("tcutd"):  # force N batch pieces of the exposed tail at ANY band
        # _dq_tail_cut refuses the cut below the 256-row band; this overrides the whole rule
        # so the D128 cells (which take a 128-row band) can be priced on the same axis.
        n = int(arm[5:])
        bwd._dq_tail_cut = lambda wgs, batch, nq, bkv: min(n, batch)
    elif arm.startswith("tcut"):  # batch pieces of the exposed tail (see _DQ_TAIL_CUT_FILLS)
        # 0 disables the cut, i.e. the tail is one whole-batch dispatch with its fold fully
        # exposed; N asks each piece to be worth N fills of the CU array at the 256-row band.
        bwd._DQ_TAIL_CUT_FILLS = int(arm[4:])
    elif arm == "tserial0":  # the EXPOSED tail fold moves to the reduce queue
        # It then runs beside the dk/dv slot fold the caller enqueues next instead of in
        # front of it (see _DQ_TAIL_SERIAL).
        bwd._DQ_TAIL_SERIAL = False
    elif arm.startswith("wide"):  # force N q_split subsets per pipeline chunk
        n = int(arm[4:])
        bwd._dq_pipe_qsp = lambda wgs, q_split, block_kv, batch=1: n
    elif arm == "nopipe":  # no pipeline at all: one dispatch, the WHOLE fold exposed
        bwd._DQ_PIPE = False
    elif arm == "noodo":  # PRICING ONLY (delta wrong): the odo/delta pass replaced by a no-op
        # dkdv reads delta, so this only prices POSITION+TRANSPORT of the auxiliary pass, which
        # the fused D128 path still launches separately. Its first q_split piece is on the
        # caller's stream in front of chunk 0 and the rest rides the side queue under it.
        bwd.build_flash_attn_bwd_odo_module = lambda **kw: (lambda *a, **k: None)
    elif arm == "nored":  # PRICING ONLY (dQ wrong): the fold replaced by a no-op
        bwd._reduce_dq_partials = lambda *a, **k: None
    elif arm.startswith("qsp"):  # the whole q-loop split count, host plan included
        n = int(arm[3:])
        bwd._qsplit_for = lambda sq, window_left=-1, head_dim=64: 1 if window_left >= 0 else n
    elif arm.startswith("nopbs"):  # a PER-BATCH chunk's hidden fold back on one plain launch
        bwd._DQ_SLICE_PER_BATCH = False
    elif arm.startswith("pbs"):  # per-batch hidden folds sliced, N slices (see _DQ_SLICE_PER_BATCH)
        bwd._DQ_SLICE_PER_BATCH = True
        if arm[3:]:
            bwd._DQ_FOLD_SLICES = int(arm[3:])
    elif arm.startswith("slices"):  # how narrowly a hidden fold is sliced
        bwd._DQ_FOLD_SLICES = int(arm[6:])
    elif arm in ("qint", "qintd"):  # slice a hidden fold on contiguous q INTERVALS, not a modulus
        # dQ stays bitwise exact either way (same rows, same ascending band walk), so this is a
        # pure order/locality arm: qint enqueues the intervals ascending -- the body walks q
        # downwards, so the lowest interval holds the freshest partials -- and qintd descending,
        # which follows the body's own write order and keeps the heaviest rows in front.
        bwd._DQ_FOLD_Q_INTERVAL = True
        bwd._DQ_FOLD_Q_ASC = arm == "qint"
    elif arm == "tslice":  # the EXPOSED tail fold sliced like a hidden one (see _DQ_TAIL_SLICE)
        bwd._DQ_TAIL_SLICE = True
    elif arm.startswith("plan:"):  # plan:<q_split>:<n1,n2,...> an explicit chunk partition
        qs, spec = arm[5:].split(":")
        qs, ns = int(qs), [int(x) for x in spec.split(",")]
        assert sum(ns) == qs, f"plan {ns} must cover q_split {qs}"
        bwd._qsplit_for = lambda sq, window_left=-1, head_dim=64: 1 if window_left >= 0 else qs
        los = [sum(ns[:i]) for i in range(len(ns))]
        bwd._pipe_chunks = lambda B, q, bkv, sq, hd=64, sbhd=False, **_: (
            [(None, lo, n) for lo, n in zip(los, ns)] if sbhd else plan(B, q, bkv, sq, hd, sbhd)
        )
    elif arm == "hbr":  # PRICING ONLY (dQ wrong): the fold READS half the bands
        red = bwd._reduce_dq_partials
        bwd._reduce_dq_partials = lambda ws, dq, bkv, *a, **k: red(ws, dq, bkv * 2, *a, **k)
    elif arm.startswith("wdiv"):  # PRICING ONLY: the partial WRITE footprint / N (band ring)
        n = int(arm[4:])
        ring = bwd._wsq_ring_for
        bwd._wsq_ring_for = lambda nb, bkv, wl, ilv, *a, **k: (
            max(1, (nb // ilv) // n) if wl < 0 else ring(nb, bkv, wl, ilv, *a, **k)
        )
    elif arm == "halfband":  # PRICING ONLY (dQ wrong): the dQ partial stream at half its bytes
        patch("hbr")
        patch("kw:wsq_nrec=2")
    elif arm == "notailred":  # PRICING ONLY (dQ wrong): the EXPOSED tail fold dropped
        # The tail is the one fold issued on the caller's stream (see _DQ_TAIL_SERIAL); every
        # hidden slice goes to the side queue. So base - notailred is the tail's exposure, which
        # no byte accounting gives: the bytes are paid either way, the position is what costs.
        red = bwd._reduce_dq_partials
        bwd._reduce_dq_partials = lambda *a, **k: (
            None if a[6] is not bwd._SIDE_STREAM.get(a[1].device) else red(*a, **k)
        )
    elif arm.startswith("red:"):  # the FOLD kernel's work-group shape (block/uc/vec/rows_per_wg)
        # Its shape knobs were all swept on the D64 body and its own docstring warns that a
        # verdict taken at one alloc_body does not transfer, so they need their own arm here.
        red = {}
        for kv in arm[4:].split(","):
            k, v = kv.split("=")
            red[k] = int(v) if v.lstrip("-").isdigit() else v
        rbuild = bwd.build_flash_attn_bwd_dqred_module
        bwd.build_flash_attn_bwd_dqred_module = lambda **kw: rbuild(**{**kw, **red})
    elif arm.startswith("ldcm"):  # the fold's partial-load CPol (see the dqred builder's ld_cm)
        # This used to set ``bwd._DQRED_LD_CM``, a name the kernel tree never defined, so the
        # arm was a sixth inert family: it read whatever base read. Route it at the builder.
        patch(f"red:ld_cm={int(arm[4:])}")
    elif arm.startswith("ilv"):  # bands sharing one dQ partial row (see _WSQ_BAND_ILV)
        # Host-level, so the body, the fold and the workspace shape all move together and dQ
        # stays exact -- a builder-only kw: override would desync the fold's derived band_ilv.
        bwd._WSQ_BAND_ILV = int(arm[3:])
    elif arm.startswith("pad"):  # bytes between dQ partial band groups (see _WSQ_BAND_PAD)
        bwd._WSQ_BAND_PAD = int(arm[3:])
    elif arm.startswith("bkv"):  # the kv band width the fused path builds
        n = int(arm[3:])
        bwd._fuse_blockkv_for = lambda skv, d=64, wl=-1: n
    elif arm.startswith("bq"):  # the q rows per q-loop trip, host planners included
        _rebind_block_q(int(arm[2:]))
    elif arm == "nofill":  # PRICING ONLY (dQ wrong): the a16 image zeroing off, odo pass kept
        odo = bwd.build_flash_attn_bwd_odo_module
        bwd.build_flash_attn_bwd_odo_module = lambda **kw: odo(**{**kw, "fill_img": False})
    elif arm == "noperm":  # PRICING ONLY (dQ wrong): the a16 un-permute pass dropped
        bwd._unpermute_dq_a16 = lambda *a2, **kw: None
    elif arm == "noatom":  # PRICING ONLY (dQ wrong): a16's atomics become stores, same bytes
        bwd._A16_NOATOM = 1
    elif arm == "qtratom":  # PRICING ONLY (dQ wrong): the a16 atomic stream at 1/4 its bytes
        bwd._A16_NOATOM = 2
    elif arm.startswith("mod:"):  # module-level constants (e.g. mod:_A16_UC=2)
        for kv in arm[4:].split(","):
            k, v = kv.split("=")
            assert hasattr(bwd, k), f"no module constant {k}"
            setattr(bwd, k, int(v) if v.lstrip("-").isdigit() else v)
    elif arm.startswith("uq:"):  # the a16 un-permute builder's kwargs (e.g. uq:uc=2)
        over = {}
        for kv in arm[3:].split(","):
            k, v = kv.split("=")
            over[k] = int(v) if v.lstrip("-").isdigit() else v
        ub = bwd.build_flash_attn_bwd_dqa16_module
        bwd.build_flash_attn_bwd_dqa16_module = lambda *a2, **kw: ub(*a2, **{**kw, **over})
    elif arm.startswith("kw:"):
        over = {}
        for kv in arm[3:].split(","):
            k, v = kv.split("=")
            over[k] = int(v) if v.lstrip("-").isdigit() else v
        build = bwd.build_flash_attn_bwd_dkdv_module
        bwd.build_flash_attn_bwd_dkdv_module = lambda **kw: build(**{**kw, **over})
    elif arm != "base":
        raise SystemExit(f"unknown arm {arm}")


def patch_all(arm):  # "a+b" applies both, so an end of the round trip can be priced alone
    for one in arm.split("+"):
        patch(one)


def main():
    a = sys.argv[1:]
    B, Hq, Hkv, S, D = (int(x) for x in a[:5])
    arm = a[5]
    reps = int(a[6]) if len(a) > 6 else 40
    W = int(a[7]) if len(a) > 7 else -1
    ws = (W, 0) if W >= 0 else (-1, -1)
    patch_all(arm)
    sh = lambda H: torch.randn(S, B, H, D, device=DEV, dtype=DT)
    q, k, v = sh(Hq), sh(Hkv), sh(Hkv)
    do = torch.randn_like(q)
    o, lse = flash_attn_sbhd_flydsl_forward_impl(q, k, v, causal=True, window_size=ws, return_lse=True)
    lse_h = lse.view(B, S, Hq).permute(0, 2, 1)
    f = lambda: flash_attn_sbhd_flydsl_backward_impl(do, q, k, v, o, lse_h, causal=True, window_size=ws)
    det = all(torch.equal(x, y) for x, y in zip(f()[:3], f()[:3]))
    for _ in range(5):
        f()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        f()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e))
    print(json.dumps({"arm": arm, "B": B, "Hq": Hq, "S": S, "D": D, "W": W, "bwd": round(best, 4), "det": det}))


if __name__ == "__main__":
    main()
