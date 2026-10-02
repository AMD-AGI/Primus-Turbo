#!/usr/bin/env python3
"""Regenerate the gb-ruler ready copies from a base file by anchored, exact replacements.

    make_ready.py bwd-benchmark  BASE OUT    # BASE = BWD job_context/op/benchmark.py (md5 292ac70e on 2026-10-02)
    make_ready.py fwd-benchmark  BASE OUT    # BASE = FWD job_context/op/benchmark.py (md5 17e8804a, B0 backup)
    make_ready.py fwd-final-yaml BASE OUT    # BASE = FWD gfx1250-flydsl-attn-fwd_final.yaml (md5 39e3c33c, A0 after fwdjob)

Every anchor must occur exactly once in BASE; otherwise it stops and names the anchor, so a harness that a round
edited after 2026-10-02 is merged by hand instead of being patched blindly. Stdlib only.
Revision 2 (2026-10-02 afternoon, after review): preimport only under `if __name__ == "__main__"`, and the scored
ruler is resolve_ruler(--ruler, --warmup-seconds): blocked only under a profiler or with --warmup-seconds 0.
"""
import sys
from pathlib import Path

BWD_BENCH = [
    ('''  L2           flushed between timed iterations, outside the event window.
''', '''  L2           flushed between timed iterations, outside the event window.
  ruler        --ruler auto (default; prepared by the A0 operator 2026-10-02 as hint h84 and NOT installed:
               the bwd job keeps the blocked ruler, whose s6/ASM 0.972 matches the e2e's 0.974): `fast` keeps
               the blocked ruler above (MIN, h83); `proxy` and `prod` run it too (reported as blk_*) and are
               then SCORED on the gb ruler = the training operating point: before EVERY timed call NG bf16
               GEMMs 32768x4096x14336 with the IMAGE hipBLASLt library (~2.15 ms each), then the call; NG =
               --gb-ng, default the CALIBRATED gbruler.GB_NG_DEFAULT (RULER.md 8.1); palindromic rounds of
               --gb-block calls per arm; median; clock witness gb_sclk per call; a burst slower than
               --gb-max-burst-ms (8 ms x NG) voids the ruler (exit 4). --ruler blk = the pre-10-02 behaviour,
               and what every shape falls back to under a profiler (rocprofv3) or with --warmup-seconds 0.
               --aa LABEL adds a byte copy of an arm as LABEL_aa (A/A floor, aa_ratio). Code: gbruler.py;
               why: it ranks FlyDSL arms like the e2e does (s6/r29: e2e 0.754, gb 0.774, blocked 0.806;
               PT/output/1002__e2e/RESULT-{realab,e2e}.md, PT/output/1002__oe/RULER.md).
'''),
    ('''sys.path.insert(0, str(HERE.parent.parent.parent.parent / "tools"))   # op-evolve/tools

import torch  # noqa: E402
''', '''sys.path.insert(0, str(HERE.parent.parent.parent.parent / "tools"))   # op-evolve/tools
sys.path.insert(0, str(HERE))                                          # gbruler.py

import gbruler  # noqa: E402  -- stdlib only; it must run BEFORE torch is imported
if __name__ == "__main__":       # a script that imports this module keeps its own BLAS environment
    gbruler.preimport(sys.argv[1:], default_shapes="fast,proxy,prod")  # image hipBLASLt env iff gb runs

import torch  # noqa: E402
'''),
    ('''def measure(shape, arms, fns, iters, warmup_seconds, causal=True, block=9, lead=4):
''', '''def measure(shape, arms, fns, iters, warmup_seconds, causal=True, block=9, lead=4, rulers=("blk",), gb=None):
'''),
    ('''    if block <= 1:                             # legacy: arms interleaved call by call
        for i in range(iters):
            order = labels if i % 2 == 0 else labels[::-1]''', '''    if "blk" not in rulers:
        pass
    elif block <= 1:                           # legacy: arms interleaved call by call
        for i in range(iters):
            order = labels if i % 2 == 0 else labels[::-1]'''),
    ('''                for _ in range(block):
                    times[label].append(timed(label))
    sclk1 = sclk_mhz()
''', '''                for _ in range(block):
                    times[label].append(timed(label))
    # gb (A0 operator 2026-10-02, h84): the SCORED figure for proxy/prod -- every timed call right after a
    # GEMM burst, i.e. at the clock the call runs at inside training. The blocked figure above stays in the
    # row as blk_*. No L2 flush here: the burst's 1.2 GB of traffic is the flush.
    gbrec = gb.run(labels, call) if "gb" in rulers else None
    if gbrec is not None:
        void = gb.check(gbrec, shape)
        if void:
            return void
    sclk1 = sclk_mhz()
'''),
    ('''    for label in labels:
        ts = sorted(times[label])
        med = ts[len(ts) // 2]''', '''    for label in labels:
        ts = sorted(gbrec[label]["ms"] if gbrec is not None else times[label])
        med = ts[len(ts) // 2]'''),
    ('''            "sclk_start": sclk0, "sclk_end": sclk1, "causal": bool(causal),
            "order": "interleaved" if block <= 1 else f"blocked{block}+lead{lead}",
        })
    return rows
''', '''            "sclk_start": sclk0, "sclk_end": sclk1, "causal": bool(causal),
            "order": (gb.order if gbrec is not None
                      else "interleaved" if block <= 1 else f"blocked{block}+lead{lead}"),
            "ruler": "gb" if gbrec is not None else "blk",
        })
        if gbrec is not None:
            rows[-1].update(gbruler.gb_fields(gbrec[label], gb.max_sclk))
            if "blk" in rulers:
                rows[-1].update(gbruler.blk_fields(times[label], c.flop, use_min=(shape == "fast")))
    gbruler.add_aa_ratios(rows)
    return rows
'''),
    ('''    ap.add_argument("--json", help="also write every row here")
    args = ap.parse_args()

    arms = resolve_arms([a for a in args.arms.split(",") if a], args.arm_path)
    if not arms:
        ap.error("no arms given")
''', '''    ap.add_argument("--json", help="also write every row here")
    gbruler.add_args(ap)
    args = ap.parse_args()

    arms = resolve_arms([a for a in args.arms.split(",") if a], args.arm_path)
    if not arms:
        ap.error("no arms given")
    arms = gbruler.add_aa_arms(arms, args.aa)  # LABEL_aa = byte copy of LABEL (A/A noise floor)
'''),
    ('''          f"device {torch.cuda.get_device_properties(0).gcnArchName}")
''', '''          f"device {torch.cuda.get_device_properties(0).gcnArchName}")
    shapes = [s for s in args.shapes.split(",") if s]
    ruler, ruler_note = gbruler.resolve_ruler(args.ruler, args.warmup_seconds)  # profiler / warmup 0 -> blk
    plan = {s: gbruler.rulers_for(s, ruler) for s in shapes}
    print(f"# ruler {args.ruler}{'' if ruler == args.ruler else ' -> ' + ruler}: "
          + "  ".join(f"{s}={'+'.join(r)} (scored {r[-1]})" for s, r in plan.items())
          + (f"  [{ruler_note}]" if ruler_note else ""))
    # BEFORE load_impl: the burst initialises hipBLASLt with the image library before any arm's _env.py runs.
    gb = (gbruler.GbRuler(torch, args, default_iters=args.iters)
          if any("gb" in r for r in plan.values()) else None)
'''),
    ('''    fns = {label: load_impl(path) for label, path in arms}

    rows = []
    for shape in [s for s in args.shapes.split(",") if s]:
        rows += measure(shape, arms, fns, args.iters, args.warmup_seconds,
                        causal=not args.non_causal, block=args.block, lead=args.lead)
        for r in rows[-len(arms):]:''', '''    fns = {label: load_impl(path) for label, path in arms}
    if gb is not None:
        void = gb.after_load()                 # env re-assigned, 3 timed bursts, witness thread started
        if void:
            gbruler.say_void(void)
            return gbruler.VOID_RC
        print("# " + gb.header(), flush=True)

    rows = []
    for shape in shapes:
        new = measure(shape, arms, fns, args.iters, args.warmup_seconds,
                      causal=not args.non_causal, block=args.block, lead=args.lead,
                      rulers=plan[shape], gb=gb)
        if isinstance(new, str):               # the gb ruler was void for this shape: no figure at all
            gbruler.say_void(new)
            return gbruler.VOID_RC
        rows += new
        for r in rows[-len(arms):]:'''),
]

FWD_BENCH = [
    ('''  counts       FLOP and bytes from the shipped tools/op_flops.py, forward (backward=False).
"""''', '''  counts       FLOP and bytes from the shipped tools/op_flops.py, forward (backward=False).
  ruler        --ruler auto (default, A0 operator 2026-10-02, hint h51): `fast` keeps the blocked ruler above;
               `proxy` and `prod` run it too (reported as blk_*) and are then SCORED on the gb ruler = the
               training operating point: before EVERY timed call NG bf16 GEMMs 32768x4096x14336 with the IMAGE
               hipBLASLt library (~2.15 ms each), then the call; NG = --gb-ng, default the CALIBRATED
               gbruler.GB_NG_DEFAULT (RULER.md 8.1); palindromic rounds of --gb-block calls per arm; median;
               clock witness gb_sclk per call; a burst slower than --gb-max-burst-ms (8 ms x NG) voids the
               ruler (exit 4). --ruler blk = the pre-10-02 behaviour, and what every shape falls back to under
               a profiler (rocprofv3) or with --warmup-seconds 0. --aa LABEL adds a byte copy of an arm as
               LABEL_aa (A/A floor, aa_ratio). Code: gbruler.py; why: r16/ASM is 1.263 in the e2e, 1.349 on
               gb at NG 10, 1.081 blocked (PT/output/1002__e2e/RESULT-{realab,e2e}.md, PT/output/1002__oe/RULER.md).
"""'''),
    ('''sys.path.insert(0, str(TOOLS))

import torch  # noqa: E402
''', '''sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(HERE))                                        # gbruler.py

import gbruler  # noqa: E402  -- stdlib only; it must run BEFORE torch is imported
if __name__ == "__main__":       # a script that imports this module keeps its own BLAS environment
    gbruler.preimport(sys.argv[1:], default_shapes="fast,proxy,prod")  # image hipBLASLt env iff gb runs

import torch  # noqa: E402
'''),
    ('''def measure(shape, labels, fns, iters, warmup_seconds, causal=True, block=9, lead=4):
''', '''def measure(shape, labels, fns, iters, warmup_seconds, causal=True, block=9, lead=4, rulers=("blk",), gb=None):
'''),
    ('''    if block <= 1:                             # legacy: arms interleaved call by call
        for i in range(iters):
            for label in (labels if i % 2 == 0 else labels[::-1]):
                times[label].append(timed(label))
    else:''', '''    if "blk" not in rulers:
        pass
    elif block <= 1:                           # legacy: arms interleaved call by call
        for i in range(iters):
            for label in (labels if i % 2 == 0 else labels[::-1]):
                times[label].append(timed(label))
    else:'''),
    ('''                for _ in range(block):
                    times[label].append(timed(label))
    sclk1 = sclk_mhz()
''', '''                for _ in range(block):
                    times[label].append(timed(label))
    # gb (A0 operator 2026-10-02, h51): the SCORED figure for proxy/prod -- every timed call right after a
    # GEMM burst, i.e. at the clock the call runs at inside training. The blocked figure above stays in the
    # row as blk_*. No L2 flush here: the burst's 1.2 GB of traffic is the flush.
    gbrec = gb.run(labels, call) if "gb" in rulers else None
    if gbrec is not None:
        void = gb.check(gbrec, shape)
        if void:
            return void
    sclk1 = sclk_mhz()
'''),
    ('''    for label in labels:
        ts = sorted(times[label])
        n = len(ts)''', '''    for label in labels:
        ts = sorted(gbrec[label]["ms"] if gbrec is not None else times[label])
        n = len(ts)'''),
    ('''            "shape": shape, "arm": label, "stat": STATISTIC, "iters": iters,''',
     '''            "shape": shape, "arm": label, "stat": STATISTIC, "iters": n if gbrec is not None else iters,'''),
    ('''            "sclk_start": sclk0, "sclk_end": sclk1, "causal": bool(causal),
            "order": "interleaved" if block <= 1 else f"blocked{block}+lead{lead}",
        })
    return rows
''', '''            "sclk_start": sclk0, "sclk_end": sclk1, "causal": bool(causal),
            "order": (gb.order if gbrec is not None
                      else "interleaved" if block <= 1 else f"blocked{block}+lead{lead}"),
            "ruler": "gb" if gbrec is not None else "blk",
        })
        if gbrec is not None:
            rows[-1].update(gbruler.gb_fields(gbrec[label], gb.max_sclk))
            if "blk" in rulers:
                rows[-1].update(gbruler.blk_fields(times[label], c.flop, ndigits=True))
    gbruler.add_aa_ratios(rows)
    return rows
'''),
    ('''    ap.add_argument("--json", help="also write every row here")
    a = ap.parse_args()

    arms = resolve_arms([x for x in a.arms.split(",") if x], a.arm_path)
    if not arms:
        ap.error("no arms given")
    for label, path in arms:
        if not (path / "impl.py").is_file():
            ap.error(f"arm {label!r}: no impl.py under {path}")
    # One module per arm for the whole process (reloading per shape let a GC'd flydsl
    # module unload mid-launch in the backward job).
    fns = {label: load_impl(path) for label, path in arms}
    for label, path in arms:
        print(f"# arm {label} -> {path}  {witness(fns[label])}")
    print(f"# ENV TORCH_BLAS_PREFER_HIPBLASLT={os.environ.get('TORCH_BLAS_PREFER_HIPBLASLT')} "
          f"HIPBLASLT_TENSILE_LIBPATH={os.environ.get('HIPBLASLT_TENSILE_LIBPATH')}")
''', '''    ap.add_argument("--json", help="also write every row here")
    gbruler.add_args(ap)
    a = ap.parse_args()

    arms = resolve_arms([x for x in a.arms.split(",") if x], a.arm_path)
    if not arms:
        ap.error("no arms given")
    arms = gbruler.add_aa_arms(arms, a.aa)     # LABEL_aa = byte copy of LABEL (A/A noise floor)
    for label, path in arms:
        if not (path / "impl.py").is_file():
            ap.error(f"arm {label!r}: no impl.py under {path}")
    shapes = [s for s in a.shapes.split(",") if s]
    ruler, ruler_note = gbruler.resolve_ruler(a.ruler, a.warmup_seconds)   # profiler / warmup 0 -> blk
    plan = {s: gbruler.rulers_for(s, ruler) for s in shapes}
    # BEFORE load_impl: the burst initialises hipBLASLt with the image library before any arm's _env.py
    # runs (this job's _env.py ASSIGNS the host library, with which the burst is void).
    gb = gbruler.GbRuler(torch, a, default_iters=a.iters) if any("gb" in r for r in plan.values()) else None
    # One module per arm for the whole process (reloading per shape let a GC'd flydsl
    # module unload mid-launch in the backward job).
    fns = {label: load_impl(path) for label, path in arms}
    if gb is not None:
        void = gb.after_load()                 # env re-assigned, 3 timed bursts, witness thread started
        if void:
            gbruler.say_void(void)
            return gbruler.VOID_RC
    for label, path in arms:
        print(f"# arm {label} -> {path}  {witness(fns[label])}")
    print(f"# ENV TORCH_BLAS_PREFER_HIPBLASLT={os.environ.get('TORCH_BLAS_PREFER_HIPBLASLT')} "
          f"HIPBLASLT_TENSILE_LIBPATH={os.environ.get('HIPBLASLT_TENSILE_LIBPATH')}  "
          f"ruler {a.ruler}{'' if ruler == a.ruler else ' -> ' + ruler}: "
          + " ".join(f"{s}={'+'.join(r)}" for s, r in plan.items())
          + (f" [{ruler_note}]" if ruler_note else "") + (f" | {gb.header()}" if gb is not None else ""))
'''),
    ('''    for shape in [s for s in a.shapes.split(",") if s]:
        new = measure(shape, labels, fns, a.iters, a.warmup_seconds, causal=not a.non_causal,
                      block=a.block, lead=a.lead)
        rows += new''', '''    for shape in shapes:
        new = measure(shape, labels, fns, a.iters, a.warmup_seconds, causal=not a.non_causal,
                      block=a.block, lead=a.lead, rulers=plan[shape], gb=gb)
        if isinstance(new, str):               # the gb ruler was void for this shape: no figure at all
            gbruler.say_void(new)
            return gbruler.VOID_RC
        rows += new'''),
]

FWD_YAML = [   # on top of the A0 move already applied by output/1002__oe/fwdjob (final.yaml md5 39e3c33c)
    ("  min_gain: 0.007  # [2026-09-25] just above this operator's 0.24-0.66% same-session noise floor.",
     "  gain_weights: {prod: 1.0, proxy: 0.25, fast: 0.0}   # A0 operator 2026-10-02 (h51, as A0 fwd job r10 and "
     "bwd D7/h83): prod ranks; fast is blk-ruled and launch-bound\n"
     "  min_gain: 0.007  # [2026-09-25] just above this operator's 0.24-0.66% same-session noise floor."),
]

RECIPES = {"bwd-benchmark": BWD_BENCH, "fwd-benchmark": FWD_BENCH, "fwd-final-yaml": FWD_YAML}


def main() -> int:
    if len(sys.argv) != 4 or sys.argv[1] not in RECIPES:
        print(__doc__)
        return 2
    s = Path(sys.argv[2]).read_text()
    for i, (old, new) in enumerate(RECIPES[sys.argv[1]]):
        n = s.count(old)
        if n != 1:
            print(f"anchor {i} occurs {n} times (want 1): {old.strip().splitlines()[0][:100]!r}\n"
                  "-> the base was edited after 2026-10-02: merge this hunk by hand", file=sys.stderr)
            return 1
        s = s.replace(old, new)
    Path(sys.argv[3]).write_text(s)
    print(f"wrote {sys.argv[3]} ({len(RECIPES[sys.argv[1]])} hunks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
