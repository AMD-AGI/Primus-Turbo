#!/usr/bin/env python3
"""Measure implementations of attn_fwd. It prints numbers and decides nothing.

    python3 benchmark.py --arms baseline,current,beat --shapes prod
    python3 benchmark.py --arm-path candidate=/abs/rounds/007/op --arms beat --shapes fast

Arms: `--arms` names directories under op/ (baseline, current, beat, eager);
`--arm-path LABEL=/abs/dir` addresses an arm BY PATH. One RESULT line per (shape, arm).
Run ONE shape per process when it matters (validation does): in the backward job,
crossing shapes inside one process faulted this card.

Method, fixed here (the card is VR-throttled to 1100 MHz and drifts):
  statistic    MEDIAN of per-iteration CUDA-event times. Never mixed with best-of-N.
  iterations   --iters, default 101 timed iterations per arm per shape (the protocol every
               figure in the spec was taken under).
  warmup       --warmup-seconds (default 8) of CONTINUOUS load per arm, no sleeps anywhere.
  ordering     BLOCKED (default --block 9 --lead 4): per round each arm runs `lead` untimed
               then `block` timed calls back to back; rounds palindromic. --block 1 restores
               the old call-by-call palindromic interleave, where at prod a call inherits the
               card state left by the previous few calls (other arms): 3-7% for FlyDSL vs
               FlyDSL, ~25% for FlyDSL vs ASM beat. Identical code then reads differently by
               position and by which other arms share the process (ruler/REPORT.md).
  inputs       one seeded input per shape, shared by every arm.
  clock        sclk read before and after the timed loop, printed on every RESULT line.
  L2           256 MB buffer zeroed before each timed call, OUTSIDE the event window.
  counts       FLOP and bytes from the shipped tools/op_flops.py, forward (backward=False).
  ruler        --ruler auto (default, A0 operator 2026-10-02, hint h51): `fast` keeps the blocked ruler above;
               `proxy` and `prod` run it too (reported as blk_*) and are then SCORED on the gb ruler = the
               training operating point: before EVERY timed call 10 bf16 GEMMs 32768x4096x14336 with the IMAGE
               hipBLASLt library (~21.5 ms), then the call; palindromic rounds of --gb-block calls per arm;
               median; clock witness gb_sclk per call; a burst slower than --gb-max-burst-ms voids the ruler
               (exit 4). --ruler blk = the pre-10-02 behaviour. --aa LABEL adds a byte copy of an arm as
               LABEL_aa (A/A floor, aa_ratio). Code: gbruler.py; why: r16/ASM is 1.263 in the e2e, 1.349 on
               gb, 1.081 blocked (PT/output/1002__e2e/RESULT-{realab,e2e}.md, PT/output/1002__oe/RULER.md).
"""
from __future__ import annotations

import os

os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "1"
os.environ["HIPBLASLT_TENSILE_LIBPATH"] = "/home/lihuzhan/.local/hipblaslt-gfx1250/gfx1250"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent.parent.parent.parent / "tools"          # op-evolve/tools
sys.path.insert(0, str(HERE / "ut"))
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(HERE))                                        # gbruler.py

import gbruler  # noqa: E402  -- stdlib only; it must run BEFORE torch is imported
gbruler.preimport(sys.argv[1:], default_shapes="fast,proxy,prod")    # image hipBLASLt env iff gb runs

import torch  # noqa: E402

import op_flops  # noqa: E402  -- the shipped tool, imported, not copied
from common import SHAPES, load_impl, make_inputs  # noqa: E402

STATISTIC = "median"
L2_FLUSH_MB = 256
ROCM_SMI = "/opt/venv/bin/rocm-smi"


def sclk_mhz():
    """Current shader clock of the card under test, or -1 if it cannot be read.

    B0 runs one container per card (fa-gN, OE_PHYS_GPU=N). rocm-smi reads sysfs for every
    card and its first sclk line is GPU[0], so read this card's pp_dpm_sclk directly.
    """
    phys = os.environ.get("OE_PHYS_GPU")
    if phys is not None:
        try:
            with open(f"/sys/class/drm/renderD{128 + 8 * int(phys)}/device/pp_dpm_sclk") as f:
                for line in f:
                    if line.rstrip().endswith("*"):
                        return int(line.split(":")[1].strip().split("Mhz")[0])
        except Exception:
            pass
        return -1
    try:
        out = subprocess.run([ROCM_SMI, "--showclocks"], capture_output=True, text=True,
                             timeout=30).stdout
        for line in out.splitlines():
            if "sclk" in line.lower() and "Mhz" in line:
                return int(line.split("(")[-1].split("Mhz")[0].strip())
    except Exception:
        pass
    return -1


def counts(shape):
    b, sq, skv, hq, hkv, d = SHAPES[shape]
    return op_flops.attention(batch=b, heads=hq, seqlen_q=sq, seqlen_kv=skv, head_dim=d,
                              kv_heads=hkv, causal="bottom-right", dtype="bf16",
                              backward=False)


def resolve_arms(names, paths):
    arms = []
    for spec in paths:
        label, _, p = spec.partition("=")
        arms.append((label, Path(p).resolve()))
    for name in names:
        arms.append((name, (HERE / name).resolve()))
    return arms


def witness(fn):
    """Which code an arm actually runs: the files its module and kernel module came from."""
    mod = sys.modules[fn.__module__]
    kern = getattr(mod, "_kern", None)
    s = f"impl={mod.__file__}"
    if kern is not None:
        # md5 of the kernel file AS LOADED: a path (e.g. op/current) is not an identity --
        # it was re-installed mid-session once and turned r4-vs-r6 into a fake "bimodality".
        md5 = hashlib.md5(Path(kern.__file__).read_bytes()).hexdigest()[:8]
        s += f" kernel={kern.__file__} md5={md5} O_VARIANT={getattr(kern, 'O_VARIANT', '?')}"
    return s


def measure(shape, labels, fns, iters, warmup_seconds, causal=True, block=9, lead=4, rulers=("blk",), gb=None):
    q, k, v = make_inputs(shape, seed=0)
    flush = torch.empty(L2_FLUSH_MB * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)

    def call(label):
        return fns[label](q, k, v, causal=causal)

    for label in labels:                       # build once, outside every timing window
        call(label)
    torch.cuda.synchronize()
    for label in labels:                       # continuous load, seconds, no pauses
        t_end = time.perf_counter() + warmup_seconds
        while time.perf_counter() < t_end:
            call(label)
        torch.cuda.synchronize()

    sclk0 = sclk_mhz()
    times = {label: [] for label in labels}
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)

    def timed(label):
        flush.zero_()
        ev0.record()
        call(label)
        ev1.record()
        ev1.synchronize()
        return ev0.elapsed_time(ev1)

    if "blk" not in rulers:
        pass
    elif block <= 1:                           # legacy: arms interleaved call by call
        for i in range(iters):
            for label in (labels if i % 2 == 0 else labels[::-1]):
                times[label].append(timed(label))
    else:
        # BLOCKED: every timed call is preceded by >= `lead` calls of the SAME arm. On this
        # card a prod call carries the state left by the previous few calls (3-7% FlyDSL vs
        # FlyDSL, ~25% vs ASM beat; ruler/REPORT.md). Interleaving call by call hands that to
        # the neighbour arm: identical code reads by position and by arm set, and the
        # higher-power arm is rewarded. Rounds are palindromic over arms, as before.
        rounds = -(-iters // block)
        for r in range(rounds):
            for label in (labels if r % 2 == 0 else labels[::-1]):
                for _ in range(lead):
                    timed(label)
                for _ in range(block):
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

    c = counts(shape)
    rows = []
    for label in labels:
        ts = sorted(gbrec[label]["ms"] if gbrec is not None else times[label])
        n = len(ts)
        med = ts[n // 2] if n % 2 else 0.5 * (ts[n // 2 - 1] + ts[n // 2])
        secs = med / 1e3
        rows.append({
            "shape": shape, "arm": label, "stat": STATISTIC, "iters": n if gbrec is not None else iters,
            "latency_ms": round(med, 5), "min_ms": round(ts[0], 5), "max_ms": round(ts[-1], 5),
            "tflops": round(c.flop / secs / 1e12, 2), "bw_gbs": round(c.bytes_min / secs / 1e9, 2),
            "flop": c.flop, "bytes_min": c.bytes_min,
            "sclk_start": sclk0, "sclk_end": sclk1, "causal": bool(causal),
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="", help="comma-separated names under op/")
    ap.add_argument("--arm-path", action="append", default=[],
                    help="LABEL=/abs/dir -- an arm addressed by PATH, repeatable")
    ap.add_argument("--shapes", default="fast,proxy,prod")
    ap.add_argument("--iters", type=int, default=101)
    ap.add_argument("--warmup-seconds", type=float, default=8.0)
    ap.add_argument("--non-causal", action="store_true")
    ap.add_argument("--block", type=int, default=9,
                    help="timed calls per arm per round (<=1: legacy call-by-call interleave)")
    ap.add_argument("--lead", type=int, default=4, help="untimed same-arm calls before each block")
    ap.add_argument("--json", help="also write every row here")
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
    plan = {s: gbruler.rulers_for(s, a.ruler) for s in shapes}
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
          f"ruler {a.ruler}: " + " ".join(f"{s}={'+'.join(r)}" for s, r in plan.items())
          + (f" | {gb.header()}" if gb is not None else ""))
    print(f"# statistic {STATISTIC}  iters {a.iters}  warmup {a.warmup_seconds}s continuous  "
          f"order {'palindromic' if a.block <= 1 else f'blocked {a.block}+lead {a.lead}, palindromic rounds'}  device {torch.cuda.get_device_properties(0).gcnArchName}",
          flush=True)

    rows = []
    labels = [label for label, _ in arms]
    for shape in shapes:
        new = measure(shape, labels, fns, a.iters, a.warmup_seconds, causal=not a.non_causal,
                      block=a.block, lead=a.lead, rulers=plan[shape], gb=gb)
        if isinstance(new, str):               # the gb ruler was void for this shape: no figure at all
            gbruler.say_void(new)
            return gbruler.VOID_RC
        rows += new
        for r in new:
            print("RESULT " + " ".join(f"{k}={v}" for k, v in r.items()), flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
