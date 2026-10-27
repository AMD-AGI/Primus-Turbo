"""stage_summary.py OUTDIR [BASE_OUTDIR]: compact table of a stage_bench.sh run (and the fused-ms delta vs BASE)."""
import csv, glob, os, sys

SHORT = {"L1 fwd dispatch_gemm (NT)": "L1 fwd", "L2 fwd gemm_combine (NT)": "L2 fwd",
         "L2 dgrad dispatch_gemm (NN)": "L2 dgrad", "dW2 variable-K": "dW2", "L1 dgrad gemm_combine (NN)": "L1 dgrad",
         "dW1 x re-dispatch + variable-K": "dW1+redisp", "SUM of the 6 GEMM stages": "SUM"}


def load(d):
    rows = {}
    for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
        for r in csv.DictReader(open(f)):
            rows[(r["shape"], SHORT.get(r["stage"], r["stage"]))] = r
    return rows


def num(r, k):
    try:
        return float(r.get(k) or "nan")
    except ValueError:
        return float("nan")


cur = load(sys.argv[1])
base = load(sys.argv[2]) if len(sys.argv) > 2 else {}
hdr = (f"{'shape':<18} {'stage':<11} {'gemm_only':>9} {'TF':>6} {'gr/dense':>8} {'comm':>7} {'GB/s':>6} "
       f"{'fused':>7} {'TF':>6} {'hidden':>7} {'hid/comm':>8} {'fus/gemm':>8} {'roofl':>6}" + ("  fused vs base" if base else ""))
print(hdr)
for (shape, stage), r in cur.items():
    line = (f"{shape:<18} {stage:<11} {num(r, 'gemm_only ms'):9.3f} {num(r, 'gemm_only TF'):6.0f} "
            f"{num(r, 'grouped/dense %'):7.1f}% {num(r, 'comm_only ms'):7.3f} {num(r, 'comm GB/s'):6.0f} "
            f"{num(r, 'fused ms'):7.3f} {num(r, 'fused TF'):6.0f} {num(r, 'hidden ms'):7.3f} "
            f"{num(r, 'hidden/comm %'):7.0f}% {num(r, 'fused/gemm_only %'):7.1f}% {num(r, 'roofline %'):5.1f}%")
    b = base.get((shape, stage))
    if b:
        line += f"  {100 * (num(r, 'fused ms') / num(b, 'fused ms') - 1):+6.2f}%"
    print(line)
    if stage == "SUM":
        step = f"  step {num(r, 'step ms'):.3f} ms ({num(r, 'step TF'):.0f} TF), non-GEMM {num(r, 'non-GEMM ms'):.3f} ms"
        if b:
            step += f"  step vs base {100 * (num(r, 'step ms') / num(b, 'step ms') - 1):+.2f}%"
        print(f"{'':<18} {step}")
