#!/bin/bash
# Ablation arms (output is WRONG by design -> no correctness gate): compile-only, fast-shape smoke run
# under AMD_SERIALIZE_KERNEL=3, then prod blocked timing vs r29 (+ r29_aa, asm). kbench per-kernel split optional.
#   abl.sh <tag> <arm>...
set -u
W=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0930__bwd; cd $W
TAG=$1; shift
OP=$(readlink -f $W/../0927__b0/ruler/bwd/oe/artifacts/job/job_context/op)
AB=""
for a in "$@"; do
  out=$(tools/compile.sh arms/$a dkdv dqg); echo "[compile $a] $out"
  echo "$out" | grep -q "RC=0" || { echo "GATE compile FAIL $a"; exit 2; }
  echo "$out" | grep -qE "spill_count:\s*[1-9]|private_segment_fixed_size:\s*[1-9]" && { echo "GATE spill/scratch FAIL $a"; exit 3; }
  AB="$AB --arm-path $a=$W/arms/$a"
done
SERIAL=1 tools/run.sh ${TAG}_smoke 900 $OP -- "/opt/venv/bin/python3 benchmark.py $AB --shapes fast --iters 3 --warmup-seconds 0 --block 1 --lead 0" || { echo "GATE smoke FAIL"; exit 4; }
tools/run.sh ${TAG}_bench 1800 $OP -- "/opt/venv/bin/python3 benchmark.py --arm-path r29=$W/arms/r29 $AB --arm-path r29_aa=$W/arms/r29_aa --arm-path asm=$OP/beat --shapes prod --json $W/runs/${TAG}_bench.json" || { echo "GATE bench FAIL"; exit 6; }
python3 - "$W/runs/${TAG}_bench.log" <<'PY'
import sys
rows={}
for l in open(sys.argv[1]):
    if l.startswith('RESULT'):
        d=dict(kv.split('=',1) for kv in l.split()[1:]); rows[d['arm']]=float(d['latency_ms'])
b=rows['r29']; a=rows['asm']
for k,v in rows.items(): print(f"BENCH {k:10s} {v:.4f} ms  vs_r29 {100*(v/b-1):+.2f}%  asm/arm {100*a/v:.1f}%")
PY
