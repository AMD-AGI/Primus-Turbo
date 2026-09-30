#!/bin/bash
# Full screen of one or more arms vs r29 (+ r29_aa, asm) on A0. Stops at the first failing gate.
#   arm.sh <tag> <arm_name>...        (arms under arms/<name>)
# 1 compile-only (no GPU): vgpr/spill/scratch gate
# 2 fast-shape validation, AMD_SERIALIZE_KERNEL=3 (toy-first rule)
# 3 prod validation (SQNR >= 50 dB each, dk/dv bitwise, dq run-to-run)
# 4 prod blocked benchmark, palindromic, r29 + r29_aa + asm in the same process
set -u
W=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0930__bwd; cd $W
TAG=$1; shift
OP=$(readlink -f $W/../0927__b0/ruler/bwd/oe/artifacts/job/job_context/op); V=$(readlink -f $W/../0927__b0/lab-kdq/OP)
AV=""; AB=""
for a in "$@"; do
  out=$(tools/compile.sh arms/$a dkdv dqg); echo "[compile $a] $out"
  echo "$out" | grep -q "RC=0" || { echo "GATE compile FAIL $a"; exit 2; }
  echo "$out" | grep -qE "spill_count:\s*[1-9]|private_segment_fixed_size:\s*[1-9]" && { echo "GATE spill/scratch FAIL $a"; exit 3; }
  AV="$AV $a=$W/arms/$a"; AB="$AB --arm-path $a=$W/arms/$a"
done
SERIAL=1 tools/run.sh ${TAG}_valfast 900 $V -- "/opt/venv/bin/python3 lab_validate.py fast 3 r29=$W/arms/r29 $AV" || { echo "GATE valfast FAIL"; exit 4; }
grep -E "^CORR|^DET|LABVAL" runs/${TAG}_valfast.log
grep -q "LABVAL fast PASS" runs/${TAG}_valfast.log || { echo "GATE valfast not PASS"; exit 4; }
tools/run.sh ${TAG}_valprod 1200 $V -- "/opt/venv/bin/python3 lab_validate.py prod 3 r29=$W/arms/r29 $AV" || { echo "GATE valprod FAIL"; exit 5; }
grep -E "^CORR|^DET|LABVAL" runs/${TAG}_valprod.log
grep -q "LABVAL prod PASS" runs/${TAG}_valprod.log || { echo "GATE valprod not PASS"; exit 5; }
tools/run.sh ${TAG}_bench 1800 $OP -- "/opt/venv/bin/python3 benchmark.py --arm-path r29=$W/arms/r29 $AB ${EXTRA_ARMS:-} --arm-path r29_aa=$W/arms/r29_aa --arm-path asm=$OP/beat --shapes prod --json $W/runs/${TAG}_bench.json" || { echo "GATE bench FAIL"; exit 6; }
python3 - "$W/runs/${TAG}_bench.log" <<'PY'
import sys,re
rows={}
for l in open(sys.argv[1]):
    if l.startswith('RESULT'):
        d=dict(kv.split('=',1) for kv in l.split()[1:]); rows[d['arm']]=float(d['latency_ms'])
b=rows['r29']; a=rows['asm']
for k,v in rows.items(): print(f"BENCH {k:10s} {v:.4f} ms  vs_r29 {100*(v/b-1):+.2f}%  asm/arm {100*a/v:.1f}%")
PY
