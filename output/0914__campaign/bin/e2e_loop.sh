#!/bin/bash
# Self-refilling e2e stream for GPU3. Alternates the A/B pair forever and appends one line
# per run to a ledger, so the comparison accumulates statistics instead of resting on one
# run each.
#
# Written because the one-shot e2e left GPU3 idle for eight minutes after it finished: the
# watchdog only restarts streams it knows about, and a run that is not a loop has nothing to
# restart. Alternating rather than batching each arm matters -- the first turbo-vs-flex
# comparison showed 20-30% step-time spread and the two arms were taken at different times,
# so the difference could not be attributed.
set -u
OUT=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0914__campaign
LEDGER=$OUT/ledgers/e2e_ab2.jsonl
STOP=/tmp/campaign.g3.stop
PIDF=$OUT/ledgers/g3.e2e.pid
touch "$LEDGER"; echo $$ > "$PIDF"; trap 'rm -f "$PIDF"' EXIT

AITER=/home/lihuzhan/code/aiter-src
# Our checkout, FIRST on PYTHONPATH. Without this the training imported the image's editable
# primus_turbo 0.4.1.dev12 (built 2026-09-09) and every e2e run today measured code that
# predates this branch -- the asm and noasm arms were byte-identical, which is why they
# differed by 0.3%. The image's editable install had to be pip-uninstalled inside the
# container first: it registers a MetaPathFinder via a .pth, and sys.meta_path is consulted
# before sys.path, so PYTHONPATH alone could not shadow it.
TURBO=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo
BASE="-e TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1 -e TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=TRITON -e TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_e2e_mt -e PRIMUS_TURBO_PATH=$TURBO"

summarise(){ # tag, logfile
  python3 - "$1" "$2" <<'PY' >> "$LEDGER"
import json,re,sys,statistics as st,pathlib
tag,log=sys.argv[1],pathlib.Path(sys.argv[2])
A=re.compile(r'\x1b\[[0-9;]*m')
tps=[];tf=[]
for l in log.read_bytes().decode('utf8','replace').splitlines():
    l=A.sub('',l)
    m=re.search(r'tps:\s*([\d,]+)',l);  tps.append(float(m.group(1).replace(',',''))) if m else None
    f=re.search(r'tflops:\s*([\d.]+)',l); tf.append(float(f.group(1))) if f else None
row={"tag":tag,"steps":len(tps)}
if len(tps)>=8:
    a=tps[-8:]
    row.update(tps_median=st.median(a), tps_min=min(a), tps_max=max(a),
               spread_pct=round((max(a)-min(a))/st.median(a)*100,2),
               tflops_median=st.median(tf[-8:]))
print(json.dumps(row))
PY
}

R=0
while true; do
  R=$((R+1))
  for arm in asm noasm flex; do
    while [ -f "$STOP" ]; do sleep 10; done
    tag="ab|r$R|$arm"
    grep -qF "\"tag\": \"$tag\"" "$LEDGER" 2>/dev/null && continue
    CFG=repro_l8b_turbo_compile.yaml
    case "$arm" in
      asm)   ENV="$BASE -e PYTHONPATH=$TURBO:$AITER" ;;
      noasm) ENV="$BASE -e PYTHONPATH=$TURBO" ;;
      # flex = attention off entirely (the v5 config sets enable_primus_turbo false), so the
      # asm/noasm/flex triple prices the ASM forward, the turbo attention stack, and the
      # GEMM fix separately instead of confounding them.
      flex)  ENV="$BASE -e PYTHONPATH=$TURBO"; CFG=repro_l8b_bf16_mbs4_seq8k_v5_compile.yaml ;;
    esac
    E2E_ENV="$ENV" bash "$OUT/bin/e2e.sh" "ab_${arm}_r$R" "$CFG" >/dev/null 2>&1
    summarise "$tag" "$OUT/logs/e2e.ab_${arm}_r$R.log"
  done
done
