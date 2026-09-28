#!/bin/bash
# Interference probe. usage: probe.sh <seg A|B> <rep>
# fwd harness on fa-g0 then bwd harness on fa-g1; in seg B the other three cards run burn.py.
set -u
SEG=$1; REP=$2
P=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/probe
FJ=/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-flydsl-attn-fwd-20260925-114644
BJ=/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-flydsl-attn-bwd-20260917-115934
KFD=(34992 30548 57865 51359)
card_busy() { for q in /sys/class/kfd/kfd/proc/*/queues/*/gpuid; do [ -f "$q" ] && [ "$(cat $q 2>/dev/null)" = "${KFD[$1]}" ] && return 0; done; return 1; }
burn_on() { for g in "$@"; do docker exec -e DURATION=1500 fa-g$g python3 $P/burn.py >/dev/null 2>&1 & done; sleep 20; }
burn_off() { for g in 0 1 2 3; do docker exec fa-g$g pkill -f burn.py 2>/dev/null; done; sleep 10; }
sclk_log() { ( while true; do echo "$(date +%T) $(grep '\*' /sys/class/drm/card$((8*$1))/device/pp_dpm_sclk) busy=$(cat /sys/class/drm/card$((8*$1))/device/gpu_busy_percent)"; sleep 1; done ) > $2 2>&1 & echo $!; }
run() {  # <gpu> <tag> <cmd>
  local g=$1 tag=$2 cmd=$3
  card_busy $g && { echo "ABORT: card $g busy before $tag"; exit 3; }
  [ "$SEG" = B ] && burn_on $(for x in 0 1 2 3; do [ $x != $g ] && echo $x; done)
  local sp=$(sclk_log $g $P/sclk_${tag}.log)
  timeout -s INT 2400 docker exec -e ARCH=gfx1250 -e FLYDSL_GPU_ARCH=gfx1250 fa-g$g bash -c "$cmd" > $P/${tag}.log 2>&1
  local rc=$?; kill $sp; [ "$SEG" = B ] && burn_off
  echo "$tag rc=$rc $(grep -c RESULT $P/${tag}.log) results"
  [ $rc -ne 0 ] && { tail -5 $P/${tag}.log; exit 4; }
}
run 0 fwd_${SEG}${REP} "cd $FJ/job_context/op && /opt/venv/bin/python3 benchmark.py --arms baseline,beat --arm-path r4=$FJ/rounds/004/op --shapes prod --iters 101 --json $P/fwd_${SEG}${REP}.json"
run 1 bwd_${SEG}${REP} "cd $BJ/job_context/op && export TORCH_BLAS_PREFER_HIPBLASLT=1 HIPBLASLT_TENSILE_LIBPATH=/home/lihuzhan/.local/hipblaslt-gfx1250/gfx1250 && /opt/venv/bin/python3 benchmark.py --arms current,beat --shapes prod --iters 51 --json $P/bwd_${SEG}${REP}.json"
timeout 20 sudo -n dmesg | tail -n 30 | grep -iE "ring buffer|failed to respond|gcvm|page fault|reset" && echo "DMESG FAULT"
exit 0
