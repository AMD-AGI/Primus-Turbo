#!/bin/bash
# serial correctness: per arm toy -> fast -> proxy -> prod, one shape per process; stop on any failure
W=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/lab2-L12
for arm in "$@"; do
  for s in toy fast proxy prod; do
    [ -f $W/card/${arm}_$s.json ] && continue
    ser=3; [ $s = prod ] && ser=0
    $W/tools/card.sh ${arm}_$s $ser $W/tools/corr_one.py $W/arms/$arm $s $W/card/${arm}_$s.json 20 || { echo "STOPPED at ${arm}_$s"; exit 1; }
    grep -q '"pass": true' $W/card/${arm}_$s.json || { echo "CORR FAIL ${arm}_$s"; exit 1; }
  done
done
echo SEQDONE
