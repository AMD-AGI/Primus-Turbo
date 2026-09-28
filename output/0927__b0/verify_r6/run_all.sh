#!/bin/bash
V=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/verify_r6
base=$(timeout 20 sudo -n dmesg | grep -c amdgpu)
for sc in "toy 0" "short_q 1" "short_q 0" "gqa4 1" "gqa4 0" "proxy 1" "prod 1"; do
  set -- $sc
  echo "== $sc $(date -u +%T)"
  $V/run_card.sh $1 $2
  now=$(timeout 20 sudo -n dmesg | grep -c amdgpu)
  if [ "$now" != "$base" ]; then echo "NEW_AMDGPU_DMESG $base -> $now"; exit 3; fi
done
echo ALL_DONE
