#!/bin/bash
# Closing champion measurement on GPU1 -- healthy (124.8 TFLOP/s verified), and the only
# other GPU work right now is op-evolve, which is fenced to its own card.
#
# GPU0 is deliberately excluded: it is the one card that failed the liveness test, and all
# 120 of today's MES timeouts are on its PCI address.
#
# Timeouts are generous on purpose. A cold process spends ~78 s on HIP init and allocation
# before any kernel runs; a 150 s bound measures the runtime's startup, not the card, and
# that mistake cost four wrong conclusions today.
set -u
R=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo
L=$R/output/0914__campaign/ledgers/final_champion.jsonl; : > "$L"
for i in 1 2 3; do
  for impl in asm fused; do
    o=$(timeout 2400 docker exec -e GPU=1 -e PYTHONPATH=/home/lihuzhan/code/aiter-src \
          -e TRITON_CACHE_DIR=/tmp/triton_cache_g1 \
          fa-repro bash -lc "cd $R && exec python3 tools/gfx1250/tune_attention.py --shape llama31-8b --impl $impl" 2>/dev/null | tail -1)
    case "$o" in '{'*) echo "{\"tag\":\"final|$impl|$i\",\"r\":$o}" >> "$L"; echo "  $impl|$i ok" ;;
                 *)    echo "{\"tag\":\"final|$impl|$i\",\"r\":null}" >> "$L"; echo "  $impl|$i FAILED" ;; esac
  done
done
echo "final champion measurement complete"
