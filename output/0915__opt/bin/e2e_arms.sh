#!/bin/bash
# The three arms. Each writes its own trace file, so "did the ASM backward engage" is READ,
# not inferred from a timing difference -- the gate says so itself when it fires.
#
#   A  converters on,  ASM backward on    what today's work is worth end to end
#   B  converters on,  ASM backward off   isolates the ASM backward (and its memory cost)
#   C  converters off                     the old baseline -- which is what every e2e number
#                                         before today actually measured, because the
#                                         converter is what installs turbo attention at all
set -u
cd /home/lihuzhan/code/2026_0903__turbo/Primus-Turbo
T=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0915__opt

run_arm() {
  local tag=$1 cfg=$2 extra=${3:-}
  E2E_TIMEOUT=1500 E2E_INNER=1440 \
  E2E_ENV="-e PRIMUS_TURBO_ASM_BWD_TRACE=1 -e PRIMUS_TURBO_ASM_BWD_TRACE_FILE=/tmp/asmbwd.$tag.trace $extra" \
    bash "$T/bin/e2e.sh" "$tag" "$cfg"
  echo "--- trace for $tag:"
  docker exec fa-repro cat "/tmp/asmbwd.$tag.trace" 2>/dev/null || echo "  (no trace file -- gate never reached)"
}

case "${1:?arm A|B|C}" in
  A) run_arm armA repro_l8b_turbo_conv.yaml ;;
  B) run_arm armB repro_l8b_turbo_conv.yaml "-e PRIMUS_TURBO_ATTN_DISABLE_ASM_BWD=1" ;;
  C) run_arm armC repro_l8b_bf16_mbs4_seq8k_turbo.yaml ;;
  *) echo "arm must be A, B or C"; exit 2 ;;
esac
