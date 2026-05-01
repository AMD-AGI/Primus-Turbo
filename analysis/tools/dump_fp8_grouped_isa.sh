#!/bin/bash
# Round-34-dm: ISA dump + spill-distribution analysis for FP8 grouped kernel.
# Usage: bash dump_fp8_grouped_isa.sh [output_dir]
# Emits spec_{tag}.s files and spill/mfma histograms.
set -e
OUT="${1:-/tmp/isa_dump}"
mkdir -p "$OUT"
cd "$OUT"

: "${THUNDERKITTENS_ROOT:=/workspace/code/HipKittens}"
export THUNDERKITTENS_ROOT

/opt/rocm/bin/hipcc \
  -std=c++20 \
  -DKITTENS_CDNA4 --offload-arch=gfx950 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math \
  -I/opt/rocm/include/hip -I/opt/rocm/include/rocrand \
  -I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype \
  $(python3 -m pybind11 --includes) \
  -shared -fPIC -w -O3 \
  --save-temps=obj \
  -c ${THUNDERKITTENS_ROOT}/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp \
  -o kernel_fp8_layouts.o 2>/dev/null

ISA="${OUT}/kernel_fp8_layouts-hip-amdgcn-amd-amdhsa-gfx950.s"
echo "=== ISA dump: $ISA ==="
ls -la "$ISA"

echo ""
echo "=== Per-spec: scratch-ops / mfma-ops / interleaved-spills (within 3 lines of mfma) ==="
for tag in "Li0ELb0ELb0" "Li0ELb1ELb0" "Li0ELb0ELb1" "Li0ELb1ELb1"; do
  awk "/_Z18grouped_rcr_kernelI${tag}EEv22grouped_layout_globals:/,/^\\s*\\.end_amdhsa_kernel\$/" "$ISA" > "${OUT}/spec_${tag}.s"
  total=$(grep -cE "scratch_(store|load)" "${OUT}/spec_${tag}.s" || echo 0)
  mfma=$(grep -cE "v_mfma" "${OUT}/spec_${tag}.s" || echo 0)
  interleaved=$(awk '
    /v_mfma/ {is_mfma[NR]=1}
    {all_lines[NR]=$0}
    END {
      for (i=1; i<=NR; i++) {
        if (all_lines[i] ~ /scratch_(store|load)/) {
          for (j=i-3; j<=i+3; j++) {
            if (is_mfma[j]) { count++; break }
          }
        }
      }
      print count+0
    }' "${OUT}/spec_${tag}.s")
  printf "  spec %-14s: %4d interleaved / %4d total scratch, %4d mfma\n" "$tag" "$interleaved" "$total" "$mfma"
done
