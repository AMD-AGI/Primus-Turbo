#!/usr/bin/env bash
# ISA gate for one or more _probe_bwdcfg arms: dump, then print the fields goal 4 records.
# usage: _isa_gate.sh <B> <Hq> <Hkv> <S> <D> <arm> [<arm> ...]
set -u
B=$1; HQ=$2; HKV=$3; SL=$4; DD=$5; shift 5
OPS="v_mfma_f32_16x16x32_bf16 v_mfma_f32_32x32x16_bf16 v_permlane16_swap_b32 ds_read_b64_tr_b16 ds_read_b128 ds_write_b128 s_barrier buffer_load_dwordx4 buffer_store_dwordx2 buffer_store_dwordx4 buffer_atomic_pk_add_bf16 buffer_atomic_add_f32 v_cvt_pk_bf16_f32 v_exp_f32 v_accvgpr_read_b32 v_accvgpr_write_b32 v_mov_b32"
for A in "$@"; do
  T=$(echo "$A" | tr ':=,+' '____')
  D=/tmp/isa_$T
  rm -rf "$D"
  FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR="$D" FLYDSL_RUNTIME_ENABLE_CACHE=0 \
    timeout -s KILL 1200 python -u _probe_bwdcfg.py "$B" "$HQ" "$HKV" "$SL" "$DD" "$A" 1 >"/tmp/dump_$T.log" 2>&1
  echo "===== $A rc=$?"
  tail -2 "/tmp/dump_$T.log"
  S="$D/flash_attn_bwd_dkdv_kernel_0/21_final_isa.s"
  if [ ! -f "$S" ]; then echo "NO_ISA at $S"; ls "$D" 2>/dev/null | head; continue; fi
  grep -E 'amdhsa_next_free_vgpr|amdhsa_accum_offset|amdhsa_private_segment_fixed_size|amdhsa_group_segment_fixed_size' "$S"
  printf 'scratch_ops %s\n' "$(grep -c 'scratch_' "$S")"
  printf 'instructions %s\n' "$(grep -cE '^[[:space:]]+(s_|v_|ds_|buffer_|global_|flat_|scratch_)' "$S")"
  for OP in $OPS; do printf '%-26s %s\n' "$OP" "$(grep -c "$OP" "$S")"; done
  grep -oE 'lgkmcnt\([0-9]+\)' "$S" | sort | uniq -c | tr '\n' ' '; echo
  grep -oE 'vmcnt\([0-9]+\)' "$S" | sort | uniq -c | tr '\n' ' '; echo
done
