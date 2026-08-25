#!/usr/bin/env bash
# One wall batch: every arm in its own process, in the order given (build the palindrome
# at the call site), each JSON line written to a file first -- a batch read through a pipe
# has come back with interleaved lines and readings 87% apart (round 17).
# usage: _wall_batch.sh <B> <Hq> <Hkv> <S> <D> <reps> <W> <arm> [<arm> ...]
set -u
B=$1; HQ=$2; HKV=$3; SL=$4; DD=$5; REPS=$6; WW=$7; shift 7
OUT=/tmp/wall_batch_$$.txt
: >"$OUT"
for A in "$@"; do
  timeout -s KILL 2400 python -u _probe_bwdcfg.py "$B" "$HQ" "$HKV" "$SL" "$DD" "$A" "$REPS" "$WW" \
    >"/tmp/one_arm_$$.txt" 2>"/tmp/one_arm_err_$$.txt"
  tail -1 "/tmp/one_arm_$$.txt" >>"$OUT" || true
  tail -2 "/tmp/one_arm_err_$$.txt" >>"$OUT"
done
echo "----- batch $B/$HQ/$HKV/$SL/$DD reps=$REPS W=$WW"
cat "$OUT"
