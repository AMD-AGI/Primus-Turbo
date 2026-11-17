#!/bin/bash
# usage (inside the container, from any cwd): run_quick.sh OUT_DIR [--unfused] [--no-check]
#   CP8=1 also runs the CP8 shapes (first compile of a new kernel ~25 min per shape at n = 8).
#   BASE=path/to/results.jsonl compares against an earlier run.
# One torchrun at a time (they would share the links) -> OUT_DIR/results.jsonl + score.txt.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
W=${TURBO_ROOT:-$(cd "$HERE/../../../.." && pwd)}
OUT=$1; shift
mkdir -p "$OUT"
: > "$OUT/results.jsonl"
if pgrep -r D,R,S -f torchrun >/dev/null; then echo "stale torchrun"; exit 1; fi
export PYTHONPATH=$W FLYDSL_EXTRA_SOURCE_DIRS=$W/primus_turbo/flydsl
ulimit -c 0
run() {
  n=$1; tag=$2; shift 2
  HIP_VISIBLE_DEVICES=$(seq -s, 0 $((n - 1))) timeout 3600 torchrun --nproc_per_node $n --master_port 2973$n \
    "$HERE/gqa_quick_bench.py" --shapes "$@" $EXTRA > "$OUT/log_$tag.txt" 2>&1 || true
  grep '^RESULT ' "$OUT/log_$tag.txt" | sed 's/^RESULT //' >> "$OUT/results.jsonl"
}
EXTRA="$*"
run 4 n4 glm-4.7:32768 glm-4.7:131072 minimax-m2.5:32768 qwen3-235b-a22b:32768 qwen3-30b-a3b:32768
if [ -n "$CP8" ]; then run 8 n8 glm-4.7:32768 minimax-m2.5:32768; fi
# MLA exit guard: the same A2AGemm / GemmA2A kernels
HIP_VISIBLE_DEVICES=0,1,2,3 timeout 2400 torchrun --nproc_per_node 4 --master_port 29741 \
  "$HERE/quick_test_bench.py" --shapes deepseek-v3:32768 > "$OUT/log_mla_n4.txt" 2>&1 || true
grep '^RESULT ' "$OUT/log_mla_n4.txt" | sed 's/^RESULT //' > "$OUT/mla_results.jsonl"
python3 "$HERE/score_gqa.py" "$OUT/results.jsonl" ${BASE:+$BASE} | tee "$OUT/score.txt"
echo "MLA exit guard (deepseek-v3 32K n=4; compare with the baseline round, must not regress):" | tee -a "$OUT/score.txt"
cat "$OUT/mla_results.jsonl" | tee -a "$OUT/score.txt"
