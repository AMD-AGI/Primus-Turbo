set -u
SC="$1"; OP="$2"
run(){ env $1 OPDIR=$OP TRITON_CACHE_DIR=$SC/tcx SWEEP="$2" python $SC/sweep3.py 2>&1 | grep -E "bwd_ms|FAIL"; }
echo "### ITT=1 matrix"
run "TRITON_HIP_USE_IN_THREAD_TRANSPOSE=1" '[{},{"waves_per_eu":0},{"schedule_hint":"attention"},{"schedule_hint":"attention","waves_per_eu":0},{"schedule_hint":"memory-bound-attention","waves_per_eu":0},{"schedule_hint":"attention","waves_per_eu":0,"kpack":2},{"num_warps":8,"schedule_hint":"attention","waves_per_eu":0}]'
echo "### ITT=1 + PINGPONG"
run "TRITON_HIP_USE_IN_THREAD_TRANSPOSE=1 TRITON_HIP_USE_BLOCK_PINGPONG=1" '[{"schedule_hint":"attention","waves_per_eu":0},{"waves_per_eu":0}]'
