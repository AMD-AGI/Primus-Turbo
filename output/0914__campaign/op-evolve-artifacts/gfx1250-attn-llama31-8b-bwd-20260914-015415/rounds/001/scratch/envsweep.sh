set -u
SC="$1"; OP="$2"
for e in "NONE" "TRITON_HIP_USE_BLOCK_PINGPONG=1" "TRITON_HIP_USE_ASYNC_COPY=0" "TRITON_HIP_USE_ASYNC_COPY=1" "AMDGCN_USE_BUFFER_OPS=0" "TRITON_HIP_USE_IN_THREAD_TRANSPOSE=1"; do
  if [ "$e" = "NONE" ]; then pre=""; else pre="$e"; fi
  out=$(env $pre OPDIR=$OP TRITON_CACHE_DIR=$SC/tc_$(echo $e|tr '=' '_') SWEEP='[{"schedule_hint":"attention","waves_per_eu":0}]' python $SC/sweep3.py 2>&1 | grep bwd_ms | head -1)
  echo "$e  ->  $out"
done
