set -u
S=/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-attn-llama31-8b-bwd-20260914-015415/rounds/003/scratch
JC=/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-attn-llama31-8b-bwd-20260914-015415/job_context
OPDIR=$JC/op/current
run() {  # run <label> <env assignments...>
  lbl="$1"; shift
  cd $JC
  env -i PATH=/usr/bin:/bin:/opt/venv/bin HOME=/root HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-1} \
    TORCH_BLAS_PREFER_HIPBLASLT=0 \
    TRITON_CACHE_DIR=$S/tc_$(echo "$lbl" | tr -c 'A-Za-z0-9' '_') \
    LABEL="$lbl" OPDIR=$OPDIR "$@" \
    python $S/knobpoint.py 2>&1 | grep -E "^RESULT|Error|error:" | head -5
}
# base slot first and last (palindromic) -- the whole sweep is one session
run "base.pre"                 TRITON_HIP_USE_IN_THREAD_TRANSPOSE=1
run "scalarize_packed_fops"    TRITON_HIP_USE_IN_THREAD_TRANSPOSE=1 AMDGCN_SCALARIZE_PACKED_FOPS=1
run "small_tensor_range"       TRITON_HIP_USE_IN_THREAD_TRANSPOSE=1 AMDGCN_ANALYZE_SMALL_TENSOR_RANGE=1
run "no_buffer_ops"            TRITON_HIP_USE_IN_THREAD_TRANSPOSE=1 AMDGCN_USE_BUFFER_OPS=0
run "pingpong"                 TRITON_HIP_USE_IN_THREAD_TRANSPOSE=1 TRITON_HIP_USE_BLOCK_PINGPONG=1
run "no_async_copy"            TRITON_HIP_USE_IN_THREAD_TRANSPOSE=1 TRITON_HIP_USE_ASYNC_COPY=0
run "itt_off"                  TRITON_HIP_USE_IN_THREAD_TRANSPOSE=0
run "base.post"                TRITON_HIP_USE_IN_THREAD_TRANSPOSE=1
