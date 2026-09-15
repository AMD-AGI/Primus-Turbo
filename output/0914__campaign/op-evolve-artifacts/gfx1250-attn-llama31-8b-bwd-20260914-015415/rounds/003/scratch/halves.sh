set -u
S=/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-attn-llama31-8b-bwd-20260914-015415/rounds/003/scratch
JC=/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-attn-llama31-8b-bwd-20260914-015415/job_context
run() {
  lbl="$1"; spec="$2"
  cd $JC
  env -i PATH=/usr/bin:/bin:/opt/venv/bin HOME=/root HIP_VISIBLE_DEVICES=1 TORCH_BLAS_PREFER_HIPBLASLT=0 \
    TRITON_CACHE_DIR=$S/tch_$(echo "$lbl" | tr -c 'A-Za-z0-9' '_') \
    PRIMUS_TURBO_FUSED_MHA_BWD_TUNE="$spec" \
    LABEL="$lbl" OPDIR=$S/probe_halves \
    python $S/knobpoint.py 2>&1 | grep -E "^RESULT|Error|error:" | head -5
}
run "probe.full"     "off"
run "probe.dkdv_only" "SKIP_DQ=1"
run "probe.dq_only"   "SKIP_DKDV=1"
run "probe.full2"    "off"
