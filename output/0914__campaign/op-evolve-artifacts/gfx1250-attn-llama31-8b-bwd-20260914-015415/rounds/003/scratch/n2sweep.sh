set -u
S=/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-attn-llama31-8b-bwd-20260914-015415/rounds/003/scratch
JC=/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-attn-llama31-8b-bwd-20260914-015415/job_context
run() {
  lbl="$1"; spec="$2"
  cd $JC
  env -i PATH=/usr/bin:/bin:/opt/venv/bin HOME=/root HIP_VISIBLE_DEVICES=1 TORCH_BLAS_PREFER_HIPBLASLT=0 \
    TRITON_CACHE_DIR=$S/tcn_$(echo "$lbl" | tr -c 'A-Za-z0-9' '_') \
    PRIMUS_TURBO_FUSED_MHA_BWD_TUNE="$spec" LABEL="$lbl" OPDIR="$S/probe_n2" \
    python $S/knobpoint.py 2>&1 | grep -E "^RESULT|rror" | head -4
}
run "n2.base"    "off"
run "n2.64"      "BLOCK_N2=64"
run "n2.128"     "BLOCK_N2=128"
run "kpack2"     "kpack=2"
run "nonkdim32"  "matrix_instr_nonkdim=32"
run "n2.64.m1.32.wpe1" "BLOCK_N2=64,waves_per_eu=1"
run "n2.base2"   "off"
