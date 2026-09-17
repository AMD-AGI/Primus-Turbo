set -u
S=/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-attn-llama31-8b-bwd-20260914-015415/rounds/003/scratch
cd /home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-attn-llama31-8b-bwd-20260914-015415/job_context
run() {
  env -i PATH=/usr/bin:/bin:/opt/venv/bin HOME=/root HIP_VISIBLE_DEVICES=1 TORCH_BLAS_PREFER_HIPBLASLT=0 \
    TRITON_CACHE_DIR=$S/tct_$1 PROBE_TILE="$2" LABEL="tile$1" OPDIR="$S/probe_tile" \
    python $S/knobpoint.py 2>&1 | grep -E "^RESULT|rror" | head -4
}
run 256a 256
run 384  384
run 512  512
run 192  192
run 256b 256
