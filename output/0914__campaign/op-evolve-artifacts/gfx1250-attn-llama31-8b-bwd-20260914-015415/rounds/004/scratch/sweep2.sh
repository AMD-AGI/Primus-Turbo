set -u
R=/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-attn-llama31-8b-bwd-20260914-015415/rounds/004
S=$R/scratch
T=$R/scratch
JC=/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-attn-llama31-8b-bwd-20260914-015415/job_context
run() {
  lbl="$1"; opdir="$2"
  cd $JC
  env -i PATH=/usr/bin:/bin:/opt/venv/bin HOME=/root HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-1} \
    TORCH_BLAS_PREFER_HIPBLASLT=0 \
    TRITON_CACHE_DIR=$T/tcD_$(echo "$lbl" | tr -c 'A-Za-z0-9' '_') \
    DOTIME=${DOTIME:-1} \
    LABEL="$lbl" OPDIR="$opdir" \
    python $S/point.py > $S/ptD_$(echo "$lbl" | tr -c 'A-Za-z0-9' '_').log 2>&1
  grep -E "^RESULT|Error|error:|Traceback" $S/ptD_$(echo "$lbl" | tr -c 'A-Za-z0-9' '_').log | head -4
}
for v in bp1 g17 g16flat g17b bp2; do
  d=$S/v_$v; [ "$v" = g17b ] && d=$S/v_g17
  run "$v" "$d"
done
