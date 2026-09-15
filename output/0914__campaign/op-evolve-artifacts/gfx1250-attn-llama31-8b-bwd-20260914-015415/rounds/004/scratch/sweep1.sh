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
    TRITON_CACHE_DIR=$T/tc_$(echo "$lbl" | tr -c 'A-Za-z0-9' '_') \
    DOTIME=${DOTIME:-1} \
    LABEL="$lbl" OPDIR="$opdir" \
    python $S/point.py > $S/pt_$(echo "$lbl" | tr -c 'A-Za-z0-9' '_').log 2>&1
  grep -E "^RESULT|Error|error:|Traceback" $S/pt_$(echo "$lbl" | tr -c 'A-Za-z0-9' '_').log | head -4
}
for v in base.pre dkdv_nolicm dkdv_unroll2 assume unroll2_nolicm assume_nolicm base.post; do
  [ -d "$T/v_$v" ] || python3 $S/mkvariant.py "$v" "$T/v_$v"
  run "$v" "$T/v_$v"
done
