set -u
B=/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-flydsl-attn-bwd-20260917-115934
S=$B/rounds/027/_scratch
export TORCH_BLAS_PREFER_HIPBLASLT=1 HIPBLASLT_TENSILE_LIBPATH=/home/lihuzhan/.local/hipblaslt-gfx1250/gfx1250
export ARCH=gfx1250 FLYDSL_GPU_ARCH=gfx1250
cd $B/job_context/op
for a in A_g74 B_g82; do
  echo "################## BUILD $a $(date -u +%FT%TZ)"
  D=$S/isa_$a; rm -rf $D; mkdir -p $D
  export FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache_r027_$a.$$
  rm -rf $FLYDSL_RUNTIME_CACHE_DIR
  find $S/arms/$a -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
  timeout 900 /opt/venv/bin/python3 $B/rounds/027/1-opt/raw/build_probe.py \
      --impl $S/arms/$a --dump-dir $D --shape prod
  echo "PROBE_RC_$a=$?"
  rm -rf $FLYDSL_RUNTIME_CACHE_DIR
done
echo "################## DONE $(date -u +%FT%TZ)"
/opt/venv/bin/rocm-smi --showpids | tail -3
