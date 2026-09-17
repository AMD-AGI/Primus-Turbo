set -u
R=/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-attn-llama31-8b-bwd-20260914-015415/rounds/004
JC=/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-attn-llama31-8b-bwd-20260914-015415/job_context
S=$R/scratch
echo "=== MAXPOWER / limits ==="
rocm-smi --showmaxpower 2>/dev/null | grep -E 'GPU\[1\]'
rocm-smi --showclkfrq 2>/dev/null | grep -A2 'GPU\[1\].*sclk' | head -5
echo "=== IDLE (3 samples) ==="
for i in 1 2 3; do rocm-smi -P -g -t 2>/dev/null | grep -E 'GPU\[1\]' | tr '\n' '|'; echo; sleep 1; done
cd $JC
env -i PATH=/usr/bin:/bin:/opt/venv/bin HOME=/root HIP_VISIBLE_DEVICES=1 TORCH_BLAS_PREFER_HIPBLASLT=0 \
  TRITON_CACHE_DIR=$S/tc_power SECS=70 python $S/powerload.py > $S/power_bench.out 2>&1 &
BP=$!
echo "=== UNDER LOAD ==="
for i in $(seq 1 60); do
  kill -0 $BP 2>/dev/null || break
  echo -n "t=$i "; rocm-smi -P -g -t 2>/dev/null | grep -E 'GPU\[1\]' | tr '\n' '|'; echo
  sleep 2
done
wait $BP
echo "=== LOAD OUTPUT ==="; cat $S/power_bench.out | tail -5
