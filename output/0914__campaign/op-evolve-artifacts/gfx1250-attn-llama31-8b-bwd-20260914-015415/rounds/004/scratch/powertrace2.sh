set -u
R=/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-attn-llama31-8b-bwd-20260914-015415/rounds/004
JC=/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-attn-llama31-8b-bwd-20260914-015415/job_context
S=$R/scratch
echo "=== device map ==="
rocm-smi --showbus 2>/dev/null | grep -E 'GPU\['
python - <<'PY'
import os, torch
os.environ.setdefault("HIP_VISIBLE_DEVICES","1")
print("torch sees", torch.cuda.device_count(), "device(s)")
p = torch.cuda.get_device_properties(0)
print("HIP_VISIBLE_DEVICES=1 -> name", p.name, "pci_bus_id", getattr(p,"pci_bus_id",None), "uuid", getattr(p,"uuid",None))
PY
echo "=== IDLE, ALL GPUS (3 samples) ==="
for i in 1 2 3; do rocm-smi -P -g -t -u 2>/dev/null | grep -E 'Power|sclk|GPU use'; echo "--"; sleep 2; done
cd $JC
env -i PATH=/usr/bin:/bin:/opt/venv/bin HOME=/root HIP_VISIBLE_DEVICES=1 TORCH_BLAS_PREFER_HIPBLASLT=0 \
  TRITON_CACHE_DIR=$S/tc_power SECS=90 python $S/powerload.py > $S/power_bench2.out 2>&1 &
BP=$!
until grep -q LOADSTART $S/power_bench2.out 2>/dev/null; do kill -0 $BP 2>/dev/null || break; sleep 2; done
echo "=== UNDER LOAD, ALL GPUS ==="
for i in $(seq 1 30); do
  kill -0 $BP 2>/dev/null || break
  echo "t=$i"; rocm-smi -P -g -t -u 2>/dev/null | grep -E 'Power|sclk|GPU use'
  sleep 2
done
wait $BP
echo "=== LOAD OUTPUT ==="; cat $S/power_bench2.out
