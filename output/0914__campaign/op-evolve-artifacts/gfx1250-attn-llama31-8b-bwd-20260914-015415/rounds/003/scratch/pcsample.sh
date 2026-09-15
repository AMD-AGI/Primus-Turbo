set -u
cd /home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-attn-llama31-8b-bwd-20260914-015415/job_context
rm -rf /home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-attn-llama31-8b-bwd-20260914-015415/rounds/003/scratch/pc/out; mkdir -p /home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-attn-llama31-8b-bwd-20260914-015415/rounds/003/scratch/pc/out
rocprofv3 --pc-sampling-beta-enabled --pc-sampling-method stochastic \
  --pc-sampling-unit cycles --pc-sampling-interval 1048576 \
  --output-format csv --output-file pc -d /home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-attn-llama31-8b-bwd-20260914-015415/rounds/003/scratch/pc/out \
  -- python op/benchmark.py --impl /home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-attn-llama31-8b-bwd-20260914-015415/rounds/003/op --shape b4_s8192_hq32_hkv8_d128 --iters 5 --warmup-s 1.0
echo "RC=$?"
find /home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-attn-llama31-8b-bwd-20260914-015415/rounds/003/scratch/pc/out -type f | head -20
