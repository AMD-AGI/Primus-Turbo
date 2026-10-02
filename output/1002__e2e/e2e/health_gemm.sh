#!/bin/bash
# Post-incident card health check (card-safety section 2a): ONE process, ONE 4096^3 bf16 matmul x 20 under a
# hard timeout, IMAGE hipBLASLt library, finite result, its usual time. Run it ONLY after an e2e stop that
# was not an unrecoverable dmesg fault (NaN stop, watchdog hang with a clean dmesg, memguard), and BEFORE any
# other card work (including op-evolve). Never run it on a card whose dmesg shows `wait for reset ack`,
# `GPU reset begin`, `ring ... timeout` or `MES ... unrecoverable` -- that card needs the user's AC cycle.
#   bash health_gemm.sh     -> HEALTH_OK <ms> | HEALTH_FAIL <why>  (exit 0 / 1)
set -u
[ -z "$(ls /sys/class/kfd/kfd/proc 2>/dev/null)" ] || { echo "HEALTH_FAIL KFD holders: $(ls /sys/class/kfd/kfd/proc | tr '\n' ' ')"; exit 1; }
BLAS_LIB=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250
OUT=$(flock -w 600 /tmp/a0-gpu0.lock timeout 120 docker exec fa-repro bash -c "ulimit -c 0; \
  export TORCH_BLAS_PREFER_HIPBLASLT=1 HIPBLASLT_TENSILE_LIBPATH=$BLAS_LIB; exec timeout --foreground -k 10 100 /opt/venv/bin/python3 -c '
import time, torch
a = torch.randn(4096, 4096, device=\"cuda\", dtype=torch.bfloat16); b = torch.randn(4096, 4096, device=\"cuda\", dtype=torch.bfloat16)
c = a @ b; torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(20): c = a @ b
torch.cuda.synchronize(); ms = (time.perf_counter() - t) / 20 * 1e3
print(\"GEMM\", round(ms, 4), \"ms finite\", bool(torch.isfinite(c).all()), torch.cuda.get_device_properties(0).gcnArchName)
'" 2>&1)
RC=$?
echo "$OUT" | tail -3
if [ $RC = 0 ] && echo "$OUT" | grep -q "finite True"; then echo "HEALTH_OK $(echo "$OUT" | grep -oE 'GEMM [0-9.]+')"; exit 0; fi
echo "HEALTH_FAIL rc=$RC"; exit 1
