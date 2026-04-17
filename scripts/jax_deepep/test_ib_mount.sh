#!/bin/bash
set -e
echo "=== Container with host ionic driver mount ==="
echo "Host: $(hostname)"

echo "--- libibverbs driver config ---"
cat /etc/libibverbs.d/ionic.driver 2>/dev/null || echo "(no ionic.driver)"
echo ""

echo "--- Checking mounted ionic libs ---"
ls -la /usr/lib/x86_64-linux-gnu/libionic* 2>/dev/null
ls -la /usr/lib/x86_64-linux-gnu/libibverbs/libionic* 2>/dev/null
echo ""

echo "--- ibv_devices ---"
ibv_devices 2>&1
echo ""

echo "--- ibv_devinfo (first device) ---"
ibv_devinfo -d ionic_0 2>&1 | head -30
echo ""

echo "--- Test ibv_rc_pingpong loopback (quick sanity) ---"
# Just check if we can at least open the device and create QP
python3 -c "
import ctypes, ctypes.util
lib = ctypes.CDLL('libibverbs.so.1')
print('libibverbs loaded OK')

# Try to get device list
num_devs = ctypes.c_int(0)
dev_list = lib.ibv_get_device_list(ctypes.byref(num_devs))
print(f'ibv_get_device_list: {num_devs.value} devices found')
if num_devs.value > 0:
    print('RDMA devices accessible from Python!')
else:
    print('WARNING: No RDMA devices visible')
lib.ibv_free_device_list(dev_list)
" 2>&1
echo "=== exit code: $? ==="
