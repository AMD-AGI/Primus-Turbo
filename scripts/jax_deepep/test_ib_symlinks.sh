#!/bin/bash
echo "--- Which libionic.so is actually loaded ---"
readlink -f /usr/lib/x86_64-linux-gnu/libionic.so
readlink -f /usr/lib/x86_64-linux-gnu/libionic.so.1
readlink -f /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so
echo "--- File sizes ---"
ls -la /usr/lib/x86_64-linux-gnu/libionic.so.1.1.54.0-185
ls -la /usr/lib/x86_64-linux-gnu/libionic.so.1.0.54.0-149.g3304be71
echo "--- md5sum ---"
md5sum /usr/lib/x86_64-linux-gnu/libionic.so.1.1.54.0-185
md5sum /usr/lib/x86_64-linux-gnu/libionic.so.1.0.54.0-149.g3304be71
echo "--- Dynamic loader check ---"
LD_DEBUG=libs python3 -c "import ctypes; ctypes.CDLL('libionic.so')" 2>&1 | grep -i ionic | head -10
