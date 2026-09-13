#!/bin/bash
# Post-reboot bring-up and sanity check for a gfx1250 node.
#
# Every step here exists because skipping it cost time at least once today.
# Run it after any reboot, before any measurement. Exits non-zero if the node is not fit.
set -u
fail=0
say(){ printf "%-46s %s\n" "$1" "$2"; }

# 1. amdgpu is blacklisted on the kernel command line on this node, so it does NOT come up
#    on its own. Without it there is no /dev/kfd and torch reports
#    "No CUDA GPUs are available" -- which reads like a dead card, not a missing modprobe.
if ! lsmod 2>/dev/null | grep -q '^amdgpu'; then
  say "amdgpu module" "NOT LOADED -- loading"
  sudo -n modprobe amdgpu || { say "modprobe amdgpu" "FAILED"; exit 1; }
  sleep 8
fi
lsmod 2>/dev/null | grep -q '^amdgpu' && say "amdgpu module" "loaded" || { say "amdgpu module" "STILL MISSING"; fail=1; }

# 2. /dev/kfd is what torch actually needs.
[ -e /dev/kfd ] && say "/dev/kfd" "present" || { say "/dev/kfd" "MISSING"; fail=1; }

# 3. A wedge survives a modprobe but not a reboot. dmesg is the ONLY safe probe here --
#    rocm-smi, ps with wchan, and docker exec all HANG on a wedged card.
w=$(timeout 20 sudo -n dmesg 2>/dev/null | tail -300 | grep -cE 'MES\(|wait for reset ack|GPU Hang' || echo 0)
[ "$w" -eq 0 ] && say "dmesg: wedge signatures" "clean" || { say "dmesg: wedge signatures" "$w FOUND -- REBOOT NEEDED"; fail=1; }

# 4. The VR throttle has returned after every reboot so far. Not fatal, but every absolute
#    number is conditional on it, so it must be recorded rather than discovered later.
clk=$(sudo -n cat /sys/class/drm/card*/device/pp_dpm_sclk 2>/dev/null | tr '\n' ' ')
case "$clk" in
  *2[0-9][0-9][0-9]Mhz*|*1[6-9][0-9][0-9]Mhz*) say "clock ceiling" "HEALTHY -- $clk" ;;
  *1100Mhz*) say "clock ceiling" "VR-THROTTLED 1100MHz (~1.65x low) -- $clk" ;;
  *) say "clock ceiling" "unknown: ${clk:-unreadable}" ;;
esac

# 5. A leftover process holding VRAM does not raise -- it makes every measurement contended
#    and low, and a low number becomes the champion the next round must beat.
p=$(timeout 30 sudo -n /opt/rocm/bin/rocm-smi --showpids 2>/dev/null | grep -cE '^[0-9]+' || echo 0)
[ "$p" -eq 0 ] && say "leftover KFD processes" "none" || { say "leftover KFD processes" "$p STILL HOLDING VRAM"; fail=1; }

# 6. GPU count, for deciding how many streams to launch.
n=$(timeout 30 sudo -n /opt/rocm/bin/rocm-smi --showid 2>/dev/null | grep -c '^GPU\[' || echo 0)
say "GPUs visible" "$n"

# 7. The BLAS gap makes end-to-end tps meaningless; check whether this image has it.
if [ -n "$(ls /opt/venv/lib/python*/site-packages/_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/TensileLibrary_lazy_gfx1250.dat 2>/dev/null)" ]; then
  say "hipBLASLt gfx1250 library" "PRESENT -- end-to-end may be valid"
else
  say "hipBLASLt gfx1250 library" "ABSENT -- dense GEMM ~27 TFLOP/s, E2E tps cannot validate attention"
fi

echo
[ "$fail" -eq 0 ] && echo "NODE READY." || echo "NODE NOT READY -- fix the lines marked above."
exit "$fail"
