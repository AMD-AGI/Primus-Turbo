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
# grep -c already prints a count; it also exits 1 when that count is zero, so a
# `|| echo 0` appends a SECOND line and every later [ ] test sees "0\n0". Drop the
# fallback and default the variable instead.
# Search the WHOLE buffer, not a tail window: on a busy node the MES lines scroll out of
# a few hundred lines within minutes (apparmor audit spam), and a tail-based check then
# reports a wedged card as clean -- which it did on the first run of this script.
w=$(timeout -k 5 20 sudo -n dmesg 2>/dev/null | grep -cE 'MES\(|wait for reset ack|GPU Hang'); w=${w:-0}
if [ "$w" -eq 0 ]; then
  say "dmesg: wedge signatures" "clean"
else
  say "dmesg: wedge signatures" "$w FOUND -- REBOOT NEEDED"
  echo
  echo "NODE WEDGED. Everything below this point either hangs or reports nothing useful"
  echo "on a wedged card, so the remaining checks are skipped. Reboot, then modprobe amdgpu."
  exit 1
fi

# 4. The VR throttle has returned after every reboot so far. Not fatal, but every absolute
#    number is conditional on it, so it must be recorded rather than discovered later.
clk=$(timeout -k 5 15 sudo -n cat /sys/class/drm/card*/device/pp_dpm_sclk 2>/dev/null | tr '\n' ' ')
case "$clk" in
  *2[0-9][0-9][0-9]Mhz*|*1[6-9][0-9][0-9]Mhz*) say "clock ceiling" "HEALTHY -- $clk" ;;
  *1100Mhz*) say "clock ceiling" "VR-THROTTLED 1100MHz (~1.65x low) -- $clk" ;;
  *) say "clock ceiling" "unknown: ${clk:-unreadable}" ;;
esac

# 5. A leftover process holding VRAM does not raise -- it makes every measurement contended
#    and low, and a low number becomes the champion the next round must beat.
# -k: rocm-smi on a wedged card ignores SIGTERM (it is blocked in amdgpu_info_ioctl),
# so a plain `timeout` never returns. SIGKILL after 5 more seconds is what actually
# bounds it. This is the command that hung the first run of this script.
# Same rocm-smi hazard as above; /sys/class/kfd/kfd/proc/ lists the processes holding the
# device and is a directory read.
p=$(ls /sys/class/kfd/kfd/proc/ 2>/dev/null | wc -l); p=${p:-0}
[ "$p" -eq 0 ] && say "leftover KFD processes" "none" || { say "leftover KFD processes" "$p STILL HOLDING VRAM"; fail=1; }

# 6. GPU count, for deciding how many streams to launch.
# NOT rocm-smi: under `timeout` the sudo parent dies but the rocm-smi child survives
# holding the pipe open, so $( ) never returns -- that is what hung this script twice.
# KFD topology is a plain sysfs read and cannot block on the driver.
n=$(grep -l . /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null | wc -l); n=${n:-0}
n=$(awk '/gfx_target_version/ && $2 != 0 {c++} END{print c+0}' /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null)
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
