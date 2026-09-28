#!/bin/bash
# Host-side high-rate sampler for GPU0 (card0): prints "epoch_s sclk_hz power_uw" whenever
# any value changes (sysfs reads are ~5 us; the driver refreshes its SMU metrics cache ~1 ms), plus a
# heartbeat line every ~50 ms. Bounded sysfs reads only (card-safety sec. 4). Stops at $2 seconds.
OUT=${1:?out}; DUR=${2:-600}
D=/sys/class/drm/card0/device; H=$(ls -d $D/hwmon/hwmon* | head -1)
end=$(( ${EPOCHREALTIME%.*} + DUR )); last=""; hb=0
exec 3>"$OUT"
while :; do
  read -r f <"$H/freq1_input"; read -r p <"$H/power1_input"
  cur="$f $p"; now=$EPOCHREALTIME
  if [ "$cur" != "$last" ] || [ "${now%.*}${now:11:2}" != "$hb" ]; then
    echo "$now $cur" >&3; last=$cur; hb="${now%.*}${now:11:2}"
  fi
  [ "${now%.*}" -ge "$end" ] && break
done
