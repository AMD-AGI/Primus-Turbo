#!/bin/bash
# Like clksamp.sh plus fclk: prints "epoch_s sclk_hz power_uw fclk_mhz" on any change + 10 ms heartbeat.
OUT=${1:?out}; DUR=${2:-600}
D=/sys/class/drm/card0/device; H=$(ls -d $D/hwmon/hwmon* | head -1)
end=$(( ${EPOCHREALTIME%.*} + DUR )); last=""; hb=0; fc=0
exec 3>"$OUT"
while :; do
  read -r f <"$H/freq1_input" 2>/dev/null; read -r p <"$H/power1_input" 2>/dev/null
  read -r -N 256 t <"$D/pp_dpm_fclk" 2>/dev/null; t=${t%%Mhz **}; fc=${t##*: }
  cur="$f $p $fc"; now=$EPOCHREALTIME
  if [ "$cur" != "$last" ] || [ "${now%.*}${now:11:2}" != "$hb" ]; then
    echo "$now $cur" >&3; last=$cur; hb="${now%.*}${now:11:2}"
  fi
  [ "${now%.*}" -ge "$end" ] && break
done
