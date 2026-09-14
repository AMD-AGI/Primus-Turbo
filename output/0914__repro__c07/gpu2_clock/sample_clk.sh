#!/bin/bash
# sample card16 (= HIP GPU 2 = 0003:04:00.0)
D=/sys/class/drm/card16/device
H=$D/hwmon/hwmon6
end=$((SECONDS+$1))
while [ $SECONDS -lt $end ]; do
  echo "$(date +%s.%N) $(cat $H/freq1_input) $(cat $H/freq2_input) $(cat $D/gpu_busy_percent) $(cat $D/mem_busy_percent) $(cat $H/temp2_input) $(cat $D/pp_dpm_sclk | tr '\n' '|')"
  sleep 0.2
done
