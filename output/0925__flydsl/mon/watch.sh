#!/bin/bash
# Emits one line per notable event: wedge signatures, op-evolve process gone, module/round change, KFD crowding.
OE=/home/lihuzhan/code/2026_0910__op-evolve/op-evolve
JOB=${1:?job}
A=$OE/artifacts/$JOB/job_context/state.yaml
last_state=""; last_dmesg=$(dmesg | wc -l); gone=0
while true; do
  n=$(dmesg | wc -l)
  if [ "$n" -gt "$last_dmesg" ]; then
    dmesg | tail -n $((n-last_dmesg)) | grep -iE "amdgpu.*(fault|timeout|hang|reset|MES|hogged|ring)|INVALIDATE_TLBS|copy_context_work_handler" | head -3 | sed 's/^/WEDGE? /'
    last_dmesg=$n
  fi
  if ! pgrep -f "op-evolve (run|resume).*gfx1250-flydsl-attn-fwd" >/dev/null; then
    gone=$((gone+1)); [ $gone -eq 2 ] && echo "PROC-GONE op-evolve fwd process not found"
  else gone=0; fi
  if [ -f "$A" ]; then
    s=$(python3 -c "import yaml;d=yaml.safe_load(open('$A'));su=d.get('setup',{});r=d.get('rounds') or [];lr=r[-1] if r else {};print('setup',su.get('job_setup',{}).get('status'),su.get('op_setup',{}).get('status'),'| round',lr.get('number',lr.get('round')),lr.get('status'),lr.get('module',''),'| best',d.get('best_round'),'| last',(d.get('lifecycle') or [{}])[-1].get('event'))" 2>/dev/null)
    [ "$s" != "$last_state" ] && echo "STATE $s" && last_state=$s
  fi
  k=$(timeout 15 rocm-smi --showpids 2>/dev/null | grep -cE "^[0-9]+ ")
  [ "${k:-0}" -gt 2 ] && echo "KFD-CROWD $k processes on card"
  sleep 30
done
