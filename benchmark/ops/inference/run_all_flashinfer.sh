#!/usr/bin/env bash
# Full FlashInfer (NVIDIA B200) bench sweep, mirroring the AMD MI355X aiter
# reference layout (smx355/) one-to-one so results compare directly:
#   MoE    -> moe_results/{model}_{quant}_tp{T}_ep{E}_var{V}.xlsx
#   decode -> pa_results/{model}_tp{T}_dp{D}_cv{CV}.xlsx
#
# Matrix matches the AMD reference exactly (none/fp8_block/fp4 x var{0,0.5,2};
# decode cv{0,0.5,1.0,1.5}); nvfp4 is added as the Blackwell-flagship extra
# (same naming convention, no AMD counterpart).
set -u
cd "$(dirname "$0")"
mkdir -p moe_results pa_results

MOE_LOG=moe_results/run_all.log
PA_LOG=pa_results/run_pa.log
: > "$MOE_LOG"
: > "$PA_LOG"

run() {  # log-file cmd...
  local log="$1"; shift
  echo "+ $*" | tee -a "$log"
  "$@" >> "$log" 2>&1 || echo "  (exit $? -- see $log)" | tee -a "$log"
}

# ----- MoE: (model -> ep degree), quants, variances -------------------------
# AMD-comparable quants + nvfp4 flagship extra.
MOE_QUANTS=(none fp8_block fp4 nvfp4)
MOE_VARS=(0 0.5 2)
# model:degree  (tp1_ep{deg} and tp{deg}_ep1, matching the AMD reference)
MOE_MODELS=(
  "deepseek-r1:8" "deepseek-v4-pro:8" "glm-5.1:8" "kimi-k2.6:8"
  "mimo-v2.5-pro:8" "qwen3-next-80b-a3b:8" "minimax-m2.7:2" "minimax-m3:4"
)
for entry in "${MOE_MODELS[@]}"; do
  model="${entry%%:*}"; deg="${entry##*:}"
  for quant in "${MOE_QUANTS[@]}"; do
    for var in "${MOE_VARS[@]}"; do
      for par in "tp1_ep${deg}" "tp${deg}_ep1"; do
        tp="${par%%_*}"; tp="${tp#tp}"; ep="${par##*ep}"
        out="moe_results/${model}_${quant}_tp${tp}_ep${ep}_var${var}.xlsx"
        run "$MOE_LOG" python3 bench_moe_flashinfer.py --model "$model" \
          --tp-size "$tp" --ep-size "$ep" --quant "$quant" --variance "$var" --check -o "$out"
      done
    done
  done
done

# ----- decode: (model -> tp/dp combos), cv ----------------------------------
PA_CVS=(0 0.5 1.0 1.5)
# model:combo[,combo...]   combo = tp{T}_dp{D}
PA_MODELS=(
  "deepseek-r1:tp8_dp1,tp8_dp8" "glm-5.1:tp8_dp1,tp8_dp8" "kimi-k2.6:tp8_dp1,tp8_dp8"
  "gpt-oss-120b:tp1_dp1" "mimo-v2.5-pro:tp8_dp1" "minimax-m2.7:tp2_dp1"
)
for entry in "${PA_MODELS[@]}"; do
  model="${entry%%:*}"; combos="${entry##*:}"
  IFS=',' read -ra combo_arr <<< "$combos"
  for combo in "${combo_arr[@]}"; do
    tp="${combo%%_*}"; tp="${tp#tp}"; dp="${combo##*dp}"
    for cv in "${PA_CVS[@]}"; do
      out="pa_results/${model}_tp${tp}_dp${dp}_cv${cv}.xlsx"
      run "$PA_LOG" python3 bench_decode_attn_flashinfer.py --model "$model" \
        --tp-size "$tp" --dp-size "$dp" --ctx-cv "$cv" --check -o "$out"
    done
  done
done

echo "ALL DONE" | tee -a "$MOE_LOG" | tee -a "$PA_LOG"
