#!/bin/bash
# 测试所有 96 个 FP8 GEMM shapes，每个 shape 测试该 layout 的所有 kernel

BIN_DIR="/workspace/composable_kernel/build/bin"
OUTPUT_FILE="/workspace/Primus-Turbo/benchmark/ops/fp8_ck_results.csv"
DETAIL_FILE="/workspace/Primus-Turbo/benchmark/ops/fp8_ck_results_detail.csv"

# CSV 头
echo "Index,Model,M,K,N,Layout,Best_Kernel,Best_TFLOPS,Latency_ms,Bandwidth_GB_s,Kernels_Tested" > "$OUTPUT_FILE"
echo "Index,Model,M,K,N,Layout,Kernel,TFLOPS,Latency_ms,Bandwidth_GB_s" > "$DETAIL_FILE"

run_all_kernels() {
    local idx=$1 model=$2 m=$3 k=$4 n=$5 layout=$6
    
    echo "[$idx/96] $model M=$m K=$k N=$n Layout=$layout"
    
    # 获取该 layout 的所有 kernel
    local kernels=($BIN_DIR/benchmark_gemm_fp8_${layout}_*)
    local num_kernels=${#kernels[@]}
    
    local best_tflops=0
    local best_kernel=""
    local best_latency=0
    local best_bw=0
    local tested=0
    
    for kernel in "${kernels[@]}"; do
        if [ -x "$kernel" ]; then
            result=$($kernel -m=$m -n=$n -k=$k -warmup=10 -repeat=50 -json_output=true 2>/dev/null)
            tflops=$(echo "$result" | grep -o '"tflops(TFlops)": [0-9.]*' | grep -o '[0-9.]*$')
            latency=$(echo "$result" | grep -o '"latency(ms)": [0-9.]*' | grep -o '[0-9.]*$')
            bw=$(echo "$result" | grep -o '"bandwidth(GB/s)": [0-9.]*' | grep -o '[0-9.]*$')
            
            if [ -n "$tflops" ]; then
                tested=$((tested + 1))
                kernel_name=$(basename "$kernel")
                
                # 保存详细结果
                echo "$idx,$model,$m,$k,$n,$layout,$kernel_name,$tflops,$latency,$bw" >> "$DETAIL_FILE"
                
                # 检查是否是最佳 (用 awk 比较浮点数)
                if awk "BEGIN {exit !($tflops > $best_tflops)}"; then
                    best_tflops=$tflops
                    best_kernel=$kernel_name
                    best_latency=$latency
                    best_bw=$bw
                fi
            fi
        fi
    done
    
    echo "  -> Best: $best_tflops TFLOPS ($best_kernel) [tested $tested kernels]"
    echo "$idx,$model,$m,$k,$n,$layout,$best_kernel,$best_tflops,$best_latency,$best_bw,$tested" >> "$OUTPUT_FILE"
}

echo "Starting FP8 GEMM benchmark for all 96 shapes (testing ALL kernels per layout)..."
echo "Each layout has ~60 kernels to test"
echo ""

# llama2-7b (12 shapes)
run_all_kernels 1 "llama2-7b" 4096 4096 12288 "rcr"
run_all_kernels 2 "llama2-7b" 4096 12288 4096 "rrr"
run_all_kernels 3 "llama2-7b" 12288 4096 4096 "crr"
run_all_kernels 4 "llama2-7b" 4096 4096 4096 "rcr"
run_all_kernels 5 "llama2-7b" 4096 4096 4096 "rrr"
run_all_kernels 6 "llama2-7b" 4096 4096 4096 "crr"
run_all_kernels 7 "llama2-7b" 4096 4096 22016 "rcr"
run_all_kernels 8 "llama2-7b" 4096 22016 4096 "rrr"
run_all_kernels 9 "llama2-7b" 22016 4096 4096 "crr"
run_all_kernels 10 "llama2-7b" 4096 11008 4096 "rcr"
run_all_kernels 11 "llama2-7b" 4096 4096 11008 "rrr"
run_all_kernels 12 "llama2-7b" 4096 4096 11008 "crr"

# llama2-70b (12 shapes)
run_all_kernels 13 "llama2-70b" 4096 8192 10240 "rcr"
run_all_kernels 14 "llama2-70b" 4096 10240 8192 "rrr"
run_all_kernels 15 "llama2-70b" 10240 4096 8192 "crr"
run_all_kernels 16 "llama2-70b" 4096 8192 8192 "rcr"
run_all_kernels 17 "llama2-70b" 4096 8192 8192 "rrr"
run_all_kernels 18 "llama2-70b" 8192 4096 8192 "crr"
run_all_kernels 19 "llama2-70b" 4096 8192 57344 "rcr"
run_all_kernels 20 "llama2-70b" 4096 57344 8192 "rrr"
run_all_kernels 21 "llama2-70b" 57344 4096 8192 "crr"
run_all_kernels 22 "llama2-70b" 4096 28672 8192 "rcr"
run_all_kernels 23 "llama2-70b" 4096 8192 28672 "rrr"
run_all_kernels 24 "llama2-70b" 8192 4096 28672 "crr"

# llama3.1-8b (12 shapes)
run_all_kernels 25 "llama3.1-8b" 8192 4096 6144 "rcr"
run_all_kernels 26 "llama3.1-8b" 8192 6144 4096 "rrr"
run_all_kernels 27 "llama3.1-8b" 6144 8192 4096 "crr"
run_all_kernels 28 "llama3.1-8b" 8192 4096 4096 "rcr"
run_all_kernels 29 "llama3.1-8b" 8192 4096 4096 "rrr"
run_all_kernels 30 "llama3.1-8b" 4096 8192 4096 "crr"
run_all_kernels 31 "llama3.1-8b" 8192 4096 28672 "rcr"
run_all_kernels 32 "llama3.1-8b" 8192 28672 4096 "rrr"
run_all_kernels 33 "llama3.1-8b" 28672 8192 4096 "crr"
run_all_kernels 34 "llama3.1-8b" 8192 14336 4096 "rcr"
run_all_kernels 35 "llama3.1-8b" 8192 4096 14336 "rrr"
run_all_kernels 36 "llama3.1-8b" 4096 8192 14336 "crr"

# llama3.1-70b (12 shapes)
run_all_kernels 37 "llama3.1-70b" 8192 8192 10240 "rcr"
run_all_kernels 38 "llama3.1-70b" 8192 10240 8192 "rrr"
run_all_kernels 39 "llama3.1-70b" 10240 8192 8192 "crr"
run_all_kernels 40 "llama3.1-70b" 8192 8192 8192 "rcr"
run_all_kernels 41 "llama3.1-70b" 8192 8192 8192 "rrr"
run_all_kernels 42 "llama3.1-70b" 8192 8192 8192 "crr"
run_all_kernels 43 "llama3.1-70b" 8192 8192 57344 "rcr"
run_all_kernels 44 "llama3.1-70b" 8192 57344 8192 "rrr"
run_all_kernels 45 "llama3.1-70b" 57344 8192 8192 "crr"
run_all_kernels 46 "llama3.1-70b" 8192 28672 8192 "rcr"
run_all_kernels 47 "llama3.1-70b" 8192 8192 28672 "rrr"
run_all_kernels 48 "llama3.1-70b" 8192 8192 28672 "crr"

# llama3.1-405b (12 shapes)
run_all_kernels 49 "llama3.1-405b" 8192 16384 18432 "rcr"
run_all_kernels 50 "llama3.1-405b" 8192 18432 16384 "rrr"
run_all_kernels 51 "llama3.1-405b" 18432 8192 16384 "crr"
run_all_kernels 52 "llama3.1-405b" 8192 16384 16384 "rcr"
run_all_kernels 53 "llama3.1-405b" 8192 16384 16384 "rrr"
run_all_kernels 54 "llama3.1-405b" 16384 8192 16384 "crr"
run_all_kernels 55 "llama3.1-405b" 8192 16384 106496 "rcr"
run_all_kernels 56 "llama3.1-405b" 8192 106496 16384 "rrr"
run_all_kernels 57 "llama3.1-405b" 106496 8192 16384 "crr"
run_all_kernels 58 "llama3.1-405b" 8192 53248 16384 "rcr"
run_all_kernels 59 "llama3.1-405b" 8192 16384 53248 "rrr"
run_all_kernels 60 "llama3.1-405b" 16384 8192 53248 "crr"

# qwen2-7b (12 shapes)
run_all_kernels 61 "qwen2-7b" 32768 3584 4608 "rcr"
run_all_kernels 62 "qwen2-7b" 32768 4608 3584 "rrr"
run_all_kernels 63 "qwen2-7b" 4608 32768 3584 "crr"
run_all_kernels 64 "qwen2-7b" 32768 3584 3584 "rcr"
run_all_kernels 65 "qwen2-7b" 32768 3584 3584 "rrr"
run_all_kernels 66 "qwen2-7b" 3584 32768 3584 "crr"
run_all_kernels 67 "qwen2-7b" 32768 3584 37888 "rcr"
run_all_kernels 68 "qwen2-7b" 32768 37888 3584 "rrr"
run_all_kernels 69 "qwen2-7b" 37888 32768 3584 "crr"
run_all_kernels 70 "qwen2-7b" 32768 18944 3584 "rcr"
run_all_kernels 71 "qwen2-7b" 32768 3584 18944 "rrr"
run_all_kernels 72 "qwen2-7b" 3584 32768 18944 "crr"

# qwen2-72b (12 shapes)
run_all_kernels 73 "qwen2-72b" 32768 8192 10240 "rcr"
run_all_kernels 74 "qwen2-72b" 32768 10240 8192 "rrr"
run_all_kernels 75 "qwen2-72b" 10240 32768 8192 "crr"
run_all_kernels 76 "qwen2-72b" 32768 8192 8192 "rcr"
run_all_kernels 77 "qwen2-72b" 32768 8192 8192 "rrr"
run_all_kernels 78 "qwen2-72b" 8192 32768 8192 "crr"
run_all_kernels 79 "qwen2-72b" 32768 8192 59136 "rcr"
run_all_kernels 80 "qwen2-72b" 32768 59136 8192 "rrr"
run_all_kernels 81 "qwen2-72b" 59136 32768 8192 "crr"
run_all_kernels 82 "qwen2-72b" 32768 29568 8192 "rcr"
run_all_kernels 83 "qwen2-72b" 32768 8192 29568 "rrr"
run_all_kernels 84 "qwen2-72b" 8192 32768 29568 "crr"

# qwen2.5-32b (12 shapes)
run_all_kernels 85 "qwen2.5-32b" 32768 5120 7168 "rcr"
run_all_kernels 86 "qwen2.5-32b" 32768 7168 5120 "rrr"
run_all_kernels 87 "qwen2.5-32b" 7168 32768 5120 "crr"
run_all_kernels 88 "qwen2.5-32b" 32768 5120 5120 "rcr"
run_all_kernels 89 "qwen2.5-32b" 32768 5120 5120 "rrr"
run_all_kernels 90 "qwen2.5-32b" 5120 32768 5120 "crr"
run_all_kernels 91 "qwen2.5-32b" 32768 5120 55296 "rcr"
run_all_kernels 92 "qwen2.5-32b" 32768 55296 5120 "rrr"
run_all_kernels 93 "qwen2.5-32b" 55296 32768 5120 "crr"
run_all_kernels 94 "qwen2.5-32b" 32768 27648 5120 "rcr"
run_all_kernels 95 "qwen2.5-32b" 32768 5120 27648 "rrr"
run_all_kernels 96 "qwen2.5-32b" 5120 32768 27648 "crr"

echo ""
echo "=========================================="
echo "All 96 shapes tested with ALL kernels!"
echo "Best results: $OUTPUT_FILE"
echo "Detail results: $DETAIL_FILE"
echo "=========================================="
