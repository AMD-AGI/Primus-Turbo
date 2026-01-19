#!/bin/bash

#SBATCH -J primus_turbo_test   # 作业名称 (Job Name)
#SBATCH -p mi3258x             # !! 指定使用 mi3258x 分区 !!
#SBATCH -o example.log          # 输出日志文件, %j 会被替换为作业ID
#SBATCH -t 00:15:00            # 请求 15 分钟的时间
#SBATCH -N 1                   # 请求 1 个节点
#SBATCH -n 1                   # 请求 1 个任务

# --- 准备工作 ---
echo "=========================================================="
echo "Job started on: $(hostname)"
echo "Job started at: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "=========================================================="

# --- 定义容器和脚本 ---
# 直接指定你本地已有的 Apptainer SIF 文件名
CONTAINER_SIF_FILE="megatron-lm_v25.8_py310.sif"
PYTHON_SCRIPT="Primus-Turbo/minimal_example.py"

# --- 检查文件是否存在 ---
# 在执行前，最好检查一下 SIF 文件和 Python 脚本是否存在
if [ ! -f "$CONTAINER_SIF_FILE" ]; then
    echo "Error: Container file '$CONTAINER_SIF_FILE' not found!"
    exit 1
fi
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script '$PYTHON_SCRIPT' not found!"
    exit 1
fi

echo "Using existing container: $CONTAINER_SIF_FILE"

# --- 执行作业 ---
echo "\nRunning python script inside the container..."
apptainer exec \
    --rocm \
    "$CONTAINER_SIF_FILE" \
    bash -c "
        echo '--- Inside Container ---';
        python $PYTHON_SCRIPT;
        echo '--- Exiting Container ---';
    "
    
# --- 完成 ---
echo "=========================================================="
echo "Job finished at: $(date)"
echo "=========================================================="