# B0 上三个容器的启动参数（docker inspect 导出，已剔除 Env）

镜像统一：`amdprimus/amdprimus:gfx1250-20260910`（A0 已有本地副本，46.7 GB，**不用 pull**）。
三个容器都是 `cmd=["sleep","infinity"]`、`entrypoint=null`、`--network host`、`--ipc host`、
`--group-add video`、`--device /dev/kfd --device /dev/dri`、`privileged=false`。差异如下：

| 容器 | 额外 cap | shm-size | 挂载 |
|---|---|---|---|
| `fa-repro`（算子测量主力） | `CAP_SYS_PTRACE` | 64 MB（默认） | `/home/lihuzhan/code` → 同路径 |
| `fa-e2e`（端到端训练） | `CAP_SYS_PTRACE` | **64 GB** | `/home/lihuzhan` → 同路径 |
| `op-evolve-...`（自主循环） | `CAP_SYS_PTRACE` + `CAP_SYS_ADMIN` | 64 MB | op-evolve 仓库路径 → 同路径 |

**A0 是单卡**：B0 上用 `HIP_VISIBLE_DEVICES` 把流围在 1 / 2 / 3 上，A0 上删掉围栏或设为 0。

```bash
# fa-repro 等价命令
docker run -d --name fa-repro \
  --network host --ipc host \
  --device /dev/kfd --device /dev/dri --group-add video \
  --cap-add CAP_SYS_PTRACE --security-opt seccomp=unconfined \
  --shm-size 64m \
  -v /home/lihuzhan/code:/home/lihuzhan/code \
  amdprimus/amdprimus:gfx1250-20260910 sleep infinity

# fa-e2e 等价命令（训练要大 shm）
docker run -d --name fa-e2e \
  --network host --ipc host \
  --device /dev/kfd --device /dev/dri --group-add video \
  --cap-add CAP_SYS_PTRACE --security-opt seccomp=unconfined \
  --shm-size 64g \
  -v /home/lihuzhan:/home/lihuzhan \
  amdprimus/amdprimus:gfx1250-20260910 sleep infinity
```

## 容器内必做（顺序不能换）

```bash
pip uninstall -y primus_turbo          # 镜像自带 editable 安装，.pth 的 MetaPathFinder 压过 PYTHONPATH
export PYTHONPATH=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo:$PYTHONPATH
python -c "import primus_turbo, sys; print(primus_turbo.__file__)"   # 必须指向 checkout
export TRITON_CACHE_DIR=/tmp/triton_cache_0                          # 每条并行流一个
export TORCH_BLAS_PREFER_HIPBLASLT=0                                 # fp32 参考实现要用
```

## 宿主机上的 rocminfo shim

`rocminfo` 只在容器里有。op-evolve 的 `tools/supervise_job.sh:120` 的 `gpu_ok()` 在**宿主机**
上跑它，不装 shim 会让 supervisor 永远在退避里睡着、一次都不重启。

```bash
mkdir -p ~/bin && cat > ~/bin/rocminfo <<'EOF'
#!/bin/sh
exec docker exec fa-repro rocminfo "$@"
EOF
chmod +x ~/bin/rocminfo && export PATH=$HOME/bin:$PATH
```

## A0 特有

`amdgpu` 在 A0 被内核参数 blacklist，重启后**必须手工** `sudo modprobe amdgpu`，
否则没有 `/dev/kfd`，torch 报 "No CUDA GPUs are available"。
重启后用 `cat /sys/class/drm/card*/device/pp_dpm_sclk` 确认限频状态（每次重启后都还在）。
