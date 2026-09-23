# `poison_allocator` 修复 —— 已写好，**尚未应用**

2026-09-23。今天一个候选 kernel 的越界**写**冲掉中断环、MES 排不空队列，
花掉一次断电周期。`poison_allocator` 本该让「没写过的元素」以 NaN 暴露出来，
它没抓住。查下来它有**两个独立缺陷**，而不是一个。

## 缺陷一：毒块太小，prod 上根本没命中

现版本毒的是 `1<<24, 1<<22, 1<<20, 1<<18, 1<<16` 个 **fp32** 元素，最大 **64 MiB**。

而 prod 的输出：

| 张量 | 形状 | dtype | 字节 |
|---|---|--:|--:|
| `dq` | [4, 8192, 32, 128] | bf16 | **256 MiB** |
| `dk` | [4, 8192, 8, 128] | bf16 | 64 MiB |
| `dv` | [4, 8192, 8, 128] | bf16 | 64 MiB |

缓存分配器按尺寸分箱，一个块只能满足**不大于它**的请求。
所以 prod 的 `dq` 永远拿不到毒块，`torch.empty` 给的是全新段，
没写过的元素读出来就是**零**——正是毒块要防的那种「像模像样的零」。

## 缺陷二：毒的位模式对输出 dtype 是错的

现版本填 **fp32** 的 NaN，即 `0x7FC00000`。但输出是 **bf16**，这四个字节被当作**两个** bf16 读：

```
0x7FC00000  ->  bf16[0] = 0x7FC0 = NaN
                bf16[1] = 0x0000 = 0.0     ← 零
```

**即使毒块命中，一半的 bf16 元素也是零。** 两个缺陷叠加，prod 上的毒块基本等于没有。

## 修法

用一个**在三种 dtype 下都是 NaN** 的字：`0x7FC07FC0`。
CPU 上已验证（无需碰卡）：

| 读作 | 结果 |
|---|---|
| fp32 | `nan` |
| bf16 低半 `0x7FC0` | exp=0xFF, man=0x40 → NaN |
| bf16 高半 `0x7FC0` | 同上 → NaN |
| fp16 `0x7FC0` | exp=0x1F, man=0x3C0 → NaN |

通过 `int32` 视图写入——用 `torch.full` 填浮点 NaN 会按各 dtype 重新编码，丢掉这个性质。
块尺寸从**实际被测的最大张量**往下按 2 的幂覆盖到 256 KiB，默认按 prod 的 256 MiB。

## ⚠ 应用它的连带后果（所以必须放在维护窗口，不能在轮次运行中改）

`build_refcache.py` 把 `ut/common.py` 的 SHA-256 写进每个缓存的 provenance，
而 `validation.py` 的 `load_reference()` 会校验它。所以：

1. **改 `ut/common.py` 会作废全部三个 refcache**（fast / proxy / prod），
   validation 会判 IGNORED 并回退到实时重算。
2. **回退到实时重算意味着 fp32 Tensile GEMM 回到测量路径**——那正是 2026-09-22
   烧掉一次断电周期的那条路径。
3. 因此必须**同一窗口内**用 `build_refcache.py --ref-device cpu` 重建 `prod.pt`。
   在 GPU 上算 prod 参考必 fault（实测两次、两种签名）。

### 应用顺序

```
1. 确认没有轮次在跑          pgrep -f 'op-evolve resume'
2. 备份                      cp op/ut/common.py op/ut/common.py.bak
3. 替换 poison_allocator     （本目录的 poison_allocator.py）
4. 自检                      python3 -c "from poison_allocator import _self_test; _self_test()"
5. 重建三个 refcache          tools/gfx1250/build_refcache.py --ref-device cpu
6. 验证 provenance 被接受     validation.py 不再报 IGNORED
```

**在此之前**：任何改 block size、split 数、grid 映射的候选都在重写地址算术，
毒块保护是失效的，必须靠离线 screen 加人工核边界顶上。
