# import torch
# import primus_turbo.pytorch as turbo

# print(f"PyTorch version: {torch.__version__}")
# print(f"Is CUDA (ROCm) available? {torch.cuda.is_available()}")
# if torch.cuda.is_available():
#     print(f"Device Name: {torch.cuda.get_device_name(0)}")

# dtype = torch.bfloat16
# device = "cuda:0"
# G = 4
# M = 128  # 128=32+16+48+32
# N = 256
# K = 512

# print("\nCreating tensors on device...")
# group_lens = torch.tensor([32, 16, 48, 32], dtype=torch.long, device=device)
# a = torch.randn(M, K, device=device, dtype=dtype)
# b = torch.randn(G, K, N, device=device, dtype=dtype)
# print("Tensors created successfully.")

# print("Performing GEMM operation with primus_turbo...")
# c = turbo.ops.grouped_gemm(a, b, group_lens, trans_b=False)
# print("GEMM operation completed.")

# print("\nOutput Tensor:")
# print(c)
# print(f"Output Tensor Shape: {c.shape}")
# print(f"Output Tensor Dtype: {c.dtype}")
# print("Script finished successfully.")
# ==========================================================================================
# import torch
# import primus_turbo.pytorch as turbo

# device = "cuda:0"
# dtype = torch.bfloat16
# M = 128
# N = 256
# K = 512

# # a [M, K]
# a = torch.randn((M, K), dtype=dtype, device=device)
# # b [K, N]
# b = torch.randn((N, K), dtype=dtype, device=device)
# # c [M, N]
# c = turbo.ops.gemm(a, b, trans_a=False, trans_b=True, out_dtype=dtype)

# print(c)
# print(c.shape)

# ==========================================================================================

import torch
import primus_turbo.pytorch as turbo
import sys
import os

# 1. 获取当前脚本 (test.py) 的绝对路径目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 拼接出 'x' 文件夹的绝对路径
x_dir = os.path.join(current_dir, 'transformers-qwen3-moe-fused')

# 3. 将 'x' 文件夹加入到系统搜索路径的前排
sys.path.insert(0, x_dir)

# --- 现在你可以直接 import x 里面的包了 ---
# 假设 x 下面直接就是 qwen3_moe_fused 文件夹
try:
    from qwen3_moe_fused.grouped_gemm.interface import grouped_gemm
    print("导入成功！")
except ImportError as e:
    print(f"导入失败: {e}")

print(f"PyTorch version: {torch.__version__}")
print(f"Is CUDA (ROCm) available? {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")

dtype = torch.float16
device = "cuda:0"
G = 4
M = 128  # 128=32+16+48+32
N = 256
K = 512

print("\nCreating tensors on device...")
group_lens = torch.tensor([32, 16, 48, 32], dtype=torch.long, device=device)
a = torch.randn(M, K, device=device, dtype=dtype)
b = torch.randn(G, N, K, device=device, dtype=dtype)
print("Tensors created successfully.")

print("Performing GEMM operation with primus_turbo...")
c = turbo.ops.grouped_gemm(a, b, group_lens, trans_b=True)
c_x = grouped_gemm(a, b, group_lens)
print("GEMM operation completed.")

print("-" * 30)
print("Starting Verification against torch.matmul...")

print("-" * 30)
print("Starting Verification (Robust Mode)...")

# 1. 计算 Reference (还是用 torch.matmul 循环)
splits = group_lens.tolist()
a_chunks = torch.split(a, splits, dim=0)
expected_chunks = []
for i in range(len(splits)):
    sub_a = a_chunks[i]
    sub_b = b[i] # trans_b=False
    expected_chunks.append(torch.matmul(sub_a, sub_b.t()))

expected_c = torch.cat(expected_chunks, dim=0)

# 2. 使用 torch.testing.assert_close 进行验证
# 这会自动根据 dtype (bfloat16) 选择合适的容忍度
try:
    # rtol=1.6e-2, atol=1e-3 是 bfloat16 的常见宽松标准
    torch.testing.assert_close(c, expected_c, rtol=1e-2, atol=1e-2)
    print("\n✅ Verification PASSED for 1e-2 using torch.testing.assert_close")
except AssertionError as e:
    print("\n❌ Verification FAILED with assert_close for 1e-2")
    print(e)

try:
    # rtol=1.6e-2, atol=1e-3 是 bfloat16 的常见宽松标准
    torch.testing.assert_close(c, expected_c, rtol=1e-3, atol=1e-3)
    print("\n✅ Verification PASSED for 1e-3 using torch.testing.assert_close")
except AssertionError as e:
    print("\n❌ Verification FAILED with assert_close for 1e-3")
    print(e)

diff = (c - expected_c).abs()
print(f"\nStats:")
print(f"Max Absolute Diff: {diff.max().item():.6f}")
print(f"Mean Absolute Diff: {diff.mean().item():.6f}")

try:
    # test for c_x
    torch.testing.assert_close(c_x, expected_c, rtol=1e-2, atol=1e-2)
    print("\n✅ Verification PASSED for c_x 1e-2 using torch.testing.assert_close")
except AssertionError as e:
    print("\n❌ Verification FAILED for c_x with assert_close for 1e-2")
    print(e)

try:
    # test for c_x
    torch.testing.assert_close(c_x, expected_c, rtol=1e-3, atol=1e-3)
    print("\n✅ Verification PASSED for c_x 1e-3 using torch.testing.assert_close")
except AssertionError as e:
    print("\n❌ Verification FAILED for c_x with assert_close for 1e-3")
    print(e)

diff_x = (c_x - expected_c).abs()
print(f"\nStats for x:")
print(f"Max Absolute Diff: {diff_x.max().item():.6f}")
print(f"Mean Absolute Diff: {diff_x.mean().item():.6f}")

# ==========================================================================================
# import torch
# import primus_turbo.pytorch as turbo
# import sys
# import os

# # --- 导入 setup (保持你原本的路径逻辑) ---
# current_dir = os.path.dirname(os.path.abspath(__file__))
# x_dir = os.path.join(current_dir, 'transformers-qwen3-moe-fused')
# sys.path.insert(0, x_dir)

# try:
#     from qwen3_moe_fused.grouped_gemm.interface import grouped_gemm
# except ImportError:
#     pass # 假设已导入

# device = "cuda:0"
# dtype = torch.bfloat16

# # 关键设置：K 非常大，且输入全为 1
# # 如果累加器是 BF16，加到 512 左右就会加不上去了
# G = 1
# M = 128
# N = 128
# K = 8192  # 远超 BF16 的精度范围 (256/512)

# print(f"\n🧪 Running Accumulation Precision Test (K={K})")
# print("-" * 40)

# # 1. 创建全 1 的张量
# # 注意：group_lens 设为 M，即不分组，或者看作只有一组，方便观测
# group_lens = torch.tensor([M], dtype=torch.long, device=device)
# a = torch.ones(M, K, device=device, dtype=dtype)       # 全是 1
# b = torch.ones(G, N, K, device=device, dtype=dtype)    # 全是 1 (trans_b=True layout)

# # 2. 理论上的正确结果 (FP32/Int)
# # 1.0 * 1.0 * K = K
# expected_value = float(K)
# print(f"理论期望值 (FP32 Accumulation): {expected_value:.1f}")

# # 3. 运行 Primus Turbo
# c_primus = turbo.ops.grouped_gemm(a, b, group_lens, trans_b=True)
# val_primus = c_primus.float().mean().item()

# # 4. 运行 Qwen Fused
# c_qwen = grouped_gemm(a, b, group_lens)
# val_qwen = c_qwen.float().mean().item()

# # 5. 运行 PyTorch 原生 (作为基准)
# # 手动 reshape 以匹配 grouped 逻辑
# c_torch = torch.matmul(a, b[0].t())
# val_torch = c_torch.float().mean().item()

# print("-" * 40)
# print(f"PyTorch Matmul 结果: {val_torch:.1f}")
# print(f"Qwen Fused 结果:     {val_qwen:.1f}")
# print(f"Primus Turbo 结果:   {val_primus:.1f}")
# print("-" * 40)

# # --- 自动判定 ---
# def judge(name, val, expected):
#     # 如果误差在 10 以内 (对于 8192 来说很小)，认为是 FP32 累加
#     if abs(val - expected) < 10:
#         return "FP32 Accumulation (High Precision)"
#     # 如果结果严重偏小 (比如只有 512 或 1000 多)，说明中间被截断了
#     elif val < expected * 0.9:
#         return "BF16 Accumulation (Low Precision / Saturation)"
#     else:
#         return "Unknown / Mixed Precision"

# print(f"结论推断:")
# print(f"PyTorch: {judge('PyTorch', val_torch, expected_value)}")
# print(f"Qwen:    {judge('Qwen', val_qwen, expected_value)}")
# print(f"Primus:  {judge('Primus', val_primus, expected_value)}")