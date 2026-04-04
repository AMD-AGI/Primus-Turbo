# Round 5: Forward NT Pipeline + Cache — 无改进

## 测试
- `.cs` cache modifier: Triton gfx950 编译失败
- `num_stages=3` (triple buffer): BM=256 时 LDS overflow (192KB > 160KB)
- 唯一可行组合: `.ca` + `ns=2` (当前默认)

## 结论
无可行的替代方案。保持现状。
