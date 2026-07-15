# primus_turbo/tuning

Primus-Turbo 的离线 autotune 工具:离线为每个 shape 挑好最优 backend 并落盘,运行时只查表。

产物存 `configs/<framework>/<arch>/<op>.json`(如 `configs/pytorch/gfx950/gemm_fp8.json`),首次 dispatch 时按 arch 自动加载(懒加载、零 env;`AUTO_TUNE=1` 时忽略资产),miss 落默认后端。

## GEMM Offline Tune

shape json 格式(m = token 数,n/k = 权重维):

```json
{"mnk": [[16, 4096, 4096], [64, 4096, 4096], [256, 4096, 4096]]}
```

python 生成:

```python
import json

mnk = [[16, 4096, 4096], [64, 4096, 4096], [256, 4096, 4096]]
json.dump({"mnk": mnk}, open("my.json", "w"))
```

运行:

```bash
python -m primus_turbo.tuning.offline_tune_gemm                 # 内置小网格
python -m primus_turbo.tuning.offline_tune_gemm --shapes my.json
```

产物固定写到 `configs/pytorch/<arch>/`(运行时自动加载的规范路径,不可指定)。每个 shape 内部枚举所有变体(fp8: dtype × format × granularity,fwd + bwd),按 dispatcher 产出 `gemm_fp8.json` 等。
