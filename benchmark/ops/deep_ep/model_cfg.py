from dataclasses import dataclass


@dataclass
class DeepEPModelCfg:
    model_name: str
    hidden_size: int
    num_experts: int
    num_topk: int
    seqlen: int
    batch_size: int

    def __post_init__(self):
        self.num_tokens = self.batch_size * self.seqlen


g_model_cfg = [
    DeepEPModelCfg(
        model_name="deepseekv3", hidden_size=7168, num_experts=256, num_topk=8, seqlen=4096, batch_size=1
    ),
    DeepEPModelCfg(
        model_name="deepseekv2", hidden_size=5120, num_experts=160, num_topk=6, seqlen=4096, batch_size=1
    ),
    DeepEPModelCfg(
        model_name="qwen3_235b", hidden_size=4096, num_experts=128, num_topk=6, seqlen=4096, batch_size=1
    ),
    DeepEPModelCfg(
        model_name="deepseek_proxy-515B",
        hidden_size=8192,
        num_experts=112,
        num_topk=8,
        seqlen=4096,
        batch_size=1,
    ),
    DeepEPModelCfg(
        model_name="deepseek_proxy-1T",
        hidden_size=8192,
        num_experts=224,
        num_topk=8,
        seqlen=4096,
        batch_size=1,
    ),
    DeepEPModelCfg(
        model_name="deepseek_proxy-2T",
        hidden_size=8192,
        num_experts=448,
        num_topk=16,
        seqlen=4096,
        batch_size=1,
    ),
]


def get_model_cfg() -> DeepEPModelCfg:
    global g_model_cfg
    return g_model_cfg
