## DeepEP (experimental)

DeepEP of Primus-Turbo is in the experimental stage.

The kernel code of DeepEP is primarily derived from ROCm internal DeepEP (it's still under development). We've added some support for Sync-Free MoE, including cuda num_tokens_per_experts (not python list) and support num_worst_token for internode to use with Turbo's GruopedGEMM to eliminate host synchronization between DeepEP dispatch and groupedgemm.

### Benchmark Usage

benchmark usage see [DeepEP benchmark](../../../benchmark/README.md)
