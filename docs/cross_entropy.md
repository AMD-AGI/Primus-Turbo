# Single-rank fused cross entropy

`primus_turbo.pytorch.ops.cross_entropy` computes per-token vocabulary cross
entropy for BF16 or FP32 logits on a GPU. It accepts `[tokens, vocab]` or any
3D layout whose leading dimensions match the int64 targets. The result is FP32.

```python
from primus_turbo.pytorch.ops import cross_entropy

# logits: [sequence, batch, vocabulary]; labels: [batch, sequence]
loss = cross_entropy(logits, labels.T.contiguous())
(loss * token_weights.T).sum().backward()
```

This is a TP=1 operator. The caller owns masking, loss reduction, distributed
normalization and any loss scaling. `ignore_index=-100` zeros the loss and
gradient for ignored tokens. Label smoothing uses the full vocabulary.
Target values must be valid vocabulary indices or the ignore index.

Forward computes the loss and two FP32 softmax statistics per token in one
main kernel. Backward reconstructs probabilities, subtracts the target,
applies the incoming gradient in FP32, and writes gradients in the input dtype.
No full-size probability/gradient tensor is materialized during forward. Full
vocabulary logits are still materialized by the model and retained until backward.

By default, forward copies logits into private saved storage in the same kernel,
preserving the original tensor. `overwrite_input=True` avoids this copy and
requires contiguous, exclusively owned logits: backward overwrites that storage
with gradients. Do not reuse it in another autograd branch or read it after
backward begins. Only one backward per forward is supported in either mode;
higher-order differentiation is unsupported. Allocation and target/gradient
layout conversions can add kernels beyond the two main CE kernels.

The kernels are ported from NVIDIA TransformerEngine commit
`76ae4b0981849d4b85a528d26f39981974b409f8`,
`transformer_engine/common/triton/cross_entropy.py`, with its Apache license and
copyright retained. This operator does not import Transformer Engine or require
its native extensions. Triton compiles the GPU kernels on first use.

Run GPU correctness tests with:

```bash
pytest tests/pytorch/ops/test_cross_entropy.py
```

Performance and convergence depend on the surrounding workload. In particular,
the FP32 gradient reconstruction differs from older TE implementations that
round intermediate probabilities to BF16; compare training convergence before
enabling it in an established recipe.
