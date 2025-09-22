import pytest
import torch

from primus_turbo.pytorch.core import TurboStream


@pytest.mark.parametrize("device", [1, "cuda", "cuda:2", torch.device("cuda:0")])
@pytest.mark.parametrize("cu_masks", [None, [0xFFFFFFFF], [0xFFFFFFFF, 0xFFFFFFFF]])
def test_turbo_stream(device, cu_masks):
    turbo_stream = TurboStream(device=device, cu_masks=cu_masks)

    x = torch.ones(10, device=device)
    y = torch.ones(10, device=device)
    out = torch.zeros_like(x)

    with torch.cuda.stream(turbo_stream.torch_stream):
        out = x + y

    turbo_stream.torch_stream.synchronize()
    assert torch.all(out == 2)
