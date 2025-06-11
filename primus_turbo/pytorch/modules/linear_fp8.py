# TODO:
"""
class BlockwiseFloat8Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        block_size: int = 128,
        device=None,
        dtype: torch.dtype = torch.float8_e4m3fn,
    ):
        super().__init__()
        # TODO: HIP float8_e4m3fn
        supported_dtypes = [
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ]
        assert (
            dtype in supported_dtypes
        ), f"Unsupported dtype: {dtype}. Supported dtypes: {supported_dtypes}"

        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        factory_kwargs = {"device": device, "dtype": dtype}

        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        self.weight_scales = nn.Parameter(
            torch.empty(
                out_features // block_size,
                in_features // block_size,
                dtype=torch.float32,
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        # TODO: reset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

"""
