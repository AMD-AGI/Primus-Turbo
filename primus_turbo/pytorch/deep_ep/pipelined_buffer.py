###############################################################################

# Copyright (c) 2025 DeepSeek. All rights reserved.

# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.

#

# See LICENSE for license information.

###############################################################################


from typing import Optional, Tuple, Union

import torch


class PipelinedBuffer:
    """
    The communication buffer for pipelined EP, which aim to overlap the ep communication and grouped-gemm computation.
    """

    def __init__(
        self,
        group_name: str,
    ) -> None:
        """
        Initialize the communication buffer.

        """

    def dispatch(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        handle: Optional[Tuple] = None,
        topk_idx: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
    ):

        if handle is None:
            # call metadata processs kernel for dispatch
            pass
        else:
            # use cached metadata to dispatch
            pass
