###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import importlib
from typing import Dict, Tuple, Type

import torch

MODEL_REGISTRY: Dict[Tuple[str, str], Type[torch.nn.Module]] = {}


def register_model(model_type: str, backend: str = "torch"):
    """
    example usage:
        @register_model("llama", "torch")
        class LlamaBasicModel(nn.Module): ...
    """

    def wrapper(cls):
        key = (model_type, backend)
        if key in MODEL_REGISTRY:
            raise ValueError(f"Duplicate registration for {key}")
        MODEL_REGISTRY[key] = cls
        return cls

    return wrapper


def build_model(config):
    importlib.import_module("models.basic_llama")
    importlib.import_module("models.turbo_llama")

    model_type = config.hf_config.model_type
    backend = config.backend

    key = (model_type, backend)
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Model {key} is not registered.")
    model_cls = MODEL_REGISTRY[key]
    return model_cls(config.hf_config)
