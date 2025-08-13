###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import random

import numpy as np
import torch
from data.build_dataset import get_dataloaders
from engine.trainer import Trainer
from models.build_model import build_model
from transformers import LlamaConfig


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# TODO:
class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def to_dict(self):
        return self.__dict__


def get_hf_torch_dtype(hf_config):
    if isinstance(hf_config.torch_dtype, str):
        try:
            torch_dtype = getattr(torch, hf_config.torch_dtype)
        except AttributeError:
            raise ValueError(f"Invalid torch_dtype string: {hf_config.torch_dtype}")
    else:
        torch_dtype = hf_config.torch_dtype
    return torch_dtype


if __name__ == "__main__":
    set_seed(42)

    model_path = "meta-llama/Llama-3.1-8B-Instruct"
    hf_config = LlamaConfig.from_pretrained(model_path)
    # === Cfg ===
    config = Config(
        model_path=model_path,
        hf_config=hf_config,
        torch_dtype=get_hf_torch_dtype(hf_config),
        dataset_name="c4",  # c4 , wikitext
        batch_size=4,
        grad_accum_nums=4,
        context_length=8192,
        max_steps=500,
        warmup_steps=100,
        lr=1e-5,
        backend="torch",  # "torch", "turbo"
    )

    # === Model ===
    model = build_model(config).cuda()

    # === Data ===
    train_loader, val_loader = get_dataloaders(config, max_train_samples=200000, max_val_samples=5000)

    # === Trainer ===
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    trainer.train()
