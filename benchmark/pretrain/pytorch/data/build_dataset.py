###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from itertools import islice

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer


class CausalLMDataset(Dataset):
    def __init__(self, token_ids: list[int], context_length: int):
        self.context_length = context_length
        total_len = (len(token_ids) // (context_length + 1)) * (context_length + 1)
        self.token_ids = torch.tensor(token_ids[:total_len], dtype=torch.long)

    def __len__(self):
        return len(self.token_ids) // (self.context_length + 1)

    def __getitem__(self, idx):
        i = idx * (self.context_length + 1)
        chunk = self.token_ids[i : i + self.context_length + 1]  # shape: [context+1]
        input_ids = chunk[:-1]  # shape: [context]
        labels = chunk[1:]  # shape: [context]
        return input_ids, labels

    def __repr__(self):
        return f"CausalLMDataset(total_tokens={len(self.token_ids)}, num_samples={len(self)})"


def load_and_tokenize(tokenizer, dataset_name: str, split: str, max_samples: int):
    if dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        text_key = "text"
        texts = [sample[text_key] for sample in dataset if sample[text_key].strip()]
        texts = texts[:max_samples]

    elif dataset_name == "c4":
        dataset = load_dataset("allenai/c4", "en", split=split, streaming=True)
        text_key = "text"

        def filtered_texts():
            for sample in dataset:
                if sample[text_key].strip():
                    yield sample[text_key]

        texts = list(islice(filtered_texts(), max_samples))

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    # Tokenize
    token_ids = []
    for text in tqdm(texts, desc=f"Encoding ({split})", ncols=80):
        ids = tokenizer.encode(text, add_special_tokens=True)
        token_ids.extend(ids)

    return token_ids


def get_dataloaders(
    config,
    max_train_samples=500000,
    max_val_samples=1000,
):
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    train_ids = load_and_tokenize(
        tokenizer,
        config.dataset_name,
        "train",
        max_train_samples,
    )
    val_ids = load_and_tokenize(
        tokenizer,
        config.dataset_name,
        "validation",
        max_val_samples,
    )

    train_dataset = CausalLMDataset(train_ids, config.context_length)
    val_dataset = CausalLMDataset(val_ids, config.context_length)

    print(train_dataset)
    print(val_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader
