###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import time
from contextlib import nullcontext

import torch
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from transformers import get_scheduler


def get_time_str():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def cross_entropy_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(pred.flatten(0, 1).float(), labels.flatten(0, 1))


class InfiniteIterator:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def next(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            return next(self.iterator)


class Trainer:
    def __init__(self, config, model, train_loader, val_loader):
        self.config = config
        self.model = model.cuda().train()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.writer = SummaryWriter()

        self.max_steps = config.max_steps + 1
        self.grad_accum_nums = config.grad_accum_nums

        self.loss_fn = cross_entropy_loss
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)

        self.scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.max_steps,
        )

        self.use_amp = hasattr(config, "torch_dtype") and config.torch_dtype in (
            torch.float16,
            torch.bfloat16,
        )

        self.train_iter = InfiniteIterator(train_loader)

    def train_step(self):
        self.optimizer.zero_grad()
        total_loss = 0.0

        for _ in range(self.grad_accum_nums):
            input_ids, labels = self.train_iter.next()
            input_ids = input_ids.cuda()
            labels = labels.cuda()

            with autocast("cuda", dtype=self.config.torch_dtype) if self.use_amp else nullcontext():
                pred = self.model(input_ids)
                loss = self.loss_fn(pred, labels)

            loss.backward()
            total_loss += loss.detach().item()

        self.optimizer.step()
        self.scheduler.step()

        return total_loss

    def train(self):
        print_interval = 10
        eval_interval = 100

        for step in range(1, self.max_steps + 1):
            loss = self.train_step()

            if step % print_interval == 0:
                print(f"[{get_time_str()}] [Step {step:06d}] : loss={loss:.4f}")
                self.writer.add_scalar("train/loss", loss, step)

            if self.val_loader and step % eval_interval == 0:
                val_loss = self.evaluate()
                # print(f"  → Eval loss: {val_loss:.4f}")
                self.writer.add_scalar("eval/loss", val_loss, step)

    def evaluate(self):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for input_ids, labels in self.val_loader:
                input_ids = input_ids.cuda()
                labels = labels.cuda()

                with autocast("cuda", dtype=self.config.torch_dtype) if self.use_amp else nullcontext():
                    pred = self.model(input_ids)
                    loss = self.loss_fn(pred, labels)
                    losses.append(loss.item())
        self.model.train()
        return sum(losses) / len(losses)
