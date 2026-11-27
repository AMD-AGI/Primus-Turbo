###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Delayed tensorwise FP8 scaling: per-GEMM scale and amax state, and the recipe that updates it.

A dynamic tensorwise cast reads its input twice, once for the amax and once to cast. A delayed
cast reads it once: it multiplies by a scale kept from earlier steps and, in the same pass, raises
an amax buffer to the input's abs-max (``quantize_fp8(..., scale=, amax_out=)``). Once per
optimizer step, ``DelayedScaling.step()`` turns the recorded amaxes into the next scales.

Usage::

    recipe = DelayedScaling()
    config = Float8QuantConfig(strategy=ScalingStrategy.DELAYED)
    linear = Float8Linear(1024, 1024, config=config, delayed_scaling=recipe)
    for batch in loader:
        loss = model(batch).sum()
        loss.backward()
        optimizer.step()
        recipe.step()

A framework with its own linear class makes the state itself::

    state = recipe.make_state(config, device)
    out = gemm_fp8(x, w, trans_b=True, config=config, scaling_state=state)

Semantics:
  - Each state holds a scale and an amax for the input, the weight and the output gradient.
  - The amax is a running max over every cast since the last ``step()``: gradient accumulation
    and activation-checkpoint recomputation fold into it, and the step uses their maximum.
    Scales only change in ``step()``, so a recomputed forward casts exactly as the first one did.
  - ``step()`` sets ``scale = fp8_max / max(amax, 1e-12)``, the dynamic cast's formula, so a
    delayed cast with a fresh amax is bit-identical to a dynamic one. An amax that is zero (the
    GEMM did not run) or not finite keeps the previous scale. It then resets the amaxes to +0.
  - Until a state's first ``step()``, its GEMM casts dynamically and records the amax, so the
    first scales come from real data. Under ``torch.compile`` this is one recompile per compiled
    region after the first step.
  - With ``refresh_interval=n`` the amaxes are recorded only in optimizer steps 0, n, 2n, ...
    (step 0 is the warm-up before the first ``step()``), and only the ``step()`` that ends such a
    step changes the scales. The other steps cast with the stored scales and skip the amax
    update in the cast. Under ``torch.compile`` the two kinds of step are two graphs; with
    the default ``n=1`` every step records and there is no second graph.
  - Each rank keeps its own amaxes, as dynamic scaling does, unless ``amax_reduction_group`` is
    given: then ``step()`` all-reduces them (max) over that group first, so the group shares scales.
  - From the first ``step()`` the recipe keeps every state's scales and amaxes in one flat buffer
    each (per device), and the states' tensors are views into them, so a step is a handful of
    kernels however many GEMMs there are. A state made, copied or moved later is packed in at
    the next step.
  - State tensors are plain attributes, not module buffers: ``state_dict`` does not save them (a
    resumed run warms up again) and DDP's buffer broadcast does not overwrite them.
"""

import copy
import weakref
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingStrategy,
    float8_e4m3,
    float8_e5m2,
)

__all__ = ["DelayedScaling", "DelayedScalingState"]

# Matches AMAX_SCALE_EPS of the dynamic tensorwise cast, so the two give the same scale.
_AMAX_EPS = 1e-12
# float32 values per 16 bytes: one row of the recipe's flat buffers.
_SLOT = 4


def _stage_fp8_max(fmt: Format) -> Tuple[float, float, float]:
    """FP8 max of the input, weight and output-gradient casts for ``fmt``."""
    fwd = float8_e5m2 if fmt == Format.E5M2 else float8_e4m3
    bwd = float8_e4m3 if fmt == Format.E4M3 else float8_e5m2
    return (torch.finfo(fwd).max, torch.finfo(fwd).max, torch.finfo(bwd).max)


class _Schedule:
    """Whether the current optimizer step records amaxes; one per recipe, shared by its states.

    A plain bool attribute, so a compiled region specialises on it instead of branching on device.
    """

    def __init__(self):
        self.record = True


class DelayedScalingState:
    """Delayed-scaling state of one tensorwise FP8 GEMM, created by ``DelayedScaling.make_state``.

    ``scale`` and ``amax`` are ``(input, weight, output gradient)`` triples of 0-d float32
    tensors. ``ready`` turns true at the state's first ``DelayedScaling.step()``; before that,
    ``gemm_fp8`` casts dynamically and only records the amaxes. From then on the tensors are views
    into the recipe's flat buffers. A ``copy.deepcopy`` of a state (or of a model holding one) is
    registered with the same recipe.
    """

    def __init__(
        self,
        config: Float8QuantConfig,
        device: Optional[torch.device] = None,
        recipe: Optional["DelayedScaling"] = None,
    ):
        self.fp8_max = _stage_fp8_max(config.format)
        self.ready = False
        self._recipe = weakref.ref(recipe) if recipe is not None else None
        self._schedule = recipe._schedule if recipe is not None else _Schedule()
        self._reset(torch.device(device) if device is not None else torch.device("cuda"))

    @property
    def recording(self) -> bool:
        """Whether this step's casts record amaxes: always before the first step, then as the
        recipe's ``refresh_interval`` says."""
        return not self.ready or self._schedule.record

    def _reset(self, device: torch.device) -> None:
        self.scale = tuple(torch.ones((), dtype=torch.float32, device=device) for _ in range(3))
        self.amax = tuple(torch.zeros((), dtype=torch.float32, device=device) for _ in range(3))

    @property
    def device(self) -> torch.device:
        return self.scale[0].device

    def to(self, device: torch.device) -> "DelayedScalingState":
        """Move to ``device`` in place; from the meta device, start over as a new state."""
        device = torch.device(device)
        if device != self.device:
            if self.device.type == "meta":
                self._reset(device)
                self.ready = False
            else:
                self.scale = tuple(t.to(device) for t in self.scale)
                self.amax = tuple(t.to(device) for t in self.amax)
            recipe = self._recipe() if self._recipe is not None else None
            if recipe is not None:
                recipe._repack = True
        return self

    def __deepcopy__(self, memo) -> "DelayedScalingState":
        new = copy.copy(self)
        new.scale = tuple(t.clone() for t in self.scale)
        new.amax = tuple(t.clone() for t in self.amax)
        recipe = self._recipe() if self._recipe is not None else None
        if recipe is not None:
            recipe._register(new)
        memo[id(self)] = new
        return new

    def __repr__(self) -> str:
        return f"DelayedScalingState(device={self.device}, ready={self.ready})"


class DelayedScaling:
    """Delayed tensorwise FP8 scaling recipe: makes per-GEMM states and updates them together.

    Args:
        amax_reduction_group: if given, ``step()`` all-reduces (max) the amaxes over this process
            group before computing scales, so its ranks cast with the same scales. Every rank of
            the group must make its states in the same order. Default None: each rank keeps its
            own, as dynamic scaling does.
        refresh_interval: record amaxes only in optimizer steps 0, n, 2n, ... and keep the
            scales in between. Default 1: every step records and updates the scales.
    """

    def __init__(self, amax_reduction_group: Optional[dist.ProcessGroup] = None, refresh_interval: int = 1):
        if (
            isinstance(refresh_interval, bool)
            or not isinstance(refresh_interval, int)
            or refresh_interval < 1
        ):
            raise ValueError(f"refresh_interval must be an int >= 1, got {refresh_interval!r}")
        self.amax_reduction_group = amax_reduction_group
        self.refresh_interval = refresh_interval
        self._schedule = _Schedule()
        self._num_steps = 0
        self._states: List[weakref.ref] = []
        # (scale, amax, fp8_max) flat buffers per device; the states' tensors are views into them.
        self._buffers: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._repack = False

    def make_state(
        self, config: Float8QuantConfig, device: Optional[torch.device] = None
    ) -> DelayedScalingState:
        """A new state for one GEMM with ``config``, updated by this recipe's ``step()``."""
        if config.strategy != ScalingStrategy.DELAYED:
            raise ValueError(f"make_state needs a ScalingStrategy.DELAYED config, got {config.strategy}")
        state = DelayedScalingState(config, device, recipe=self)
        self._register(state)
        return state

    def _register(self, state: DelayedScalingState) -> None:
        self._states.append(weakref.ref(state))
        self._repack = True

    def states(self) -> List[DelayedScalingState]:
        """The live states, in the order they were made."""
        live = [s for s in (r() for r in self._states) if s is not None]
        if len(live) != len(self._states):
            self._states = [weakref.ref(s) for s in live]
        return live

    def _pack(self) -> None:
        """Copy every state into flat per-device buffers and make its tensors views of them.

        One update over a flat buffer costs a handful of launches, where one per state's tensor
        would cost thousands for a large model. Runs at the first step and after a state is made,
        copied or moved; the views keep the states' values.

        Each value takes a 16-byte row, so every view starts 16-byte aligned. Inductor assumes
        the outputs of an op that takes an unaligned input are unaligned too, which would carry
        from the casts through the GEMM outputs and slow down every kernel that reads them.
        """
        by_device = {}
        for s in self.states():
            by_device.setdefault(s.device, []).append(s)
        self._buffers = []
        for device in sorted(by_device, key=str):
            group = by_device[device]
            rows = 3 * len(group)
            scale = torch.zeros(rows, _SLOT, dtype=torch.float32, device=device)
            amax = torch.zeros(rows, _SLOT, dtype=torch.float32, device=device)
            scale[:, 0] = torch.stack([t for s in group for t in s.scale])
            amax[:, 0] = torch.stack([t for s in group for t in s.amax])
            fp8_max = torch.tensor(
                [m for s in group for m in s.fp8_max], dtype=torch.float32, device=device
            ).unsqueeze(1)
            for i, s in enumerate(group):
                s.scale = tuple(scale[3 * i + j, 0] for j in range(3))
                s.amax = tuple(amax[3 * i + j, 0] for j in range(3))
                s.ready = True
            self._buffers.append((scale, amax, fp8_max))
        self._repack = False

    @torch.no_grad()
    def step(self) -> None:
        """Turn the amaxes recorded since the last call into scales, then reset them to +0.

        Call once per optimizer step, after the last backward that should count toward it and
        before the next forward. Launches a few small kernels and no host synchronisation; after
        a step that did not record (``refresh_interval``), none unless a state was added.
        """
        # States made, copied or moved since the last step recorded their amaxes while not ready.
        update = self._schedule.record or self._repack
        if self._repack:
            self._pack()
        if update:
            for scale, amax, fp8_max in self._buffers:
                if self.amax_reduction_group is not None:
                    dist.all_reduce(amax, op=dist.ReduceOp.MAX, group=self.amax_reduction_group)
                new_scale = fp8_max / torch.clamp_min(amax, _AMAX_EPS)
                scale.copy_(torch.where(torch.isfinite(amax) & (amax > 0), new_scale, scale))
                amax.zero_()
        self._num_steps += 1
        self._schedule.record = self._num_steps % self.refresh_interval == 0
