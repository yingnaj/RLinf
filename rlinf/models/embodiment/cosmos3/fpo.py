# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Typed, variable-history replay batches for causal FPO actor updates."""

from dataclasses import dataclass, fields, is_dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class CausalFPOBatch:
    """One actor's valid chunks, before shuffling or any optimizer update.

    Examples are CPU-owned Cosmos CausalFPOExample records. Advantages are
    detached, already grouped across ranks. Padding must be removed by the
    collector before equal-sized actor shards are constructed.
    """

    examples: tuple[Any, ...]
    advantages: torch.Tensor
    behavior_version: str

    def validate(self, expected_version: str | None = None) -> None:
        _validate_cpu_record(self)
        if not self.examples or self.advantages.shape != (len(self.examples),):
            raise ValueError("Expected nonempty examples and advantages [chunks]")
        if self.advantages.device.type != "cpu" or self.advantages.requires_grad:
            raise ValueError("Replay advantages must be detached CPU tensors")
        if not torch.isfinite(self.advantages).all():
            raise ValueError("Replay advantages must be finite")
        if expected_version is not None and self.behavior_version != expected_version:
            raise ValueError("Replay behavior version differs from the actor")
        probe_counts = set()
        for example in self.examples:
            example.rollout.validate()
            example.probes.validate(example.rollout)
            if example.rollout.policy_version != self.behavior_version:
                raise ValueError("Replay batch contains mixed behavior versions")
            q = len(example.probes.sigmas)
            probe_counts.add(q)
            if (
                example.old_cfm.shape != (q,)
                or not torch.isfinite(example.old_cfm).all()
            ):
                raise ValueError("Expected finite old CFM [probes]")
            if (example.old_cfm < 0).any():
                raise ValueError("Old CFM must be nonnegative")
        if len(probe_counts) != 1:
            raise ValueError("Replay batch contains different probe counts")


def recompute_old_cfm(policy, examples, *, micro_batch_size: int):
    """Freeze all old scores using the same grad-enabled eval forward as updates."""
    if micro_batch_size < 1:
        raise ValueError("micro_batch_size must be positive")
    policy.eval()
    old = []
    with torch.enable_grad():
        for start in range(0, len(examples), micro_batch_size):
            output = policy(fpo_examples=examples[start : start + micro_batch_size])
            old.append(output["cfm"].detach().float().cpu().clone())
            del output
    return torch.cat(old)


def fpo_update_loss(policy, examples, old_cfm, advantages, *, config):
    """Run the wrapped policy forward; preserve the native chunk/probe reduction."""
    from cosmos_framework.model.generator.algorithm.rl.fpo import fpo_loss

    current = policy(fpo_examples=examples)["cfm"]
    return fpo_loss(
        current,
        old_cfm.to(current.device),
        advantages.to(current.device),
        config=config,
    )


def _validate_cpu_record(value):
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.requires_grad:
            raise ValueError("Every replay tensor must be detached and on CPU")
    elif is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            _validate_cpu_record(getattr(value, item.name))
    elif isinstance(value, dict):
        for item in value.values():
            _validate_cpu_record(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _validate_cpu_record(item)


def grouped_suffix_advantages(
    rewards, discounts, valid, *, group_ids, member_ids, group_size
):
    """Compute [trajectories, chunks] advantages after gathering group metadata.

    Group IDs must identify a common task, reset and goal. This function verifies
    complete membership; the collector owns verification of those conditions.
    """
    from cosmos_framework.model.generator.algorithm.rl.fpo import (
        discounted_chunk_returns,
        group_relative_advantages,
    )

    if (
        rewards.ndim != 2
        or group_ids.shape != rewards.shape[:1]
        or member_ids.shape != group_ids.shape
    ):
        raise ValueError(
            "Expected rewards [trajectories, chunks] and one group/member ID per trajectory"
        )
    if (
        group_size < 2
        or group_ids.dtype != torch.int64
        or member_ids.dtype != torch.int64
    ):
        raise ValueError("Expected group_size >= 2 and int64 group/member IDs")
    returns = discounted_chunk_returns(rewards, discounts, valid=valid)
    advantages = torch.zeros_like(returns)
    for group in group_ids.unique():
        rows = torch.where(group_ids == group)[0].to(rewards.device)
        members = member_ids[group_ids == group].sort().values
        expected = torch.arange(group_size, device=members.device)
        if not torch.equal(members, expected):
            raise ValueError("Each group must contain every member exactly once")
        advantages[rows] = group_relative_advantages(
            returns[rows][None], valid=valid[rows][None]
        )[0]
    return advantages
