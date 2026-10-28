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

"""Episode identity, executed feedback, and complete-group FPO collection."""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from rlinf.models.embodiment.cosmos3.fpo import (
    CausalFPOBatch,
    _validate_cpu_record,
    grouped_suffix_advantages,
)


@dataclass(frozen=True)
class EpisodeKey:
    version: int
    group: int
    member: int
    task: str
    reset_seed: int
    instruction: str


@dataclass(frozen=True)
class ActionRequest:
    episode: EpisodeKey
    chunk: int
    observation_step: int
    actions: np.ndarray

    def validate(self, block_size, action_dim):
        if self.chunk < 0 or self.observation_step != self.chunk * block_size:
            raise ValueError("Action request must address the next complete block")
        if not isinstance(self.actions, np.ndarray) or self.actions.shape != (
            block_size,
            action_dim,
        ):
            raise ValueError("Expected complete physical action block [T,D]")
        if self.actions.dtype != np.float32 or not np.isfinite(self.actions).all():
            raise ValueError("Physical actions must be finite float32 CPU arrays")


@dataclass(frozen=True)
class ExecutionFeedback:
    episode: EpisodeKey
    chunk: int
    observation_step: int
    frames: tuple[dict, ...]
    state: np.ndarray
    rewards: tuple[float, ...]
    terminated: bool
    truncated: bool
    success: bool
    timestamps: tuple[float, ...]

    @property
    def executed_steps(self):
        return len(self.frames)

    def validate(self, requested_steps, fps):
        n = self.executed_steps
        if (
            not 1 <= n <= requested_steps
            or len(self.rewards) != n
            or len(self.timestamps) != n
        ):
            raise ValueError(
                "Feedback requires exactly one real frame/reward/timestamp per executed control"
            )
        if n < requested_steps and not (self.terminated or self.truncated):
            raise ValueError("Partial blocks require episode termination or truncation")
        if (
            self.state.ndim != 1
            or not np.isfinite(self.state).all()
            or not np.isfinite(self.rewards).all()
        ):
            raise ValueError("Measured state and rewards must be finite")
        expected = (np.arange(n) + self.observation_step + 1) / fps
        if not np.allclose(self.timestamps, expected, atol=1e-9, rtol=0):
            raise ValueError("Feedback timestamps must match actual control steps")
        if self.success and not self.terminated:
            raise ValueError("Success must terminate the first-version task contract")


@dataclass(frozen=True)
class CollectedEpisode:
    key: EpisodeKey
    examples: tuple[Any, ...]
    feedback: tuple[ExecutionFeedback, ...]


def collect_actor_shards(
    episodes,
    *,
    version,
    group_size,
    world_size,
    global_batch_size,
    gamma,
    block_size,
    fps,
):
    """Normalize complete groups before distributing distinct, valid chunks.

    Timeouts end finite-horizon episodes (zero bootstrap). A tail smaller than
    one global batch is discarded only after group returns are computed, and
    its size is returned as a metric. No padded or duplicated training records.
    """
    if not 0 <= gamma <= 1 or not math.isfinite(gamma):
        raise ValueError("gamma must be finite and in [0,1]")
    if world_size < 1 or global_batch_size < 1 or global_batch_size % world_size:
        raise ValueError("global batch must divide equally across actor ranks")
    keys = [e.key for e in episodes]
    if not keys or len({(k.group, k.member) for k in keys}) != len(keys):
        raise ValueError("Missing episodes or duplicate group members")
    for group in {k.group for k in keys}:
        members = [k for k in keys if k.group == group]
        if sorted(k.member for k in members) != list(range(group_size)):
            raise ValueError("Every reset group must have complete unique membership")
        if (
            len({(k.version, k.task, k.reset_seed, k.instruction) for k in members})
            != 1
        ):
            raise ValueError(
                "A group must share task, reset, instruction and behavior version"
            )
    count = max(len(e.examples) for e in episodes)
    rewards = torch.zeros(len(episodes), count)
    discounts, valid = (
        torch.zeros_like(rewards),
        torch.zeros_like(rewards, dtype=torch.bool),
    )
    for row, episode in enumerate(episodes):
        if (
            episode.key.version != version
            or not episode.examples
            or len(episode.examples) != len(episode.feedback)
        ):
            raise ValueError(
                "Expected nonempty matched episodes from the current behavior version"
            )
        _validate_cpu_record(episode)
        CausalFPOBatch(
            episode.examples, torch.zeros(len(episode.examples)), str(version)
        ).validate(str(version))
        for col, feedback in enumerate(episode.feedback):
            feedback.validate(block_size, fps)
            if (
                feedback.episode != episode.key
                or feedback.chunk != col
                or feedback.observation_step != col * block_size
            ):
                raise ValueError(
                    "Feedback identity or step order differs from its replay"
                )
            done = feedback.terminated or feedback.truncated
            if done != (col == len(episode.feedback) - 1):
                raise ValueError("Episodes must finish exactly at their last feedback")
            rewards[row, col] = sum(
                gamma**i * r for i, r in enumerate(feedback.rewards)
            )
            discounts[row, col] = 0 if done else gamma**feedback.executed_steps
            valid[row, col] = True
    advantages = grouped_suffix_advantages(
        rewards,
        discounts,
        valid,
        group_ids=torch.tensor([k.group for k in keys], dtype=torch.int64),
        member_ids=torch.tensor([k.member for k in keys], dtype=torch.int64),
        group_size=group_size,
    )
    examples = [x for e in episodes for x in e.examples]
    values = advantages[valid]
    used = len(examples) // global_batch_size * global_batch_size
    if not used:
        return [], {"valid_chunks": len(examples), "need_more_groups": True}
    shards = []
    for rank in range(world_size):
        shard = CausalFPOBatch(
            tuple(examples[rank:used:world_size]),
            values[:used][rank::world_size].clone(),
            str(version),
        )
        shard.validate(str(version))
        shards.append(shard)
    return shards, {
        "valid_chunks": len(examples),
        "trained_chunks": used,
        "discarded_tail_chunks": len(examples) - used,
        "success_rate": sum(e.feedback[-1].success for e in episodes) / len(episodes),
        "zero_advantages": bool(values[:used].eq(0).all()),
    }
