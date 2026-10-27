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

"""Batched RoboTwin physical states and normalized causal action endpoints."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch


class RobotwinCausalAdapter:
    """Convert at the environment boundary without modifying FPO endpoints.

    The environment must supply measured states. Native dictionaries can carry
    EEF poses; joint vectors cannot provide EEF channels or infer them via FK.
    State normalization belongs to the causal batch builder, using statistics
    identical to those supplied here for action decoding.
    """

    def __init__(
        self,
        native_adapter: Any,
        statistics: Any,
        *,
        control_mode: str,
        state_format: str,
    ) -> None:
        if control_mode not in native_adapter.available_control_modes:
            raise ValueError(
                f"Unsupported control_mode={control_mode!r}; "
                f"available: {native_adapter.available_control_modes}"
            )
        if state_format not in ("native", "joint_vector"):
            raise ValueError("state_format must be 'native' or 'joint_vector'")
        if state_format == "joint_vector" and native_adapter.state_mask[14:28].any():
            raise ValueError("joint_vector cannot supply active EEF state channels")
        statistics.validate(
            native_adapter.template.width,
            native_adapter.state_mask,
            native_adapter.action_mask,
        )
        self.native = native_adapter
        self.statistics = statistics
        self.control_mode = control_mode
        self.state_format = state_format

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "RobotwinCausalAdapter":
        """Load explicit source statistics and an optional robot profile."""
        from cosmos_framework.data.generator.action.action_state_template import (
            ActionStateTemplate55,
        )
        from cosmos_framework.data.generator.action.block_state import BlockStatistics
        from cosmos_framework.utils.robotwin import RobotwinAdapter

        for key in ("statistics_path", "robot_config_path", "dataset_root"):
            path = config.get(key)
            if path is not None and not Path(path).is_absolute():
                raise ValueError(f"robotwin.{key} must be an absolute path")
        if not config.get("statistics_path"):
            raise ValueError("robotwin.statistics_path is required")
        statistics = BlockStatistics.load(config["statistics_path"])
        native = RobotwinAdapter.from_statistics(
            ActionStateTemplate55(),
            statistics,
            config.get("robot_config_path"),
            dataset_root=config.get("dataset_root"),
        )
        return cls(
            native,
            statistics,
            control_mode=config["control_mode"],
            state_format=config["state_format"],
        )

    @property
    def action_type(self) -> str:
        """RoboTwin take_action controller name; EEF needs env-side support."""
        return {"joint": "qpos", "eef": "ee"}[self.control_mode]

    @property
    def action_dim(self) -> int:
        """Environment command width, independent of the model's 55D width."""
        return (
            sum(self.native.dofs.values()) + 2 if self.control_mode == "joint" else 16
        )

    def pack_states(
        self,
        states: Sequence[Mapping[str, Any]] | np.ndarray | torch.Tensor,
        *,
        state_source: str,
    ) -> torch.Tensor:
        """Return unnormalized measured anchors [B,55] on CPU.

        joint_vector uses [left joints, left gripper, right joints, right gripper].
        The caller must verify the environment's state source before declaring it
        measured; command targets are rejected even when the shapes match.
        """
        if state_source != "measured":
            raise ValueError("Causal RL requires measured state, not command targets")
        if self.state_format == "joint_vector":
            values = torch.as_tensor(states).detach().float().cpu().numpy()
            left, right = self.native.dofs.values()
            if values.ndim != 2 or values.shape[1] != left + right + 2:
                raise ValueError(f"Expected joint states [B,{left + right + 2}]")
            states = [
                {
                    "joint_action": {
                        "left_arm": row[:left],
                        "left_gripper": row[left],
                        "right_arm": row[left + 1 : left + 1 + right],
                        "right_gripper": row[-1],
                    }
                }
                for row in values
            ]
        if not isinstance(states, (list, tuple)) or not states:
            raise ValueError("Expected a nonempty batch of native state dictionaries")
        packed = torch.from_numpy(
            np.stack([self.native.assemble_state(state) for state in states])
        )
        return self.native.template.sanitize(packed, self.native.state_mask)

    def decode_actions(
        self, normalized_actions: torch.Tensor, anchors: torch.Tensor
    ) -> torch.Tensor:
        """Decode one complete block per batch item to CPU commands [B,T,D].

        Each anchor is the measured physical state used when sampling that block.
        Normalized endpoints remain untouched for CFM training. No execution
        prefix selection or controller clipping is performed here.
        """
        action = torch.as_tensor(normalized_actions).detach().float().cpu()
        anchor = torch.as_tensor(anchors).detach().float().cpu()
        if (
            action.ndim != 3
            or action.shape[-1] != 55
            or action.shape[0] < 1
            or action.shape[1] < 1
            or anchor.shape != (action.shape[0], 55)
        ):
            raise ValueError("Expected actions [B,T,55] and physical anchors [B,55]")
        mask = self.native.action_mask
        if not torch.isfinite(action[..., mask]).all():
            raise ValueError("Nonfinite normalized action in a valid channel")
        anchor = self.native.template.sanitize(anchor, self.native.state_mask)
        delta = self.statistics.action.denormalize(action, mask)
        absolute = self.native.template.decode_action_delta(
            delta, anchor[:, None, :], mask
        )
        return torch.from_numpy(
            np.stack(
                [
                    self.native.split_actions(block.numpy())[self.control_mode]
                    for block in absolute
                ]
            )
        )
