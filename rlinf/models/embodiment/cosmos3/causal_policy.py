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

"""Causal Cosmos3 replay forward and prepared-input sampling for FPO."""

from typing import Any

import torch

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.cosmos3.cosmos3_policy import Cosmos3Policy


class CausalCosmos3Policy(Cosmos3Policy):
    """Keep CFM differentiable through the RLinf/FSDP root forward."""

    _no_split_modules = ["CausalActionNetwork", "MoTDecoderLayer"]

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointWrapper,
        )

        if any(isinstance(module, CheckpointWrapper) for module in model.net.modules()):
            # Shard the outer checkpoint unit, as native Cosmos does, so its
            # recomputation sees materialized parameters throughout the block.
            self._no_split_modules = ["CausalActionNetwork", "CheckpointWrapper"]
        self.robotwin_adapter = None

    def configure_robotwin(self, config):
        """Attach the physical state/action adapter without changing sampling."""
        from rlinf.models.embodiment.cosmos3.robotwin_adapter import (
            RobotwinCausalAdapter,
        )

        self.robotwin_adapter = RobotwinCausalAdapter.from_config(config)

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        if forward_type != ForwardType.DEFAULT:
            raise ValueError(f"Unsupported causal Cosmos3 forward: {forward_type}")
        return self.default_forward(**kwargs)

    def default_forward(self, *, fpo_examples, **kwargs):
        from cosmos_framework.model.generator.algorithm.rl.causal_fpo import (
            compute_action_cfm,
        )

        if any(
            kwargs.get(key, False)
            for key in ("compute_logprobs", "compute_values", "compute_entropy")
        ):
            raise ValueError(
                "Causal FPO returns CFM, not logprobs, values or entropy; use the FPO actor"
            )
        if not fpo_examples:
            raise ValueError("Expected at least one FPO example")
        if len({x.rollout.policy_version for x in fpo_examples}) != 1:
            raise ValueError("FPO forward cannot mix behavior versions")
        if len({len(x.probes.sigmas) for x in fpo_examples}) != 1:
            raise ValueError("FPO forward requires equal probe counts")
        self._sync_runtime_device()
        return {
            "cfm": torch.stack(
                [
                    compute_action_cfm(self.omni, example.rollout, example.probes)
                    for example in fpo_examples
                ]
            )
        }

    @torch.no_grad()
    def sample_fpo_batch(
        self,
        batches: list[dict[str, Any]],
        *,
        policy_version: str,
        seeds: list[int],
        probe_seeds: list[int],
        num_probes: int,
        **sampling_options,
    ):
        """Sample prepared causal inputs; return native outputs and CPU examples.

        The caller owns measured-state packing, session routing and conversion
        of model actions to environment commands. No robot semantics are inferred.
        """
        from cosmos_framework.model.generator.algorithm.rl.causal_fpo import (
            prepare_fpo_example,
            sample_causal_fpo_rollout,
        )

        if (
            not batches
            or len(batches) != len(seeds)
            or len(batches) != len(probe_seeds)
        ):
            raise ValueError(
                "batches, seeds and probe_seeds must have equal nonzero lengths"
            )
        self._sync_runtime_device()
        outputs, examples = [], []
        for batch, seed, probe_seed in zip(batches, seeds, probe_seeds, strict=True):
            result, rollout = sample_causal_fpo_rollout(
                self.omni,
                batch,
                policy_version=policy_version,
                seed=[seed],
                **sampling_options,
            )
            example = prepare_fpo_example(
                self.omni,
                rollout,
                num_probes=num_probes,
                probe_seed=probe_seed,
                behavior_version=policy_version,
            )
            outputs.append(result)
            examples.append(example)
        return outputs, examples

    def predict_action_batch(self, **kwargs):
        raise NotImplementedError(
            "RoboTwin camera/history batch construction and the FPO collector are "
            "required. Use robotwin_adapter for state/action conversion and "
            "sample_fpo_batch with prepared causal inputs."
        )
