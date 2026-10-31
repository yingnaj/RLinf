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

"""FSDP updates for explicit causal FPO replay batches."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from rlinf.models.embodiment.cosmos3.fpo import (
    CausalFPOBatch,
    fpo_update_loss,
    recompute_old_cfm,
)
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import append_to_dict
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor


class EmbodiedFPOFSDPActor(EmbodiedFSDPActor):
    """Reuse FSDP ownership with CFM batches rather than token log probabilities.

    A collector or staged replay driver must submit equal-sized valid shards
    and globally grouped advantages. Ordinary PPO Trajectory is not accepted.
    """

    def save_checkpoint(self, save_path: str, step: int = 0) -> None:
        if getattr(self, "_fpo_batch", None) is not None or getattr(
            self, "_fpo_update_started", False
        ):
            raise RuntimeError("FPO checkpoints require a completed update boundary")
        super().save_checkpoint(save_path, step)
        if self._rank == 0:
            state = {
                "format_version": 1,
                "version": self.version,
                "consumed_version": getattr(self, "_fpo_consumed_version", None),
                "optimizer_steps": self.optimizer_steps,
            }
            path = Path(save_path) / "fpo_state.json"
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(state))
            temporary.replace(path)
        if dist.is_initialized():
            dist.barrier()

    def load_checkpoint(self, load_path: str) -> None:
        state = json.loads((Path(load_path) / "fpo_state.json").read_text())
        if state.get("format_version") != 1:
            raise ValueError("Expected FPO checkpoint format_version=1")
        super().load_checkpoint(load_path)
        self.set_global_step(state["version"])
        self.optimizer_steps = state["optimizer_steps"]
        self._fpo_consumed_version = state["consumed_version"]
        self._fpo_batch = None
        self._fpo_update_started = False

    def set_fpo_batch(self, batch: CausalFPOBatch) -> None:
        if not isinstance(batch, CausalFPOBatch):
            raise TypeError(
                "Expected CausalFPOBatch; ordinary PPO trajectories cannot carry causal replay"
            )
        if getattr(self, "_fpo_batch", None) is not None:
            raise RuntimeError(
                "Consume the pending FPO batch before submitting another"
            )
        batch.validate(expected_version=str(self.version))
        if getattr(self, "_fpo_consumed_version", None) == self.version:
            raise RuntimeError(
                "This behavior version was already consumed; synchronize a new version"
            )
        self._fpo_batch = batch

    async def recv_rollout_trajectories(self, input_channel) -> None:
        batch = await input_channel.get(async_op=True).async_wait()
        self.set_fpo_batch(batch)

    def compute_advantages_and_returns(self):
        if getattr(self, "_fpo_batch", None) is None:
            raise RuntimeError(
                "Submit a grouped CausalFPOBatch before computing metrics"
            )
        advantages = self._fpo_batch.advantages
        return {"fpo/advantage_mean": advantages.float().mean().item()}

    def run_training(self):
        from cosmos_framework.model.generator.algorithm.rl.fpo import FPOConfig

        batch = getattr(self, "_fpo_batch", None)
        if batch is None:
            raise RuntimeError("No CausalFPOBatch submitted")
        batch.validate(expected_version=str(self.version))
        if getattr(self, "_fpo_update_started", False):
            raise RuntimeError(
                "An FPO update failed after starting; restore the last committed checkpoint"
            )
        if self.enable_sft_co_train or self.cfg.algorithm.get("entropy_bonus", 0) != 0:
            raise ValueError(
                "Causal FPO does not support SFT co-training or entropy bonuses"
            )
        config = FPOConfig(**dict(self.cfg.algorithm.get("fpo", {})))
        micro = int(self.cfg.actor.micro_batch_size)
        local_batch = int(self.cfg.actor.global_batch_size) // self._world_size
        count = len(batch.examples)
        if micro < 1 or local_batch < 1 or self.update_epoch < 1:
            raise ValueError(
                "FPO requires positive micro/global batch sizes and update_epoch"
            )
        if count % local_batch or local_batch % micro:
            raise ValueError(
                "FPO chunk count must divide into complete global/micro batches"
            )
        geometry = (count, len(batch.examples[0].probes.sigmas), batch.behavior_version)
        if dist.is_initialized():
            geometries = [None] * dist.get_world_size()
            dist.all_gather_object(geometries, geometry)
            if any(item != geometry for item in geometries):
                raise ValueError(
                    "FPO ranks must agree on chunk count, probe count and policy version"
                )
        active = torch.tensor(int(batch.advantages.ne(0).any()), device=self.device)
        if dist.is_initialized():
            dist.all_reduce(active)
        if not active.item():
            self._fpo_batch = None
            self._fpo_consumed_version = self.version
            return {"fpo/skipped_zero_advantages": 1.0}
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)
        self.model.eval()
        with self.amp_context:
            old_cfm = recompute_old_cfm(
                self.model, batch.examples, micro_batch_size=micro
            )
        metrics = {}
        self._fpo_update_started = True
        generator = torch.Generator(device="cpu").manual_seed(
            int(self.cfg.actor.seed) + self.version + self._rank
        )
        for _ in range(self.update_epoch):
            order = torch.randperm(count, generator=generator).tolist()
            for start in range(0, count, local_batch):
                self.optimizer.zero_grad()
                for offset in range(0, local_batch, micro):
                    ids = order[start + offset : start + offset + micro]
                    examples = tuple(batch.examples[i] for i in ids)
                    backward_ctx = self.before_micro_batch(
                        self.model, is_last_micro_batch=offset + micro == local_batch
                    )
                    with backward_ctx:
                        with self.amp_context:
                            loss, data = fpo_update_loss(
                                self.model,
                                examples,
                                old_cfm[ids],
                                batch.advantages[ids],
                                config=config,
                            )
                        self.grad_scaler.scale(
                            loss / self.gradient_accumulation
                        ).backward()
                    append_to_dict(
                        metrics, {key: value.item() for key, value in data.items()}
                    )
                grad_norm, lr_list = self.optimizer_step()
                append_to_dict(
                    metrics, {"actor/grad_norm": grad_norm, "actor/lr": lr_list[0]}
                )
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        self._fpo_batch = None
        self._fpo_update_started = False
        self._fpo_consumed_version = self.version
        result = {key: float(np.mean(value)) for key, value in metrics.items()}
        if dist.is_initialized():
            result = all_reduce_dict(result, op=dist.ReduceOp.AVG)
        return result


class OnlineCausalFPOActor(EmbodiedFPOFSDPActor):
    """Production bootstrap, full-weight synchronization and boundary commits."""

    def model_provider_func(self):
        from rlinf.models.embodiment.cosmos3.runtime import (
            bootstrap_cosmos,
            compatible_dcp_metadata,
        )

        bootstrap_cosmos(self.torch_device_type)
        with compatible_dcp_metadata(self.cfg.causal.compatible_dcp):
            return super().model_provider_func()

    def init_worker(self):
        super().init_worker()
        self.param_names_need_sync = list(self.model.state_dict())
        from rlinf.models.embodiment.cosmos3.runtime import asset_signature

        self._asset_signature = asset_signature(self.cfg.actor.model)

    def get_rollout_state_dict(self):
        import hashlib

        from torch.distributed.tensor import DTensor

        state = super().get_rollout_state_dict()
        digest = hashlib.sha256()
        for name in sorted(state):
            value = state[name]
            value = value.full_tensor() if isinstance(value, DTensor) else value
            digest.update(name.encode())
            digest.update(str((tuple(value.shape), value.dtype)).encode())
            digest.update(
                value.detach()
                .cpu()
                .contiguous()
                .reshape(-1)
                .view(torch.uint8)
                .numpy()
                .tobytes()
            )
        self._sync_digest = digest.hexdigest()
        return state

    def synchronization_report(self):
        return {
            "version": self.version,
            "sha256": self._sync_digest,
            "assets": self._asset_signature,
        }

    def prepare_online_run(self, first_version, last_version):
        from rlinf.models.embodiment.cosmos3.runtime import prepare_checkpoint_directory

        failure = [None]
        if self._rank == 0:
            try:
                prepare_checkpoint_directory(
                    self.cfg.runner.checkpoint_dir, first_version, last_version
                )
            except (OSError, ValueError) as error:
                failure[0] = f"{type(error).__name__}: {error}"
        dist.broadcast_object_list(failure, src=0)
        if failure[0] is not None:
            raise RuntimeError(failure[0])

    def run_training(self):
        before = self.optimizer_steps
        metrics = super().run_training()
        metrics["fpo/optimizer_steps_total"] = self.optimizer_steps
        metrics["fpo/optimizer_steps_this_version"] = self.optimizer_steps - before
        return metrics

    def optimizer_step(self):
        grad_norm, lr_list = super().optimizer_step()
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            raise RuntimeError(
                "Nonfinite FPO gradient; restore the last committed boundary"
            )
        return grad_norm, lr_list

    def validate_behavior(self):
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        with self.amp_context:
            actual = recompute_old_cfm(
                self.model,
                self._fpo_batch.examples,
                micro_batch_size=self.cfg.actor.micro_batch_size,
            )
        expected = torch.stack(
            [example.old_cfm for example in self._fpo_batch.examples]
        )
        torch.testing.assert_close(
            actual,
            expected,
            rtol=self.cfg.causal.cfm_rtol,
            atol=self.cfg.causal.cfm_atol,
        )
        return {"cfm_max_abs": float((actual - expected).abs().max())}

    def save_online(self, path, manifest):
        import uuid

        target_dir = Path(path)
        exists = [target_dir.exists() if self._rank == 0 else None]
        dist.broadcast_object_list(exists, src=0)
        if exists[0]:
            raise FileExistsError(
                "Checkpoint destination already exists; use a fresh run directory or resume"
            )
        staging = [
            str(target_dir.with_name(f".{target_dir.name}.partial-{uuid.uuid4().hex}"))
            if self._rank == 0
            else None
        ]
        dist.broadcast_object_list(staging, src=0)
        self.save_checkpoint(staging[0], step=self.version)
        if self._rank == 0:
            target = Path(staging[0]) / "COMMITTED.json"
            target.write_text(
                json.dumps(
                    dict(manifest, version=self.version, assets=self._asset_signature),
                    sort_keys=True,
                )
            )
            Path(staging[0]).rename(target_dir)
        dist.barrier()

    def restore_online(self, path, fingerprint):
        manifest = json.loads((Path(path) / "COMMITTED.json").read_text())
        if manifest["config_sha256"] != fingerprint:
            raise ValueError(
                "Checkpoint configuration/asset contract differs from this run"
            )
        if (
            manifest["assets"] != self._asset_signature
            or manifest["actor_world_size"] != self._world_size
        ):
            raise ValueError(
                "Restoring with changed model assets or actor world size is unsupported"
            )
        self.load_checkpoint(path)
        if self.version != manifest["version"]:
            raise ValueError("Actor version differs from committed runner boundary")
        if self.enable_offload:
            self.offload_param_and_grad()
            self.offload_optimizer()
        return manifest
