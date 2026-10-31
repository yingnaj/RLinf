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

"""Independent causal policy replicas and a replay collector on the model node."""

import gc
import os

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.hybrid_engines.weight_syncer import WeightSyncer
from rlinf.models.embodiment.cosmos3.online_fpo import (
    ActionRequest,
    CollectedEpisode,
    EpisodeKey,
    collect_actor_shards,
    reset_seed_for_group,
)
from rlinf.models.embodiment.cosmos3.runtime import (
    bootstrap_cosmos,
    compatible_dcp_metadata,
)
from rlinf.scheduler import Worker


class CausalFPORolloutWorker(Worker):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.version = None

    def init_worker(self):
        bootstrap_cosmos(self.torch_device_type)
        self.torch_platform.set_device(int(os.environ["LOCAL_RANK"]))
        self.device = torch.device(
            f"{self.torch_device_type}:{os.environ['LOCAL_RANK']}"
        )
        # Replicas sample independently; DCP/Cosmos still require a default group.
        import torch.distributed as dist

        if not dist.is_initialized():
            backend = {"npu": "hccl", "cuda": "nccl", "cpu": "gloo"}[
                self.torch_device_type
            ]
            dist.init_process_group(
                backend, store=dist.HashStore(), rank=0, world_size=1
            )
        from cosmos_framework.data.generator.action.online_batch import (
            ObservedCausalBatchBuilder,
        )

        from rlinf.models import get_model

        with compatible_dcp_metadata(self.cfg.causal.compatible_dcp):
            self.policy = get_model(self.cfg.actor.model).cpu().eval()
        adapter = self.policy.robotwin_adapter
        if adapter.control_mode != "joint" or adapter.state_format != "joint_vector":
            raise ValueError(
                "Online FPO first version requires measured joint-vector control"
            )
        if adapter.action_dim != self.cfg.env.action_dim:
            raise ValueError("Robot action dimension differs from environment contract")
        self.builder = ObservedCausalBatchBuilder(
            template=adapter.native.template,
            statistics=adapter.statistics,
            state_mask=adapter.native.state_mask,
            action_mask=adapter.native.action_mask,
            **OmegaConf.to_container(self.cfg.causal.batch, resolve=True),
        )
        if (
            self.builder.geometry.temporal_compression_factor
            != self.policy.omni.tokenizer_vision_gen.temporal_compression_factor
        ):
            raise ValueError("Batch geometry differs from the checkpoint VAE")
        import tomllib
        from pathlib import Path

        from rlinf.models.embodiment.cosmos3.runtime import asset_signature

        self._asset_signature = asset_signature(self.cfg.actor.model)
        with Path(self.cfg.actor.model.cosmos_sft_toml).open("rb") as stream:
            action_recipe = tomllib.load(stream)["action"]
        for field in ("actions_per_block", "video_stride", "resolution"):
            if str(action_recipe[field]) != str(self.cfg.causal.batch[field]):
                raise ValueError(
                    f"Online {field} differs from the paired training recipe"
                )
        self.syncer = WeightSyncer.create(self.cfg.weight_syncer)

    async def sync_model_from_actor(self):
        async def receive():
            return await self.broadcast(
                None,
                groups=[
                    (self.cfg.actor.group_name, 0),
                    (self._group_name, list(range(self._world_size))),
                ],
                src=(self.cfg.actor.group_name, 0),
                async_op=True,
                options=self.syncer.comm_options,
            ).async_wait()

        if not self.syncer.receiver_initialized():
            await self.syncer.init_receiver(
                state_dict=self.policy.state_dict(), recv=receive, send=None
            )
        self.version = int(await self.syncer.apply(self.policy, receive))
        import hashlib

        digest = hashlib.sha256()
        state = self.policy.state_dict()
        for name in sorted(state):
            value = state[name]
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
        return {
            "rank": self._rank,
            "version": self.version,
            "sha256": digest.hexdigest(),
            "assets": self._asset_signature,
        }

    def _rpc(self, requests, response, **message):
        requests.put(dict(message, rank=self._rank))
        result = response.get()
        if "error" in result:
            raise RuntimeError(result["error"])
        return result["result"]

    def collect(self, version, round_id, requests, response, replay_channel):
        if self.version != version:
            raise ValueError(
                "Rollout must acknowledge the current weight version before sampling"
            )
        self.policy.to(self.device).eval()
        self.policy._sync_runtime_device()
        episodes = []
        try:
            group = round_id * self._world_size + self._rank
            reset_seed = reset_seed_for_group(
                self.cfg.causal.reset_seeds,
                version=version,
                round_index=round_id % self.cfg.causal.max_collect_rounds,
                rank=self._rank,
                world_size=self._world_size,
            )
            for member in range(self.cfg.causal.group_size):
                key = EpisodeKey(
                    version,
                    group,
                    member,
                    self.cfg.env.task_config.task_name,
                    reset_seed,
                    self.cfg.causal.instruction,
                )
                initial = self._rpc(requests, response, op="reset", episode=key)
                if initial["episode"] != key or initial["observation_step"] != 0:
                    raise ValueError("Reset response differs from requested episode")
                frames, state = list(initial["frames"]), initial["state"]
                examples, feedbacks = [], []
                chunk = 0
                while True:
                    anchors = self.policy.robotwin_adapter.pack_states(
                        np.asarray(state)[None], state_source="measured"
                    )
                    batch = self.builder(
                        frames=frames, state=anchors[0], prompt=key.instruction
                    )
                    # Unique deterministic seeds include version, group, member and chunk.
                    import hashlib

                    identity = f"{self.cfg.actor.seed}:{key}:{chunk}".encode()
                    seed = int.from_bytes(
                        hashlib.sha256(identity).digest()[:8], "little"
                    ) % (2**31)
                    _, records = self.policy.sample_fpo_batch(
                        [batch],
                        policy_version=str(version),
                        seeds=[seed],
                        probe_seeds=[seed ^ 0x5A5A5A5A],
                        num_probes=self.cfg.causal.num_probes,
                        causal_block_size=self.builder.geometry.latent_frames_per_block,
                        causal_history_blocks=self.builder.history_blocks,
                        causal_use_kv_cache=False,
                        **OmegaConf.to_container(
                            self.cfg.causal.sampling, resolve=True
                        ),
                    )
                    record = records[0]
                    # Replay retains the full endpoint; only current rows become environment commands.
                    mask = record.rollout.action_mask.any(-1)
                    endpoint = record.rollout.endpoint_action[mask]
                    commands = self.policy.robotwin_adapter.decode_actions(
                        endpoint[None], anchors
                    )[0].numpy()
                    request = ActionRequest(
                        key,
                        chunk,
                        chunk * self.cfg.env.actions_per_block,
                        commands.astype(np.float32),
                    )
                    request.validate(
                        self.cfg.env.actions_per_block, self.cfg.env.action_dim
                    )
                    feedback = self._rpc(requests, response, op="step", request=request)
                    feedback.validate(self.cfg.env.actions_per_block, self.cfg.env.fps)
                    if (
                        feedback.episode != key
                        or feedback.chunk != chunk
                        or feedback.observation_step != request.observation_step
                    ):
                        raise ValueError(
                            "Feedback does not belong to the pending sample"
                        )
                    examples.append(record)
                    feedbacks.append(feedback)
                    if feedback.terminated or feedback.truncated:
                        break
                    frames.extend(feedback.frames)
                    frames = frames[
                        -(
                            self.builder.history_blocks * self.cfg.env.actions_per_block
                            + 1
                        ) :
                    ]
                    state = feedback.state
                    chunk += 1
                episodes.append(
                    CollectedEpisode(key, tuple(examples), tuple(feedbacks))
                )
            replay_channel.put(episodes)
            return {"rank": self._rank, "episodes": len(episodes), "version": version}
        finally:
            self.policy.cpu()
            self.policy._sync_runtime_device()
            gc.collect()
            self.torch_platform.empty_cache()


class CausalFPOCollectorWorker(Worker):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.episodes = []
        self.version = None

    def gather(self, version, replay_channels, actor_channels, *, train):
        if self.version != version:
            if self.episodes:
                raise ValueError("Unconsumed replay from a previous behavior version")
            self.version = version
        for channel in replay_channels:
            self.episodes.extend(channel.get())
        shards, metrics = collect_actor_shards(
            self.episodes,
            version=version,
            group_size=self.cfg.causal.group_size,
            world_size=len(actor_channels),
            global_batch_size=self.cfg.actor.global_batch_size,
            gamma=self.cfg.causal.gamma,
            block_size=self.cfg.env.actions_per_block,
            fps=self.cfg.env.fps,
        )
        if not shards:
            return metrics
        if train:
            for channel, shard in zip(actor_channels, shards, strict=True):
                channel.put(shard)
        self.episodes = []
        return metrics
