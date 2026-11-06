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
import json
import os
from pathlib import Path

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
from rlinf.utils.metric_utils import collect_memory_metrics


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
        if adapter.control_mode != "joint":
            raise ValueError("Online FPO requires joint command execution")
        if adapter.state_format not in ("joint_vector", "native"):
            raise ValueError(
                "Online FPO requires measured joint-vector or native joint+EEF state"
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

        from rlinf.models.embodiment.cosmos3.runtime import asset_signature

        self._asset_signature = asset_signature(self.cfg.actor.model)
        with Path(self.cfg.actor.model.cosmos_sft_toml).open("rb") as stream:
            action_recipe = tomllib.load(stream)["action"]
        for field in ("actions_per_block", "video_stride", "resolution"):
            if str(action_recipe[field]) != str(self.cfg.causal.batch[field]):
                raise ValueError(
                    f"Online {field} differs from the paired training recipe"
                )
        # Diagnostic evidence: decode each sampled video latent and persist the
        # predicted block as mp4. Off by default because the VAE pass is not
        # needed for training.
        self.save_predicted_videos = bool(
            self.cfg.causal.get("save_predicted_video", False)
        )
        # E5a/E5c diagnostics: force one gripper command channel closed after
        # decoding. ``expected_arm`` closes from the first chunk (E5a), while
        # ``expected_arm_at_contact`` keeps the model command until the arm is
        # over the bottle and has descended to grasp height (E5c).
        self.gripper_override = str(
            self.cfg.causal.get("gripper_override", "none") or "none"
        ).lower()
        if self.gripper_override not in (
            "none",
            "expected_arm",
            "expected_arm_at_contact",
            "left",
            "right",
        ):
            raise ValueError(
                "causal.gripper_override must be one of "
                "none|expected_arm|expected_arm_at_contact|left|right"
            )
        self.gripper_contact_xy = float(self.cfg.causal.get("gripper_contact_xy", 0.06))
        self.gripper_contact_dz = float(self.cfg.causal.get("gripper_contact_dz", 0.15))
        if self.gripper_contact_xy <= 0 or self.gripper_contact_dz <= 0:
            raise ValueError("causal gripper contact thresholds must be positive")
        # Whether env feedback carries measured qpos or RoboTwin drive targets.
        # The verified reference client uses "command"; measured is the RL
        # default. The adapter records the declared source on every anchor.
        self.state_source = str(
            self.cfg.env.get("state_source", "measured") or "measured"
        ).lower()
        if self.state_source not in ("measured", "command"):
            raise ValueError("env.state_source must be 'measured' or 'command'")
        predicted_video_dir = self.cfg.causal.get("predicted_video_dir", None)
        self.predicted_video_dir = (
            Path(predicted_video_dir)
            if predicted_video_dir
            else Path(self.cfg.runner.output_dir) / "predicted_videos"
        )
        if not self.predicted_video_dir.is_absolute():
            raise ValueError("causal.predicted_video_dir must be absolute")
        # When observation evidence is enabled, put the prediction next to the
        # episode's observed.mp4/chunk_*_commands.npy instead of a flat folder.
        evidence_dir = self.cfg.env.get("evidence_dir", None)
        self.evidence_dir = (
            Path(evidence_dir)
            if self.cfg.env.get("record_video", False) and evidence_dir
            else None
        )
        if self.evidence_dir is not None and not self.evidence_dir.is_absolute():
            raise ValueError("env.evidence_dir must be absolute")
        # Decode a bounded tail instead of the whole re-encoded history window:
        # causal VAE decode cost grows with the episode, while the current
        # block only needs a few preceding latents for temporal context.
        self.predicted_video_history_blocks = int(
            self.cfg.causal.get("predicted_video_history_blocks", 2)
        )
        if self.predicted_video_history_blocks < 0:
            raise ValueError("causal.predicted_video_history_blocks must be >= 0")
        self.predicted_block_frames = (
            self.builder.geometry.latent_frames_per_block
            * self.policy.omni.tokenizer_vision_gen.temporal_compression_factor
        )
        # Whether to mark retained history state rows as valid using the anchors
        # measured at earlier chunks. Training always has valid history states;
        # the original online builder left them masked, which changes the state
        # token sequence. Opt-in so the default stays unchanged.
        self.fill_state_history = bool(self.cfg.causal.get("state_history", False))
        self.syncer = WeightSyncer.create(self.cfg.weight_syncer)

    @Worker.timer("sync_model_from_actor")
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

    def adopt_version(self, version):
        """Set the behavior version for inference-only runs without an actor."""
        if type(version) is not int or version < 0:
            raise ValueError("behavior version must be a nonnegative integer")
        self.version = version
        return version

    def _rpc(self, requests, response, **message):
        requests.put(dict(message, rank=self._rank))
        result = response.get()
        if "error" in result:
            raise RuntimeError(result["error"])
        for key, value in result["metrics"].items():
            if key.startswith("time/"):
                self.env_metrics[key] = self.env_metrics.get(key, 0.0) + value
            else:
                self.env_metrics[key] = max(self.env_metrics.get(key, 0.0), value)
        return result["result"]

    def _write_predicted_video(
        self,
        latent,
        *,
        version,
        group,
        member,
        chunk,
        fps,
        seed,
        episode,
        observation_step,
        run_id,
    ):
        """Decode the sampled video latent into the episode's output directory.

        Evidence must never fail a training rollout, so decode/encode errors are
        reported and skipped. The manifest is per episode (or per rank when
        episode evidence is disabled) to avoid interleaved appends.
        """
        from cosmos_framework.inference.causal_action.video import (
            save_predicted_video,
        )

        if self.evidence_dir is not None and run_id:
            out_dir = self.evidence_dir / run_id / f"v{version}_g{group}_m{member}"
            name = f"predicted_c{chunk:04d}.mp4"
            manifest = out_dir / "predicted.jsonl"
        else:
            out_dir = self.predicted_video_dir
            name = f"v{version:04d}_g{group:03d}_m{member:02d}_c{chunk:03d}.mp4"
            manifest = out_dir / f"predicted_videos_rank{self._rank}.jsonl"
        try:
            blocks = self.builder.geometry.latent_frames_per_block
            keep = (self.predicted_video_history_blocks + 1) * blocks
            if latent.shape[2] > keep:
                latent = latent[:, :, -keep:]
            metadata = save_predicted_video(
                self.policy.omni,
                latent,
                out_dir / name,
                fps=fps,
                tail_frames=self.predicted_block_frames,
            )
        except Exception as error:
            self.log_warning(
                f"Predicted video decode failed ({name}): "
                f"{type(error).__name__}: {error}"
            )
            return
        record = {
            "version": version,
            "group": group,
            "member": member,
            "chunk": chunk,
            "observation_step": observation_step,
            "seed": seed,
            "task": episode.task,
            "reset_seed": episode.reset_seed,
            **metadata,
        }
        with manifest.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(
            f"[video] {metadata['path']} frames={metadata['num_frames']} fps={fps}",
            flush=True,
        )

    def _contact_reached(self, native_state, arm, bottle_point):
        """Whether the target arm is over the bottle at grasp height.

        The trigger uses the measured EEF pose of the current chunk boundary:
        horizontal distance to the bottle's initial functional point and the
        vertical gap to it (a closed finger pair can only grasp once the hand
        has descended toward the bottle).
        """
        if native_state is None or bottle_point is None:
            return False
        pose = native_state.get("endpose", {}).get(f"{arm}_endpose")
        if pose is None:
            return False
        pose = np.asarray(pose, dtype=np.float64).reshape(-1)
        bottle = np.asarray(bottle_point, dtype=np.float64).reshape(-1)
        if pose.size < 3 or bottle.size < 3:
            return False
        xy = float(np.linalg.norm(pose[:2] - bottle[:2]))
        dz = float(pose[2] - bottle[2])
        return xy <= self.gripper_contact_xy and dz <= self.gripper_contact_dz

    def _write_gripper_override(self, run_id, key, chunk, arm, raw_gripper):
        """Record the model's raw gripper command before E5a forces it closed."""
        if self.evidence_dir is None or not run_id:
            return
        out_dir = (
            self.evidence_dir
            / str(run_id)
            / f"v{key.version}_g{key.group}_m{key.member}"
        )
        if not out_dir.is_dir():
            return
        record = {
            "chunk": int(chunk),
            "arm": arm,
            "channel": 6 if arm == "left" else 13,
            "raw_min": float(np.min(raw_gripper)),
            "raw_max": float(np.max(raw_gripper)),
            "raw_mean": float(np.mean(raw_gripper)),
            "forced": 0.0,
        }
        with (out_dir / "gripper_override.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    @Worker.timer("collect")
    def collect(
        self,
        version,
        round_id,
        requests,
        response,
        replay_channel,
        run_id=None,
        publish_replay=True,
    ):
        if self.version != version:
            raise ValueError(
                "Rollout must acknowledge the current weight version before sampling"
            )
        self.env_metrics = {}
        self.torch_platform.synchronize()
        self.torch_platform.reset_peak_memory_stats()
        self.policy.to(self.device).eval()
        self.policy._sync_runtime_device()
        episodes = []
        episode_meta = {}
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
                # The environment may specialize the instruction (for example
                # naming the left/right arm from the task's qpose_tag).
                prompt = initial.get("instruction", key.instruction)
                override_arm = None
                contact_close = False
                if self.gripper_override in (
                    "expected_arm",
                    "expected_arm_at_contact",
                ):
                    qpose_tag = initial.get("qpose_tag")
                    if qpose_tag not in (0, 1):
                        raise ValueError(
                            "causal.gripper_override requires the environment "
                            "qpose_tag to be 0 (left) or 1 (right)"
                        )
                    override_arm = "left" if int(qpose_tag) == 0 else "right"
                    # E5a closes from the first chunk; E5c latches closed once
                    # the arm is over the bottle at grasp height.
                    contact_close = self.gripper_override == "expected_arm"
                elif self.gripper_override in ("left", "right"):
                    override_arm = self.gripper_override
                    contact_close = True
                bottle_point = initial.get("bottle_point")
                episode_meta[key] = {
                    "instruction": prompt,
                    "qpose_tag": initial.get("qpose_tag"),
                    "model_id": initial.get("model_id"),
                    "override_arm": override_arm,
                    "contact_chunk": 0 if contact_close else None,
                }
                frames, state = list(initial["frames"]), initial["state"]
                native_state = initial.get("native_state")
                examples, feedbacks = [], []
                anchor_history = []
                chunk = 0
                while True:
                    adapter = self.policy.robotwin_adapter
                    if adapter.state_format == "native":
                        if native_state is None:
                            raise ValueError(
                                "native state_format requires env native_state "
                                "(joint_action + endpose); enable "
                                "env.task_config.data_type.endpose"
                            )
                        model_state = native_state
                    else:
                        model_state = np.asarray(state)[None]
                    anchors = adapter.pack_states(
                        model_state, state_source=self.state_source
                    )
                    batch = self.builder(
                        frames=frames,
                        state=anchors[0],
                        prompt=prompt,
                        state_history=anchor_history
                        if self.fill_state_history
                        else None,
                    )
                    if self.fill_state_history:
                        anchor_history.append(anchors[0].clone())
                    # Unique deterministic seeds include version, group, member and chunk.
                    import hashlib

                    identity = f"{self.cfg.actor.seed}:{key}:{chunk}".encode()
                    seed = int.from_bytes(
                        hashlib.sha256(identity).digest()[:8], "little"
                    ) % (2**31)
                    with self.worker_timer("policy_inference"):
                        outputs, records = self.policy.sample_fpo_batch(
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
                        self.torch_platform.synchronize()
                    record = records[0]
                    # Replay retains the full endpoint; only current rows become environment commands.
                    mask = record.rollout.action_mask.any(-1)
                    endpoint = record.rollout.endpoint_action[mask]
                    commands = adapter.decode_actions(endpoint[None], anchors)[
                        0
                    ].numpy()
                    predicted_video = predicted_video_fps = None
                    if self.cfg.env.get("record_predicted_video", False):
                        geometry = self.builder.geometry
                        predicted_video = self.policy.decode_predicted_video(
                            outputs[0],
                            num_frames=geometry.actions_per_block
                            // geometry.video_stride,
                        )
                        predicted_video_fps = float(batch["conditioning_fps"][0])
                    if (
                        override_arm is not None
                        and not contact_close
                        and self.gripper_override == "expected_arm_at_contact"
                    ):
                        contact_close = self._contact_reached(
                            native_state, override_arm, bottle_point
                        )
                        if contact_close:
                            episode_meta[key]["contact_chunk"] = chunk
                    if override_arm is not None and contact_close:
                        channel = 6 if override_arm == "left" else 13
                        raw_gripper = commands[:, channel].copy()
                        commands[:, channel] = 0.0
                        self._write_gripper_override(
                            run_id, key, chunk, override_arm, raw_gripper
                        )

                    if os.environ.get("CAUSAL_GRIPPER_DEBUG") == "1":
                        debug_dir = os.path.join(
                            self.cfg.runner.output_dir, "gripper_debug"
                        )
                        os.makedirs(debug_dir, exist_ok=True)
                        np.savez(
                            os.path.join(
                                debug_dir,
                                f"v{version:04d}_g{group:03d}_m{member:02d}_c{chunk:03d}.npz",
                            ),
                            endpoint_norm=endpoint[:, [28, 29]].numpy(),
                            anchor_phys=anchors[0, [28, 29]].numpy(),
                            commands_phys=commands[:, [6, 13]],
                        )
                        print(
                            f"[gripraw] v{version} g{group} m{member} c{chunk} "
                            f"L[min/max]={endpoint[:, 28].min():+.3f}/{endpoint[:, 28].max():+.3f} "
                            f"R[min/max]={endpoint[:, 29].min():+.3f}/{endpoint[:, 29].max():+.3f}",
                            flush=True,
                        )

                    request = ActionRequest(
                        key,
                        chunk,
                        chunk * self.cfg.env.actions_per_block,
                        commands.astype(np.float32),
                        predicted_video=predicted_video,
                        predicted_video_fps=predicted_video_fps,
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
                    if self.save_predicted_videos:
                        self._write_predicted_video(
                            outputs[0]["vision"][0],
                            version=version,
                            group=group,
                            member=member,
                            chunk=chunk,
                            fps=float(batch["conditioning_fps"][0]),
                            seed=seed,
                            episode=key,
                            observation_step=request.observation_step,
                            run_id=run_id,
                        )
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
                    native_state = feedback.native_state
                    chunk += 1
                # Frames were already folded into the local history window;
                # the collector never reads them. Strip them before the replay
                # transfer so a full episode does not ship ~GBs of RGB.
                episodes.append(
                    CollectedEpisode(
                        key,
                        tuple(examples),
                        tuple(feedback.without_frames() for feedback in feedbacks),
                    )
                )
            # Scalar-only summaries are safe to deserialize on the driver even
            # when the rollout and driver use different NumPy major versions.
            summaries = []
            for episode in episodes:
                feedback = episode.feedback
                meta = episode_meta.get(episode.key, {})
                summaries.append(
                    {
                        "version": episode.key.version,
                        "group": episode.key.group,
                        "member": episode.key.member,
                        "reset_seed": episode.key.reset_seed,
                        "instruction": meta.get("instruction"),
                        "qpose_tag": meta.get("qpose_tag"),
                        "model_id": meta.get("model_id"),
                        "state_source": self.state_source,
                        "gripper_override": self.gripper_override,
                        "override_arm": meta.get("override_arm"),
                        "contact_chunk": meta.get("contact_chunk"),
                        "chunks": len(episode.examples),
                        "executed_steps": int(
                            sum(entry.executed_steps for entry in feedback)
                        ),
                        "reward_sum": float(
                            sum(sum(entry.rewards) for entry in feedback)
                        ),
                        "success": bool(feedback[-1].success),
                        "terminated": bool(feedback[-1].terminated),
                        "truncated": bool(feedback[-1].truncated),
                    }
                )
            with self.worker_timer("send_replay"):
                if publish_replay:
                    replay_channel.put(episodes)
            self.torch_platform.synchronize()
            return {
                "rank": self._rank,
                "episodes": len(episodes),
                "version": version,
                "metrics": {
                    f"rollout/{k}": v
                    for k, v in collect_memory_metrics(self.torch_platform).items()
                },
                "env_metrics": self.env_metrics,
                "summaries": summaries,
            }
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

    @Worker.timer("gather")
    def gather(self, version, replay_channels, actor_channels, *, train):
        if self.version != version:
            if self.episodes:
                raise ValueError("Unconsumed replay from a previous behavior version")
            self.version = version
        for channel in replay_channels:
            self.episodes.extend(channel.get())
        with self.worker_timer("build_shards"):
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
        metrics["collector_memory"] = collect_memory_metrics()
        if train:
            for channel, shard in zip(actor_channels, shards, strict=True):
                channel.put(shard)
        self.episodes = []
        return metrics
