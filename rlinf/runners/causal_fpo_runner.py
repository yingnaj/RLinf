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

"""Synchronous on-policy causal FPO orchestration over RLinf workers/channels."""

import hashlib
import json
import os
import time
import uuid
from pathlib import Path

from omegaconf import OmegaConf

from rlinf.scheduler import Channel, Cluster, PackedPlacementStrategy
from rlinf.utils.logging import get_logger


def validate_causal_fpo_config(cfg):
    from rlinf.models.embodiment.cosmos3.cosmos3_config import validate_cosmos3_variant
    from rlinf.scheduler import ClusterConfig

    validate_cosmos3_variant(cfg.actor.model)
    schema = ClusterConfig.from_dict_cfg(cfg.cluster)
    labels = {g.label: g.node_ranks for g in schema.node_groups}
    if labels.get("sim") != [0] or labels.get("ascend") != [1]:
        raise ValueError("Expected sim at node 0 and ascend at node 1")
    if cfg.runner.mode not in {"train", "frozen"}:
        raise ValueError("runner.mode must be train or frozen")
    roles = cfg.cluster.component_placement
    for role, label in (("env", "sim"), ("actor", "ascend"), ("rollout", "ascend")):
        if roles[role].node_group != label:
            raise ValueError(f"{role} must be placed on {label}")
    if cfg.cluster.num_nodes != 2 or cfg.rollout.pipeline_stage_num != 1:
        raise ValueError(
            "First online recipe requires two nodes and one synchronous pipeline"
        )
    import math

    if cfg.runner.timeout_seconds <= 0 or not math.isfinite(cfg.runner.timeout_seconds):
        raise ValueError("Stage timeout must be finite and positive")
    if not 0 <= cfg.causal.gamma <= 1 or not math.isfinite(cfg.causal.gamma):
        raise ValueError("gamma must be finite and in [0,1]")
    if cfg.causal.num_probes < 1 or cfg.causal.sampling.num_steps < 1:
        raise ValueError("Probe count and sampling steps must be positive")
    if any(
        type(seed) is not int or seed < 0 or seed >= 2**32
        for seed in cfg.causal.reset_seeds
    ):
        raise ValueError("Reset seeds must be uint32 integers")
    if any(
        not math.isfinite(cfg.causal[key]) or cfg.causal[key] < 0
        for key in ("cfm_rtol", "cfm_atol")
    ):
        raise ValueError("CFM acceptance tolerances must be finite and nonnegative")
    if (
        cfg.env.auto_reset
        or cfg.env.ignore_terminations
        or cfg.env.skip_intermediate_renders
        or cfg.env.center_crop
    ):
        raise ValueError(
            "Native causal execution requires explicit reset, termination and uncropped full feedback"
        )
    if cfg.actor.fsdp_config.strategy != "fsdp2" or not cfg.actor.enable_offload:
        raise ValueError("Shared-device stages require FSDP2 and Actor offload")
    if (
        cfg.weight_syncer.type != "bucket"
        or cfg.weight_syncer.bucket.bucket_device != "cpu"
        or cfg.weight_syncer.bucket.bucket_dtype is not None
    ):
        raise ValueError("Use full CPU bucket synchronization without dtype conversion")
    if (
        cfg.actor.model.robotwin is None
        or cfg.actor.model.robotwin.control_mode != "joint"
        or cfg.actor.model.robotwin.state_format != "joint_vector"
    ):
        raise ValueError(
            "Expected explicit joint-vector RoboTwin adapter configuration"
        )
    if (
        not cfg.causal.instruction.strip()
        or not cfg.causal.reset_seeds
        or cfg.causal.group_size < 2
    ):
        raise ValueError(
            "Specify task instruction, validated reset seeds and group_size >= 2"
        )
    if (
        cfg.env.actions_per_block != cfg.causal.batch.actions_per_block
        or cfg.env.fps != cfg.causal.batch.fps
    ):
        raise ValueError("Environment and model block geometry/control FPS must agree")
    if (
        cfg.env.max_episode_steps < 1
        or cfg.runner.max_versions < 1
        or cfg.causal.max_collect_rounds < 1
    ):
        raise ValueError("Episode/version/collection limits must be positive")
    if (
        cfg.actor.micro_batch_size < 1
        or cfg.actor.global_batch_size < 1
        or cfg.algorithm.update_epoch < 1
    ):
        raise ValueError("Batch sizes and update_epoch must be positive")
    if cfg.algorithm.get("entropy_bonus", 0) or cfg.actor.get(
        "enable_sft_co_train", False
    ):
        raise ValueError("Causal FPO does not support entropy or SFT co-training")
    for path in (
        cfg.actor.model.cosmos_sft_toml,
        cfg.actor.model.model_path,
        cfg.actor.model.robotwin.statistics_path,
        cfg.runner.checkpoint_dir,
        cfg.env.assets_path,
    ):
        if not Path(path).is_absolute():
            raise ValueError("All node-owned asset/checkpoint paths must be absolute")
    return cfg


def config_fingerprint(cfg):
    # Resume/output directories and the stopping horizon may change on restart.
    contract = OmegaConf.to_container(cfg, resolve=True)
    contract.pop("hydra", None)
    contract.pop("runner", None)
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


class CausalFPORunner:
    def __init__(self, cfg):
        self.cfg = validate_causal_fpo_config(cfg)
        self.groups, self.channels = [], []
        self.logger = get_logger()

    def wait(self, handle):
        deadline = time.monotonic() + self.cfg.runner.timeout_seconds
        while not handle.done():
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Causal FPO stage timed out; restore the last committed boundary"
                )
            time.sleep(0.1)
        return handle.wait()

    def run(self):
        import ray

        from rlinf.workers.actor.fsdp_fpo_actor_worker import OnlineCausalFPOActor
        from rlinf.workers.env.causal_robotwin_worker import CausalRoboTwinEnvWorker
        from rlinf.workers.rollout.causal_fpo_worker import (
            CausalFPOCollectorWorker,
            CausalFPORolloutWorker,
        )

        if os.environ.get("RAY_ADDRESS", "auto") == "local":
            raise ValueError(
                "Online training must use the existing two-node Ray cluster"
            )
        os.environ.setdefault("RAY_ADDRESS", "auto")
        Cluster.NAMESPACE = "CausalFPO_" + uuid.uuid4().hex
        cluster = Cluster(cluster_cfg=self.cfg.cluster)
        from rlinf.utils.placement import HybridComponentPlacement

        placement = HybridComponentPlacement(self.cfg, cluster)
        actor_size, rollout_size = (
            placement.get_world_size("actor"),
            placement.get_world_size("rollout"),
        )
        if placement.get_world_size("env") != 1:
            raise ValueError("First online version uses one GPU environment service")
        if self.cfg.actor.global_batch_size % (
            actor_size * self.cfg.actor.micro_batch_size
        ):
            raise ValueError("Global batch must divide by actor ranks times microbatch")

        def launch(cls, name, strategy):
            group = cls.create_group(self.cfg)
            self.groups.append(group)
            return group.launch(cluster=cluster, name=name, placement_strategy=strategy)

        def channel(name, node_rank):
            ch = Channel.create(name=name, node_rank=node_rank)
            self.channels.append(ch)
            return ch

        try:
            actor = launch(
                OnlineCausalFPOActor,
                self.cfg.actor.group_name,
                placement.get_strategy("actor"),
            )
            self.wait(actor.init_worker())
            fingerprint = config_fingerprint(self.cfg)
            version = 0
            if self.cfg.runner.resume:
                states = self.wait(
                    actor.restore_online(self.cfg.runner.resume, fingerprint)
                )
                if any(s != states[0] for s in states):
                    raise ValueError("Actor ranks disagree on restored manifest")
                version = states[0]["version"]
            rollout = launch(
                CausalFPORolloutWorker,
                self.cfg.rollout.group_name,
                placement.get_strategy("rollout"),
            )
            self.wait(rollout.init_worker())
            env = launch(
                CausalRoboTwinEnvWorker,
                self.cfg.env.group_name,
                placement.get_strategy("env"),
            )
            collector = launch(
                CausalFPOCollectorWorker,
                "CausalCollector",
                PackedPlacementStrategy(
                    self.cfg.causal.collector_device,
                    self.cfg.causal.collector_device,
                    node_group="ascend",
                ),
            )
            requests = channel("env_requests", 0)
            responses = [channel(f"env_response_{r}", 1) for r in range(rollout_size)]
            replays = [channel(f"replay_{r}", 1) for r in range(rollout_size)]
            shards = [channel(f"actor_batch_{r}", 1) for r in range(actor_size)]
            serving = env.serve(requests, responses, Cluster.NAMESPACE)
            output = Path(self.cfg.runner.output_dir)
            output.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(self.cfg, output / "resolved.yaml")
            while version < self.cfg.runner.max_versions:
                receive = rollout.sync_model_from_actor()
                self.wait(actor.sync_model_to_rollout())
                acknowledgements = self.wait(receive)
                expected = self.wait(actor.synchronization_report())
                if (
                    any(item != expected[0] for item in expected)
                    or expected[0]["version"] != version
                ):
                    raise ValueError(
                        "Actor ranks disagree on synchronized weights/assets/version"
                    )
                if any(
                    a["version"] != version
                    or a["sha256"] != expected[0]["sha256"]
                    or a["assets"] != expected[0]["assets"]
                    for a in acknowledgements
                ):
                    raise ValueError(
                        "Rollout weight hash/version acknowledgement differs from Actor"
                    )
                train = self.cfg.runner.mode == "train"
                for round_index in range(self.cfg.causal.max_collect_rounds):
                    tasks = [
                        rollout.execute_on(r).collect(
                            version,
                            version * self.cfg.causal.max_collect_rounds + round_index,
                            requests,
                            responses[r],
                            replays[r],
                        )
                        for r in range(rollout_size)
                    ]
                    gathered = collector.gather(version, replays, shards, train=train)
                    for task in tasks:
                        self.wait(task)
                    metrics = self.wait(gathered)[0]
                    if not metrics.get("need_more_groups"):
                        break
                else:
                    raise RuntimeError(
                        "Insufficient valid chunks; increase collection limit"
                    )
                if train:
                    submitted = [
                        actor.execute_on(r).recv_rollout_trajectories(shards[r])
                        for r in range(actor_size)
                    ]
                    for task in submitted:
                        self.wait(task)
                    metrics["behavior_alignment"] = self.wait(actor.validate_behavior())
                    metrics["actor"] = self.wait(actor.run_training())
                version += 1
                self.wait(actor.set_global_step(version))
                if train:
                    checkpoint = str(
                        Path(self.cfg.runner.checkpoint_dir) / f"version_{version:06d}"
                    )
                    self.wait(
                        actor.save_online(
                            checkpoint,
                            {
                                "config_sha256": fingerprint,
                                "actor_world_size": actor_size,
                            },
                        )
                    )
                    metrics["checkpoint"] = checkpoint
                metrics["version"] = version
                metrics["run_id"] = Cluster.NAMESPACE
                with (output / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(metrics) + "\n")
                self.logger.info("Causal FPO version %s: %s", version, metrics)
            requests.put({"op": "stop"})
            self.wait(serving)
        finally:
            for group in self.groups + [c._channel_worker_group for c in self.channels]:
                for info in group.worker_info_list:
                    try:
                        ray.kill(info.worker, no_restart=True)
                    except Exception as error:
                        self.logger.warning("Worker cleanup failed: %s", error)
            ray.shutdown()
