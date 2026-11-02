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
import math
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from omegaconf import OmegaConf

from rlinf.scheduler import Channel, Cluster, ClusterConfig, PackedPlacementStrategy
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.logging import get_logger
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import collect_memory_metrics, print_metrics_table


def validate_causal_fpo_config(cfg):
    from rlinf.models.embodiment.cosmos3.cosmos3_config import validate_cosmos3_variant
    from rlinf.scheduler import ClusterConfig

    for field, minimum in (
        ("runner.max_versions", 1),
        ("causal.group_size", 2),
        ("causal.max_collect_rounds", 1),
        ("causal.num_probes", 1),
        ("causal.collector_device", 0),
        ("causal.sampling.num_steps", 1),
        ("causal.batch.actions_per_block", 1),
        ("causal.batch.video_stride", 1),
        ("causal.batch.temporal_compression_factor", 1),
        ("causal.batch.history_blocks", 1),
        ("env.actions_per_block", 1),
        ("env.action_dim", 1),
        ("env.max_episode_steps", 1),
        ("actor.micro_batch_size", 1),
        ("actor.global_batch_size", 1),
        ("actor.seed", 0),
        ("algorithm.update_epoch", 1),
    ):
        value = OmegaConf.select(cfg, field)
        if type(value) is not int or value < minimum:
            raise ValueError(f"{field} must be an integer >= {minimum}")
    for field, allow_zero in (
        ("runner.timeout_seconds", False),
        ("env.fps", False),
        ("causal.batch.fps", False),
        ("causal.sampling.shift", False),
        ("causal.sampling.guidance", True),
        ("actor.optim.lr", False),
        ("actor.optim.clip_grad", False),
    ):
        value = OmegaConf.select(cfg, field)
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value < 0
            or (not allow_zero and value == 0)
        ):
            raise ValueError(
                f"{field} must be finite and {'nonnegative' if allow_zero else 'positive'}"
            )
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
    if cfg.causal.batch.actions_per_block % (
        cfg.causal.batch.video_stride * cfg.causal.batch.temporal_compression_factor
    ):
        raise ValueError(
            "Action block must divide into complete video/VAE latent intervals"
        )
    roles = {"ego_view": ("head",), "concat_view": ("head", "left", "right")}
    if cfg.causal.batch.viewpoint not in roles:
        raise ValueError("Expected ego_view or concat_view camera layout")
    cameras = cfg.causal.batch.cameras
    required = [cameras.get(role) for role in roles[cfg.causal.batch.viewpoint]]
    if any(not name or name not in cfg.env.camera_names for name in required) or len(
        set(required)
    ) != len(required):
        raise ValueError(
            "Every policy camera must map to a distinct environment camera"
        )
    if cfg.env.get("record_video", False) and not {
        "head_camera",
        "left_camera",
        "right_camera",
    }.issubset(set(cfg.env.camera_names)):
        raise ValueError(
            "Observation video requires head_camera and both wrist cameras"
        )
    if type(cfg.env.get("record_predicted_video", False)) is not bool:
        raise ValueError("env.record_predicted_video must be a boolean")
    if (
        cfg.env.get("record_video", False)
        or cfg.env.get("record_predicted_video", False)
    ) and not cfg.env.get("evidence_dir"):
        raise ValueError("Video recording requires env.evidence_dir")
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
        cfg.actor.model.robotwin.robot_config_path,
        cfg.runner.output_dir,
        cfg.runner.checkpoint_dir,
        cfg.env.assets_path,
    ):
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError("All node-owned asset/checkpoint paths must be absolute")
    for path in (
        cfg.runner.resume,
        cfg.actor.model.get("wan_vae_path"),
        cfg.env.get("evidence_dir"),
    ):
        if path is not None and (
            not isinstance(path, str) or not Path(path).is_absolute()
        ):
            raise ValueError(
                "Resume, VAE and evidence paths must be absolute when configured"
            )
    return cfg


def config_fingerprint(cfg):
    # Resume/output directories and the stopping horizon may change on restart.
    contract = OmegaConf.to_container(cfg, resolve=True)
    contract.pop("hydra", None)
    contract.pop("runner", None)
    # Prediction recording can be toggled without changing the training contract.
    contract["env"].pop("record_predicted_video", None)
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


class CausalFPORunner:
    def __init__(self, cfg):
        self.cfg = validate_causal_fpo_config(cfg)
        self.groups, self.channels = [], []
        self.logger = get_logger()
        self.timer = ScopedTimer(
            reduction="max",
            sync_cuda=False,
            trace=bool(OmegaConf.select(cfg, "cluster.tracer.enable", default=False)),
        )
        self.metric_logger = None
        self.worker_metrics = {}

    def init_metric_logger(self):
        # Derive defaults in a copy so explicit paths and the saved recipe survive.
        logging_cfg = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=True))
        logger = logging_cfg.runner.get("logger", {})
        logging_cfg.runner.logger = OmegaConf.merge(
            {"log_path": self.cfg.runner.output_dir}, logger
        )
        if logging_cfg.runner.logger.log_path is None:
            logging_cfg.runner.logger.log_path = self.cfg.runner.output_dir
        self.metric_logger = MetricLogger(logging_cfg)

    def record_worker_metrics(self, role, rank, metrics):
        target = self.worker_metrics.setdefault(role, {}).setdefault(rank, {})
        for key, value in metrics.items():
            if key.startswith("time/"):
                target[key] = target.get(key, 0.0) + value
            elif "/memory/" in key:
                target[key] = max(target.get(key, 0.0), value)
            else:
                target[key] = value

    def log_metrics(self, metrics, start_time, start_version):
        """Write one completed behavior version to RLinf backends and the audit log."""
        version = metrics["version"]
        durations = self.timer.consume_durations()
        scalars = {f"time/{key}": value for key, value in durations.items()}
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and key != "version":
                prefix = (
                    "env"
                    if key
                    in {
                        "success_rate",
                        "episodes",
                        "executed_steps",
                        "episode_length",
                        "reward",
                    }
                    else "rollout"
                )
                scalars[f"{prefix}/{key}"] = float(value)
        scalars.update(
            {f"train/driver/{k}": v for k, v in collect_memory_metrics().items()}
        )
        scalars["train/behavior_version"] = version
        for role, ranked in self.worker_metrics.items():
            values = {}
            for rank, data in ranked.items():
                if self.metric_logger.per_worker_log:
                    group_name = (
                        self.cfg[role].group_name
                        if role != "collector"
                        else "CausalCollector"
                    )
                    self.metric_logger.log(
                        data, version, worker_group_name=group_name, rank=rank
                    )
                for key, value in data.items():
                    values.setdefault(key, []).append(value)
            for key, items in values.items():
                # Expose the slowest/largest rank, and mean training scalars.
                scalars[key] = (
                    max(items)
                    if key.startswith("time/")
                    or "/memory/" in key
                    or key.endswith("cfm_max_abs")
                    else sum(items) / len(items)
                )
        for key, count in (
            ("episodes", metrics["episodes"]),
            ("env_steps", metrics["executed_steps"]),
            ("chunks", metrics["valid_chunks"]),
        ):
            scalars[f"rollout/{key}_per_second"] = count / max(
                durations["generate_rollouts"], 1e-12
            )
        scalars["env/steps_per_second"] = metrics["executed_steps"] / max(
            durations["step"], 1e-12
        )
        if "actor" in metrics:
            updates = max(
                item["fpo/optimizer_steps_this_version"] for item in metrics["actor"]
            )
            trained = (
                metrics["trained_chunks"] * self.cfg.algorithm.update_epoch
                if updates
                else 0
            )
            scalars["train/actor/chunks_per_second"] = trained / max(
                durations["actor_training"], 1e-12
            )
        metrics["scalars"] = scalars
        metrics["workers"] = self.worker_metrics
        output = Path(self.cfg.runner.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics) + "\n")
        self.metric_logger.log(scalars, version)
        print_metrics_table(
            version - 1,
            self.cfg.runner.max_versions,
            start_time,
            scalars,
            start_step=start_version,
            log_path=self.metric_logger.log_path,
        )
        self.logger.info("Causal FPO version %s metrics saved to %s", version, output)

    def wait(self, handle):
        deadline = time.monotonic() + self.cfg.runner.timeout_seconds
        while not handle.done():
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Causal FPO stage timed out; restore the last committed boundary"
                )
            time.sleep(0.1)
        return handle.wait()

    @contextmanager
    def profile_version(self, version, actor, rollout, env, requests, responses):
        """Use RLinf's configured profiler windows at behavior-version boundaries."""
        profile = ClusterConfig.from_dict_cfg(self.cfg.cluster).profiling
        active = profile is not None and profile.should_profile_step(version)
        groups = [
            group
            for group in (actor, rollout)
            if active and profile.profiles_worker_group(group.worker_group_name)
        ]
        profile_env = active and profile.profiles_worker_group(env.worker_group_name)

        def env_command(op):
            # serve() owns the env event loop; route profiling through its queues.
            for rank, request in enumerate(requests):
                self.wait(
                    request.put(
                        {"op": op, "rank": rank, "step": version}, async_op=True
                    )
                )
            for response in responses[: len(requests)]:
                result = self.wait(response.get(async_op=True))
                if "error" in result:
                    raise RuntimeError(result["error"])

        for group in groups:
            self.wait(group.start_profile(version))
        if profile_env:
            env_command("start_profile")
        try:
            yield
        finally:
            for group in groups:
                self.wait(group.stop_profile())
            if profile_env:
                env_command("stop_profile")

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
        env_size = placement.get_world_size("env")
        if env_size < 1 or rollout_size % env_size:
            raise ValueError(
                "Rollout replicas must divide evenly across GPU environment workers"
            )
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
            self.init_metric_logger()
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
            if self.cfg.runner.mode == "train":
                self.wait(
                    actor.prepare_online_run(version + 1, self.cfg.runner.max_versions)
                )
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
            requests = [channel(f"env_requests_{r}", 0) for r in range(env_size)]
            responses = [channel(f"env_response_{r}", 1) for r in range(rollout_size)]
            replays = [channel(f"replay_{r}", 1) for r in range(rollout_size)]
            shards = [channel(f"actor_batch_{r}", 1) for r in range(actor_size)]
            serving = env.serve(requests, responses, Cluster.NAMESPACE)
            output = Path(self.cfg.runner.output_dir)
            output.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(self.cfg, output / "resolved.yaml")
            start_version, start_time = version, time.time()
            while version < self.cfg.runner.max_versions:
                with (
                    self.profile_version(
                        version, actor, rollout, env, requests, responses
                    ),
                    self.timer("step", trace_args={"step_idx": version}),
                ):
                    self.worker_metrics = {}
                    with self.timer("sync_weights"):
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
                    with self.timer("generate_rollouts"):
                        for round_index in range(self.cfg.causal.max_collect_rounds):
                            tasks = [
                                rollout.execute_on(r).collect(
                                    version,
                                    version * self.cfg.causal.max_collect_rounds
                                    + round_index,
                                    requests[r % env_size],
                                    responses[r],
                                    replays[r],
                                )
                                for r in range(rollout_size)
                            ]
                            gathered = collector.gather(
                                version, replays, shards, train=train
                            )
                            for rank, task in enumerate(tasks):
                                report = self.wait(task)[0]
                                self.record_worker_metrics(
                                    "rollout", rank, report["metrics"]
                                )
                                self.record_worker_metrics(
                                    "env", rank % env_size, report["env_metrics"]
                                )
                            metrics = self.wait(gathered)[0]
                            if not metrics.get("need_more_groups"):
                                self.record_worker_metrics(
                                    "collector",
                                    0,
                                    {
                                        f"rollout/collector/{k}": v
                                        for k, v in metrics["collector_memory"].items()
                                    },
                                )
                                break
                        else:
                            raise RuntimeError(
                                "Insufficient valid chunks; increase collection limit"
                            )
                    if train:
                        with self.timer("recv_trajectories"):
                            submitted = [
                                actor.execute_on(r).recv_rollout_trajectories(shards[r])
                                for r in range(actor_size)
                            ]
                            for task in submitted:
                                self.wait(task)
                        with self.timer("validate_behavior"):
                            metrics["behavior_alignment"] = self.wait(
                                actor.validate_behavior()
                            )
                        with self.timer("actor_training"):
                            metrics["actor"] = self.wait(actor.run_training())
                        for rank, report in enumerate(metrics["actor"]):
                            self.record_worker_metrics(
                                "actor",
                                rank,
                                {f"train/{k}": v for k, v in report.items()},
                            )
                        for rank, report in enumerate(metrics["behavior_alignment"]):
                            self.record_worker_metrics(
                                "actor",
                                rank,
                                {f"train/actor/{k}": v for k, v in report.items()},
                            )
                    version += 1
                    self.wait(actor.set_global_step(version))
                    if train:
                        with self.timer("save_checkpoint"):
                            checkpoint = str(
                                Path(self.cfg.runner.checkpoint_dir)
                                / f"version_{version:06d}"
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
                for role, group in (
                    ("actor", actor),
                    ("rollout", rollout),
                    ("collector", collector),
                ):
                    for rank, report in enumerate(
                        self.wait(group.pop_execution_times())
                    ):
                        self.record_worker_metrics(
                            role,
                            rank,
                            {f"time/{role}/{k}": v for k, v in report.items()},
                        )
                self.log_metrics(metrics, start_time, start_version)
            for request_channel in requests:
                request_channel.put({"op": "stop"})
            self.wait(serving)
        finally:
            for group in self.groups + [c._channel_worker_group for c in self.channels]:
                for info in group.worker_info_list:
                    try:
                        ray.kill(info.worker, no_restart=True)
                    except Exception as error:
                        self.logger.warning("Worker cleanup failed: %s", error)
            try:
                if self.metric_logger is not None:
                    self.metric_logger.finish()
            finally:
                ray.shutdown()
