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

"""Run frozen causal-FPO inference on the two-node cluster.

This entry point keeps the exact training-time batch builder and sampler but
drops the actor, collector, weight synchronization and optimizer: one
``CausalRoboTwinEnvWorker`` runs on the sim node and one
``CausalFPORolloutWorker`` runs on the Ascend node. It is intended for
single-task policy/state ablations, not for training.

Examples::

    python examples/embodiment/causal_fpo_infer.py +infer.episodes=8

    python examples/embodiment/causal_fpo_infer.py +infer.episodes=8 \\
        causal.instruction="Use the correct arm to adjust the bottle upright" \\
        causal.sampling.shift=5.0 causal.sampling.num_steps=20 \\
        env.max_episode_steps=450 env.task_config.step_lim=450 \\
        causal.state_history=true

The runner prints one JSON record per episode and writes a summary under
``<runner.output_dir>/infer/<run_id>.json``. Observation/prediction evidence is
written by the existing workers when ``env.record_video`` and
``causal.save_predicted_video`` are enabled.
"""

import json
import os
import time
import uuid
from pathlib import Path

import hydra
import ray
from omegaconf import OmegaConf

from rlinf.runners.causal_fpo_runner import validate_causal_fpo_config
from rlinf.scheduler import Channel, Cluster
from rlinf.utils.placement import HybridComponentPlacement


def wait(handle, timeout_seconds):
    """Wait for an RLinf worker handle with a finite deadline."""
    deadline = time.monotonic() + float(timeout_seconds)
    while not handle.done():
        if time.monotonic() >= deadline:
            raise TimeoutError("Causal FPO inference stage timed out")
        time.sleep(0.1)
    return handle.wait()


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="robotwin_causal_production",
)
def main(cfg):
    infer = cfg.get("infer", {})
    episodes = infer.get("episodes", 1)
    version = infer.get("version", 0)
    device_index = infer.get("device_index", 0)
    timeout_seconds = infer.get("timeout_seconds", cfg.runner.timeout_seconds)
    if type(episodes) is not int or episodes < 1:
        raise ValueError("infer.episodes must be a positive integer")
    if type(version) is not int or version < 0:
        raise ValueError("infer.version must be a nonnegative integer")
    if type(device_index) is not int or device_index < 0:
        raise ValueError("infer.device_index must be a nonnegative integer")

    if os.environ.get("RAY_ADDRESS", "auto") == "local":
        raise ValueError("Inference must use the existing two-node Ray cluster")
    os.environ.setdefault("RAY_ADDRESS", "auto")

    # One process per role. Node-group labels and per-node env_configs stay from
    # the recipe, so the sim node and the Ascend node keep their interpreters
    # and PYTHONPATH.
    device = str(device_index)
    cfg.cluster.component_placement.env.placement = device
    cfg.cluster.component_placement.rollout.placement = device
    if "actor" in cfg.cluster.component_placement:
        cfg.cluster.component_placement.actor.placement = device
    cfg.runner.mode = "frozen"
    # validate_causal_fpo_config() encodes the production contract, including a
    # >=2 group size. Validate it, then collect a single member per episode.
    validation_group_size = int(cfg.causal.group_size)
    if validation_group_size < 2:
        cfg.causal.group_size = 2
    validate_causal_fpo_config(cfg)
    cfg.causal.group_size = 1

    from rlinf.workers.env.causal_robotwin_worker import CausalRoboTwinEnvWorker
    from rlinf.workers.rollout.causal_fpo_worker import CausalFPORolloutWorker

    run_id = str(
        infer.get("run_id") or f"CausalFPOInfer_{time.strftime('%Y%m%d-%H%M%S')}"
    )
    Cluster.NAMESPACE = "CausalFPOInfer_" + uuid.uuid4().hex
    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = HybridComponentPlacement(cfg, cluster)
    env_size = placement.get_world_size("env")
    rollout_size = placement.get_world_size("rollout")
    if env_size != 1 or rollout_size != 1:
        raise ValueError(
            f"Expected one env and one rollout worker, got {env_size}/{rollout_size}"
        )

    groups = []

    def launch(cls, name, role):
        group = cls.create_group(cfg)
        groups.append(group)
        return group.launch(
            cluster=cluster,
            name=name,
            placement_strategy=placement.get_strategy(role),
        )

    channels = []
    serving = None
    summary = {
        "run_id": run_id,
        "version": version,
        "config": {
            "instruction": cfg.causal.instruction,
            "instruction_template": cfg.env.get("instruction_template", None),
            "sampling": OmegaConf.to_container(cfg.causal.sampling, resolve=True),
            "max_episode_steps": cfg.env.max_episode_steps,
            "state_history": bool(cfg.causal.get("state_history", False)),
            "gripper_override": str(cfg.causal.get("gripper_override", "none")),
            "diagnostics": bool(cfg.env.get("diagnostics", False)),
        },
        "episodes": [],
    }
    try:
        env = launch(CausalRoboTwinEnvWorker, "CausalFPOInferEnv", "env")
        rollout = launch(CausalFPORolloutWorker, "CausalFPOInferRollout", "rollout")
        requests = [
            Channel.create(f"infer_env_requests_{rank}", node_rank=0)
            for rank in range(env_size)
        ]
        responses = [
            Channel.create(f"infer_env_response_{rank}", node_rank=1)
            for rank in range(rollout_size)
        ]
        replays = [
            Channel.create(f"infer_replay_{rank}", node_rank=1)
            for rank in range(rollout_size)
        ]
        channels.extend(requests + responses + replays)

        wait(rollout.init_worker(), timeout_seconds)
        wait(rollout.execute_on(0).adopt_version(version), timeout_seconds)
        serving = env.serve(requests, responses, run_id)

        for episode in range(episodes):
            result = wait(
                rollout.execute_on(0).collect(
                    version,
                    episode,
                    requests[0],
                    responses[0],
                    replays[0],
                    run_id,
                    # Scalar summaries come back in the RPC result; replay
                    # payloads contain NumPy arrays from the Ascend image and
                    # are not needed by the driver.
                    False,
                ),
                timeout_seconds,
            )
            # RLinf returns one result per worker in the group.
            if isinstance(result, (list, tuple)):
                result = result[0] if result else {}
            if not isinstance(result, dict) or result.get("episodes", 0) < 1:
                raise RuntimeError(f"Rollout produced no episode: {result!r}")
            records = result.get("summaries", [])
            if not records:
                raise RuntimeError("Rollout returned no episode summary")
            for record in records:
                summary["episodes"].append(record)
                print("[infer] " + json.dumps(record), flush=True)

        requests[0].put({"op": "stop"})
        wait(serving, timeout_seconds)
        serving = None
    except Exception as error:
        summary["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            if serving is not None:
                try:
                    requests[0].put({"op": "stop"})
                except Exception:
                    pass
            for group in groups + [
                channel._channel_worker_group for channel in channels
            ]:
                for info in group.worker_info_list:
                    try:
                        ray.kill(info.worker, no_restart=True)
                    except Exception:
                        pass
            try:
                ray.shutdown()
            except Exception:
                # Ray 2.59 can hit a circular import in ray.dag when shutdown
                # races the RLinf signal handler; the driver is exiting anyway.
                pass
        finally:
            evidence_dir = cfg.env.get("evidence_dir")
            if evidence_dir:
                summary["evidence_dir"] = str(Path(evidence_dir) / run_id)
            output = Path(cfg.runner.output_dir) / "infer"
            output.mkdir(parents=True, exist_ok=True)
            summary_path = output / f"{run_id}.json"
            summary_path.write_text(json.dumps(summary, indent=2))
            print(f"[infer] summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
