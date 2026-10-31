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

"""CPU-only native UniPC replay through real RLinf collector/Channel workers."""

import argparse
import json
import os
import tempfile
import time
import uuid
from pathlib import Path

import ray
from omegaconf import OmegaConf

from rlinf.scheduler import Channel, Cluster, NodePlacementStrategy, Worker
from rlinf.scheduler.hardware import AcceleratorType
from rlinf.workers.env.causal_robotwin_worker import CausalRoboTwinEnvWorker
from rlinf.workers.rollout.causal_fpo_worker import CausalFPOCollectorWorker


class CPUOnlyWorker(Worker):
    """Opt out of node accelerator inheritance before Worker reads its env."""

    def __new__(cls, *args, **kwargs):
        if (
            os.environ.get("CLUSTER_NODE_RANK") is not None
            and "ActorClass(" not in cls.__name__
        ):
            os.environ["ACCELERATOR_TYPE"] = AcceleratorType.NO_ACCEL.value
            os.environ["VISIBLE_DEVICES"] = ""
        return super().__new__(cls, *args, **kwargs)


class ReplayProducer(CPUOnlyWorker):
    def produce(self, channel):
        import numpy as np
        import torch
        from cosmos_framework.model.generator.algorithm.rl.causal_fpo import (
            configure_fpo_trainable_parameters,
        )
        from cosmos_framework.model.generator.algorithm.rl.causal_fpo_test import (
            PackedSequence,
            TinyModel,
        )

        from rlinf.models.embodiment.cosmos3.causal_policy import CausalCosmos3Policy
        from rlinf.models.embodiment.cosmos3.online_fpo import (
            CollectedEpisode,
            EpisodeKey,
            ExecutionFeedback,
        )

        class TinyCosmos(torch.nn.Module, TinyModel):
            def __init__(self):
                torch.nn.Module.__init__(self)
                TinyModel.__init__(self)

        PackedSequence.to_cuda = lambda self: None
        model = TinyCosmos()
        configure_fpo_trainable_parameters(model.net)
        policy = CausalCosmos3Policy(model, ema_enabled=False).eval()
        _, examples = policy.sample_fpo_batch(
            [{}, {}],
            policy_version="0",
            seeds=[42, 43],
            probe_seeds=[13, 14],
            num_probes=2,
        )
        episodes = []
        for member, example in enumerate(examples):
            key = EpisodeKey(0, 0, member, "test", 42, "CPU protocol test")
            feedback = ExecutionFeedback(
                key,
                0,
                0,
                ({},),
                np.zeros(14, np.float32),
                (float(member == 0),),
                member == 0,
                member != 0,
                member == 0,
                (1 / 30,),
            )
            episodes.append(CollectedEpisode(key, (example,), (feedback,)))
        channel.put(episodes)
        return len(episodes)


class ReplayConsumer(CPUOnlyWorker):
    def consume(self, channel):
        import hashlib

        import torch

        replay = channel.get()
        replay.validate("0")
        digest = hashlib.sha256()
        for example in replay.examples:
            digest.update(
                example.rollout.endpoint_action.contiguous()
                .view(torch.uint8)
                .numpy()
                .tobytes()
            )
        return {
            "advantages": replay.advantages.tolist(),
            "endpoint_sha256": digest.hexdigest(),
        }


class ToyRoboTwinTask:
    """External simulator stand-in with measured joints and a terminal control."""

    def __init__(self):
        from types import SimpleNamespace

        import numpy as np

        self.robot = SimpleNamespace()
        for side in ("left", "right"):
            entity = SimpleNamespace(qpos=np.zeros(2))
            entity.get_qpos = lambda entity=entity: entity.qpos
            entity.get_active_joints = lambda: [0, 1]
            setattr(self.robot, f"{side}_entity", entity)
            setattr(self.robot, f"{side}_arm_joints", [0])
            setattr(self.robot, f"{side}_gripper", [(1, 1.0, 0.0)])
            setattr(self.robot, f"{side}_gripper_scale", [0.0, 1.0])

    def setup_demo(self, **kwargs):
        self.seed = kwargs["seed"]
        self.take_action_cnt, self.step_lim, self.eval_success = 0, 1, False

    def set_instruction(self, instruction):
        self.instruction = instruction

    def get_obs(self):
        import numpy as np

        return {
            "observation": {
                "head_camera": {
                    "rgb": np.full((4, 4, 3), self.seed % 256, dtype=np.uint8)
                }
            }
        }

    def take_action(self, action, action_type):
        assert action_type == "qpos"
        time.sleep(0.1)
        self.robot.left_entity.qpos[:] = action[:2]
        self.robot.right_entity.qpos[:] = action[2:]
        self.take_action_cnt += 1
        self.eval_success = True

    def close_env(self, **kwargs):
        pass


class ToyEnvironmentWorker(CausalRoboTwinEnvWorker, CPUOnlyWorker):
    def placement_report(self):
        return {
            "rank": self._rank,
            "device_type": self.torch_device_type or "cpu",
            "has_accelerator": self.has_accelerator,
        }

    def _make_env(self, run_id):
        from rlinf.envs.sim.robotwin.causal_env import CausalRoboTwinEnv

        return CausalRoboTwinEnv(
            self.cfg.env, task_factory=ToyRoboTwinTask, run_id=run_id
        )


class CPUCollector(CausalFPOCollectorWorker, CPUOnlyWorker):
    pass


class EnvironmentClient(CPUOnlyWorker):
    def execute(self, requests, responses):
        import numpy as np

        from rlinf.models.embodiment.cosmos3.online_fpo import ActionRequest, EpisodeKey

        request_channel = requests[self._rank % len(requests)]
        response = responses[self._rank]
        # Exercise reset, execution, duplicate delivery and session reuse.
        for version in range(2):
            key = EpisodeKey(version, self._rank, 0, "toy", 42 + self._rank, "Move")
            request_channel.put({"op": "reset", "rank": self._rank, "episode": key})
            initial = response.get()["result"]
            assert initial["episode"] == key
            assert initial["frames"][0]["head_camera"][0, 0, 0] == key.reset_seed
            action = np.full((32, 4), (self._rank + 1) / 10, dtype=np.float32)
            request = ActionRequest(key, 0, 0, action)
            for _ in range(2):
                request_channel.put(
                    {"op": "step", "rank": self._rank, "request": request}
                )
                feedback = response.get()["result"]
                assert feedback.episode == key and feedback.executed_steps == 1
                assert feedback.terminated and feedback.success
                np.testing.assert_allclose(feedback.state, action[0])
        return {"rollout_rank": self._rank, "env_rank": self._rank % len(requests)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-workers", type=int, default=8)
    parser.add_argument("--rollout-clients", type=int, default=None)
    args = parser.parse_args()
    os.environ["RAY_ADDRESS"] = "local"
    os.environ["RLINF_NODE_RANK"] = "0"
    os.environ["RAY_TMPDIR"] = tempfile.mkdtemp(prefix="causal-fpo-cpu-")
    os.environ["RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES"] = str(128 * 1024 * 1024)
    os.environ["OMP_NUM_THREADS"] = "1"
    groups, channels = [], []
    if args.env_workers < 1:
        parser.error("--env-workers must be positive")
    client_count = args.rollout_clients or args.env_workers
    if client_count < args.env_workers or client_count % args.env_workers:
        parser.error("Rollout clients must divide evenly across environment workers")
    result = {"status": "running", "scope": "cpu_replay_and_environment_routing_only"}

    def wait(handle):
        deadline = time.monotonic() + 180
        while not handle.done():
            if time.monotonic() > deadline:
                raise TimeoutError("CPU Worker/Channel smoke timed out")
            time.sleep(0.1)
        return handle.wait()

    try:
        Cluster.NAMESPACE = "CausalFPOCPU_" + uuid.uuid4().hex[:8]
        cluster = Cluster(num_nodes=1)
        cfg = OmegaConf.create(
            {
                "causal": {"group_size": 2, "gamma": 1.0},
                "actor": {"global_batch_size": 2},
                "env": {"actions_per_block": 32, "fps": 30.0},
            }
        )
        for cls, name, parameters in (
            (ReplayProducer, "producer", ()),
            (CPUCollector, "collector", (cfg,)),
            (ReplayConsumer, "consumer", ()),
        ):
            group = cls.create_group(*parameters)
            groups.append(group)
            group.launch(
                cluster, name=name, placement_strategy=NodePlacementStrategy([0])
            )
        for name in ("episodes", "shard0", "shard1"):
            channels.append(Channel.create(name, node_rank=0))
        gathering = groups[1].gather(0, channels[:1], channels[1:], train=True)
        wait(groups[0].produce(channels[0]))
        metrics = wait(gathering)[0]
        received = [wait(groups[2].consume(ch))[0] for ch in channels[1:]]
        assert (
            received[0]["advantages"][0] > 0.99 and received[1]["advantages"][0] < -0.99
        )
        assert received[0]["endpoint_sha256"] != received[1]["endpoint_sha256"]
        cfg.env = OmegaConf.create(
            {
                "task_config": {"task_name": "toy"},
                "max_episode_steps": 1,
                "camera_names": ["head_camera"],
                "action_dim": 4,
                "actions_per_block": 32,
                "fps": 30.0,
            }
        )
        envs = ToyEnvironmentWorker.create_group(cfg)
        groups.append(envs)
        envs.launch(
            cluster,
            name="envs",
            placement_strategy=NodePlacementStrategy([0] * args.env_workers),
        )
        clients = EnvironmentClient.create_group()
        groups.append(clients)
        clients.launch(
            cluster,
            name="clients",
            placement_strategy=NodePlacementStrategy([0] * client_count),
        )
        requests = [
            Channel.create(f"env_requests_{r}", node_rank=0)
            for r in range(args.env_workers)
        ]
        responses = [
            Channel.create(f"env_response_{r}", node_rank=0)
            for r in range(client_count)
        ]
        channels.extend(requests + responses)
        placement_reports = wait(envs.placement_report())
        assert all(
            report["device_type"] == "cpu" and not report["has_accelerator"]
            for report in placement_reports
        )
        serving = envs.serve(requests, responses, "cpu-routing")
        routing = wait(clients.execute(requests, responses))
        for request in requests:
            request.put({"op": "stop"})
        wait(serving)
        assert len(routing) == client_count
        assert all(
            item["env_rank"] == item["rollout_rank"] % args.env_workers
            for item in routing
        )
        result.update(
            status="passed",
            metrics=metrics,
            shards=received,
            environment_routing=routing,
            environment_placement=placement_reports,
        )
    except Exception as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        for group in groups + [c._channel_worker_group for c in channels]:
            for info in group.worker_info_list:
                ray.kill(info.worker, no_restart=True)
        ray.shutdown()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
