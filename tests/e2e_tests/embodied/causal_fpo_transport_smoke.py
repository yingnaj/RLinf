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
from rlinf.workers.rollout.causal_fpo_worker import CausalFPOCollectorWorker


class ReplayProducer(Worker):
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


class ReplayConsumer(Worker):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["RAY_ADDRESS"] = "local"
    os.environ["RLINF_NODE_RANK"] = "0"
    os.environ["RAY_TMPDIR"] = tempfile.mkdtemp(prefix="causal-fpo-cpu-")
    os.environ["RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES"] = str(128 * 1024 * 1024)
    os.environ["OMP_NUM_THREADS"] = "1"
    groups, channels = [], []
    result = {"status": "running", "scope": "cpu_replay_transport_only"}

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
            (CausalFPOCollectorWorker, "collector", (cfg,)),
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
        result.update(status="passed", metrics=metrics, shards=received)
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
