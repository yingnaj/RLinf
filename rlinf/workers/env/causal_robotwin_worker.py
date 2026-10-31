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

"""Process-isolated native RoboTwin sessions with rank-owned request queues."""

import json
import os
from pathlib import Path

from omegaconf import OmegaConf

from rlinf.envs.sim.robotwin.causal_env import CausalRoboTwinEnv
from rlinf.scheduler import Worker


class CausalRoboTwinEnvWorker(Worker):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def _make_env(self, run_id):
        return CausalRoboTwinEnv(
            OmegaConf.create(OmegaConf.to_container(self.cfg.env, resolve=True)),
            run_id=run_id,
        )

    def serve(self, requests, responses, run_id):
        if len(requests) != self._world_size or len(responses) % self._world_size:
            raise ValueError(
                "Expected one request queue per env and evenly assigned rollouts"
            )
        request_channel = requests[self._rank]
        # RLinf starts each isolated worker with only its assigned CUDA device
        # visible, before native RoboTwin/SAPIEN is imported in _create_task.
        if self.has_accelerator:
            self.torch_platform.set_device(int(os.environ["LOCAL_RANK"]))
        identity = {
            "env_rank": self._rank,
            "env_world_size": self._world_size,
            "pid": os.getpid(),
            "device_type": self.torch_device_type,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "local_rank": os.environ.get("LOCAL_RANK"),
        }
        self.log_info(f"Causal RoboTwin environment placement: {identity}")
        if self.cfg.env.get("record_video", False):
            evidence = Path(self.cfg.env.evidence_dir) / run_id
            evidence.mkdir(parents=True, exist_ok=True)
            (evidence / f"env_worker_{self._rank}.json").write_text(
                json.dumps(identity, indent=2)
            )
        sessions = {}
        try:
            while True:
                message = request_channel.get()
                if message["op"] == "stop":
                    return
                rank = message["rank"]
                try:
                    if (
                        not 0 <= rank < len(responses)
                        or rank % self._world_size != self._rank
                    ):
                        raise ValueError(
                            "Rollout request reached a different environment owner"
                        )
                    if rank not in sessions:
                        sessions[rank] = self._make_env(run_id)
                    env = sessions[rank]
                    if message["op"] == "reset":
                        result = env.reset(message["episode"])
                    elif message["op"] == "step":
                        result = env.step(message["request"])
                    else:
                        raise ValueError("Unknown causal environment operation")
                    responses[rank].put({"result": result})
                except Exception as error:
                    if 0 <= rank < len(responses):
                        responses[rank].put(
                            {"error": f"{type(error).__name__}: {error}"}
                        )
                    raise
        finally:
            for env in sessions.values():
                env.close()
