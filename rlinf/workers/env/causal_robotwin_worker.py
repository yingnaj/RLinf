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

"""CPU observation/action channels for native RoboTwin causal sessions."""

from omegaconf import OmegaConf

from rlinf.envs.sim.robotwin.causal_env import CausalRoboTwinEnv
from rlinf.scheduler import Worker


class CausalRoboTwinEnvWorker(Worker):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def serve(self, requests, responses, run_id):
        sessions = {}
        try:
            while True:
                message = requests.get()
                if message["op"] == "stop":
                    return
                rank = message["rank"]
                try:
                    if rank not in sessions:
                        sessions[rank] = CausalRoboTwinEnv(
                            OmegaConf.create(
                                OmegaConf.to_container(self.cfg.env, resolve=True)
                            ),
                            run_id=run_id,
                        )
                    env = sessions[rank]
                    if message["op"] == "reset":
                        result = env.reset(message["episode"])
                    elif message["op"] == "step":
                        result = env.step(message["request"])
                    else:
                        raise ValueError("Unknown causal environment operation")
                    responses[rank].put({"result": result})
                except Exception as error:
                    responses[rank].put({"error": f"{type(error).__name__}: {error}"})
                    raise
        finally:
            for env in sessions.values():
                env.close()
