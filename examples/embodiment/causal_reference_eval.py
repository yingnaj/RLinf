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

"""Minimal E7 path: current RoboTwin env against the reference streaming server.

The reference Cosmos causal server (`c00670587/cosmos-framework`) runs on the
Ascend node as a root Ray actor; the GPU head drives the current
``CausalRoboTwinEnv`` and speaks the v2 msgpack/WebSocket protocol directly.
This avoids the reference ``robotwin_client.py`` dependency on
``RoboTwin/scripts/eval_policy_xpolicylab.py``, which the current checkout does
not have.

Example::

    python examples/embodiment/causal_reference_eval.py +reference.episodes=1

The script sets the per-episode arm instruction template by default (same as
E4b), starts the reference server, runs the seeds, saves observed/current
evidence plus the reference prediction video, and writes a JSON summary under
``<runner.output_dir>/reference_eval/<run_id>.json``.
"""

import json
import os
import socket
import subprocess
import time
from pathlib import Path

import hydra
import msgpack
import numpy as np
import ray
from omegaconf import OmegaConf
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from rlinf.envs.sim.robotwin.causal_env import CausalRoboTwinEnv
from rlinf.models.embodiment.cosmos3.online_fpo import ActionRequest, EpisodeKey

DEFAULT_SERVER_REPO = "/efs-mi-pay-gy1/c00670587/cosmos-framework"
DEFAULT_SERVER_PYTHON = (
    "/efs-mi-east4-1/apps/miniconda3/envs/cosmos-rlinf-ascend/bin/python"
)
DEFAULT_ASCEND_IP = "7.150.2.73"
DEFAULT_INSTRUCTION_TEMPLATE = (
    "Lift the bottle head-up from the table with the {arm} arm."
)
CAMERA_MAP = {
    "head_camera": "cam_high",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
}


# --------------------------------------------------------------------------- #
# v2 wire format (copied from the reference causal_action/wire.py)
# --------------------------------------------------------------------------- #
def _encode(value):
    if isinstance(value, (np.ndarray, np.generic)):
        if value.dtype.kind not in "biuf":
            raise ValueError(f"Unsupported NumPy dtype: {value.dtype}")
        if isinstance(value, np.generic):
            return {
                b"__npgeneric__": True,
                b"dtype": value.dtype.str,
                b"data": value.item(),
            }
        return {
            b"__ndarray__": True,
            b"dtype": value.dtype.str,
            b"shape": value.shape,
            b"data": value.tobytes(),
        }
    raise TypeError(f"Unsupported wire value: {type(value).__name__}")


def _decode(value):
    if b"__ndarray__" in value or b"__npgeneric__" in value:
        dtype = np.dtype(value[b"dtype"])
        if dtype.kind not in "biuf":
            raise ValueError(f"Unsupported NumPy dtype: {dtype}")
        if b"__ndarray__" in value:
            return np.frombuffer(value[b"data"], dtype=dtype).reshape(value[b"shape"])
        return dtype.type(value[b"data"])
    return value


def packb(value):
    return msgpack.packb(value, default=_encode, use_bin_type=True)


def unpackb(value):
    return msgpack.unpackb(value, object_hook=_decode, raw=False)


# --------------------------------------------------------------------------- #
# World xyz+wxyz  <->  dataset xyz+xyzw (reference client conversion)
# --------------------------------------------------------------------------- #
class RobotwinPoseConverter:
    def __init__(self, *, eef_pose_frame="world", T_world_base=None):
        if eef_pose_frame not in ("base", "world"):
            raise ValueError("eef_pose_frame must be 'base' or 'world'")
        self.eef_pose_frame = eef_pose_frame
        self.T_world_base = None
        if eef_pose_frame == "base":
            if not isinstance(T_world_base, dict):
                raise ValueError(
                    "Base-frame EEF requires T_world_base with position and quaternion_wxyz"
                )
            position = np.asarray(T_world_base.get("position"), dtype=np.float64)
            quaternion = np.asarray(
                T_world_base.get("quaternion_wxyz"), dtype=np.float64
            )
            if position.shape != (3,) or quaternion.shape != (4,):
                raise ValueError(
                    "T_world_base needs position[3] and quaternion_wxyz[4]"
                )
            if (
                not np.isfinite(position).all()
                or not np.isfinite(quaternion).all()
                or np.linalg.norm(quaternion) < 1e-8
            ):
                raise ValueError(
                    "T_world_base must be finite with a nonzero quaternion"
                )
            quaternion = quaternion / np.linalg.norm(quaternion)
            self.T_world_base = {
                "position": position.tolist(),
                "quaternion_wxyz": quaternion.tolist(),
            }
            self._base_position, self._base_quaternion = position, quaternion
        elif T_world_base is not None:
            raise ValueError("T_world_base only applies to eef_pose_frame='base'")

    def _transform_pose(self, pose, *, to_world):
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape[-1:] != (7,):
            raise ValueError("EEF pose must contain seven values")
        if self.eef_pose_frame == "world":
            return pose.astype(np.float32)
        rotation = self._base_quaternion.copy()
        xyz = pose[..., :3]
        if not to_world:
            rotation[1:] *= -1
            xyz = xyz - self._base_position
        uv = np.cross(rotation[1:], xyz)
        xyz = xyz + 2 * (rotation[0] * uv + np.cross(rotation[1:], uv))
        if to_world:
            xyz = xyz + self._base_position
        q = pose[..., 3:7]
        w = rotation[:1] * q[..., :1] - (rotation[1:] * q[..., 1:]).sum(
            axis=-1, keepdims=True
        )
        v = (
            rotation[:1] * q[..., 1:]
            + q[..., :1] * rotation[1:]
            + np.cross(rotation[1:], q[..., 1:])
        )
        return np.concatenate([xyz, w, v], axis=-1).astype(np.float32)

    def to_dataset(self, pose):
        pose = self._transform_pose(pose, to_world=False)
        return pose[..., [0, 1, 2, 4, 5, 6, 3]]


# --------------------------------------------------------------------------- #
# Reference server launched on Ascend as a root Ray actor
# --------------------------------------------------------------------------- #
@ray.remote
class ReferenceServer:
    def __init__(self):
        self.proc = None
        self.log_path = None

    def start(
        self,
        command,
        server_repo,
        alias_dir,
        log_path,
        port,
        timeout_seconds,
        ascend_device=None,
    ):
        env = dict(os.environ)
        existing = env.get("PYTHONPATH", "")
        prefix = f"{alias_dir}:" if alias_dir else ""
        env["PYTHONPATH"] = prefix + server_repo + (":" + existing if existing else "")
        env["COSMOS_DEVICE"] = "npu"
        if ascend_device is not None:
            env["ASCEND_RT_VISIBLE_DEVICES"] = str(ascend_device)
        # The reference server is normally launched by torchrun; when spawned
        # directly it still calls distributed.init(), so provide the env
        # rendezvous variables for one rank and pick a free dist port.
        env["RANK"] = "0"
        env["LOCAL_RANK"] = "0"
        env["WORLD_SIZE"] = "1"
        env["HF_HUB_OFFLINE"] = "1"
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            env["MASTER_ADDR"] = "127.0.0.1"
            env["MASTER_PORT"] = str(probe.getsockname()[1])
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path = str(path)
        log = open(path, "wb")
        self.proc = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
            # The reference model config contains repo-relative JSON paths.
            cwd=server_repo,
        )
        deadline = time.time() + float(timeout_seconds)
        while time.time() < deadline:
            if self.proc.poll() is not None:
                return {
                    "ok": False,
                    "returncode": self.proc.returncode,
                    "log_tail": self._tail(),
                }
            try:
                with socket.create_connection(("127.0.0.1", int(port)), timeout=1):
                    pass
                return {"ok": True, "pid": self.proc.pid, "log_path": str(path)}
            except OSError:
                time.sleep(1)
        return {"ok": False, "error": "timeout", "log_tail": self._tail()}

    def _tail(self, limit=4000):
        try:
            with open(self.log_path, "rb") as stream:
                stream.seek(0, 2)
                size = stream.tell()
                stream.seek(max(0, size - 20000))
                return stream.read().decode(errors="replace")[-limit:]
        except Exception as error:
            return f"{type(error).__name__}: {error}"

    def status(self):
        return {
            "alive": bool(self.proc is not None and self.proc.poll() is None),
            "log_tail": self._tail(),
        }

    def stop(self):
        if self.proc is None:
            return {"stopped": True}
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
        return {"stopped": True}


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_server_command(cfg, reference):
    model = cfg.actor.model
    processor = reference.get("processor", None)
    if not processor:
        for override in model.get("cosmos_overrides", []):
            if "tokenizer_type=" in override:
                processor = override.split("tokenizer_type=", 1)[1]
                break
    if not processor:
        raise ValueError("No processor path; set +reference.processor=...")
    checkpoint = reference.get("checkpoint", None) or model.model_path
    vae = reference.get("vae", None) or model.wan_vae_path
    statistics = reference.get("statistics", None) or model.robotwin.statistics_path
    return [
        str(reference.get("server_python", DEFAULT_SERVER_PYTHON)),
        "-m",
        "cosmos_framework.inference.causal_action.server",
        "--checkpoint",
        str(checkpoint),
        "--processor",
        str(processor),
        "--vae",
        str(vae),
        "--statistics",
        str(statistics),
        "--actions-per-block",
        "32",
        "--execute-steps",
        "32",
        "--video-stride",
        "4",
        "--history",
        "32",
        "--steps",
        str(int(reference.get("steps", 20))),
        "--guidance",
        str(float(reference.get("guidance", 1.0))),
        "--shift",
        str(float(reference.get("shift", 5.0))),
        "--fps",
        "30",
        "--resolution",
        str(int(reference.get("resolution", 480))),
        "--host",
        "0.0.0.0",
        "--port",
        str(int(reference.get("port", 8000))),
    ]


# --------------------------------------------------------------------------- #
# Observation/state conversion and client loop
# --------------------------------------------------------------------------- #
def camera_images(frame):
    return {target: np.asarray(frame[source]) for source, target in CAMERA_MAP.items()}


def converted_state(native_state, converter):
    if not isinstance(native_state, dict):
        raise ValueError("Current env did not return a native_state")
    endpose = {
        key: converter.to_dataset(np.asarray(value))
        for key, value in native_state["endpose"].items()
    }
    return {"joint_action": dict(native_state["joint_action"]), "endpose": endpose}


def run_client(cfg, reference, run_id, server_info):
    from websockets.sync.client import connect

    env_cfg = OmegaConf.create(OmegaConf.to_container(cfg.env, resolve=True))
    if env_cfg.get("instruction_template", None) is None:
        env_cfg.instruction_template = str(
            reference.get("instruction_template", DEFAULT_INSTRUCTION_TEMPLATE)
        )
    env = CausalRoboTwinEnv(env_cfg, run_id=run_id)
    robot_cfg = json.loads(Path(cfg.actor.model.robotwin.robot_config_path).read_text())
    converter = RobotwinPoseConverter(
        **{
            key: robot_cfg[key]
            for key in ("eef_pose_frame", "T_world_base")
            if key in robot_cfg
        }
    )
    seeds = [int(seed) for seed in cfg.causal.reset_seeds]
    episodes = int(reference.get("episodes", 1))
    if episodes < 1:
        raise ValueError("reference.episodes must be a positive integer")
    uri = reference.get("server_uri", None) or (
        f"ws://{reference.get('ascend_ip', DEFAULT_ASCEND_IP)}:"
        f"{int(reference.get('port', 8000))}"
    )
    summary = {
        "run_id": run_id,
        "server_uri": uri,
        "server": json_safe(server_info),
        "config": {
            "episodes": episodes,
            "instruction_template": env_cfg.instruction_template,
            "shift": float(reference.get("shift", 5.0)),
            "steps": int(reference.get("steps", 20)),
            "guidance": float(reference.get("guidance", 1.0)),
            "max_episode_steps": int(cfg.env.max_episode_steps),
        },
        "episodes": [],
    }
    try:
        with connect(
            uri,
            compression=None,
            max_size=None,
            ping_interval=None,
            open_timeout=60,
            # The driver environment sets http_proxy/https_proxy; websockets >=15
            # would route the ws:// handshake through it and fail with
            # "HTTP/1.0 500 handshakefailed". Connect to the Ascend node directly.
            proxy=None,
        ) as conn:
            metadata = unpackb(conn.recv())
            if metadata.get("protocol") != "cosmos-causal-action-v2":
                raise ValueError(f"Unexpected server protocol: {metadata}")
            if metadata.get("quaternion_order") != "xyzw":
                raise ValueError("Reference server requires xyzw quaternions")
            if metadata.get("eef_pose_frame") != "dataset":
                raise ValueError("Reference server requires dataset-frame EEF poses")
            if "joint" not in metadata.get("available_control_modes", []):
                raise ValueError("Reference server has no joint control mode")
            summary["server_metadata"] = json_safe(metadata)
            block = int(metadata["actions_per_block"])
            count = int(metadata["execute_steps"])
            width = int(metadata["left_arm_dof"]) + int(metadata["right_arm_dof"]) + 2
            if block != int(cfg.env.actions_per_block) or count != block:
                raise ValueError(f"Server block/count {block}/{count} differs from env")
            if width != int(cfg.env.action_dim):
                raise ValueError(f"Server action width {width} differs from env")

            def request(payload):
                conn.send(packb(payload))
                response = unpackb(conn.recv())
                if "error" in response:
                    raise RuntimeError(response["error"])
                return response

            for index in range(episodes):
                seed = seeds[index % len(seeds)]
                key = EpisodeKey(
                    0,
                    index,
                    0,
                    cfg.env.task_config.task_name,
                    seed,
                    cfg.causal.instruction,
                )
                initial = env.reset(key)
                instruction = initial.get("instruction", cfg.causal.instruction)
                qpose_tag = initial.get("qpose_tag")
                request(
                    {
                        "reset": True,
                        "prompt": instruction,
                        "control_mode": "joint",
                        "record_video": True,
                    }
                )
                payload = {
                    "obs": camera_images(initial["frames"][0]),
                    "state": converted_state(initial["native_state"], converter),
                    "observation_step": 0,
                }
                chunk = 0
                executed = 0
                success = terminated = truncated = False
                while True:
                    response = request(payload)
                    actions = np.asarray(response["action"]["joint"], dtype=np.float32)
                    if (
                        actions.shape != (block, width)
                        or not np.isfinite(actions).all()
                    ):
                        raise ValueError(
                            f"Reference server returned invalid actions {actions.shape}"
                        )
                    if int(response["execute_steps"]) != count:
                        raise ValueError("Reference server execute_steps changed")
                    feedback = env.step(
                        ActionRequest(
                            key,
                            chunk,
                            chunk * block,
                            actions.copy(),
                        )
                    )
                    executed += int(feedback.executed_steps)
                    success = bool(feedback.success)
                    terminated = bool(feedback.terminated)
                    truncated = bool(feedback.truncated)
                    if terminated or truncated:
                        break
                    if int(feedback.executed_steps) != count:
                        raise ValueError("Partial block without termination/truncation")
                    payload = {
                        "obs": [camera_images(frame) for frame in feedback.frames],
                        "state": converted_state(feedback.native_state, converter),
                        "executed_steps": count,
                        "observation_step": executed,
                    }
                    chunk += 1

                evidence = getattr(env, "evidence", None)
                observed_path = (
                    str(Path(evidence) / "observed.mp4")
                    if evidence is not None
                    else None
                )
                prediction_path = None
                prediction_meta = {}
                try:
                    finish = request({"finish_video": True})
                    if evidence is not None and "video_mp4" in finish:
                        prediction_path = Path(evidence) / "reference_prediction.mp4"
                        prediction_path.write_bytes(finish["video_mp4"])
                        prediction_meta = {
                            "prediction_video_path": str(prediction_path),
                            "prediction_video_frames": finish.get("video_frames"),
                            "prediction_video_fps": finish.get("video_fps"),
                        }
                except Exception as error:
                    prediction_meta = {
                        "prediction_error": f"{type(error).__name__}: {error}"
                    }
                bottle = None
                try:
                    bottle = [
                        float(value)
                        for value in env.task.bottle.get_functional_point(0)
                    ]
                except Exception:
                    pass
                record = {
                    "seed": seed,
                    "qpose_tag": qpose_tag,
                    "instruction": instruction,
                    "success": success,
                    "terminated": terminated,
                    "truncated": truncated,
                    "executed_steps": executed,
                    "chunks": chunk + 1,
                    "observed_video": observed_path,
                    "bottle_functional_point": bottle,
                    **prediction_meta,
                }
                summary["episodes"].append(record)
                print("[e7] " + json.dumps(json_safe(record)), flush=True)
    finally:
        try:
            env.close()
        except Exception:
            pass
    return summary


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="robotwin_causal_production",
)
def main(cfg):
    reference = cfg.get("reference", {})
    episodes = int(reference.get("episodes", 1))
    if episodes < 1:
        raise ValueError("reference.episodes must be a positive integer")
    run_id = str(
        reference.get("run_id", None)
        or f"CausalFPOReference_{time.strftime('%Y%m%d-%H%M%S')}"
    )
    server_uri = reference.get("server_uri", None)
    actor = None
    ray_initialized = False
    server_info = None
    try:
        if server_uri is None:
            os.environ.setdefault("RAY_ADDRESS", "auto")
            ray.init(address="auto", logging_level="ERROR")
            ray_initialized = True
            ascend_ip = str(reference.get("ascend_ip", DEFAULT_ASCEND_IP))
            nodes = [node for node in ray.nodes() if node["Alive"]]
            ascend = [node for node in nodes if node["NodeManagerAddress"] == ascend_ip]
            if not ascend:
                raise RuntimeError(f"Ascend Ray node {ascend_ip} is not alive")
            command = build_server_command(cfg, reference)
            log_path = reference.get("server_log", None) or (
                Path(cfg.runner.output_dir).parent
                / "reference_eval"
                / f"server_{run_id}.log"
            )
            # The checkpoint metadata was pickled by Python 3.13 and references
            # pathlib._local; the Ascend python is 3.12. A sitecustomize alias
            # lets torch DCP unpickle it without touching the checkpoint.
            alias_dir = (
                Path(cfg.runner.output_dir).parent / "reference_eval" / "pathlib_alias"
            )
            alias_dir.mkdir(parents=True, exist_ok=True)
            (alias_dir / "sitecustomize.py").write_text(
                "import pathlib, sys\n"
                'sys.modules.setdefault("pathlib._local", pathlib)\n'
            )
            actor = ReferenceServer.options(
                num_cpus=0.01,
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ascend[0]["NodeID"], soft=False
                ),
            ).remote()
            server_info = ray.get(
                actor.start.remote(
                    command,
                    str(reference.get("server_repo", DEFAULT_SERVER_REPO)),
                    str(alias_dir),
                    str(log_path),
                    int(reference.get("port", 8000)),
                    float(reference.get("server_timeout", 900)),
                    reference.get("ascend_device", None),
                )
            )
            print(
                "[e7] server: " + json.dumps(json_safe(server_info), indent=2),
                flush=True,
            )
            if not server_info.get("ok"):
                raise RuntimeError(
                    "Reference server failed to start. Log tail:\n"
                    + str(server_info.get("log_tail", ""))
                )
        summary = run_client(cfg, reference, run_id, server_info)
        summary["server"] = json_safe(server_info)
        output = Path(
            reference.get("summary_dir", None)
            or Path(cfg.runner.output_dir).parent / "reference_eval"
        )
        output.mkdir(parents=True, exist_ok=True)
        summary_path = output / f"{run_id}.json"
        summary_path.write_text(json.dumps(json_safe(summary), indent=2))
        print(f"[e7] summary: {summary_path}", flush=True)
    finally:
        if actor is not None:
            try:
                print(
                    "[e7] stop server: "
                    + json.dumps(json_safe(ray.get(actor.stop.remote()))),
                    flush=True,
                )
            except Exception:
                pass
            try:
                ray.kill(actor, no_restart=True)
            except Exception:
                pass
        if ray_initialized:
            try:
                ray.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    main()
