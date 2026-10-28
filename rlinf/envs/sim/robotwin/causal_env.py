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

"""Strict-seed native RoboTwin execution with measured joint feedback."""

import importlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np

from rlinf.models.embodiment.cosmos3.online_fpo import ActionRequest, ExecutionFeedback


class CausalRoboTwinEnv:
    """One native task per session, without automatic reset or padded frames.

    The joint-only first version reads articulation qpos rather than
    RoboTwin's command-target observation vector. Failed resets propagate;
    substituting a different seed would invalidate grouped advantages.
    """

    def __init__(self, cfg, *, task_factory=None, run_id=None):
        import uuid

        self.run_id = run_id or uuid.uuid4().hex
        self.cfg = cfg
        self.task_factory = task_factory
        self.task = None
        self.episode = None
        self.cached_request = self.cached_feedback = None
        self.elapsed = self.chunk = 0
        self.done = True
        self.video = None
        self.evidence = None

    def _create_task(self):
        import yaml

        os.environ["ASSETS_PATH"] = self.cfg.assets_path
        from envs._GLOBAL_CONFIGS import CONFIGS_PATH
        from omegaconf import OmegaConf

        args = OmegaConf.to_container(self.cfg.task_config, resolve=True)
        with open(Path(CONFIGS_PATH) / "_embodiment_config.yml") as stream:
            profiles = yaml.safe_load(stream)
        embodiments = args["embodiment"]
        if len(embodiments) not in (1, 3):
            raise ValueError("RoboTwin embodiment must have one or three entries")
        for side, index in (("left", 0), ("right", 0 if len(embodiments) == 1 else 1)):
            folder = (
                Path(self.cfg.assets_path) / profiles[embodiments[index]]["file_path"]
            )
            args[f"{side}_robot_file"] = str(folder)
            with open(folder / "config.yml") as stream:
                args[f"{side}_embodiment_config"] = yaml.safe_load(stream)
        args["dual_arm_embodied"] = len(embodiments) == 1
        if len(embodiments) == 3:
            args["embodiment_dis"] = embodiments[2]
        with open(Path(CONFIGS_PATH) / "_camera_config.yml") as stream:
            camera = yaml.safe_load(stream)[args["camera"]["head_camera_type"]]
        args.update(
            head_camera_h=camera["h"],
            head_camera_w=camera["w"],
            eval_mode=True,
            eval_video_log=False,
            render_freq=0,
        )
        name = args["task_name"]
        task = getattr(importlib.import_module(f"envs.{name}"), name)()
        return task, args

    def reset(self, episode):
        if (
            episode.task != self.cfg.task_config.task_name
            or not episode.instruction.strip()
        ):
            raise ValueError(
                "Episode task/instruction differs from environment contract"
            )
        self.close()
        random.seed(episode.reset_seed)
        np.random.seed(episode.reset_seed)
        if self.task_factory is None:
            self.task, args = self._create_task()
        else:
            self.task, args = self.task_factory(), {}
        self.task.setup_demo(
            now_ep_num=episode.reset_seed, seed=episode.reset_seed, is_test=True, **args
        )
        self.task.step_lim = min(
            int(self.task.step_lim), int(self.cfg.max_episode_steps)
        )
        self.task.set_instruction(instruction=episode.instruction)
        self.episode, self.elapsed, self.chunk, self.done = episode, 0, 0, False
        self.cached_request = self.cached_feedback = None
        image, state = self.observe()
        if self.cfg.get("record_video", False):
            import cv2

            self.evidence = (
                Path(self.cfg.evidence_dir)
                / self.run_id
                / f"v{episode.version}_g{episode.group}_m{episode.member}"
            )
            self.evidence.mkdir(parents=True, exist_ok=False)
            canvas = self._video_frame(image)
            self.video = cv2.VideoWriter(
                str(self.evidence / "observed.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.cfg.fps,
                (canvas.shape[1], canvas.shape[0]),
            )
            if not self.video.isOpened():
                raise RuntimeError("Could not open observation video encoder")
            self.video.write(canvas)
            (self.evidence / "episode.json").write_text(
                json.dumps(episode.__dict__, indent=2)
            )
        return {
            "episode": episode,
            "frames": (image,),
            "state": state,
            "observation_step": 0,
        }

    def observe(self):
        raw = self.task.get_obs()
        images = {}
        for key in self.cfg.camera_names:
            image = np.asarray(raw["observation"][key]["rgb"])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"Invalid RGB observation from {key}")
            images[key] = image.copy()
        left = self._measured_joints("left")
        right = self._measured_joints("right")
        state = np.asarray(left + right, dtype=np.float32)
        if state.shape != (self.cfg.action_dim,) or not np.isfinite(state).all():
            raise ValueError(
                "Measured joint getter does not match configured command width"
            )
        return images, state

    def _measured_joints(self, side):
        robot = self.task.robot
        entity = getattr(robot, f"{side}_entity")
        active = list(entity.get_active_joints())
        qpos = entity.get_qpos()
        joints = getattr(robot, f"{side}_arm_joints")
        values = [float(qpos[active.index(joint)]) for joint in joints]
        # RoboTwin's "real_jointState" getters still append the cached gripper
        # command. Invert set_gripper's mapping using the actual base-joint qpos.
        joint, multiplier, bias = getattr(robot, f"{side}_gripper")[0]
        closed, opened = getattr(robot, f"{side}_gripper_scale")
        if multiplier == 0 or opened == closed:
            raise ValueError("Cannot invert measured gripper joint mapping")
        physical = (float(qpos[active.index(joint)]) - bias) / multiplier
        values.append((physical - closed) / (opened - closed))
        return values

    def step(self, request: ActionRequest):
        request.validate(self.cfg.actions_per_block, self.cfg.action_dim)
        if request.episode != self.episode:
            raise ValueError("Action belongs to a different episode/version")
        if (
            self.cached_request is not None
            and request.chunk == self.cached_request.chunk
        ):
            if (
                request.observation_step != self.cached_request.observation_step
                or not np.array_equal(request.actions, self.cached_request.actions)
            ):
                raise ValueError("Conflicting duplicate action request")
            return self.cached_feedback
        if (
            self.done
            or request.chunk != self.chunk
            or request.observation_step != self.elapsed
        ):
            raise ValueError("Stale/out-of-order action, or episode already finished")
        started = time.perf_counter()
        frames, rewards, times = [], [], []
        success = terminated = truncated = False
        for action in request.actions:
            if self.task.eval_success or self.elapsed >= self.task.step_lim:
                raise RuntimeError("Environment ended without terminal feedback")
            before = self.task.take_action_cnt
            self.task.take_action(action, action_type="qpos")
            if self.task.take_action_cnt != before + 1:
                raise RuntimeError("RoboTwin did not execute exactly one control")
            image, state = self.observe()
            self.elapsed += 1
            success = bool(self.task.eval_success)
            terminated = success
            truncated = not success and self.elapsed >= self.task.step_lim
            frames.append(image)
            if self.video is not None:
                self.video.write(self._video_frame(image))
            rewards.append(float(success))
            times.append(self.elapsed / self.cfg.fps)
            if terminated or truncated:
                break
        feedback = ExecutionFeedback(
            self.episode,
            self.chunk,
            request.observation_step,
            tuple(frames),
            state,
            tuple(rewards),
            terminated,
            truncated,
            success,
            tuple(times),
        )
        feedback.validate(self.cfg.actions_per_block, self.cfg.fps)
        self.chunk += 1
        self.done = terminated or truncated
        self.cached_request = ActionRequest(
            request.episode,
            request.chunk,
            request.observation_step,
            request.actions.copy(),
        )
        self.cached_feedback = feedback
        if self.evidence is not None:
            np.save(
                self.evidence / f"chunk_{request.chunk:04d}_commands.npy",
                request.actions,
            )
            record = {
                "chunk": request.chunk,
                "observation_step": request.observation_step,
                "executed_steps": feedback.executed_steps,
                "state_source": "measured",
                "measured_state": state.tolist(),
                "rewards": rewards,
                "control_timestamps": times,
                "terminated": terminated,
                "truncated": truncated,
                "success": success,
                "execution_seconds": time.perf_counter() - started,
            }
            with (self.evidence / "feedback.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
        if self.done and self.video is not None:
            self.video.release()
            self.video = None
        return feedback

    @staticmethod
    def _video_frame(images):
        import cv2

        head = images["head_camera"]
        h, w = head.shape[:2]
        left = cv2.resize(images["left_camera"], (w // 2, h // 2))
        right = cv2.resize(images["right_camera"], (w - w // 2, h // 2))
        return cv2.cvtColor(
            np.concatenate((head, np.concatenate((left, right), axis=1)), axis=0),
            cv2.COLOR_RGB2BGR,
        )

    def close(self):
        if self.video is not None:
            self.video.release()
            self.video = None
        self.evidence = None
        if self.task is not None:
            self.task.close_env(clear_cache=True)
            self.task = None
