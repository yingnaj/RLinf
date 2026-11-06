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
from collections.abc import Mapping
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
        # E5a diagnostics: record the bottle functional point every control step
        # so "grasped but did not lift" can be separated from "never touched".
        self.diagnostics = bool(self.cfg.get("diagnostics", False))
        # Reference-client parity: the verified RoboTwin service feeds RoboTwin's
        # drive-target ``joint_action`` (command state), while the RL recipe has
        # so far fed measured qpos. Keep measured as the default and allow the
        # command source for controlled A/B runs.
        self.state_source = str(
            self.cfg.get("state_source", "measured") or "measured"
        ).lower()
        if self.state_source not in ("measured", "command"):
            raise ValueError("env.state_source must be 'measured' or 'command'")
        # Optional per-reset-seed instruction overrides, used to replay the exact
        # prompts recorded by a reference run.
        self.instruction_map = {}
        instruction_map_path = self.cfg.get("instruction_map_path", None)
        if instruction_map_path:
            self.instruction_map = {
                str(seed): str(text)
                for seed, text in json.loads(
                    Path(instruction_map_path).read_text()
                ).items()
            }

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

    def _effective_instruction(self, episode):
        """Return the instruction sent to the policy and the task's arm tag.

        ``instruction_template`` may contain ``{arm}``; the arm is derived from
        the task's ``qpose_tag`` (0 = left, 1 = right), matching RoboTwin's
        ``play_once`` arm selection. Without a template the configured episode
        instruction is used unchanged.
        """
        qpose_tag = getattr(self.task, "qpose_tag", None)
        if qpose_tag is not None:
            qpose_tag = int(qpose_tag)
        override = self.instruction_map.get(str(episode.reset_seed))
        if override is not None:
            return override, qpose_tag
        template = self.cfg.get("instruction_template", None)
        if template is None:
            return episode.instruction, qpose_tag
        if qpose_tag not in (0, 1):
            raise ValueError("instruction_template requires task.qpose_tag in {0, 1}")
        arm = "left" if qpose_tag == 0 else "right"
        return str(template).format(arm=arm), qpose_tag

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
        instruction, qpose_tag = self._effective_instruction(episode)
        self.task.set_instruction(instruction=instruction)
        model_id = getattr(self.task, "model_id", None)
        if model_id is not None:
            model_id = int(model_id)
        self.episode, self.elapsed, self.chunk, self.done = episode, 0, 0, False
        self.cached_request = self.cached_feedback = None
        image, state, native_state = self.observe()
        if self.cfg.get("record_video", False) or self.cfg.get(
            "record_predicted_video", False
        ):
            self.evidence = (
                Path(self.cfg.evidence_dir)
                / self.run_id
                / f"v{episode.version}_g{episode.group}_m{episode.member}"
            )
            self.evidence.mkdir(parents=True, exist_ok=False)
            record = dict(vars(episode))
            record["instruction"] = instruction
            record["qpose_tag"] = qpose_tag
            record["model_id"] = model_id
            record["state_source"] = self.state_source
            (self.evidence / "episode.json").write_text(json.dumps(record, indent=2))
        if self.cfg.get("record_video", False):
            import cv2

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
        return {
            "episode": episode,
            "frames": (image,),
            "state": state,
            "native_state": native_state,
            "instruction": instruction,
            "qpose_tag": qpose_tag,
            "model_id": model_id,
            # Initial grasp target for the E5c contact-triggered gripper close.
            "bottle_point": self._bottle_point(),
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
        if self.state_source == "command":
            left = self._command_joints(raw, "left")
            right = self._command_joints(raw, "right")
        else:
            left = self._measured_joints("left")
            right = self._measured_joints("right")
        state = np.asarray(left + right, dtype=np.float32)
        if state.shape != (self.cfg.action_dim,) or not np.isfinite(state).all():
            raise ValueError(
                "Joint state getter does not match configured command width"
            )
        return images, state, self._native_state(raw, left, right)

    def _command_joints(self, raw, side):
        """Return RoboTwin's drive-target ``joint_action`` rows for one arm.

        This mirrors the reference client's ``state_source=command``: arm values
        are the last commanded drive targets and the gripper is RoboTwin's cached
        gripper command (not the measured finger joint).
        """
        joint = raw.get("joint_action")
        if not isinstance(joint, Mapping):
            raise ValueError("RoboTwin observation must contain joint_action")
        arm = np.asarray(joint.get(f"{side}_arm"), dtype=np.float32).reshape(-1)
        gripper = np.asarray(joint.get(f"{side}_gripper"), dtype=np.float32).reshape(-1)
        expected = self.cfg.action_dim // 2 - 1
        if arm.size != expected or gripper.size != 1:
            raise ValueError(
                f"{side} command state has {arm.size}+{gripper.size} values, "
                f"expected {expected}+1"
            )
        return [*arm.tolist(), float(gripper[0])]

    def _native_state(self, raw, left, right):
        """Native joint+EEF state for the model; None when endpose is disabled.

        The model checkpoint is trained on the EEF-enriched dataset, whose state
        contains both measured joints and EEF poses. ``data_type.endpose`` must
        stay enabled for a native-format adapter; otherwise the worker raises.
        """
        endpose = raw.get("endpose")
        # RoboTwin keeps an empty ``endpose`` mapping when data_type.endpose is
        # off, which is the documented joint_vector recipe.
        if not endpose:
            return None
        if not isinstance(endpose, Mapping):
            raise ValueError("RoboTwin endpose observation must be a mapping")
        poses = {}
        for side in ("left", "right"):
            key = f"{side}_endpose"
            if key not in endpose:
                raise ValueError(
                    f"Missing {key}; enable env.task_config.data_type.endpose"
                )
            # Detach from the renderer-owned observation buffer.
            pose = np.asarray(endpose[key], dtype=np.float32).copy()
            if pose.shape != (7,) or not np.isfinite(pose).all():
                raise ValueError(f"{key} must be a finite 7-D pose")
            poses[key] = pose
        # Field groups are ordered left arm, right arm, left gripper, right
        # gripper. Consumers read these by name; the 55-D template layout is
        # fixed and does not depend on dictionary order.
        return {
            "joint_action": {
                "left_arm": np.asarray(left[:-1], dtype=np.float32),
                "right_arm": np.asarray(right[:-1], dtype=np.float32),
                "left_gripper": np.float32(left[-1]),
                "right_gripper": np.float32(right[-1]),
            },
            "endpose": poses,
        }

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

    def _bottle_point(self):
        """Return the task bottle functional point, or None when unavailable."""
        bottle = getattr(self.task, "bottle", None)
        if bottle is None:
            return None
        try:
            point = np.asarray(bottle.get_functional_point(0), dtype=np.float64)
        except Exception:
            return None
        return point.reshape(-1).tolist() if np.isfinite(point).all() else None

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
                or not np.array_equal(
                    request.predicted_video, self.cached_request.predicted_video
                )
                or request.predicted_video_fps
                != self.cached_request.predicted_video_fps
            ):
                raise ValueError("Conflicting duplicate action request")
            return self.cached_feedback
        if (
            self.done
            or request.chunk != self.chunk
            or request.observation_step != self.elapsed
        ):
            raise ValueError("Stale/out-of-order action, or episode already finished")
        if self.cfg.get("record_predicted_video", False):
            if request.predicted_video is None:
                raise ValueError(
                    "Predicted video recording requires frames in every action request"
                )
            self._save_predicted_video(request)
        started = time.perf_counter()
        frames, rewards, times = [], [], []
        bottle_points = []
        success = terminated = truncated = False
        for action in request.actions:
            if self.task.eval_success or self.elapsed >= self.task.step_lim:
                raise RuntimeError("Environment ended without terminal feedback")
            before = self.task.take_action_cnt
            self.task.take_action(action, action_type="qpos")
            if self.task.take_action_cnt != before + 1:
                raise RuntimeError("RoboTwin did not execute exactly one control")
            image, state, native_state = self.observe()
            if self.diagnostics:
                bottle_points.append(self._bottle_point())
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
            native_state=native_state,
        )
        feedback.validate(self.cfg.actions_per_block, self.cfg.fps)
        self.chunk += 1
        self.done = terminated or truncated
        self.cached_request = ActionRequest(
            request.episode,
            request.chunk,
            request.observation_step,
            request.actions.copy(),
            predicted_video=(
                None
                if request.predicted_video is None
                else request.predicted_video.copy()
            ),
            predicted_video_fps=request.predicted_video_fps,
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
                "state_source": self.state_source,
                "measured_state": state.tolist(),
                "measured_endpose": None
                if native_state is None
                else {
                    key: value.tolist()
                    for key, value in native_state["endpose"].items()
                },
                "rewards": rewards,
                "control_timestamps": times,
                "terminated": terminated,
                "truncated": truncated,
                "success": success,
                "execution_seconds": time.perf_counter() - started,
            }
            if self.diagnostics:
                record["diagnostics"] = {
                    "qpose_tag": getattr(self.task, "qpose_tag", None),
                    "bottle_functional_point": bottle_points[-1]
                    if bottle_points
                    else None,
                    "bottle_trajectory": bottle_points,
                }
            with (self.evidence / "feedback.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
        if self.done and self.video is not None:
            self.video.release()
            self.video = None
        return feedback

    def _save_predicted_video(self, request):
        import cv2

        video = request.predicted_video
        writer = cv2.VideoWriter(
            str(self.evidence / f"chunk_{request.chunk:04d}_predicted.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            request.predicted_video_fps,
            (video.shape[2], video.shape[1]),
        )
        try:
            if not writer.isOpened():
                raise RuntimeError("Could not open predicted video encoder")
            # Retain the full prediction even if environment execution terminates early.
            for frame in video:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()

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
