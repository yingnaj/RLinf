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

"""Causal Cosmos configuration and real model-side replay adapter tests."""

import io
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment.cosmos3.cosmos3_config import (
    load_cosmos3_experiment,
    validate_cosmos3_variant,
)
from rlinf.models.embodiment.cosmos3.fpo import (
    CausalFPOBatch,
    fpo_update_loss,
    grouped_suffix_advantages,
    recompute_old_cfm,
)


def causal_config(**overrides):
    return OmegaConf.create(
        {
            "cosmos_variant": "causal_action",
            "training_mode": "fpo",
            "cosmos_sft_toml": "/tmp/recipe.toml",
            "ema_enabled": False,
            "disable_cosmos_parallelism": True,
            "keys_to_skip_loading": [],
            "keys_to_select": [],
            **overrides,
        }
    )


@pytest.mark.parametrize(
    "override",
    [
        {"cosmos_variant": "unknown"},
        {"cosmos_variant": "standard"},
        {"cosmos_sft_toml": None},
        {"keys_to_skip_loading": ["action2llm"]},
        {"disable_cosmos_parallelism": False},
        {"ema_enabled": True},
        {"keys_to_select": ["moe_gen"]},
        {"training_mode": "sft", "robotwin": {}},
    ],
)
def test_causal_config_rejects_incompatible_sft_defaults(override):
    with pytest.raises(ValueError):
        validate_cosmos3_variant(causal_config(**override))


def test_standard_config_keeps_existing_defaults():
    validate_cosmos3_variant(OmegaConf.create({}))
    validate_cosmos3_variant(causal_config())


@pytest.fixture
def robotwin_adapter():
    pytest.importorskip("cosmos_framework.utils.robotwin")
    from cosmos_framework.data.generator.action.action_state_template import (
        ActionStateTemplate55,
    )
    from cosmos_framework.data.generator.action.block_state import (
        BlockStatistics,
        Quantiles,
    )
    from cosmos_framework.utils.robotwin import RobotwinAdapter

    from rlinf.models.embodiment.cosmos3.robotwin_adapter import RobotwinCausalAdapter

    mask = torch.zeros(55, dtype=torch.bool)
    mask[:6] = mask[7:13] = mask[28:30] = True
    low, high = torch.full((55,), -2.0), torch.full((55,), 2.0)
    low[28:30], high[28:30] = 0, 1
    stats = BlockStatistics(
        Quantiles(torch.full((55,), -10.0), torch.full((55,), 10.0), mask),
        Quantiles(low, high, mask),
        {"sources": [{"state_mask": mask.tolist(), "action_mask": mask.tolist()}]},
    )
    native = RobotwinAdapter(ActionStateTemplate55(), state_mask=mask, action_mask=mask)
    return RobotwinCausalAdapter(
        native, stats, control_mode="joint", state_format="joint_vector"
    )


def test_robotwin_joint_endpoint_decoding_keeps_full_block_and_replay(robotwin_adapter):
    adapter = robotwin_adapter
    raw = torch.arange(28, dtype=torch.float32).reshape(2, 14) / 10
    before = raw.clone()
    anchors = adapter.pack_states(raw, state_source="measured")
    torch.testing.assert_close(anchors[:, :6], raw[:, :6])
    torch.testing.assert_close(anchors[:, 7:13], raw[:, 7:13])
    torch.testing.assert_close(anchors[:, 28], raw[:, 6])
    torch.testing.assert_close(anchors[:, 29], raw[:, 13])
    assert not anchors[:, [6, 13, 30, 54]].any()
    # Zero normalized joints denote zero deltas; grippers denote absolute 0.5.
    endpoint = torch.zeros(2, 32, 55, requires_grad=True)
    snapshot = endpoint.detach().clone()
    commands = adapter.decode_actions(endpoint, anchors)
    assert commands.shape == (2, 32, 14) and commands.device.type == "cpu"
    assert not commands.requires_grad
    expected = raw[:, None].expand(-1, 32, -1).clone()
    expected[:, :, [6, 13]] = 0.5
    torch.testing.assert_close(commands, expected)
    torch.testing.assert_close(endpoint, snapshot)
    torch.testing.assert_close(raw, before)
    assert adapter.action_type == "qpos" and adapter.action_dim == 14


def test_robotwin_rejects_commands_wrong_width_and_nonfinite_values(robotwin_adapter):
    adapter = robotwin_adapter
    with pytest.raises(ValueError, match="measured"):
        adapter.pack_states(torch.zeros(1, 14), state_source="command")
    with pytest.raises(ValueError, match="joint states"):
        adapter.pack_states(torch.zeros(1, 55), state_source="measured")
    raw = torch.zeros(1, 14)
    raw[0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        adapter.pack_states(raw, state_source="measured")
    endpoint = torch.zeros(1, 32, 55)
    endpoint[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        adapter.decode_actions(endpoint, torch.zeros(1, 55))
    with pytest.raises(ValueError, match="anchors"):
        adapter.decode_actions(torch.zeros(2, 32, 55), torch.zeros(1, 55))


def test_robotwin_config_attaches_to_policy_and_requires_absolute_assets(
    policy, robotwin_adapter, tmp_path
):
    path = tmp_path / "stats.json"
    robotwin_adapter.statistics.save(path)
    config = {
        "statistics_path": str(path),
        "control_mode": "joint",
        "state_format": "joint_vector",
    }
    policy.configure_robotwin(config)
    assert policy.robotwin_adapter.action_dim == 14
    with pytest.raises(ValueError, match="absolute path"):
        policy.configure_robotwin(dict(config, statistics_path="stats.json"))
    with pytest.raises(ValueError, match="control_mode"):
        policy.configure_robotwin(dict(config, control_mode="eef"))


def test_robotwin_eef_decoding_applies_anchor_and_base_transform():
    pytest.importorskip("cosmos_framework.utils.robotwin")
    from cosmos_framework.data.generator.action.action_state_template import (
        ActionStateTemplate55,
    )
    from cosmos_framework.data.generator.action.block_state import (
        BlockStatistics,
        Quantiles,
    )
    from cosmos_framework.utils.robotwin import RobotwinAdapter

    from rlinf.models.embodiment.cosmos3.robotwin_adapter import RobotwinCausalAdapter

    mask = torch.zeros(55, dtype=torch.bool)
    mask[14:30] = True
    native = RobotwinAdapter(
        ActionStateTemplate55(),
        state_mask=mask,
        action_mask=mask,
        eef_pose_frame="base",
        T_world_base={"position": [10, 20, 30], "quaternion_wxyz": [1, 0, 0, 0]},
    )
    quantiles = Quantiles(torch.full((55,), -2.0), torch.full((55,), 2.0), mask)
    stats = BlockStatistics(quantiles, quantiles)
    with pytest.raises(ValueError, match="EEF state"):
        RobotwinCausalAdapter(
            native, stats, control_mode="eef", state_format="joint_vector"
        )
    adapter = RobotwinCausalAdapter(
        native, stats, control_mode="eef", state_format="native"
    )
    state = {
        "endpose": {
            "left_endpose": [11, 22, 33, 1, 0, 0, 0],
            "right_endpose": [14, 25, 36, 1, 0, 0, 0],
            "left_gripper": 0.1,
            "right_gripper": 0.9,
        }
    }
    anchor = adapter.pack_states([state], state_source="measured")
    normalized = torch.zeros(1, 32, 55)
    normalized[..., [20, 27]] = 0.5  # Identity delta quaternion after denormalization.
    normalized[..., [14, 21]] = 0.25  # +0.5m x in the base frame.
    normalized[..., 28:30] = 0.2  # Absolute 0.4, independent of the state gripper.
    command = adapter.decode_actions(normalized, anchor)
    torch.testing.assert_close(
        command[0, 0],
        torch.tensor(
            [
                11.5,
                22,
                33,
                1,
                0,
                0,
                0,
                0.4,
                14.5,
                25,
                36,
                1,
                0,
                0,
                0,
                0.4,
            ]
        ),
    )
    assert command.shape == (1, 32, 16) and adapter.action_type == "ee"


@pytest.fixture
def policy(monkeypatch):
    native = pytest.importorskip(
        "cosmos_framework.model.generator.algorithm.rl.causal_fpo_test"
    )
    from cosmos_framework.model.generator.algorithm.rl.causal_fpo import (
        configure_fpo_trainable_parameters,
    )

    from rlinf.models.embodiment.cosmos3.causal_policy import CausalCosmos3Policy

    class TinyCosmos(torch.nn.Module, native.TinyModel):
        def __init__(self):
            torch.nn.Module.__init__(self)
            native.TinyModel.__init__(self)

    monkeypatch.setattr(native.PackedSequence, "to_cuda", lambda self: None)
    torch.manual_seed(42)
    model = TinyCosmos()
    configure_fpo_trainable_parameters(model.net)
    return CausalCosmos3Policy(model, ema_enabled=False).eval()


def sample(policy):
    return policy.sample_fpo_batch(
        [{}, {}],
        policy_version="0",
        seeds=[42, 43],
        probe_seeds=[13, 14],
        num_probes=2,
    )[1]


def test_native_unipc_replay_survives_transport_and_preserves_forward(policy):
    examples = sample(policy)
    batch = CausalFPOBatch(tuple(examples), torch.tensor([1.0, -1.0]), "0")
    batch.validate("0")
    stream = io.BytesIO()
    torch.save(batch, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=False)
    restored.validate("0")
    calls = []
    hook = policy.register_forward_hook(lambda *args: calls.append(True))
    with torch.no_grad():
        old = recompute_old_cfm(policy, restored.examples, micro_batch_size=1)
    new = policy(fpo_examples=restored.examples)["cfm"]
    hook.remove()
    assert len(calls) == 3
    assert new.requires_grad and not old.requires_grad
    torch.testing.assert_close(old, new)
    assert policy.omni.sampler.tensor_kwargs["device"] == torch.device("cpu")
    with pytest.raises(ValueError, match="version"):
        restored.validate("1")
    with pytest.raises(ValueError, match="detached"):
        replace(restored, advantages=restored.advantages.requires_grad_()).validate()


def test_fpo_adapter_matches_native_loss_gradients_and_updates_only_action(policy):
    from cosmos_framework.model.generator.algorithm.rl.causal_fpo import causal_fpo_loss
    from cosmos_framework.model.generator.algorithm.rl.fpo import FPOConfig

    examples = sample(policy)
    old = recompute_old_cfm(policy, examples, micro_batch_size=1)
    advantages = torch.tensor([1.0, -1.0])
    config = FPOConfig()
    loss, metrics = fpo_update_loss(policy, examples, old, advantages, config=config)
    assert metrics["fpo/surrogate_ratio"] == 1
    loss.backward()
    grads = {
        n: p.grad.clone() for n, p in policy.named_parameters() if p.grad is not None
    }
    assert grads and any(g.abs().sum() > 0 for g in grads.values())
    policy.zero_grad()
    native_loss, _ = causal_fpo_loss(policy.omni, examples, advantages, config=config)
    native_loss.backward()
    torch.testing.assert_close(loss, native_loss)
    for name, p in policy.named_parameters():
        if name in grads:
            torch.testing.assert_close(grads[name], p.grad)
    before = {n: p.detach().clone() for n, p in policy.named_parameters()}
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad], lr=0.01
    )
    optimizer.step()
    changed = []
    for name, p in policy.named_parameters():
        if not p.requires_grad:
            torch.testing.assert_close(p, before[name], rtol=0, atol=0)
        elif not torch.equal(p, before[name]):
            changed.append(name)
    assert changed


def test_fpo_forward_rejects_ppo_and_training_mode(policy):
    examples = sample(policy)
    with pytest.raises(ValueError, match="not logprobs"):
        policy(fpo_examples=examples, compute_logprobs=True)
    policy.train()
    with pytest.raises(ValueError, match="eval"):
        policy(fpo_examples=examples)


def test_suffix_groups_use_member_ids_and_ignore_padding():
    pytest.importorskip("cosmos_framework.model.generator.algorithm.rl.fpo")
    rewards = torch.tensor([[1.0, 100.0], [0.0, 2.0], [0.0, 0.0], [2.0, 0.0]])
    valid = torch.tensor([[True, False], [True, True], [True, True], [True, True]])
    discounts = torch.full_like(rewards, 0.5)
    groups = torch.tensor([1, 2, 1, 2])
    members = torch.tensor([1, 0, 0, 1])
    adv = grouped_suffix_advantages(
        rewards, discounts, valid, group_ids=groups, member_ids=members, group_size=2
    )
    torch.testing.assert_close(
        adv,
        torch.tensor([[1.0, 0.0], [-1.0, 1.0], [-1.0, 0.0], [1.0, -1.0]]),
        atol=3e-6,
        rtol=0,
    )
    with pytest.raises(ValueError, match="every member"):
        grouped_suffix_advantages(
            rewards,
            discounts,
            valid,
            group_ids=groups,
            member_ids=torch.zeros(4, dtype=torch.long),
            group_size=2,
        )


def test_causal_toml_resolves_model_and_data_from_one_source(tmp_path):
    pytest.importorskip("cosmos_framework.configs.toml_config.sft_config")
    path = tmp_path / "causal.toml"
    path.write_text(
        '[job]\ntask="vfm"\nexperiment="action_causal_midtrain_edge"\n'
        "[action]\nactions_per_block=16\nvideo_stride=4\n"
    )
    cfg = causal_config(cosmos_sft_toml=str(path))
    resolved = load_cosmos3_experiment(cfg)
    assert resolved.data_setting.action.actions_per_block == 16
    assert resolved.model.config.max_action_dim == 55
    cfg.cosmos_sft_toml = "relative.toml"
    with pytest.raises(ValueError, match="absolute"):
        load_cosmos3_experiment(cfg)


def test_standard_recipe_still_resolves_without_causal_configuration(
    monkeypatch, tmp_path
):
    pytest.importorskip("cosmos_framework.configs.base.config")
    monkeypatch.setenv("LIBERO_ROOT", str(tmp_path))
    resolved = load_cosmos3_experiment(OmegaConf.create({}))
    target = resolved.model._target_
    name = target if isinstance(target, str) else target.__name__
    assert name.split(".")[-1] == "OmniMoTModel"


@pytest.mark.parametrize("zero_advantages", [False, True])
def test_actor_update_matches_full_batch_and_consumes_version(policy, zero_advantages):
    from cosmos_framework.model.generator.algorithm.rl.fpo import FPOConfig

    from rlinf.workers.actor.fsdp_fpo_actor_worker import EmbodiedFPOFSDPActor

    examples = tuple(sample(policy))
    advantages = torch.zeros(2) if zero_advantages else torch.tensor([1.0, -1.0])
    batch = CausalFPOBatch(examples, advantages, "0")
    reference = deepcopy(policy)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3)
    old = recompute_old_cfm(reference, examples, micro_batch_size=1)
    if not zero_advantages:
        for _ in range(2):
            reference_optimizer.zero_grad()
            loss, _ = fpo_update_loss(
                reference, examples, old, advantages, config=FPOConfig()
            )
            loss.backward()
            reference_optimizer.step()

    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    steps = []
    forwards = []
    hook = policy.register_forward_hook(lambda *args: forwards.append(len(steps)))

    def optimizer_step():
        steps.append(1)
        optimizer.step()
        return 0.0, [1e-3]

    # Exercise the production update loop with local optimizer/context objects;
    # Ray scheduling, device offload and FSDP collectives need separate acceptance.
    actor = SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "algorithm": {"fpo": {}},
                "actor": {"micro_batch_size": 1, "global_batch_size": 2, "seed": 42},
            }
        ),
        version=0,
        enable_sft_co_train=False,
        _world_size=1,
        _rank=0,
        update_epoch=2,
        gradient_accumulation=2,
        device=torch.device("cpu"),
        is_weight_offloaded=False,
        is_optimizer_offloaded=False,
        model=policy,
        amp_context=nullcontext(),
        optimizer=optimizer,
        grad_scaler=torch.amp.GradScaler("cpu", enabled=False),
        before_micro_batch=lambda *args, **kwargs: nullcontext(),
        optimizer_step=optimizer_step,
        lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0),
    )
    EmbodiedFPOFSDPActor.set_fpo_batch(actor, batch)
    with pytest.raises(RuntimeError, match="pending"):
        EmbodiedFPOFSDPActor.set_fpo_batch(actor, batch)
    metrics = EmbodiedFPOFSDPActor.run_training(actor)
    hook.remove()
    if zero_advantages:
        assert metrics["fpo/skipped_zero_advantages"] == 1
        assert not steps and not forwards
    else:
        assert len(steps) == 2
        # Two old-score forwards precede all four current-score forwards.
        assert forwards == [0, 0, 0, 0, 1, 1]
    for actual, expected in zip(
        policy.parameters(), reference.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected)
    with pytest.raises(RuntimeError, match="already consumed"):
        EmbodiedFPOFSDPActor.set_fpo_batch(actor, batch)


def test_online_batch_uses_real_history_and_fresh_anchor(robotwin_adapter):
    import numpy as np
    from cosmos_framework.data.generator.action.online_batch import (
        ObservedCausalBatchBuilder,
    )

    adapter = robotwin_adapter
    builder = ObservedCausalBatchBuilder(
        template=adapter.native.template,
        statistics=adapter.statistics,
        state_mask=adapter.native.state_mask,
        action_mask=adapter.native.action_mask,
        resolution="480",
    )
    frames = [
        {
            key: np.full((8, 8, 3), i, dtype=np.uint8)
            for key in ("head_camera", "left_camera", "right_camera")
        }
        for i in range(65)
    ]
    state = torch.zeros(55)
    state[0] = 0.7
    state.requires_grad_(True)
    batch = builder(frames=frames, state=state, prompt="Adjust the bottle.")
    plan = batch["sequence_plan"][0]
    metadata = plan.causal_action_metadata
    assert isinstance(batch["ai_caption"][0], str)
    assert batch["causal_action_current_block"] == 1
    assert plan.condition_frame_indexes_vision == [0, 1, 2]
    assert batch["video"][0].shape[2] == 17
    assert batch["video"][0][0, 0, 0, 0, 0] == 32
    assert batch["video"][0][0, 0, 8, 0, 0] == 64
    assert not batch["video"][0][:, :, 9:].any()
    assert torch.equal(metadata.anchors[-1], state)
    assert not metadata.anchors.requires_grad
    assert not metadata.state_mask[0].any()
    assert metadata.state_mask[-1].any()
    with pytest.raises(ValueError, match="complete"):
        builder(frames=frames[:17], state=state, prompt="Adjust the bottle.")
    invalid_state = state.clone()
    invalid_state[adapter.native.state_mask] = float("nan")
    with pytest.raises(ValueError, match="finite measured"):
        builder(frames=frames[:1], state=invalid_state, prompt="Adjust the bottle.")
    first = builder(frames=frames[:1], state=state, prompt="Adjust the bottle.")
    assert first["sequence_plan"][0].condition_frame_indexes_vision == [0]
    assert first["causal_action_current_block"] == 0
    for history_blocks in (True, 1.5, 0):
        with pytest.raises(ValueError, match="positive integer"):
            ObservedCausalBatchBuilder(
                template=adapter.native.template,
                statistics=adapter.statistics,
                state_mask=adapter.native.state_mask,
                action_mask=adapter.native.action_mask,
                history_blocks=history_blocks,
            )


def make_online_episode(
    example, member=0, *, version=0, success=True, seed=42, group=0, chunks=1
):
    import numpy as np

    from rlinf.models.embodiment.cosmos3.online_fpo import (
        CollectedEpisode,
        EpisodeKey,
        ExecutionFeedback,
    )

    key = EpisodeKey(
        version, group, member, "adjust_bottle", seed, "Adjust the bottle."
    )
    feedbacks = []
    for chunk in range(chunks):
        last = chunk == chunks - 1
        feedbacks.append(
            ExecutionFeedback(
                key,
                chunk,
                chunk * 32,
                tuple({} for _ in range(32)),
                np.zeros(14, dtype=np.float32),
                tuple([0.0] * 31 + [float(success and last)]),
                last and success,
                last and not success,
                last and success,
                tuple((np.arange(32) + chunk * 32 + 1) / 30),
            )
        )
    return CollectedEpisode(key, (example,) * chunks, tuple(feedbacks))


def test_online_collector_real_replay_and_complete_groups(policy):
    from rlinf.models.embodiment.cosmos3.online_fpo import collect_actor_shards

    records = sample(policy)
    episodes = [
        make_online_episode(records[0], 0),
        make_online_episode(records[1], 1, success=False),
    ]
    options = {
        "version": 0,
        "group_size": 2,
        "world_size": 2,
        "global_batch_size": 2,
        "gamma": 1.0,
        "block_size": 32,
        "fps": 30.0,
    }
    shards, metrics = collect_actor_shards(episodes, **options)
    assert shards[0].examples[0] is records[0]
    assert shards[1].examples[0] is records[1]
    torch.testing.assert_close(
        shards[0].advantages, torch.tensor([1.0]), atol=1e-5, rtol=0
    )
    torch.testing.assert_close(
        shards[1].advantages, torch.tensor([-1.0]), atol=1e-5, rtol=0
    )
    assert metrics["success_rate"] == 0.5
    for invalid in (
        [episodes[0]],
        [episodes[0], episodes[0]],
        [
            episodes[0],
            replace(episodes[1], key=replace(episodes[1].key, reset_seed=43)),
        ],
    ):
        with pytest.raises(ValueError):
            collect_actor_shards(invalid, **options)
    with pytest.raises(ValueError, match="version"):
        collect_actor_shards(episodes, **dict(options, version=1))
    more, metrics = collect_actor_shards(episodes, **dict(options, global_batch_size=4))
    assert not more and metrics["need_more_groups"]


def test_online_collector_zero_advantage_and_partial_terminal(policy):
    from rlinf.models.embodiment.cosmos3.online_fpo import collect_actor_shards

    records = sample(policy)
    episodes = [make_online_episode(records[i], i, success=False) for i in range(2)]
    feedback = episodes[0].feedback[0]
    partial = replace(
        feedback,
        frames=feedback.frames[:3],
        rewards=feedback.rewards[:3],
        timestamps=feedback.timestamps[:3],
    )
    episodes[0] = replace(episodes[0], feedback=(partial,))
    shards, metrics = collect_actor_shards(
        episodes,
        version=0,
        group_size=2,
        world_size=2,
        global_batch_size=2,
        gamma=0.9,
        block_size=32,
        fps=30.0,
    )
    assert metrics["zero_advantages"] and all(s.advantages.eq(0).all() for s in shards)
    assert (
        shards[0].examples[0].rollout.endpoint_action.shape
        == records[0].rollout.endpoint_action.shape
    )
    with pytest.raises(ValueError, match="Partial"):
        replace(partial, truncated=False).validate(32, 30.0)


def test_native_robotwin_execution_is_measured_exact_and_idempotent():
    import numpy as np

    from rlinf.envs.sim.robotwin.causal_env import CausalRoboTwinEnv
    from rlinf.models.embodiment.cosmos3.online_fpo import ActionRequest, EpisodeKey

    class Task:
        def __init__(self):
            self.robot = SimpleNamespace(
                get_left_arm_real_jointState=lambda: [999.0] * 7,
                get_right_arm_real_jointState=lambda: [999.0] * 7,
            )
            for side, arm_value, gripper_qpos in (
                ("left", 1.0, 0.03),
                ("right", 2.0, 0.07),
            ):
                entity = SimpleNamespace(
                    qpos=np.array([arm_value] * 6 + [gripper_qpos]),
                    get_active_joints=lambda: list(range(7)),
                )
                entity.get_qpos = lambda entity=entity: entity.qpos
                setattr(self.robot, f"{side}_entity", entity)
                setattr(self.robot, f"{side}_arm_joints", list(range(6)))
                setattr(self.robot, f"{side}_gripper", [(6, 1.0, 0.0)])
                setattr(self.robot, f"{side}_gripper_scale", [0.01, 0.09])

        def setup_demo(self, **kwargs):
            self.take_action_cnt, self.step_lim, self.eval_success = 0, 64, False
            self.seed = kwargs["seed"]

        def set_instruction(self, instruction):
            self.instruction = instruction

        def get_obs(self):
            return {
                "joint_action": {"vector": np.full(14, 999.0)},
                "observation": {
                    key: {
                        "rgb": np.full((4, 4, 3), self.take_action_cnt, dtype=np.uint8)
                    }
                    for key in ("head_camera", "left_camera", "right_camera")
                },
            }

        def take_action(self, action, action_type):
            assert action_type == "qpos"
            self.take_action_cnt += 1
            self.eval_success = self.take_action_cnt == 3

        def close_env(self, **kwargs):
            pass

    cfg = OmegaConf.create(
        {
            "task_config": {"task_name": "adjust_bottle"},
            "max_episode_steps": 64,
            "camera_names": ["head_camera", "left_camera", "right_camera"],
            "action_dim": 14,
            "actions_per_block": 32,
            "fps": 30.0,
        }
    )
    env = CausalRoboTwinEnv(cfg, task_factory=Task)
    key = EpisodeKey(0, 0, 0, "adjust_bottle", 42, "Adjust the bottle.")
    reset = env.reset(key)
    np.testing.assert_allclose(reset["state"], [1.0] * 6 + [0.25] + [2.0] * 6 + [0.75])
    env.task.robot.left_entity.qpos[-1] = 0.05
    assert env.observe()[1][6] == pytest.approx(0.5)
    request = ActionRequest(key, 0, 0, np.zeros((32, 14), np.float32))
    feedback = env.step(request)
    assert feedback.executed_steps == 3 and feedback.success and feedback.terminated
    assert feedback.rewards == (0.0, 0.0, 1.0)
    assert feedback.frames[-1]["head_camera"][0, 0, 0] == 3
    assert env.step(request) is feedback and env.task.take_action_cnt == 3
    with pytest.raises(ValueError, match="Conflicting"):
        env.step(replace(request, actions=np.ones((32, 14), np.float32)))
    with pytest.raises(ValueError, match="episode"):
        env.step(replace(request, episode=replace(key, version=1)))
    with pytest.raises(ValueError, match="finished"):
        env.step(replace(request, chunk=1, observation_step=32))
    env.reset(replace(key, version=1))
    assert env.task.take_action_cnt == 0
    env.close()


@pytest.mark.parametrize(
    "recipe",
    [
        "robotwin_adjust_bottle_causal_cosmos3_fpo_2nodes",
        "robotwin_causal_production_8gpu_8npu",
    ],
)
def test_online_recipe_and_resume_contract(recipe):
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    from rlinf.runners.causal_fpo_runner import (
        config_fingerprint,
        validate_causal_fpo_config,
    )

    config_dir = Path(__file__).resolve().parents[2] / "examples/embodiment/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        cfg = compose(
            config_name=recipe,
            overrides=[
                "causal.instruction=Adjust bottle",
                "causal.reset_seeds=[42,43,44,45,46,47,48,49]",
                "actor.model.cosmos_sft_toml=/assets/model.toml",
                "actor.model.model_path=/assets/model",
                "actor.model.robotwin.statistics_path=/assets/stats.json",
                "actor.model.robotwin.robot_config_path=/assets/robot.json",
            ],
        )
    validate_causal_fpo_config(cfg)
    fingerprint = config_fingerprint(cfg)
    cfg.runner.resume = "/checkpoints/version_000001"
    cfg.runner.max_versions = 2
    assert config_fingerprint(cfg) == fingerprint
    cfg.causal.batch.video_stride = 8
    assert config_fingerprint(cfg) != fingerprint
    cfg.env.fps = 20
    with pytest.raises(ValueError, match="FPS"):
        validate_causal_fpo_config(cfg)


def test_online_restore_rejects_uncommitted_or_changed_assets(tmp_path):
    import json

    from rlinf.workers.actor.fsdp_fpo_actor_worker import OnlineCausalFPOActor

    actor = SimpleNamespace(_asset_signature="assets", _world_size=2)
    with pytest.raises(FileNotFoundError):
        OnlineCausalFPOActor.restore_online(actor, str(tmp_path), "config")
    target = tmp_path / "COMMITTED.json"
    target.write_text(json.dumps({"config_sha256": "wrong"}))
    with pytest.raises(ValueError, match="contract"):
        OnlineCausalFPOActor.restore_online(actor, str(tmp_path), "config")
    target.write_text(
        json.dumps(
            {"config_sha256": "config", "assets": "changed", "actor_world_size": 2}
        )
    )
    with pytest.raises(ValueError, match="assets"):
        OnlineCausalFPOActor.restore_online(actor, str(tmp_path), "config")


@pytest.fixture
def production_config():
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    config_dir = Path(__file__).resolve().parents[2] / "examples/embodiment/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        return compose(
            config_name="robotwin_causal_production_8gpu_8npu",
            overrides=[
                "causal.instruction=Adjust bottle",
                "causal.reset_seeds=[42,43,44,45,46,47,48,49]",
                "actor.model.cosmos_sft_toml=/assets/model.toml",
                "actor.model.model_path=/assets/model",
                "actor.model.robotwin.statistics_path=/assets/stats.json",
                "actor.model.robotwin.robot_config_path=/assets/robot.json",
            ],
        )


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("causal.batch.history_blocks", 1.5, "history_blocks"),
        ("causal.batch.history_blocks", True, "history_blocks"),
        ("causal.num_probes", True, "num_probes"),
        ("actor.micro_batch_size", 1.5, "micro_batch_size"),
        ("env.action_dim", 0, "action_dim"),
        ("env.fps", 0, "fps"),
        ("causal.batch.fps", float("nan"), "fps"),
        ("causal.sampling.guidance", float("inf"), "guidance"),
        ("causal.sampling.shift", 0, "shift"),
        ("actor.optim.lr", float("nan"), "optim.lr"),
        ("actor.optim.clip_grad", 0, "clip_grad"),
        ("causal.batch.video_stride", 3, "latent intervals"),
        ("causal.batch.viewpoint", "unknown", "camera layout"),
        ("causal.batch.cameras.left", "head_camera", "distinct"),
        ("env.camera_names", ["head_camera"], "camera"),
        ("actor.model.robotwin.robot_config_path", "relative.json", "absolute"),
        ("runner.output_dir", "relative", "absolute"),
        ("runner.resume", "relative", "absolute"),
        ("actor.model.wan_vae_path", "relative", "absolute"),
    ],
)
def test_production_config_rejects_invalid_runtime_contract(
    production_config, field, value, match
):
    from rlinf.runners.causal_fpo_runner import validate_causal_fpo_config

    OmegaConf.update(production_config, field, value)
    with pytest.raises(ValueError, match=match):
        validate_causal_fpo_config(production_config)


def test_reset_windows_cover_long_seed_lists_and_preserve_fixed_eight():
    from rlinf.models.embodiment.cosmos3.online_fpo import reset_seed_for_group

    seeds = list(range(100, 132))
    windows = [
        [
            reset_seed_for_group(seeds, version=v, round_index=0, rank=r, world_size=8)
            for r in range(8)
        ]
        for v in range(4)
    ]
    assert [seed for window in windows for seed in window] == seeds
    assert (
        len(
            {
                reset_seed_for_group(
                    seeds, version=0, round_index=1, rank=r, world_size=8
                )
                for r in range(8)
            }
        )
        == 8
    )
    for version in range(4):
        assert [
            reset_seed_for_group(
                seeds[:8], version=version, round_index=0, rank=r, world_size=8
            )
            for r in range(8)
        ] == seeds[:8]


def test_online_asset_signature_detects_changed_statistics(tmp_path):
    from rlinf.models.embodiment.cosmos3.runtime import asset_signature

    model = tmp_path / "model"
    model.mkdir()
    (model / ".metadata").write_bytes(b"metadata")
    toml, stats = tmp_path / "recipe.toml", tmp_path / "stats.json"
    toml.write_text("[action]\n")
    stats.write_text("{}")
    cfg = OmegaConf.create(
        {
            "model_path": str(model),
            "cosmos_sft_toml": str(toml),
            "robotwin": {"statistics_path": str(stats)},
        }
    )
    before = asset_signature(cfg)
    stats.write_text('{"changed": true}')
    assert before != asset_signature(cfg)


def test_checkpoint_preflight_rejects_occupied_boundaries_before_sampling(tmp_path):
    from rlinf.models.embodiment.cosmos3.runtime import prepare_checkpoint_directory

    root = tmp_path / "run"
    prepare_checkpoint_directory(root, 1, 100)
    # An abandoned staging directory is not a committed target boundary.
    (root / ".version_000001.partial-abandoned").mkdir()
    prepare_checkpoint_directory(root, 1, 100)
    existing = root / "version_000002"
    existing.mkdir()
    (existing / "COMMITTED.json").write_text('{"version": 2}')
    with pytest.raises(FileExistsError, match="version_000002"):
        prepare_checkpoint_directory(root, 1, 100)
    assert (existing / "COMMITTED.json").read_text() == '{"version": 2}'
    prepare_checkpoint_directory(root, 3, 100)
    with pytest.raises(ValueError, match="advance"):
        prepare_checkpoint_directory(root, 3, 2)


def test_two_online_versions_resample_and_reject_stale_replay(policy):
    from rlinf.models.embodiment.cosmos3.online_fpo import collect_actor_shards
    from rlinf.workers.actor.fsdp_fpo_actor_worker import EmbodiedFPOFSDPActor

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad], lr=1e-3
    )
    actual_steps = []

    def step():
        optimizer.step()
        actual_steps.append(1)
        return 0.0, [1e-3]

    actor = SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "algorithm": {"fpo": {}},
                "actor": {"micro_batch_size": 1, "global_batch_size": 2, "seed": 42},
            }
        ),
        version=0,
        enable_sft_co_train=False,
        _world_size=1,
        _rank=0,
        update_epoch=1,
        gradient_accumulation=2,
        device=torch.device("cpu"),
        is_weight_offloaded=False,
        is_optimizer_offloaded=False,
        model=policy,
        amp_context=nullcontext(),
        optimizer=optimizer,
        grad_scaler=torch.amp.GradScaler("cpu", enabled=False),
        before_micro_batch=lambda *args, **kwargs: nullcontext(),
        optimizer_step=step,
        lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0),
    )
    old_batch = None
    for version in range(2):
        actor.version = version
        if old_batch is not None:
            with pytest.raises(ValueError, match="version"):
                EmbodiedFPOFSDPActor.set_fpo_batch(actor, old_batch)
        _, records = policy.sample_fpo_batch(
            [{}, {}],
            policy_version=str(version),
            seeds=[42 + version, 43 + version],
            probe_seeds=[13, 14],
            num_probes=2,
        )
        episodes = [
            make_online_episode(records[i], i, version=version, success=i == 0)
            for i in range(2)
        ]
        shards, _ = collect_actor_shards(
            episodes,
            version=version,
            group_size=2,
            world_size=1,
            global_batch_size=2,
            gamma=1.0,
            block_size=32,
            fps=30.0,
        )
        before = {
            n: p.detach().clone()
            for n, p in policy.named_parameters()
            if p.requires_grad
        }
        EmbodiedFPOFSDPActor.set_fpo_batch(actor, shards[0])
        EmbodiedFPOFSDPActor.run_training(actor)
        assert actor._fpo_consumed_version == version
        assert any(
            not torch.equal(before[n], p)
            for n, p in policy.named_parameters()
            if p.requires_grad
        )
        old_batch = shards[0]
    assert len(actual_steps) == 2
