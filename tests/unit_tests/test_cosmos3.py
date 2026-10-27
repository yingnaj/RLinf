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
