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

"""Cosmos3 SFT model-config validation.

The Cosmos3 model is built from cosmos-framework's own experiment recipe
(``make_config`` + ``override(experiment=<SKU>)`` + ``instantiate``).

Model defaults (model_type, precision, load_to_device, ema_enabled,
disable_cosmos_parallelism, keys_to_skip_loading, keys_to_select,
lr_multipliers, action_normalization, ...) live in the Hydra config group
``examples/sft/config/model/cosmos3.yaml`` -- pull them in via
``defaults: - model/cosmos3@actor.model``. Recipe yamls override only
deployment-specific fields (``model_path``, ``wan_vae_path``,
``max_samples_per_batch``).

Dataset-recipe knobs (``image_size``, ``chunk_length``, ``action_space``,
``fps``, ...) are owned by the cosmos experiment recipe (resolved by cosmos);
rlinf does not read them from ``actor.model``, so they are deliberately NOT
duplicated here or in ``model/cosmos3.yaml``.
"""

from pathlib import Path

from omegaconf import DictConfig, open_dict

from rlinf.utils.logging import get_logger

logger = get_logger()


def validate_cosmos3_variant(model_cfg: DictConfig) -> None:
    """Validate the explicit causal model and checkpoint ownership contract."""
    variant = model_cfg.get("cosmos_variant", "standard")
    if variant not in ("standard", "causal_action"):
        raise ValueError(f"Unsupported cosmos_variant={variant!r}")
    mode = model_cfg.get("training_mode", "sft")
    if mode not in ("sft", "fpo"):
        raise ValueError(f"Unsupported Cosmos3 training_mode={mode!r}")
    if mode == "fpo" and variant != "causal_action":
        raise ValueError("FPO requires cosmos_variant=causal_action")
    if model_cfg.get("robotwin") is not None and mode != "fpo":
        raise ValueError("robotwin adapter requires training_mode=fpo")
    if variant == "causal_action":
        if not model_cfg.get("cosmos_sft_toml"):
            raise ValueError(
                "causal_action requires cosmos_sft_toml for shared model/data configuration"
            )
        if not model_cfg.get("disable_cosmos_parallelism", True):
            raise ValueError("RLinf must own FSDP; disable Cosmos internal parallelism")
        if model_cfg.get("keys_to_skip_loading", []):
            raise ValueError(
                "Causal warm-start requires complete weights; keys_to_skip_loading must be empty"
            )
    if mode == "fpo":
        if model_cfg.get("ema_enabled", False):
            raise ValueError("FPO actor and rollout must use the same net; disable EMA")
        if model_cfg.get("keys_to_select", []):
            raise ValueError(
                "FPO selects native action interfaces; remove SFT keys_to_select"
            )


def load_cosmos3_experiment(model_cfg: DictConfig, *, fallback_experiment=None):
    """Resolve the same source configuration for the model and packed loader."""
    validate_cosmos3_variant(model_cfg)
    extra = list(model_cfg.get("cosmos_overrides", []))
    toml = model_cfg.get("cosmos_sft_toml")
    if toml:
        from cosmos_framework.configs.toml_config.sft_config import (
            load_experiment_from_toml,
        )

        if not Path(toml).is_absolute():
            raise ValueError(
                "cosmos_sft_toml must be absolute so Ray workers resolve the same file"
            )
        config = load_experiment_from_toml(toml, extra_overrides=extra)
    else:
        from cosmos_framework.configs.base.config import make_config
        from cosmos_framework.utils.config_helper import override

        experiment = model_cfg.get(
            "cosmos_experiment", fallback_experiment or "action_policy_libero_nano"
        )
        config = override(make_config(), ["--", f"experiment={experiment}", *extra])
    if model_cfg.get("cosmos_variant") == "causal_action":
        target = config.model._target_
        name = target if isinstance(target, str) else target.__name__
        if name.split(".")[-1] != "OmniMoTCausalActionModel":
            raise ValueError(f"Expected OmniMoTCausalActionModel, resolved {name}")
    return config


def validate_cosmos3_sft_model_cfg(model_cfg: DictConfig) -> DictConfig:
    """Validate a Cosmos3 SFT cfg."""

    validate_cosmos3_variant(model_cfg)
    with open_dict(model_cfg):
        model_path = model_cfg.get("model_path", None)
        assert model_path, "Cosmos3 SFT requires actor.model.model_path"

    logger.info(
        "Cosmos3 SFT model cfg validated: model_path=%s "
        "ema_enabled=%s disable_cosmos_parallelism=%s (keys_to_select=%d, "
        "lr_multipliers=%d, skip=%d)",
        model_path,
        model_cfg.get("ema_enabled"),
        model_cfg.get("disable_cosmos_parallelism"),
        len(model_cfg.get("keys_to_select", [])),
        len(model_cfg.get("lr_multipliers", {})),
        len(model_cfg.get("keys_to_skip_loading", [])),
    )
    return model_cfg
