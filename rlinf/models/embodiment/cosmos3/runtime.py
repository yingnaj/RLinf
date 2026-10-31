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

"""Worker-local device setup and read-only legacy DCP metadata compatibility."""

import os
import pathlib
import pickle
from contextlib import contextmanager


def bootstrap_cosmos(device_type):
    os.environ["COSMOS_DEVICE"] = device_type
    if device_type == "npu":
        from torch_npu.contrib import transfer_to_npu  # noqa: F401


@contextmanager
def compatible_dcp_metadata(enabled=False):
    if not enabled:
        yield
        return
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.metadata import StorageMeta

    original = FileSystemReader.read_metadata

    class CompatibleUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module in {"pathlib._local", "pathlib._abc"} and name in {
                "Path",
                "PosixPath",
                "WindowsPath",
                "PurePath",
                "PurePosixPath",
                "PureWindowsPath",
            }:
                return getattr(pathlib, name)
            return super().find_class(module, name)

    def read_metadata(reader, *args, **kwargs):
        try:
            return original(reader, *args, **kwargs)
        except ModuleNotFoundError as error:
            if error.name not in {"pathlib._local", "pathlib._abc"}:
                raise
            with reader.fs.create_stream(
                reader._get_metadata_path(kwargs.get("rank")), "rb"
            ) as stream:
                metadata = CompatibleUnpickler(stream).load()
            if getattr(metadata, "storage_meta", None) is None:
                metadata.storage_meta = StorageMeta()
            metadata.storage_meta.load_id = reader.load_id
            return metadata

    FileSystemReader.read_metadata = read_metadata
    try:
        yield
    finally:
        FileSystemReader.read_metadata = original


def asset_signature(model_cfg):
    """Identify local configuration/statistics and the immutable base DCP inventory."""
    import hashlib
    import json
    from pathlib import Path

    files = [Path(model_cfg.cosmos_sft_toml), Path(model_cfg.robotwin.statistics_path)]
    if model_cfg.robotwin.get("robot_config_path"):
        files.append(Path(model_cfg.robotwin.robot_config_path))
    checkpoint = Path(model_cfg.model_path)
    if not checkpoint.is_dir():
        raise ValueError("Online causal FPO requires a complete native DCP directory")
    inventory = []
    for path in sorted(checkpoint.rglob("*")):
        if path.is_file():
            stat = path.stat()
            inventory.append(
                (str(path.relative_to(checkpoint)), stat.st_size, stat.st_mtime_ns)
            )
            if path.name in {".metadata", "config.yaml"}:
                files.append(path)
    if not any(path.name == ".metadata" for path in files):
        raise ValueError("Base checkpoint has no DCP metadata")
    hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files
    }
    return hashlib.sha256(
        json.dumps(
            {"files": hashes, "dcp_inventory": inventory}, sort_keys=True
        ).encode()
    ).hexdigest()


def prepare_checkpoint_directory(path, first_version, last_version):
    """Reject completed target boundaries before collecting expensive rollouts."""
    from pathlib import Path

    if first_version < 1 or last_version < first_version:
        raise ValueError("Training target must advance beyond the restored boundary")
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    for version in range(first_version, last_version + 1):
        target = root / f"version_{version:06d}"
        if target.exists():
            raise FileExistsError(
                f"Checkpoint target already exists: {target}; resume the latest boundary or use a fresh run directory"
            )
