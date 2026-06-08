# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path


MODEL_DIR = Path(os.getenv("MODEL_DIR", "/home/model"))

_HF_MODEL_REPOS = {
    "InternVL3-1B-hf": "OpenGVLab/InternVL3-1B-hf",
    "LLaVA-NeXT-Video-7B-hf": "llava-hf/LLaVA-NeXT-Video-7B-hf",
    "Qwen2-VL-2B-Instruct": "Qwen/Qwen2-VL-2B-Instruct",
    "Qwen2-VL-7B-Instruct": "Qwen/Qwen2-VL-7B-Instruct",
    "Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B",
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen2.5-Omni-7B": "Qwen/Qwen2.5-Omni-7B",
    "Qwen3-0.6B": "Qwen/Qwen3-0.6B",
    "Qwen3-4B": "Qwen/Qwen3-4B",
    "Qwen3-4B-Instruct-2507": "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Qwen3-VL-30B-A3B-Instruct": "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "Video-LLaVA-7B-hf": "LanguageBind/Video-LLaVA-7B-hf",
    "gemma-2-2b-it": "google/gemma-2-2b-it",
    "gemma-3-4b-it": "google/gemma-3-4b-it",
    "gemma-4-31B-it": "google/gemma-4-31B-it",
    "llava-1.5-7b-hf": "llava-hf/llava-1.5-7b-hf",
    "llava-v1.6-vicuna-7b-hf": "llava-hf/llava-v1.6-vicuna-7b-hf",
    "paligemma-3b-pt-224": "google/paligemma-3b-pt-224",
    "phi-4": "microsoft/phi-4",
    "pixtral-12b": "mistral-community/pixtral-12b",
    "tiny-random-Llama-3": "llamafactory/tiny-random-Llama-3",
    "tiny-random-Llama-3-lora": "llamafactory/tiny-random-Llama-3-lora",
    "tiny-random-Llama-3-pissa": "llamafactory/tiny-random-Llama-3-pissa",
    "tiny-random-Llama-3-valuehead": "llamafactory/tiny-random-Llama-3-valuehead",
    "tiny-random-Llama-4": "llamafactory/tiny-random-Llama-4",
    "tiny-random-qwen3": "llamafactory/tiny-random-qwen3",
}


def _get_hf_cache_roots() -> list[Path]:
    roots = []
    for path in (
        os.getenv("HUGGINGFACE_HUB_CACHE"),
        os.getenv("HF_HUB_CACHE"),
        os.getenv("TRANSFORMERS_CACHE"),
        str(Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"),
        str(MODEL_DIR / ".hf_home" / "hub"),
        "/root/.cache/huggingface/hub",
    ):
        if path is None:
            continue

        root = Path(path)
        if root not in roots:
            roots.append(root)

    return roots


def _repo_id_to_cache_dir(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def _resolve_snapshot(cache_dir: Path) -> Path | None:
    snapshots_dir = cache_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return None

    main_ref = cache_dir / "refs" / "main"
    if main_ref.is_file():
        snapshot = snapshots_dir / main_ref.read_text().strip()
        if snapshot.is_dir():
            return snapshot

    snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()]
    if not snapshots:
        return None

    return max(snapshots, key=lambda path: path.stat().st_mtime)


def _get_cached_model_path(dirname: str) -> Path | None:
    repo_id = _HF_MODEL_REPOS.get(dirname)
    cache_names = [_repo_id_to_cache_dir(repo_id)] if repo_id is not None else []
    cache_names.append(f"models--*--{dirname}")

    for root in _get_hf_cache_roots():
        if not root.is_dir():
            continue

        candidates = []
        for cache_name in cache_names:
            candidates.extend(root.glob(cache_name))

        for candidate in candidates:
            snapshot = _resolve_snapshot(candidate)
            if snapshot is not None:
                return snapshot

    return None


def get_model_path(env_name: str, dirname: str) -> str:
    if env_name in os.environ:
        return os.environ[env_name]

    model_path = MODEL_DIR / dirname
    if model_path.exists():
        return str(model_path)

    cached_model_path = _get_cached_model_path(dirname)
    if cached_model_path is not None:
        return str(cached_model_path)

    return str(model_path)
