#!/usr/bin/env bash
set -euo pipefail

# Wrap local model directories downloaded by huggingface-cli --local-dir or
# modelscope download --local_dir into a Hugging Face cache layout.
#
# After wrapping, Transformers can resolve the original repo id, for example:
#   AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
#   AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
#
# Usage:
#   LOCAL_MODEL_DIRS=/home/jyx/models:/home/jyx/hf_models \
#   HF_CACHE_HOME=/home/jyx/models/.hf_home \
#   ./examples/ascend/wrap_modelscope_models_as_hf_cache.sh
#
# The script does not copy model files. It creates HF cache refs and symlinks.

LOCAL_MODEL_DIRS="${LOCAL_MODEL_DIRS:-${MODELSCOPE_LOCAL_DIR:-${LOCAL_DIR:-/your/local/dir}}}"
HF_CACHE_HOME="${HF_CACHE_HOME:-${LOCAL_MODEL_DIRS%%:*}/.hf_home}"
SNAPSHOT_NAME="${SNAPSHOT_NAME:-local}"
ALLOW_MISSING="${ALLOW_MISSING:-1}"
INCLUDE_HF_TOKEN_MODELS="${INCLUDE_HF_TOKEN_MODELS:-0}"

missing_models=()

models=(
  "llamafactory/tiny-random-Llama-3"
  "llamafactory/tiny-random-Llama-3-lora"
  "llamafactory/tiny-random-Llama-3-valuehead"
  "llamafactory/tiny-random-Llama-3-pissa"
  "llamafactory/tiny-random-Llama-4"
  "llamafactory/tiny-random-qwen3"
  "Qwen/Qwen3-4B-Instruct-2507"
  "Qwen/Qwen3-0.6B"
  "Qwen/Qwen2.5-0.5B"
  "Qwen/Qwen3-8B"
  "Qwen/Qwen2.5-7B-Instruct"
  "Qwen/Qwen2-VL-2B-Instruct"
  "Qwen/Qwen2-VL-7B-Instruct"
  "Qwen/Qwen2.5-Omni-7B"
  "Qwen/Qwen3-VL-30B-A3B-Instruct"
  "google/gemma-4-31B-it"
  "microsoft/phi-4"
  "OpenGVLab/InternVL3-1B-hf"
  "llava-hf/llava-1.5-7b-hf"
  "llava-hf/llava-v1.6-vicuna-7b-hf"
  "llava-hf/LLaVA-NeXT-Video-7B-hf"
  "mistral-community/pixtral-12b"
  "LanguageBind/Video-LLaVA-7B-hf"
)

hf_token_models=(
  "google/gemma-2-2b-it"
  "google/gemma-3-4b-it"
  "google/paligemma-3b-pt-224"
  "meta-llama/Meta-Llama-3-8B-Instruct"
)

# Local directory fallbacks for repos downloaded from ModelScope substitutes.
# Format: original_hf_repo_id|local_dir_repo_id
repo_aliases=(
  "meta-llama/Meta-Llama-3-8B-Instruct|LLM-Research/Meta-Llama-3-8B-Instruct"
  "meta-llama/Meta-Llama-3-8B-Instruct|AI-ModelScope/Meta-Llama-3-8B-Instruct"
  "microsoft/phi-4|LLM-Research/phi-4"
  "mistral-community/pixtral-12b|mistralai/Pixtral-12B-2409"
  "mistral-community/pixtral-12b|LLM-Research/Pixtral-12B-2409"
  "mistral-community/pixtral-12b|AI-ModelScope/pixtral-12b"
  "LanguageBind/Video-LLaVA-7B-hf|PKU-YuanLab/Video-LLaVA-7B"
  "llava-hf/llava-1.5-7b-hf|swift/llava-1.5-7b-hf"
  "llava-hf/llava-v1.6-vicuna-7b-hf|swift/llava-v1.6-vicuna-7b-hf"
  "google/gemma-2-2b-it|LLM-Research/gemma-2-2b-it"
  "google/gemma-3-4b-it|LLM-Research/gemma-3-4b-it"
  "google/paligemma-3b-pt-224|AI-ModelScope/paligemma-3b-pt-224"
)

split_local_roots() {
  local old_ifs="$IFS"
  IFS=":"
  read -r -a LOCAL_ROOTS <<< "$LOCAL_MODEL_DIRS"
  IFS="$old_ifs"
}

candidate_names_for_repo() {
  local repo_id="$1"
  local escaped_repo_id="${repo_id//\//--}"
  local basename="${repo_id##*/}"

  echo "$escaped_repo_id"
  echo "$basename"

  for alias_pair in "${repo_aliases[@]}"; do
    local target_repo="${alias_pair%%|*}"
    local source_repo="${alias_pair##*|}"
    if [[ "$target_repo" == "$repo_id" ]]; then
      echo "${source_repo//\//--}"
      echo "${source_repo##*/}"
    fi
  done
}

find_model_dir() {
  local repo_id="$1"
  local root
  local candidate

  for root in "${LOCAL_ROOTS[@]}"; do
    [[ -n "$root" ]] || continue

    while IFS= read -r candidate; do
      if [[ -d "$root/$candidate" ]]; then
        readlink -f "$root/$candidate"
        return 0
      fi
    done < <(candidate_names_for_repo "$repo_id")

    # ModelScope cache_dir sometimes creates owner/repo_name directories.
    if [[ -d "$root/$repo_id" ]]; then
      readlink -f "$root/$repo_id"
      return 0
    fi
  done

  return 1
}

wrap_one_model() {
  local repo_id="$1"
  local source_dir
  local cache_repo_dir
  local snapshot_link

  if ! source_dir="$(find_model_dir "$repo_id")"; then
    if [[ "$ALLOW_MISSING" == "1" ]]; then
      echo "Skipping missing model: $repo_id"
      missing_models+=("$repo_id")
      return 0
    fi

    echo "Missing local model directory for $repo_id" >&2
    echo "Search roots: $LOCAL_MODEL_DIRS" >&2
    echo "Expected directory names include:" >&2
    candidate_names_for_repo "$repo_id" | sed "s/^/  /" >&2
    exit 1
  fi

  cache_repo_dir="$HF_CACHE_HOME/hub/models--${repo_id//\//--}"
  snapshot_link="$cache_repo_dir/snapshots/$SNAPSHOT_NAME"

  mkdir -p "$cache_repo_dir/refs" "$cache_repo_dir/snapshots"
  printf "%s" "$SNAPSHOT_NAME" > "$cache_repo_dir/refs/main"

  if [[ -L "$snapshot_link" ]]; then
    rm "$snapshot_link"
  elif [[ -e "$snapshot_link" ]]; then
    echo "Refusing to overwrite existing non-symlink path: $snapshot_link" >&2
    exit 1
  fi

  ln -s "$source_dir" "$snapshot_link"
  echo "Wrapped $repo_id"
  echo "  source: $source_dir"
  echo "  cache : $snapshot_link"
}

split_local_roots
mkdir -p "$HF_CACHE_HOME/hub"

for model in "${models[@]}"; do
  wrap_one_model "$model"
done

if [[ "$INCLUDE_HF_TOKEN_MODELS" == "1" ]]; then
  for model in "${hf_token_models[@]}"; do
    wrap_one_model "$model"
  done
else
  echo "Skipped HF-token-gated models. Set INCLUDE_HF_TOKEN_MODELS=1 to wrap them:"
  printf "  %s\n" "${hf_token_models[@]}"
fi

if [[ "${#missing_models[@]}" -gt 0 ]]; then
  echo
  echo "Missing models skipped:"
  printf "  %s\n" "${missing_models[@]}"
fi

cat <<EOF

Done. Use these environment variables before running tests:

export HF_HOME="$HF_CACHE_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

EOF
