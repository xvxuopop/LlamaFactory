#!/usr/bin/env bash
# Run the full-size random Ling model on two Ascend devices.
set -eo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source /home/miniconda3/etc/profile.d/conda.sh
conda activate jyx_llamafactory_py312
source "$HOME/cann/ascend-toolkit/set_env.sh"

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
: "${ASCEND_RT_VISIBLE_DEVICES:?Set two available NPU IDs explicitly}"
export ASCEND_RT_VISIBLE_DEVICES
IFS=',' read -r -a ling_devices <<< "$ASCEND_RT_VISIBLE_DEVICES"
if [[ ${#ling_devices[@]} -ne 2 || "${ling_devices[0]}" == "${ling_devices[1]}" ]]; then
  echo "Set ASCEND_RT_VISIBLE_DEVICES to two distinct available NPU IDs." >&2
  exit 1
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

exec torchrun --standalone --nnodes=1 --nproc-per-node=2 \
  -m llamafactory.v1.trainers.sft_trainer \
  examples/ling_stacked/train_ling_fsdpturbo_ep_lora.yaml "$@"
