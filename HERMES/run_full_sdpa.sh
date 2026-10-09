#!/usr/bin/env bash

set -o pipefail

cd /home/workspace/zsc/HERMES

source /home/workspace/ypq/miniconda3/etc/profile.d/conda.sh
conda activate hermes-qwen

export PYTHONPATH=/home/workspace/zsc/HERMES:${PYTHONPATH:-}
export HF_HOME=/home/workspace/zsc/hf_cache
unset TRANSFORMERS_CACHE

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ANNO="data/ovobench/ovobench_realtime_backeward.json"
SAVE_DIR="results/hermes_mmduet2_3b/ovobench_full_sdpa"
LOG_DIR="logs/hermes_mmduet2_full_sdpa"

mkdir -p "${SAVE_DIR}" "${LOG_DIR}"

run_chunk() {
    local gpu="$1"
    local chunk="$2"
    local log="${LOG_DIR}/chunk${chunk}.log"

    echo
    echo "=================================================="
    echo "Starting chunk ${chunk} on physical GPU ${gpu}"
    echo "Time: $(date)"
    echo "=================================================="

    CUDA_VISIBLE_DEVICES="${gpu}" \
    python -u -m video_qa.hermes_vqa \
      --model qwen2.5_vl_3b \
      --sample_fps 0.5 \
      --kv_size 6000 \
      --streaming true \
      --anno_path "${ANNO}" \
      --save_dir "${SAVE_DIR}" \
      --debug false \
      --num_chunks 4 \
      --chunk_idx "${chunk}" \
      2>&1 | tee -a "${log}"

    local status=${PIPESTATUS[0]}

    echo
    echo "Chunk ${chunk} finished with status ${status}"
    echo "Time: $(date)"

    return "${status}"
}

gpu_worker() {
    local gpu="$1"
    shift

    local chunk
    for chunk in "$@"; do
        run_chunk "${gpu}" "${chunk}" || return $?
    done
}

gpu_worker 0 0 2 &
PID_GPU0=$!

gpu_worker 1 1 3 &
PID_GPU1=$!

wait "${PID_GPU0}"
STATUS_GPU0=$?

wait "${PID_GPU1}"
STATUS_GPU1=$?

echo
echo "=================================================="
echo "GPU 0 worker status: ${STATUS_GPU0}"
echo "GPU 1 worker status: ${STATUS_GPU1}"
echo "Finished at: $(date)"
echo "=================================================="

if (( STATUS_GPU0 != 0 || STATUS_GPU1 != 0 )); then
    exit 1
fi
