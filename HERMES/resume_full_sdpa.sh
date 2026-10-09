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
    echo "Resume chunk ${chunk} on physical GPU ${gpu}"
    echo "Start time: $(date)"
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
    echo "End time: $(date)"

    return "${status}"
}

# 保持之前的GPU分配：
# GPU 1继续chunk 1
# GPU 0继续chunk 2
run_chunk 1 1 &
PID_CHUNK1=$!

run_chunk 0 2 &
PID_CHUNK2=$!

wait "${PID_CHUNK1}"
STATUS_CHUNK1=$?

wait "${PID_CHUNK2}"
STATUS_CHUNK2=$?

echo
echo "Chunk 1 status: ${STATUS_CHUNK1}"
echo "Chunk 2 status: ${STATUS_CHUNK2}"

# 前两项都完成后，用GPU 1执行尚未开始的chunk 3。
if (( STATUS_CHUNK1 == 0 && STATUS_CHUNK2 == 0 )); then
    run_chunk 1 3
    STATUS_CHUNK3=$?
else
    echo "Chunk 1或chunk 2仍有错误，暂不启动chunk 3。"
    exit 1
fi

echo
echo "=================================================="
echo "Chunk 1 status: ${STATUS_CHUNK1}"
echo "Chunk 2 status: ${STATUS_CHUNK2}"
echo "Chunk 3 status: ${STATUS_CHUNK3}"
echo "All resume jobs finished at: $(date)"
echo "=================================================="

exit "${STATUS_CHUNK3}"
