#!/usr/bin/env bash
set -euo pipefail
set -x

# Qwen3.5 HF -> Megatron-Core checkpoint conversion via mbridge.
# Direct run:
#   bash open/verl/scripts/convert_qwen3_5_hf_to_mcore.sh
#
# Override examples:
#   HF_MODEL_PATH=/data/yanguo.sun/Qwen3.5-9B \
#   OUTPUT_PATH=/data/yanguo.sun/Qwen3.5-9B-mcore-tp1-pp4 \
#   TP=1 PP=4 NPROC_PER_NODE=4 \
#   bash open/verl/scripts/convert_qwen3_5_hf_to_mcore.sh

WORKSPACE_DIR=${WORKSPACE_DIR:-/mnt/moer-train/public/yanguo.sun/qwen35_verl/}

HF_MODEL_PATH=${HF_MODEL_PATH:-/mnt/moer-train/public/yanguo.sun/qwen35_verl/Qwen3.5-9B}
TP=${TP:-4}
PP=${PP:-1}
CP=${CP:-1}
EP=${EP:-1}
OUTPUT_PATH=${OUTPUT_PATH:-/mnt/moer-train/public/yanguo.sun/qwen35_verl/Qwen3.5-9B-mcore-tp${TP}-pp${PP}-cp${CP}-ep${EP}}

NPROC_PER_NODE=${NPROC_PER_NODE:-$((TP * PP * CP))}
DTYPE=${DTYPE:-bf16}
VISION_ATTN_IMPLEMENTATION=${VISION_ATTN_IMPLEMENTATION:-eager}
MASTER_PORT=${MASTER_PORT:-29577}

export MUSA_VISIBLE_DEVICES=${MUSA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export ACCELERATOR_BACKEND=${ACCELERATOR_BACKEND:-musa}
export MCCL_PROTOS=${MCCL_PROTOS:-2}
export MCCL_CHECK_POINTERS=${MCCL_CHECK_POINTERS:-0}
# export MCCL_DEBUG=${MCCL_DEBUG:-INFO}
# export MCCL_DEBUG_SUBSYS=${MCCL_DEBUG_SUBSYS:-ALL}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-INFO}
export FLA_USE_CUDA_JITERATOR=${FLA_USE_CUDA_JITERATOR:-0}
export QWEN3_5_VISION_ATTN_IMPLEMENTATION=${QWEN3_5_VISION_ATTN_IMPLEMENTATION:-$VISION_ATTN_IMPLEMENTATION}

export MEGATRON_PATH=${MEGATRON_PATH:-${WORKSPACE_DIR}/Megatron-LM}
export VERL_PATH=${VERL_PATH:-${WORKSPACE_DIR}/verl}
export MBRIDGE_PATH=${MBRIDGE_PATH:-${WORKSPACE_DIR}/mbridge}
export MUSA_PATCH_PATH=${MUSA_PATCH_PATH:-${WORKSPACE_DIR}/megatron-lm-musa-patch}
export PYTHONPATH=${MBRIDGE_PATH}:${VERL_PATH}:${MEGATRON_PATH}:${MUSA_PATCH_PATH}:${PYTHONPATH:-}

cd "${WORKSPACE_DIR}"

python3 - <<PY
from pathlib import Path
hf = Path("${HF_MODEL_PATH}")
out = Path("${OUTPUT_PATH}")
if not hf.exists():
    raise SystemExit(f"HF_MODEL_PATH does not exist: {hf}")
out.parent.mkdir(parents=True, exist_ok=True)
print(f"HF_MODEL_PATH={hf}")
print(f"OUTPUT_PATH={out}")
print(f"TP=${TP}, PP=${PP}, CP=${CP}, EP=${EP}, NPROC_PER_NODE=${NPROC_PER_NODE}")
PY

torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  verl/scripts/mbridge_convert_qwen3_5_hf_to_mcore.py \
  --hf-model-path "${HF_MODEL_PATH}" \
  --output-path "${OUTPUT_PATH}" \
  --tp-size "${TP}" \
  --pp-size "${PP}" \
  --cp-size "${CP}" \
  --ep-size "${EP}" \
  --dtype "${DTYPE}" \
  --vision-attn-implementation "${VISION_ATTN_IMPLEMENTATION}" \
  --trust-remote-code \
  --memory-efficient-load \
  --overwrite
