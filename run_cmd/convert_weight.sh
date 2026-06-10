export MEGATRON_PATH=/home/Megatron-LM
export VERL_PATH=/home/verl
export PYTHONPATH=${MEGATRON_PATH}:${VERL_PATH}:$PYTHONPATH

# HF_MODEL_PATH='/home/dist/zhaoping/LLMs/Qwen3-1.7B'
# DIST_CKPT_PATH='/home/dist/zhaoping/LLMs/MCORE/Qwen3-1.7B-mcore'

# HF_MODEL_PATH='/home/dist/zhaoping/LLMs/Qwen2-7B-Instruct'
# DIST_CKPT_PATH='/home/dist/zhaoping/LLMs/MCORE/Qwen2-7B-Instruct'

# HF_MODEL_PATH='/home/dist/zhaoping/LLMs/Qwen3-30B-A3B'
# DIST_CKPT_PATH='/home/dist/zhaoping/LLMs/MCORE/Qwen3-30B-A3B'

# HF_MODEL_PATH='/home/dist/zhaoping/LLMs/DeepSeek-V2-Lite'
# DIST_CKPT_PATH='/home/dist/zhaoping/LLMs/MCORE/DeepSeek-V2-Lite'

# HF_MODEL_PATH='/home/dist/zhaoping/LLMs/Qwen3-8B'
# DIST_CKPT_PATH='/home/dist/zhaoping/LLMs/MCORE/Qwen3-8B'

# HF_MODEL_PATH='/mnt/seed17/001688/models/DeepSeek-V2-Lite'
# DIST_CKPT_PATH='/mnt/seed17/001688/zhaoping/LLMs/MCORE/DeepSeek-V2-Lite'

# HF_MODEL_PATH='/mnt/seed17/001688/models/Qwen3-30B-A3B'
# DIST_CKPT_PATH='/mnt/seed17/001688/zhaoping/LLMs/MCORE/Qwen3-30B-A3B'

HF_MODEL_PATH='/mnt/seed17/001688/models/DeepSeek-V2'
DIST_CKPT_PATH='/mnt/seed17/001688/zhaoping/LLMs/MCORE/DeepSeek-V2'

# HF_MODEL_PATH='/mnt/seed17/001688/models/DeepSeek-V2-Chat'
# DIST_CKPT_PATH='/mnt/seed17/001688/zhaoping/LLMs/MCORE/DeepSeek-V2-Chat_single'


# /home/dist/zhaoping/LLMs 存储了大量LLM模型参数
# python -u ../scripts/converter_hf_to_mcore.py --hf_model_path $HF_MODEL_PATH --output_path $DIST_CKPT_PATH

# ========== MCCL 环境优化 ==========
export MCCL_DEBUG=INFO
export MCCL_BUFF_SIZE=268435456      # 256MB
export MCCL_MIN_NCHANNELS=1
export MCCL_TIMEOUT=1800000          # 30分钟超时
export PYTORCH_NO_MUSA_MEMORY_CACHING=1

# ========== 设备绑定 ==========
export MUSA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=29501             # 避免端口冲突

# DeepSeek-V2
torchrun --nproc_per_node 8 --nnodes 1 --node_rank 0 ../scripts/converter_hf_to_mcore.py --hf_model_path $HF_MODEL_PATH --output_path $DIST_CKPT_PATH