 # zhaoping: 需要替换为自己真实的模型参数路径
# zhaoping: HF_MODEL_PATH 用于 tokenizer/config/rollout；MCORE_MODEL_PATH 用于 Megatron actor/ref 初始化权重。
HF_MODEL_PATH='/data/yanguo.sun/Qwen3.5-9B'
# MCORE_MODEL_PATH='/data/yanguo.sun/Qwen3.5-9B-mcore-tp1-pp4-cp1-ep1'
MCORE_MODEL_PATH='/data/yanguo.sun/Qwen3.5-9B-mcore-tp4-pp1-cp1-ep1'

export MUSA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:/usr/local/musa/lib:${LD_LIBRARY_PATH:-}
export ACCELERATOR_BACKEND="musa"
export MUSA_LAUNCH_BLOCKING=0
export MUSA_ENABLE_LLC_OPT=1
export PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True
export MCCL_IB_GID_INDEX=3
export MCCL_NET_SHARED_BUFFERS=0
export MCCL_PROTOS=2
export MCCL_CHECK_POINTERS=0
export GLOO_SOCKET_IFNAME=bond0
export TP_SOCKET_IFNAME=bond0
export VLLM_PATCH_MUSA_CUSTOM_OPS=1
export VERL_LOGGING_LEVEL=INFO #INFO
export HYDRA_FULL_ERROR=1
export MEGATRON_PATH=/home/Megatron-LM
export VERL_PATH=/home/verl
export MUSA_PATCH_PATH=/home/megatron-lm-musa-patch
export SGLANG_PATH=/home/sglang/python
export PYTHONPATH=${MEGATRON_PATH}:${VERL_PATH}:${MUSA_PATCH_PATH}:${SGLANG_PATH}:${PYTHONPATH:-}
export FLA_USE_CUDA_JITERATOR=0
export QWEN3_5_VISION_ATTN_IMPLEMENTATION=sdpa
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1


# Must match the converted MCore checkpoint layout in MCORE_MODEL_PATH.
TP=${TP:-4}
PP=${PP:-1}
CP=${CP:-1}
# Keep Megatron actor/ref TP separate from SGLang rollout TP.
ROLLOUT_TP=${ROLLOUT_TP:-4}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.30}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-8}
ROLLOUT_MAX_MAMBA_CACHE_SIZE=${ROLLOUT_MAX_MAMBA_CACHE_SIZE:-16}
SGLANG_CUDA_GRAPH_BS=${SGLANG_CUDA_GRAPH_BS:-[1,2,3,4,5,6,7,8]}
SGLANG_CHUNKED_PREFILL_SIZE=${SGLANG_CHUNKED_PREFILL_SIZE:-4096}
SGLANG_MAX_PREFILL_TOKENS=${SGLANG_MAX_PREFILL_TOKENS:-8192}
SGLANG_ATTENTION_BACKEND=${SGLANG_ATTENTION_BACKEND:-fa3}
SGLANG_MM_ATTENTION_BACKEND=${SGLANG_MM_ATTENTION_BACKEND:-fa3}
SGLANG_MOE_RUNNER_BACKEND=${SGLANG_MOE_RUNNER_BACKEND:-deep_gemm}
SGLANG_SAMPLING_BACKEND=${SGLANG_SAMPLING_BACKEND:-flashinfer}
SGLANG_SPECULATIVE_ALGORITHM=${SGLANG_SPECULATIVE_ALGORITHM:-NEXTN}
SGLANG_SPECULATIVE_NUM_STEPS=${SGLANG_SPECULATIVE_NUM_STEPS:-1}
SGLANG_SPECULATIVE_EAGLE_TOPK=${SGLANG_SPECULATIVE_EAGLE_TOPK:-1}
SGLANG_SPECULATIVE_NUM_DRAFT_TOKENS=${SGLANG_SPECULATIVE_NUM_DRAFT_TOKENS:-2}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
MAX_TOTAL_TOKEN_LEN=${MAX_TOTAL_TOKEN_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}

ALL_OFFLOAD=${ALL_OFFLOAD:-True}

rollout_name="sglang"
project_name='verl_grpo_qwen3_5_35b_geo3k'
exp_name='qwen3_5_35b_megatron'
adv_estimator=grpo

# zhaoping: 数据集路径，需要自己配置
DATASET_PATH="/data/yanguo.sun/qwen35_verl/geo3k_verl"
train_files=$DATASET_PATH/train.parquet
test_files=$DATASET_PATH/test.parquet

# zhaoping:需要指定到 Verl 中对应config路径
CONFIG_PATH=$VERL_PATH/verl/trainer/config
LOG_DIR=/data/yanguo.sun/qwen35_verl/verl/logs
LOG_FILE=$LOG_DIR/qwen3.5-9b_grpo.log
mkdir -p "$LOG_DIR"

nohup env PYTHONPATH="$PYTHONPATH" \
    MUSA_VISIBLE_DEVICES="$MUSA_VISIBLE_DEVICES" \
    ACCELERATOR_BACKEND="$ACCELERATOR_BACKEND" \
    PYTORCH_MUSA_ALLOC_CONF="${PYTORCH_MUSA_ALLOC_CONF:-expandable_segments:True}" \
    RAY_LOGGING_LEVEL=WARNING \
    RAY_DEDUP_LOGS=0 \
    RAY_ADDRESS="localhost:65379" \
python3 -u -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='ppo_megatron_trainer_demo.yaml' \
    algorithm.adv_estimator=grpo \
    data.train_files=$train_files \
    data.val_files=$test_files \
    data.train_batch_size=8 \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.prompt_key=prompt \
    data.image_key=images \
    data.truncation='error' \
    data.shuffle=False \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=$HF_MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOTAL_TOKEN_LEN} \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.vanilla_mbridge=True \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=${MCORE_MODEL_PATH} \
    actor_rollout_ref.actor.megatron.use_remove_padding=False \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TP} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${PP} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${CP} \
    actor_rollout_ref.actor.megatron.param_offload=${ALL_OFFLOAD} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${ALL_OFFLOAD} \
    actor_rollout_ref.actor.megatron.grad_offload=${ALL_OFFLOAD} \
    actor_rollout_ref.actor.megatron.dtype=bfloat16 \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.01 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_z_loss_coeff=0.001 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION} \
    actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS} \
    actor_rollout_ref.rollout.max_model_len=${MAX_TOTAL_TOKEN_LEN} \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.top_k=50 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.enforce_eager=False \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_custom_all_reduce=True \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_radix_cache=True \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_cuda_graph_padding=False \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.cuda_graph_bs="${SGLANG_CUDA_GRAPH_BS}" \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_overlap_schedule=True \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=${SGLANG_ATTENTION_BACKEND} \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.mm_attention_backend=${SGLANG_MM_ATTENTION_BACKEND} \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.moe_runner_backend=${SGLANG_MOE_RUNNER_BACKEND} \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.sampling_backend=${SGLANG_SAMPLING_BACKEND} \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.chunked_prefill_size=${SGLANG_CHUNKED_PREFILL_SIZE} \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.max_prefill_tokens=${SGLANG_MAX_PREFILL_TOKENS} \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.speculative_algorithm=${SGLANG_SPECULATIVE_ALGORITHM} \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.speculative_num_steps=${SGLANG_SPECULATIVE_NUM_STEPS} \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.speculative_eagle_topk=${SGLANG_SPECULATIVE_EAGLE_TOPK} \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.speculative_num_draft_tokens=${SGLANG_SPECULATIVE_NUM_DRAFT_TOKENS} \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.max_mamba_cache_size=${ROLLOUT_MAX_MAMBA_CACHE_SIZE} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${MAX_TOTAL_TOKEN_LEN} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${MAX_TOTAL_TOKEN_LEN} \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.ref.megatron.dist_checkpointing_path=${MCORE_MODEL_PATH} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TP} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${PP} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${CP} \
    actor_rollout_ref.ref.megatron.param_offload=${ALL_OFFLOAD} \
    actor_rollout_ref.ref.megatron.use_remove_padding=False \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl_grpo_example_gsm8k_math' \
    trainer.experiment_name='Qwen3.5-9B_megatron_sglang' \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=False \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.total_epochs=10 $@ \
    > "$LOG_FILE" 2>&1 &

