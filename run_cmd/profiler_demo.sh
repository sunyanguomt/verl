set -x

# 直接使用下载的模型参数和mcore参数
HF_MODEL_PATH='/mnt/seed17/001688/models/Qwen3-1.7B'
DIST_CKPT_PATH='/mnt/seed17/001688/zhaoping/LLMs/MCORE/Qwen3-1.7B-mcore'

export MUSA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
# export MUSA_VISIBLE_DEVICES='7'
# export MUSA_EXECUTION_TIMEOUT=30000
export ACCELERATOR_BACKEND="musa"
export MCCL_PROTOS=2
export MCCL_CHECK_POINTERS=0

# export MCCL_IB_GID_INDEX=3
# export MUSA_BLOCK_SCHEDULE_MODE=1
# export MCCL_ALGOS=1
# export MCCL_BUFFSIZE=20480000


# export ACCELERATE_USE_FSDP=1
# export FSDP_CPU_RAM_EFFICIENT_LOADING=1
export VERL_LOGGING_LEVEL=INFO #INFO
export HYDRA_FULL_ERROR=1
#export MUSA_USERQ=1

# export MUSA_PATCH_PATH=/mnt/seed17/001688/zhaoping/Code/verl-musa-patch
export MEGATRON_PATH=/home/Megatron-LM
export VERL_PATH=/home/verl
export PYTHONPATH=${MEGATRON_PATH}:${VERL_PATH}:${MUSA_PATCH_PATH}:$PYTHONPATH


DATASET_PATH="/mnt/seed17/001688/zhaoping/Data/AM-Thinking-v1-RL-Dataset"
train_files=$DATASET_PATH/math_train.parquet
test_files=$DATASET_PATH/math_test.parquet

# 需要指定到 Verl 中对应config路径
CONFIG_PATH=$VERL_PATH/verl/trainer/config


# # 解决保存问题
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_SAFE_SERIALIZATION=1

env PYTHONPATH="$PYTHONPATH" \
    MUSA_VISIBLE_DEVICES="$MUSA_VISIBLE_DEVICES" \
    ACCELERATOR_BACKEND="$ACCELERATOR_BACKEND" \
    RAY_LOGGING_LEVEL=WARNING \
    RAY_DEDUP_LOGS=0 \
    RAY_ADDRESS="localhost:65379" \
python3 -u -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='ppo_megatron_trainer_demo.yaml'\
    algorithm.adv_estimator=grpo \
    global_profiler.tool='torch' \
    global_profiler.steps='[2,4]' \
    global_profiler.save_path=$VERL_PATH/run_cmd/outputs/profile \
    data.train_files=$train_files \
    data.val_files=$test_files \
    data.train_batch_size=16 \
    data.max_prompt_length=1024 \
    data.max_response_length=32 \
    data.filter_overlong_prompts=True \
    data.prompt_key=prompt \
    data.truncation='error' \
    actor_rollout_ref.model.path=$HF_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.profiler.enable=True \
    actor_rollout_ref.actor.profiler.all_ranks=False \
    actor_rollout_ref.actor.profiler.ranks=[0] \
    actor_rollout_ref.actor.profiler.tool_config.torch.contents=['cpu','cuda','memory','shapes','stack'] \
    actor_rollout_ref.actor.profiler.tool_config.torch.discrete=True \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.top_k=100 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_k=50 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.9 \
    actor_rollout_ref.rollout.profiler.enable=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.ref.profiler.enable=True \
    actor_rollout_ref.ref.profiler.all_ranks=False \
    actor_rollout_ref.ref.profiler.ranks=[0] \
    actor_rollout_ref.ref.profiler.tool_config.torch.contents=['cpu','cuda','memory','shapes','stack'] \
    actor_rollout_ref.ref.profiler.tool_config.torch.discrete=True \
    actor_rollout_ref.ref.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    algorithm.use_kl_in_reward=False \
    trainer.device='musa' \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl_grpo_example_gsm8k_math' \
    trainer.experiment_name='Qwen3_1.7b_megatron_sglang_profiler_demo' \
    trainer.n_gpus_per_node=1 \
    trainer.val_before_train=False \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.total_epochs=10 $@ \
    2>&1 | tee ../logs/run_ppo_profiler_demo.log