#!/bin/bash

# ============================================================================
# 简化的Ray多节点训练脚本
# ============================================================================

# 修改为你的实际项目路径
cd xxx

# 环境变量设置
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export N_GPUS=8
export ROLLOUT_TP_SIZE=4
export VLLM_ATTENTION_BACKEND=XFORMERS
export WITHLENGTH=0
export REFINEDREWARD=0
export COARSEREWARD=1
export STRICTMATCH=0
export CORRECTMAX1=1
export MAX1STEP30MAX3=0
export SCHEDULEREWARD=0
export SCHEDULELENGTH=0
export RESPONSE_HALF_REWARD=0
export TOOL_REWARD_VERSION=1
export ERRORMAX=0
export NUM_NODES=4
export USE_KL_LOSS=False
export CLIP_RATIO_HIGH=0.28
export TRAIN_BATCH_SIZE=512
export VAL_BATCH_SIZE=512
export LR=1e-6
export GROUP_SIZE=16
export HYDRA_FULL_ERROR=1
export BASE_MODEL="Qwen/Qwen3-32B"
export PROJECT_NAME="Grpo_Tool"
export VAL_BEFORE_TRAIN=True


# 需要改变的
export STEPS_SAVE=150
export SETPS_TEST=50
export EPOCHS=5
export DATA_DIR="xxx"
export TRAIN_FILE=$DATA_DIR/train.parquet
export VAL_FILE=$DATA_DIR/test.parquet
export EXPERIMENT_NAME="xxx"



# 提交Ray任务
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env=verl/trainer/runtime_env.yaml \
    --no-wait \
    -- \
    python -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files=$TRAIN_FILE \
        data.val_files=$VAL_FILE \
        data.train_batch_size=512 \
        data.val_batch_size=512 \
        data.max_prompt_length=4096 \
        data.max_response_length=1024 \
        data.filter_overlong_prompts=False \
        data.truncation=left \
        actor_rollout_ref.model.path=$BASE_MODEL \
        actor_rollout_ref.actor.clip_ratio_low=0.2 \
        actor_rollout_ref.actor.clip_ratio_high=0.28 \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=256 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.n=16 \
        actor_rollout_ref.ref.strategy=fsdp \
        actor_rollout_ref.actor.strategy=fsdp \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        trainer.critic_warmup=0 \
        trainer.logger='[console,wandb]' \
        trainer.project_name=$PROJECT_NAME \
        trainer.experiment_name=$EXPERIMENT_NAME \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=4 \
        trainer.save_freq=$STEPS_SAVE \
        trainer.test_freq=$SETPS_TEST \
        trainer.val_before_train=True \
        trainer.total_epochs=5 \
        trainer.default_local_dir=xxx/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}