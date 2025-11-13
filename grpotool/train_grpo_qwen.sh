export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export N_GPUS=8
export ROLLOUT_TP_SIZE=8
# export VLLM_ATTENTION_BACKEND=XFORMERS

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


# Change
export USE_KL_LOSS=False
export VAL_BEFORE_TRAIN=True
export CLIP_RATIO_HIGH=0.28
export TRAIN_BATCH_SIZE=256
export VAL_BATCH_SIZE=256
export LR=1e-6
export STEPS_SAVE=90 # To Change
export EPOCHS=4
export HYDRA_FULL_ERROR=1
export GROUP_SIZE=16
export DATA_DIR="xxxx" 
export TRAIN_FILE=$DATA_DIR/train.parquet  # To Change
export VAL_FILE=$DATA_DIR/test.parquet
export BASE_MODEL="Qwen/Qwen3-8B"
export PROJECT_NAME="Grpo_Tool"
export EXPERIMENT_NAME="xxxx" # e.g., "grpo-qwen2.5-3b"  # To Change

bash ./examples/grpo_trainer/grpo_qwen_eightgpus.sh
