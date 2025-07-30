#!/bin/bash
export LOGDIR=checkpoints
mkdir -p $LOGDIR

DATASET="sudoku"
RUN_NAME=${DATASET}_sft_bs12
MODEL_PATH=/home/jyjang/d1/LLaDA-sft-s1k-merged
NUM_ITER=8 # number of policy gradient inner updates iterations

accelerate launch \
    --config_file accelerate.yaml \
    --main_process_port 12346 diffu_grpo_train.py \
    --config slurm_scripts/train.yaml \
    --model_path $MODEL_PATH \
    --num_iterations $NUM_ITER \
    --dataset $DATASET \
    --run_name $RUN_NAME \
    --output_dir checkpoints/$RUN_NAME \
    --temperature 0.3
