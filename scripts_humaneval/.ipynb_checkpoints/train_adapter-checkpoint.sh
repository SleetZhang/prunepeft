#!/bin/bash

# Bottleneck Adapter训练脚本 - HumanEval数据集
# 使用Bottleneck适配器

cd /root/loraga

DATASET="codefeedback"
TEST_DATASET="humaneval"
ADAPTER_TYPES="bottleneck"
BOTTLENECK_SIZE=32
BOTTLENECK_DROPOUT=0.1
INIT_BOTTLENECK_WEIGHTS=true
ADAPTER_LAYERS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"
TARGET_MODULES="q_proj,v_proj,k_proj,o_proj"
SAMPLE_SIZE=128
SEED=420
BIAS="none"
MODEL_PATH=""
ADAPTER_PATH=""
SAVE_PATH="./save/adapter_humaneval4"
HUMANEVAL_RESULT_PATH="./eval_results/adapter_humaneval4"
STAGE=2,3

python examples/controller.py \
    --dataset "$DATASET" \
    --adapter_types "$ADAPTER_TYPES" \
    --adapter_layers "$ADAPTER_LAYERS" \
    --bottleneck_size "$BOTTLENECK_SIZE" \
    --bottleneck_dropout "$BOTTLENECK_DROPOUT" \
    --bottleneck_learning_rate 5e-5 \
    --init_bottleneck_weights "$INIT_BOTTLENECK_WEIGHTS" \
    --target_modules "$TARGET_MODULES" \
    --sample_size "$SAMPLE_SIZE" \
    --seed "$SEED" \
    --bias "$BIAS" \
    --test_dataset "$TEST_DATASET" \
    --humaneval_result_path "$HUMANEVAL_RESULT_PATH" \
    ${MODEL_PATH:+--model_path "$MODEL_PATH"} \
    ${ADAPTER_PATH:+--adapter_path "$ADAPTER_PATH"} \
    ${SAVE_PATH:+--save_path "$SAVE_PATH"} \
    --stage "$STAGE"
