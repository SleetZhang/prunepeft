#!/bin/bash

# PrunePEFT训练脚本 - HumanEval数据集
# 使用混合LoRA和Bottleneck适配器，带迭代剪枝

cd /root/loraga

DATASET="codefeedback"
TEST_DATASET="humaneval"
ADAPTER_TYPES="lora,bottleneck"
LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.1
BOTTLENECK_SIZE=32
BOTTLENECK_DROPOUT=0.1
INIT_BOTTLENECK_WEIGHTS=true
ADAPTER_LAYERS=""
LORA_LAYERS=""
TARGET_MODULES="q_proj,v_proj,k_proj,o_proj"
SAMPLE_SIZE=128
SEED=42
BIAS="none"
PRUNING_ROUNDS=4
MODULES_PER_ROUND=8
MODEL_PATH=""
ADAPTER_PATH=""
SAVE_PATH="./save/prunepeft_humaneval"
HUMANEVAL_RESULT_PATH="./eval_results/prunepeft_humaneval"
STAGE=0,1,2,3

python examples/controller.py \
    --dataset "$DATASET" \
    --adapter_types "$ADAPTER_TYPES" \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --bottleneck_size "$BOTTLENECK_SIZE" \
    --bottleneck_dropout "$BOTTLENECK_DROPOUT" \
    --init_bottleneck_weights "$INIT_BOTTLENECK_WEIGHTS" \
    --adapter_layers "$ADAPTER_LAYERS" \
    --lora_layers "$LORA_LAYERS" \
    --target_modules "$TARGET_MODULES" \
    --sample_size "$SAMPLE_SIZE" \
    --seed "$SEED" \
    --bias "$BIAS" \
    --pruning_rounds "$PRUNING_ROUNDS" \
    --modules_per_round "$MODULES_PER_ROUND" \
    --test_dataset "$TEST_DATASET" \
    --humaneval_result_path "$HUMANEVAL_RESULT_PATH" \
    ${MODEL_PATH:+--model_path "$MODEL_PATH"} \
    ${ADAPTER_PATH:+--adapter_path "$ADAPTER_PATH"} \
    ${SAVE_PATH:+--save_path "$SAVE_PATH"} \
    --stage "$STAGE"
