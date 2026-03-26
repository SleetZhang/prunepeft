#!/bin/bash

# DoRA训练脚本 - HumanEval数据集
# 使用DoRA (Weight-Decomposed LoRA) 适配器

cd /root/loraga

DATASET="codefeedback"
TEST_DATASET="humaneval"
ADAPTER_TYPES="dora"
LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.1
TARGET_MODULES="q_proj,v_proj,k_proj,o_proj"
SAMPLE_SIZE=128
SEED=42
BIAS="none"
MODEL_PATH=""
ADAPTER_PATH=""
SAVE_PATH="./save/dora_humaneval"
HUMANEVAL_RESULT_PATH="./eval_results/dora_humaneval"
STAGE=2,3

python examples/controller.py \
    --dataset "$DATASET" \
    --adapter_types "$ADAPTER_TYPES" \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
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
