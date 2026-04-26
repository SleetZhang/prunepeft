# 自动化微调npu跑通

## 一、gpu流程

1.租卡：
 a800 cuda12.4 pytorch 2.5.1 python3.12
 克隆仓库：

```
git clone https://ghfast.top/https://github.com/SleetZhang/prunepeft.git
```

2.下载模型：
 huggingface命令行工具：

```
pip install -U huggingface_hub
```

环境变量：

```
export HF_ENDPOINT=https://hf-mirror.com
```

下载模型：hf auth login行不通
 要先在huggingface申请

```
hf download meta-llama/Llama-2-7b-hf \
  --token= \
  --local-dir /root/autodl-tmp/models/Llama-2-7b-hf
```

创建虚拟环境：

```
conda create -n prunepeft python=3.10 -y
conda init bash && source /root/.bashrc
conda activate prunepeft
pip install --upgrade pip
pip install modelscope#huggingface模型没下好

modelscope download \
  --model "shakechen/Llama-2-7b-hf" \
  --local_dir /root/autodl-tmp/ckpts/pretrained/Llama-2-7b-hf
  
```

安装pytorch：（python3.10）2.5.1报错，升2.6.0

```
#报错
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

#验证
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
PY
```

安装依赖：

```
#这个路径不对且需要python3.12
pip install -r requirements-ref.txt

pip install -r requirements.txt
pip install -e peft
```

缓存目录：

```
export HF_ENDPOINT=https://hf-mirror.com  
export HF_HOME=/root/autodl-tmp/hf  
export HF_DATASETS_CACHE=/root/autodl-tmp/hf/datasets  
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf/transformers
```

单独下载一个数据集：

```
python - <<'PY'  
from datasets import load_dataset  
ds = load_dataset("meta-math/MetaMathQA", split="train")  
print(ds)  
PY
```

先跑通stage0,1：

```
CUDA_VISIBLE_DEVICES=0 python examples/controller.py \
--dataset meta_math \
--adapter_types lora,bottleneck \
--lora_rank 8 \
--lora_alpha 16 \
--lora_dropout 0.1 \
--bottleneck_size 32 \
--bottleneck_dropout 0.1 \
--sample_size 64 \
--pruning_rounds 1 \
--modules_per_round 2 \
--learning_rate 2e-4 \
--bottleneck_learning_rate 2e-4 \
--test_dataset none \
--save_path /root/autodl-tmp/prunepeft_smoke \
--stage 0,1
```

报错： 没拿到llama的通行证，需要改代码，拿到后再解决，改了代码之后解决了
 跑通全流程：

```
CUDA_VISIBLE_DEVICES=0 python examples/controller.py \
  --dataset meta_math \
  --adapter_types lora,bottleneck \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.1 \
  --bottleneck_size 32 \
  --bottleneck_dropout 0.1 \
  --sample_size 128 \
  --pruning_rounds 4 \
  --modules_per_round 8 \
  --learning_rate 2e-4 \
  --bottleneck_learning_rate 2e-4 \
  --test_dataset gsm8k \
  --save_path /root/autodl-tmp/prunepeft_full \
  --stage 0,1,2,3
```

时间：3h

## 二、**NPU跑通**

0.克隆仓库

```
git clone https://ghfast.top/https://github.com/SleetZhang/prunepeft.git
```

1.修改了 examples/controller.py  examples/data.py   examples/utils.py    peft/src/peft/tuners/lora/layer.py peft/src/peft/tuners/prunepeft/lora_layer.py  后两个不改不影响跑通
 2.下载模型（modelscope），下载数据集：安装完依赖再下载

```
modelscope download \
  --model "shakechen/Llama-2-7b-hf" \
  --local_dir /root/autodl-tmp/ckpts/pretrained/Llama-2-7b-hf
  
```

3.安装依赖：requirements.txt注释掉flash-attn

```
pip install -r requirements.txt
pip install -e peft

export HF_ENDPOINT=https://hf-mirror.com
python - <<'PY'
from datasets import load_dataset
ds = load_dataset("meta-math/MetaMathQA", split="train")
print(ds)
ds.to_json("/root/autodl-tmp/datasets/metamath_train.jsonl")
print("saved:", "/root/autodl-tmp/datasets/metamath_train.jsonl")
PY

export HF_ENDPOINT=https://hf-mirror.com
export PRUNEPEFT_TOKENIZER_PATH=/root/autodl-tmp/ckpts/pretrained/Llama-2-7b-hf
export PRUNEPEFT_METAMATH_PATH=/root/autodl-tmp/datasets/metamath_train.jsonl
```

4.冒烟测试：（缺少数据集有报错）

```
ASCEND_RT_VISIBLE_DEVICES=0 python examples/controller.py \
  --dataset meta_math \
  --adapter_types lora,bottleneck \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.1 \
  --bottleneck_size 32 \
  --bottleneck_dropout 0.1 \
  --sample_size 64 \
  --pruning_rounds 1 \
  --modules_per_round 2 \
  --learning_rate 2e-4 \
  --bottleneck_learning_rate 2e-4 \
  --test_dataset none \
  --save_path /root/autodl-tmp/prunepeft_smoke_npu \
  --stage 0,1
```

5.全流程：（跑通了）

```
#huggingface 镜像设置不能缺，缺少测评阶段会报错
export HF_ENDPOINT=https://hf-mirror.com
export PRUNEPEFT_TOKENIZER_PATH=/root/autodl-tmp/ckpts/pretrained/Llama-2-7b-hf
export PRUNEPEFT_METAMATH_PATH=/root/autodl-tmp/datasets/metamath_train.jsonl

ASCEND_RT_VISIBLE_DEVICES=0 python examples/controller.py \
  --dataset meta_math \
  --adapter_types lora,bottleneck \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.1 \
  --bottleneck_size 32 \
  --bottleneck_dropout 0.1 \
  --sample_size 128 \
  --pruning_rounds 4 \
  --modules_per_round 8 \
  --learning_rate 2e-4 \
  --bottleneck_learning_rate 2e-4 \
  --test_dataset gsm8k \
  --save_path /root/autodl-tmp/prunepeft_full_npu \
  --stage 0,1,2,3

#复现指令可行  准确率0.4632
ASCEND_RT_VISIBLE_DEVICES=0 python examples/controller.py \
  --dataset meta_math \
  --adapter_types lora,bottleneck \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.1 \
  --bottleneck_size 32 \
  --bottleneck_dropout 0.1 \
  --init_bottleneck_weights true \
  --adapter_layers "" \
  --lora_layers "" \
  --target_modules "q_proj,v_proj,k_proj,o_proj" \
  --sample_size 128 \
  --seed 42 \
  --bias none \
  --pruning_rounds 4 \
  --modules_per_round 8 \
  --test_dataset gsm8k \
  --save_path /root/autodl-tmp/prunepeft_full_npu_4_7 \
  --stage 0,1,2,3


#只训练lora 0.4503
ASCEND_RT_VISIBLE_DEVICES=0 python examples/controller.py \
  --dataset meta_math \
  --adapter_types lora \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.1 \
  --target_modules "q_proj,v_proj,k_proj,o_proj" \
  --sample_size 128 \
  --seed 42 \
  --bias none \
  --test_dataset gsm8k \
  --save_path /root/autodl-tmp/lora_npu_4_7 \
  --stage 2,3
#只训练dora 0.4193
ASCEND_RT_VISIBLE_DEVICES=0 python examples/controller.py \
  --dataset meta_math \
  --adapter_types dora \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.1 \
  --target_modules "q_proj,v_proj,k_proj,o_proj" \
  --sample_size 128 \
  --seed 42 \
  --bias none \
  --test_dataset gsm8k \
  --save_path /root/autodl-tmp/lora_npu_4_7 \
  --stage 2,3
  
#只训练adapter 
  ASCEND_RT_VISIBLE_DEVICES=0 python examples/controller.py \
  --dataset meta_math \
  --adapter_types bottleneck \
  --adapter_layers "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" \
  --bottleneck_size 32 \
  --bottleneck_dropout 0.1 \
  --bottleneck_learning_rate 5e-5 \
  --init_bottleneck_weights true  \
  --target_modules "q_proj,v_proj,k_proj,o_proj" \
  --sample_size 128 \
  --seed 42 \
  --bias none \
  --test_dataset gsm8k \
  --save_path /root/autodl-tmp/adapter_npu_4_7 \
  --stage 2,3

#复现指令不可行，剪枝0轮（原因是指令之间有空行）
  

DATASET="meta_math"

TEST_DATASET="gsm8k"

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

SAVE_PATH="./save/prunepeft_gsm8k"

STAGE=0,1,2,3

  

ASCEND_RT_VISIBLE_DEVICES=0 python examples/controller.py \

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

    ${MODEL_PATH:+--model_path "$MODEL_PATH"} \

    ${ADAPTER_PATH:+--adapter_path "$ADAPTER_PATH"} \

    ${SAVE_PATH:+--save_path "$SAVE_PATH"} \

    --stage "$STAGE"
```

时间：10h/

## 三、学习代码

0.npu代码修改：

```
# 第一步：import torch后，立即import torch_npu 
# 第二步：cuda()函数换为npu() 
#除多了import torch_npu和替换了cuda→npu，其他均相同，如torch.cuda→torch.npu
import torch 
import torch_npu

x = torch.randn(10000, 10000).cuda()
x = torch.randn(10000, 10000).npu()
```

1.warm up：
 先把每层都插入两个adapter，adapter_types为适配器哪些类型，target_modules然后决定lora插入每层的哪些模块，如qkvo，其他adapter由模型决定插入哪些模块
 adapter_layers/lora_layers：指定 哪些层要加 LoRA adapter
 流程：先都插入两个adapter，跑8步，初始没有warmup_results，跑完第一轮warmup才有。
 跑完8步根据梯度/激活/权重本身，得到各个层6种剪枝策略的top_p排名，然后根据排名信息去统计各个块中各种剪枝策略的计数信息，然后利用计数信息去横向和纵向对比去确定最终各个区域适合的剪枝策略
 例如：

```
rankings 如下：
 `` 剪枝方法     剪枝层排名
 gradient: [("lora_2", ...), ("lora_5", ...), ("lora_18", ...)] 
 snip: [("lora_3", ...), ("adapter_10", ...), ("lora_20", ...)] ```
 
 得到每个区域每种剪枝方法的计数信息
 { "lora 0-20%": {"gradient": 1, "snip": 1}, 
 "lora 20-50%": {"gradient": 1}, 
 "lora 50-80%": {"gradient": 1},
  "lora 80-100%": {"snip": 1}, 
  "adapter 20-50%":{"snip": 1}, ... }
  
  最终横向纵向确定每个区域适合的剪枝策略
  假设纵向得出 snip 最集中在 
lora 0-20%： lora 0-20% → {gradient} ∪ {snip} = {gradient, snip} ←获得两个策略 
lora 20-50% → {snip} ← snip被抢走但横向已分配，不受影响

存入final_allocation

有了warmup_results以后在迭代剪枝时计算排名会多计算混合策略的剪枝排名
```

2.final_allocation: dict | None  指定每个 block （8个块）偏好用哪些剪枝策略来综合打分。

```
final_allocation = {  
"LoRA 0-20%": ["gradient", "snip"],  
"LoRA 20-50%": ["activation", "minimum_weight"],  
"Adapter 0-20%": ["zeros", "values_below_threshold"]  
}
```

3.迭代剪枝：
 固定使用rankings['block']中的混合得分来进行剪枝，剪枝是通过删除lora/adapter的索引序号，每一次都用新的信息初始化base_model
 4.命令行参数：
 --dataset 指定训练集   meta-math/MetaMathQA
 --test_dataset gsm8k 指定测试集，测试集写死了三种，
 gsm8k（数学推理），直接输出准确率
 humaneval（代码生成），评估不是靠文字匹配而是靠运行结果
 mt-bench（多轮对话）Alpaca 数据集作为测试集，且不自动计算分数，需要后续调用 GPT-4 作为裁判打分