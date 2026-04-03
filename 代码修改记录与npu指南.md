# 代码修改记录

## 1.examples/utils.py

目的：删除cuda相关代码，改为device，新增清理缓存函数

```
第28行
-	torch.cuda.manual_seed(seed)
-   torch.backends.cudnn.deterministic = True
-    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def clear_device_cache():
    """Clear cache on available accelerator backend."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.empty_cache()
        
第190行减
model_config['use_safetensors'] = False

第324行加
device = next(model.parameters()).device

第333行减加
inputs = {k: v.cuda() for k, v in inputs.items()}
inputs = {k: v.to(device) for k, v in inputs.items()}

第349减加
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
inputs = tokenizer(input_text, return_tensors="pt").to(device)

```



## 2.examples/data.py

目的：从本地下载的模型加载llama和tokenizer，绕过huggingface鉴权，新增加载本地提前下载好的训练数据集

```
第10行加
def _resolve_llama_tokenizer_source() -> tuple[str, bool]:
    """
    Resolve tokenizer source path/id for Llama-2-7b.

    Priority:
    1) PRUNEPEFT_TOKENIZER_PATH env var
    2) /root/autodl-tmp/ckpts/pretrained/Llama-2-7b-hf
    3) meta-llama/Llama-2-7b-hf (remote)
    """
    local_candidates = [
        os.environ.get("PRUNEPEFT_TOKENIZER_PATH", "").strip(),
        "/root/autodl-tmp/ckpts/pretrained/Llama-2-7b-hf",
    ]
    for path in local_candidates:
        if path and os.path.isdir(path):
            return path, True
    return "meta-llama/Llama-2-7b-hf", False


def _load_local_metamath_from_path(path: str) -> Dataset:
    """
    Load local MetaMath-style dataset from a json/jsonl file.
    Required fields per sample: query, response. Optional: type.
    """
    import json

    if not os.path.exists(path):
        raise FileNotFoundError(f"PRUNEPEFT_METAMATH_PATH not found: {path}")

    samples = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "data" in data:
            samples = data["data"]
        elif isinstance(data, list):
            samples = data
        else:
            raise ValueError("Unsupported JSON structure for local metamath file.")
    else:
        raise ValueError("PRUNEPEFT_METAMATH_PATH must be .json or .jsonl")

    for s in samples:
        s.setdefault("type", "GSM")
    return Dataset.from_list(samples)


def _load_metamath_dataset(split="train") -> Dataset:
    """
    Load MetaMath dataset with offline fallbacks:
    1) PRUNEPEFT_METAMATH_PATH local file
    2) HF Hub
    3) Synthetic tiny dataset if PRUNEPEFT_ENABLE_SYNTHETIC_DATA=1
    """
    local_path = os.environ.get("PRUNEPEFT_METAMATH_PATH", "").strip()
    if local_path:
        log.info(f"Loading local MetaMath dataset from: {local_path}")
        return _load_local_metamath_from_path(local_path)

    try:
        return load_dataset("meta-math/MetaMathQA", split=split)
    except Exception as e:
        if os.environ.get("PRUNEPEFT_ENABLE_SYNTHETIC_DATA", "0") == "1":
            log.warning(
                f"Failed to load MetaMath from hub ({type(e).__name__}). "
                "Falling back to synthetic dataset for smoke testing."
            )
            synthetic = [
                {
                    "query": "If Tom has 3 apples and buys 2 more, how many apples does he have?",
                    "response": "Tom has 3 + 2 = 5 apples. The answer is: 5",
                    "type": "GSM",
                },
                {
                    "query": "What is 7 * 8?",
                    "response": "7 * 8 = 56. The answer is: 56",
                    "type": "GSM",
                },
            ]
            return Dataset.from_list(synthetic)
        raise
        
第379加减
-dataset = load_dataset("meta-math/MetaMathQA", split="train")
dataset = _load_metamath_dataset(split="train")

第382减加
-tokenizer = AutoTokenizer.from_pretrained("/root/ckpt/pretrained/Llama-2-7b-hf")
tokenizer_src, local_only = _resolve_llama_tokenizer_source()
tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, local_files_only=local_only)

第427减加
-dataset = load_dataset("meta-math/MetaMathQA", split="train")
dataset = _load_metamath_dataset(split="train")

第430减加
-tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer_src, local_only = _resolve_llama_tokenizer_source()
tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, local_files_only=local_only)

第475减加
-dataset = load_dataset("meta-math/MetaMathQA", split="train")
dataset = _load_metamath_dataset(split="train")
 
第478 515 523 减加
-tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer_src, local_only = _resolve_llama_tokenizer_source()
tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, local_files_only=local_only)

第560减加
-model_id = "/root/loraga/ckpts/pretrained/Llama-2-7b-hf"
model_id, _ = _resolve_llama_tokenizer_source()


```



## 3.examples/controller.py

目的：导入清理cache函数并使用

```
第60加
clear_device_cache,

第479加
if isinstance(adapter_types, str):
        adapter_types = [t.strip() for t in adapter_types.split(",") if t.strip()]

第775减加
-torch.cuda.empty_cache()
clear_device_cache()

第1027减加
-torch.cuda.empty_cache()
clear_device_cache()

第1256加
 else:

```



## 4.peft/src/peft/tuners/prunepeft/lora_layer.py和peft/src/peft/tuners/lora/layer.py

目的：cuda改为device

```
第390行减加
-if lora_A.device.type == "cuda":
-                lora_B = lora_B.to(lora_A.device)
if lora_A.device.type != "cpu":
    target_device = lora_A.device
elif lora_B.device.type != "cpu":
    target_device = lora_B.device
elif hasattr(torch, "npu") and torch.npu.is_available():
    target_device = torch.device("npu")

接着else后
	-if lora_B.device.type != "cuda":
	 -  lora_B = lora_B.to("cuda")
	ora_A = lora_A.to(lora_B.device)
    target_device = torch.device("cuda")
lora_A = lora_A.to(target_device)
lora_B = lora_B.to(target_device)

```





## 5.运行指南

## PrunePEFT 在昇腾 910B（64GB）跑通指南（HDK 25.2.0 / CANN 8.3.RC2 / torch-npu 2.7.1）

### 1. 当前仓库已做的 NPU 兼容改动

为了能在 NPU 上运行，仓库已做以下修改：

1. `seed_everything` 不再无条件调用 `torch.cuda.manual_seed`，只有 CUDA 可用时才调用。  
2. 新增 `clear_device_cache()`，会根据后端选择 `torch.cuda.empty_cache()` 或 `torch.npu.empty_cache()`。  
3. `controller.py` 中所有 `torch.cuda.empty_cache()` 已替换为 `clear_device_cache()`。  
4. `model_inference()` 不再写死 `.cuda()`/`.to("cuda")`，改为跟随 `model` 实际 device（可为 `npu:0`）。
5. `peft/src/peft/tuners/{lora,prunepeft}/lora_layer.py` 中 DoRA 的 `ephemeral_gpu_offload` 分支，已从写死 `cuda` 改为“优先当前参数设备，其次 npu，再 fallback cuda”。

> 这些改动确保主流程不会因硬编码 CUDA API 在昇腾上直接崩掉。

### 为什么之前看起来“不改 peft 也能跑”？

- 你的主流程参数是 `bf16` + 非量化路径，很多 `peft` 里 CUDA/bitsandbytes 代码分支（如 4bit/8bit）根本不会被触发。
- 但这不代表 `peft` 完全没有 CUDA 相关逻辑；在某些可选分支（例如 DoRA 的 offload）仍可能踩到硬编码，因此这次也补了对应兼容处理。

---

### 2. 环境准备（你当前版本）

你给出的版本是：
- HDK: 25.2.0
- CANN: 8.3.RC2
- Python: 3.11
- PyTorch/torch-npu: 2.7.1

建议先验证：

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('has_npu:', hasattr(torch, 'npu'))
print('npu_available:', torch.npu.is_available() if hasattr(torch, 'npu') else False)
if hasattr(torch, 'npu') and torch.npu.is_available():
    print('device_count:', torch.npu.device_count())
PY
```

---

### 3. 安装依赖建议（NPU）

### 3.1 先安装项目最小依赖

```bash
pip install -r requirements.txt
pip install -e peft
```

### 3.2 不建议在昇腾环境安装/启用的项

- `flash-attn`（CUDA 生态）
- `bitsandbytes` 的 CUDA 量化路径（NPU 通常不可用）

本仓库默认主流程配置 `model_dtype="bf16"`，不走 int8/nf4 量化路径，可以先不碰上述组件。

---

### 4. 本地模型与数据集（关键）

### 4.1 模型路径

`controller.py` 默认模型路径：

```text
/root/autodl-tmp/ckpts/pretrained/Llama-2-7b-hf
```

确保该目录存在，或做软链。

### 4.2 避免 gated 仓库 403

如果你已有本地 Llama2，强烈建议：

```bash
export PRUNEPEFT_TOKENIZER_PATH=/root/autodl-tmp/ckpts/pretrained/Llama-2-7b-hf
```

数据处理时会优先走这个本地 tokenizer 路径，避免访问 `meta-llama/Llama-2-7b-hf` 远端鉴权。

### 4.3 无法联网下载 `meta-math/MetaMathQA` 时怎么做

如果你的环境访问 HuggingFace 报 `Network is unreachable`，现在支持两种离线方式：

1. 指定本地 MetaMath 文件（json/jsonl，字段至少含 `query`、`response`）：

```bash
export PRUNEPEFT_METAMATH_PATH=/path/to/metamath.jsonl
```

2. 仅做冒烟时，启用极小合成数据回退：

```bash
export PRUNEPEFT_ENABLE_SYNTHETIC_DATA=1
```

> 建议：先用合成数据把流程链路（0,1,2,3）跑通，再替换为真实本地数据文件。

### 4.4 我本地没有 MetaMath，如何下载？

#### 方式 A：可联网直下（推荐）

```bash
# 可选：国内环境建议镜像
export HF_ENDPOINT=https://hf-mirror.com

python - <<'PY'
from datasets import load_dataset
ds = load_dataset("meta-math/MetaMathQA", split="train")
print(ds)
ds.to_json("/root/autodl-tmp/datasets/metamath_train.jsonl")
print("saved:", "/root/autodl-tmp/datasets/metamath_train.jsonl")
PY
```

下载后训练前设置：

```bash
export PRUNEPEFT_METAMATH_PATH=/root/autodl-tmp/datasets/metamath_train.jsonl
```

---

### 5. 运行命令（先冒烟再全流程）

### 5.1 冒烟（建议）

```bash
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

### 5.2 全流程

```bash
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
```

---

### 6. 昇腾常见问题

1. **RuntimeError: CUDA xxx not found**  
   说明还有 CUDA 硬编码路径，先确认你用的是本仓库最新代码。

2. **403 gated repo / tokenizer 下载失败**  
   设置 `PRUNEPEFT_TOKENIZER_PATH` 指向本地模型目录。

3. **meta_math 下载慢/失败**  
   先预下载到本地 cache，再启动训练。

---

### 7. 建议

先用 1 张 910B 把 `stage 0,1` 跑通，再扩展多卡。多卡前建议先确认 `accelerate` 在 NPU 环境的 launcher 配置。

---

### 8. 结果如何解读（你贴的日志属于“已跑通”）

如果你看到下面这些关键信号，说明流程闭环完成：

1. GSM8K 评估循环跑完（例如进度到 100%）。  
2. 打印 `Acc: ...`（如 `0.4632`）。  
3. 打印 `✅ 评估完成，GSM8K准确率: ...`。

这表示 `stage 0,1,2,3` 至少已经执行到评估收尾，不是中途崩溃。

---

### 9. 为什么 NPU 10 小时、A800 3 小时？

同样流程下出现 2~4 倍时长差异是常见现象，通常由以下因素叠加：

1. **算子生态差异**：某些算子在 CUDA 上优化更成熟，NPU 侧可能走较慢实现。  
2. **图模式/编译开销**：NPU 首轮图编译与 shape 适配会额外耗时。  
3. **DataLoader 与 CPU 预处理瓶颈**：当前脚本的数据预处理较重，若 CPU 侧跟不上会拖慢 NPU。  
4. **通信与并行策略**：单卡与多卡配置、accumulation、launch 参数都会显著影响吞吐。  
5. **评估阶段开销**：GSM8K 是逐批生成式评估，NPU 在 decode 场景不一定比 A800 快。

### 优化建议（按优先级）

1. 先在 NPU 跑 `--stage 0,1`，确认剪枝链路；再单独跑 `--stage 2` 和 `--stage 3`，拆分定位慢点。  
2. 降低评估频率/单独评估：先 `--test_dataset none` 完成训练，再单独跑评估。  
3. 提高数据侧吞吐：缓存数据、减少在线 tokenize 压力、检查 CPU 利用率。  
4. 调整 batch/accumulation：在不 OOM 的前提下增大吞吐。  
5. 若你上多卡，单独调优 `accelerate` 与 NPU 通信参数。