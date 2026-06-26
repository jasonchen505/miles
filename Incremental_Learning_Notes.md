# Miles框架增量学习笔记

## 基于3090复现过程中的新知识点

---

## 目录

1. [第一轮：环境搭建与基础理解](#第一轮环境搭建与基础理解)
2. [第二轮：核心流程深入](#第二轮核心流程深入)
3. [第三轮：高级特性与调试](#第三轮高级特性与调试)
4. [第四轮：工程落地与优化](#第四轮工程落地与优化)
5. [对比前两轮文档的增量点](#对比前两轮文档的增量点)

---

## 第一轮：环境搭建与基础理解

### 新知识点 1：3090硬件限制的深层理解

**之前理解**：3090是消费级GPU，显存24GB

**深入理解**：

```python
# 3090的关键限制
3090_限制 = {
    "FP8": "不支持，需要H100/H200",
    "NVLink": "不支持，只有PCIe 4.0 x16",
    "Transformer Engine": "部分支持，某些kernel不可用",
    "FlashAttention-3": "不支持，需要FA2或FA1",
    "内存带宽": "936 GB/s，远低于H100的3.35 TB/s",
}

# 影响分析
影响 = {
    "训练速度": "比H100慢约5-10倍",
    "最大模型": "约1.5B（BF16），远小于H100的70B+",
    "并行策略": "只能用DP，不能用TP（无NVLink）",
    "精度": "只能用BF16，不能用FP8",
}
```

**实际影响**：
- 需要选择小模型（0.5B-1.5B）
- 不能使用FP8统一训练
- 不能使用Tensor Parallelism
- 权重同步较慢（PCIe vs NVLink）

### 新知识点 2：FSDP后端的价值

**之前理解**：FSDP是实验性特性，不推荐使用

**深入理解**：

```python
# FSDP的优势（在3090上）
FSDP优势 = {
    "无需checkpoint转换": "直接使用HuggingFace格式",
    "更简单的配置": "不需要Megatron的复杂参数",
    "CPU offload": "可以将优化器状态offload到CPU",
    "更好的兼容性": "支持更多模型架构",
}

# FSDP的劣势
FSDP劣势 = {
    "吞吐量较低": "比Megatron慢约20-30%",
    "不支持TP/PP/CP/EP": "只能用数据并行",
    "实验性": "可能有bug",
}
```

**实际应用**：
```bash
# 使用FSDP的配置
--train-backend fsdp
--gradient-checkpointing
--fsdp-cpu-offload  # 如果显存不够
```

### 新知识点 3：Colocate模式的显存管理

**之前理解**：Colocate模式让训练和推理共享GPU

**深入理解**：

```python
# Colocate模式的显存分配
显存分配 = {
    "SGLang": {
        "KV Cache": "~40-60% 显存",
        "模型权重": "~10-15% 显存",
        "CUDA Graph": "~5-10% 显存",
    },
    "Megatron": {
        "模型权重": "~10-15% 显存",
        "优化器状态": "~20-30% 显存",
        "梯度": "~10-15% 显存",
        "激活值": "~10-20% 显存",
    },
}

# 关键配置
--sglang-mem-fraction-static 0.6  # SGLang占60%显存
# 剩余40%给Megatron
```

**实际调优**：
```python
# 如果OOM，逐步降低
sglang_mem_fraction = [0.7, 0.6, 0.5, 0.4]
max_tokens_per_gpu = [8192, 4096, 2048, 1024]
```

---

## 第二轮：核心流程深入

### 新知识点 4：四旋钮不变量的实际意义

**之前理解**：`rollout_batch_size × n_samples_per_prompt = global_batch_size × num_steps_per_rollout`

**深入理解**：

```python
# 实际意义
四旋钮含义 = {
    "rollout_batch_size": "每次rollout采样的prompt数量",
    "n_samples_per_prompt": "每个prompt生成的response数量",
    "global_batch_size": "每次optimizer step使用的样本数",
    "num_steps_per_rollout": "每次rollout执行的optimizer step次数",
}

# 设计选择的trade-off
设计选择 = {
    "增大rollout_batch_size": "提高GPU利用率，但增加内存",
    "增大n_samples_per_prompt": "提高组内方差，但增加计算",
    "增大global_batch_size": "提高训练稳定性，但可能降低学习速度",
    "增大num_steps_per_rollout": "提高样本效率，但增加off-policy程度",
}
```

**3090上的配置策略**：
```python
# 3090推荐配置
配置 = {
    "rollout_batch_size": 8,      # 小batch，减少内存
    "n_samples_per_prompt": 4,    # 适中的样本数
    "global_batch_size": 32,      # 8 * 4 = 32
    "num_steps_per_rollout": 1,   # 严格on-policy
}
```

### 新知识点 5：Dynamic Sampling的必要性

**之前理解**：Dynamic Sampling过滤reward方差为0的组

**深入理解**：

```python
# 问题场景
问题 = {
    "场景1": "所有response都对，advantage=0，梯度=0",
    "场景2": "所有response都错，advantage=0，梯度=0",
    "场景3": "reward全部相同，无法区分好坏",
}

# Dynamic Sampling的解决
解决方案 = {
    "oversample": "生成更多样本（over-sampling-batch-size）",
    "filter": "过滤无效组（check_reward_nonzero_std）",
    "resample": "补充被过滤的样本",
}

# 代码实现
def check_reward_nonzero_std(args, samples):
    rewards = [sample.reward for sample in samples]
    std = torch.tensor(rewards, dtype=torch.float64).std()
    return std > 1e-8  # 只保留有方差的组
```

**实际效果**：
```python
# 假设oversample=32, rollout_batch_size=8
# 过滤比例约50-70%
# 最终有效样本率约30-50%
```

### 新知识点 6：TIS/MIS的数学原理

**之前理解**：TIS用于校正off-policy偏差

**深入理解**：

```python
# 数学原理
# 设 π_old 是生成样本的策略，π_new 是当前训练的策略
# importance ratio: ρ = π_new / π_old

# 问题：ρ可能非常大或非常小
# 解决：截断ρ到合理范围

# TIS实现
def tis(pg_loss, train_log_probs, rollout_log_probs, clip_low, clip_high):
    # 计算importance ratio
    ratio = torch.exp(train_log_probs - rollout_log_probs)
    
    # 截断
    weights = torch.clamp(ratio, min=clip_low, max=clip_high)
    
    # 加权loss
    return pg_loss * weights

# MIS的改进
def mis(pg_loss, train_log_probs, rollout_log_probs, lower_bound, upper_bound):
    ratio = torch.exp(train_log_probs - rollout_log_probs)
    
    # Mask模式：超出范围的直接丢弃
    in_range = (ratio >= lower_bound) & (ratio <= upper_bound)
    weights = ratio * in_range.float()
    
    return pg_loss * weights
```

**关键参数**：
```python
TIS参数 = {
    "tis_clip": 2.0,        # 上界
    "tis_clip_low": 0.5,    # 下界（1/tis_clip）
    "tis_level": "token",   # token级或sequence级
    "tis_mode": "clip",     # clip/truncate/mask
}
```

---

## 第三轮：高级特性与调试

### 新知识点 7：R3的工程实现细节

**之前理解**：R3记录并replay routing decisions

**深入理解**：

```python
# R3的数据流
数据流 = {
    "推理时": {
        "1": "SGLang计算router logits",
        "2": "选择top-k experts",
        "3": "记录routing decisions到meta_info",
        "4": "返回给Miles",
    },
    "训练时": {
        "1": "从sample.rollout_routed_experts读取",
        "2": "在forward pass中注入routing",
        "3": "跳过router计算，直接使用记录的routing",
    },
}

# 内存开销
内存开销 = "(seq_len - 1) × num_layers × top_k × 4 bytes"
# 32K tokens, 60 layers, top_k=8: 约60MB/sample
```

**3090上的考虑**：
```python
# 3090不支持FP8，所以不需要R3
# 但如果使用MoE模型，仍然需要R3来稳定训练
```

### 新知识点 8：调试工具的使用

**之前理解**：Miles提供debug-rollout-only和debug-train-only

**深入理解**：

```python
# 完整的调试工作流
调试流程 = {
    "步骤1": {
        "命令": "--debug-rollout-only --save-debug-rollout-data /tmp/debug.pt",
        "目的": "保存几条正常的rollout数据",
        "检查": "生成的文本是否正常",
    },
    "步骤2": {
        "命令": "--debug-train-only --load-debug-rollout-data /tmp/debug.pt",
        "目的": "用固定数据训练，排除推理随机性",
        "检查": "loss是否正常下降",
    },
    "步骤3": {
        "命令": "--debug-determinism",
        "目的": "开启确定性模式，便于复现",
        "检查": "多次运行结果是否一致",
    },
}

# 关键指标监控
监控指标 = {
    "loss": "应该稳定下降",
    "reward": "应该稳定上升",
    "grad_norm": "应该小于clip_grad",
    "kl": "应该保持较小",
    "pg_clipfrac": "应该小于0.5",
    "train_rollout_logprob_abs_diff": "应该保持稳定",
}
```

### 新知识点 9：常见问题的诊断

**之前理解**：FAQ中列出了常见问题

**深入理解**：

```python
# 问题诊断流程
诊断流程 = {
    "训练崩溃": {
        "检查1": "grad_norm是否NaN/Inf",
        "检查2": "loss是否spike",
        "检查3": "kl是否突然增大",
        "解决": "降低lr，增加kl_coef，使用tis",
    },
    "推理乱码": {
        "检查1": "chat template是否正确",
        "检查2": "checkpoint是否加载成功",
        "检查3": "stop token是否配置",
        "解决": "检查配置，重新转换checkpoint",
    },
    "OOM": {
        "检查1": "显存使用情况",
        "检查2": "batch size和序列长度",
        "检查3": "sglang-mem-fraction-static",
        "解决": "降低batch size，降低mem-fraction",
    },
    "训练太慢": {
        "检查1": "rollout_time vs train_time",
        "检查2": "GPU利用率",
        "检查3": "网络带宽",
        "解决": "调整资源分配，优化配置",
    },
}
```

---

## 第四轮：工程落地与优化

### 新知识点 10：3090上的性能优化

**之前理解**：3090比H100慢，需要小模型

**深入理解**：

```python
# 3090优化策略
优化策略 = {
    "模型选择": {
        "推荐": "Qwen2.5-0.5B, Qwen3-0.6B",
        "可尝试": "Qwen3-1.7B + LoRA",
        "不推荐": "3B+ 模型",
    },
    "批处理优化": {
        "use-dynamic-batch-size": True,
        "max-tokens-per-gpu": 4096,  # 根据显存调整
        "rollout-max-response-len": 512,  # 短序列
    },
    "显存优化": {
        "sglang-mem-fraction-static": 0.6,
        "gradient-checkpointing": True,
        "fsdp-cpu-offload": True,  # 如果需要
    },
    "并行策略": {
        "tensor-model-parallel-size": 1,  # 无NVLink
        "pipeline-model-parallel-size": 1,
        "context-parallel-size": 1,
        "使用DP": "8卡数据并行",
    },
}
```

### 新知识点 11：LoRA在3090上的应用

**之前理解**：LoRA减少可训练参数，节省显存

**深入理解**：

```python
# LoRA的优势
LoRA优势 = {
    "显存节省": "优化器状态只针对LoRA参数",
    "训练速度": "梯度计算量减少",
    "模型大小": "可以训练更大的模型",
}

# 3090上的LoRA配置
LoRA配置 = {
    "lora-rank": 16,  # 较小的rank
    "lora-alpha": 32,  # 通常2x rank
    "lora-dropout": 0.0,  # RL训练不需要
    "target-modules": "all-linear",  # 或指定模块
}

# 可以训练的模型大小
模型大小 = {
    "无LoRA": "0.5B-1.5B",
    "LoRA rank=16": "1.5B-3B",
    "LoRA rank=8": "3B-4B",
}
```

### 新知识点 12：Checkpoint管理最佳实践

**之前理解**：checkpoint用于恢复训练

**深入理解**：

```python
# Checkpoint管理策略
管理策略 = {
    "保存频率": {
        "开发阶段": "--save-interval 5",
        "验证阶段": "--save-interval 10",
        "生产阶段": "--save-interval 20",
    },
    "目录结构": {
        "load": "/path/to/checkpoints",  # 恢复点
        "save": "/path/to/checkpoints",  # 保存点（通常相同）
    },
    "版本管理": {
        "方法1": "按日期命名目录",
        "方法2": "按实验名称命名",
        "方法3": "使用git管理代码，checkpoint独立管理",
    },
}

# 3090上的考虑
3090考虑 = {
    "存储空间": "checkpoint可能很大，确保有足够空间",
    "保存速度": "PCIe较慢，保存可能需要更长时间",
    "恢复测试": "定期测试checkpoint恢复是否正常",
}
```

---

## 对比前两轮文档的增量点

### 相比第一轮文档（LLM_Interview_Preparation_Guide.md）

| 维度 | 第一轮理解 | 增量理解 |
|------|------------|----------|
| **硬件适配** | 知道3090是消费级GPU | 深入理解3090的具体限制（无FP8、无NVLink、PCIe带宽） |
| **模型选择** | 知道要选小模型 | 明确0.5B-1.5B的范围，以及LoRA可以扩展到3B-4B |
| **配置优化** | 知道要调整batch size | 理解四旋钮不变量的实际意义和trade-off |
| **调试方法** | 知道有debug工具 | 掌握完整的调试工作流（分离训练/推理、确定性模式） |

### 相比第二轮文档（LLM_Interview_Deep_Dive_Guide.md）

| 维度 | 第二轮理解 | 增量理解 |
|------|------------|----------|
| **TIS/MIS** | 知道原理和代码实现 | 理解数学原理，掌握参数调优方法 |
| **Dynamic Sampling** | 知道过滤策略 | 理解实际过滤比例和效果 |
| **R3** | 知道解决MoE不稳定 | 理解工程实现细节和内存开销 |
| **问题诊断** | 知道常见问题 | 掌握系统化的诊断流程 |

### 新增的实践知识点

```python
# 1. 环境搭建
新增 = {
    "Docker配置": "ipc=host, shm-size=32g, ulimit配置",
    "依赖管理": "Miles使用patched版本的SGLang和Megatron",
    "GPU验证": "nvidia-smi, torch.cuda.is_available()",
}

# 2. 训练启动
新增 = {
    "Ray配置": "head node启动, GPU数量配置",
    "环境变量": "PYTHONPATH, CUDA_DEVICE_MAX_CONNECTIONS",
    "任务提交": "ray job submit, runtime-env-json",
}

# 3. 监控和调试
新增 = {
    "日志位置": "trainer stdout, SGLang logs, Ray workers",
    "关键指标": "loss, reward, kl, grad_norm, pg_clipfrac",
    "调试命令": "nvidia-smi dmon, py-spy, ray timeline",
}

# 4. 问题解决
新增 = {
    "OOM": "降低batch size, mem-fraction, 使用gradient checkpointing",
    "训练不稳定": "降低lr, 增加kl_coef, 使用tis",
    "推理乱码": "检查chat template, checkpoint加载",
    "同步失败": "增加NCCL_TIMEOUT, 检查网络",
}
```

---

## 实践心得总结

### 3090复现的关键要点

```python
关键要点 = {
    "1. 选择合适的模型": "0.5B-1.5B，或使用LoRA",
    "2. 合理分配显存": "SGLang 60%, Megatron 40%",
    "3. 使用小batch": "rollout_batch_size=8, max_tokens=4096",
    "4. 短序列": "rollout_max_response_len=512",
    "5. 监控关键指标": "loss, reward, kl, grad_norm",
    "6. 掌握调试方法": "分离训练/推理，确定性模式",
}
```

### 学习路径建议

```python
学习路径 = {
    "Day 1": "环境搭建，验证GPU",
    "Day 2-3": "运行最小配置，理解核心流程",
    "Day 4-5": "完整流程，checkpoint管理",
    "Day 6-7": "高级特性（TIS, Dynamic Sampling）",
    "Day 8-10": "LoRA，自定义扩展",
}
```

### 面试准备补充

```python
面试补充 = {
    "硬件适配": "能讲清楚3090的限制和优化策略",
    "配置调优": "能解释四旋钮不变量和trade-off",
    "问题诊断": "能描述系统化的诊断流程",
    "工程实践": "能分享实际复现的经验和教训",
}
```

---

*本笔记基于8卡3090的实际复现过程整理，记录了从环境搭建到高级特性的完整学习路径。*
