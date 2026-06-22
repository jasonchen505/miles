# LLM & Agent 应用及后训练面试准备指南

## 基于Miles RL框架的深度学习与面试要点

---

## 目录

1. [Miles框架概述与架构理解](#1-miles框架概述与架构理解)
2. [RLHF/RLAIF核心概念深度解析](#2-rlhfrlaif核心概念深度解析)
3. [训练-推理架构设计要点](#3-训练-推理架构设计要点)
4. [关键技术细节与面试深挖点](#4-关键技术细节与面试深挖点)
5. [Agent与多轮交互训练](#5-agent与多轮交互训练)
6. [低精度训练与量化技术](#6-低精度训练与量化技术)
7. [分布式系统与工程实践](#7-分布式系统与工程实践)
8. [常见面试问题与参考答案](#8-常见面试问题与参考答案)
9. [实习生项目介绍建议](#9-实习生项目介绍建议)

---

## 1. Miles框架概述与架构理解

### 1.1 什么是Miles？

Miles是一个**企业级的大规模强化学习训练框架**，专门针对LLM后训练（Post-Training）场景优化。它是slime框架的fork，集成了：
- **SGLang**: 高性能推理引擎
- **Megatron-LM**: 大规模分布式训练框架
- **Ray**: 分布式任务调度

**核心定位**: "A journey of thousand miles begins with a single rollout." —— 专注于底层系统优化，使大规模RL训练稳定、高效、可复现。

### 1.2 四大核心对象

```
┌─────────────────────────────────────────────────────────────┐
│                    Miles Training Loop                       │
├─────────────────────────────────────────────────────────────┤
│  1. Prompt Dataset (数据源)                                  │
│           ↓                                                  │
│  2. Rollout/SGLang Engines (推理生成)                        │
│           ↓                                                  │
│  3. Reward Model/Function (奖励计算)                         │
│           ↓                                                  │
│  4. Actor Model/Megatron (策略训练)                          │
│           ↓                                                  │
│      P2P Weight Sync → 回到Step 2                           │
└─────────────────────────────────────────────────────────────┘
```

**面试关键点**:
- 能清晰解释每个组件的职责和数据流
- 理解为什么需要将训练和推理解耦
- 知道权重同步的必要性（on-policy要求）

### 1.3 四旋钮不变量（Four-Knob Invariant）

```
rollout_batch_size × n_samples_per_prompt = global_batch_size × num_steps_per_rollout
```

**含义**:
- 左边：每次rollout生成的样本总数
- 右边：每次训练消耗的样本总数
- **必须相等**，否则系统报错

**面试深挖**:
- Q: 为什么要保持这个不变量？
- A: 确保每个生成的样本都被训练消耗，避免数据浪费或训练不足
- Q: 如果想做off-policy训练怎么办？
- A: 增大`num_steps_per_rollout`，多次使用同一批rollout数据

---

## 2. RLHF/RLAIF核心概念深度解析

### 2.1 GRPO (Group Relative Policy Optimization)

**算法原理**:
```python
# GRPO核心思想
advantages = rewards - mean(rewards_in_group)  # 组内相对优势
loss = -advantages * ratio(new_policy, old_policy)  # 策略梯度
loss = clip(loss, 1-eps, 1+eps)  # PPO裁剪
```

**Miles实现细节** (`miles/backends/training_utils/loss_hub/losses.py`):
- 支持多种优势估计器：`grpo`, `gspo`, `ppo`, `reinforce_plus_plus`
- 支持非对称裁剪：`eps_clip`和`eps_clip_high`可不同（DAPO风格）
- 支持per-sample和per-token两种loss reduction

**面试深挖点**:
1. **GRPO vs PPO的区别**:
   - GRPO不需要value network，直接用组内相对奖励作为优势
   - PPO需要额外的critic网络估计baseline
   
2. **为什么GRPO适合LLM**:
   - LLM生成的样本通常有明确的正确/错误标签
   - 组内比较天然适合这种binary reward场景
   - 节省了训练critic的计算开销

3. **裁剪机制的作用**:
   - 防止策略更新过大导致训练崩溃
   - 非对称裁剪可以更灵活地控制探索-利用平衡

### 2.2 KL散度约束

**三种KL使用方式**:

| 方式 | 实现 | 作用 |
|------|------|------|
| KL作为loss项 | `--use-kl-loss --kl-loss-coef 0.01` | 直接约束策略不偏离reference太远 |
| KL作为监控 | `--use-kl-loss --kl-loss-coef 0.0` | 只观察KL，不参与梯度计算 |
| KL在reward中 | reward - kl_coef * kl | 在奖励中加入KL惩罚 |

**Miles支持的KL类型** (`--kl-loss-type`):
- `low_var_kl`: Schulman k3估计器，方差更低
- 标准KL散度

**面试深挖**:
- Q: 为什么需要KL约束？
- A: 防止reward hacking，确保模型不会为了获得高reward而生成无意义的文本
- Q: KL约束太强会有什么问题？
- A: 模型无法学习新知识，训练效果受限

### 2.3 TIS (Truncated Importance Sampling)

**问题背景**: 当训练策略和推理策略不一致时（off-policy），会产生偏差。

**Miles实现** (`miles/backends/training_utils/loss_hub/corrections.py`):

```python
# Vanilla TIS
tis = exp(train_log_probs - rollout_log_probs)  # importance ratio
tis_weights = clamp(tis, low, high)  # 截断
pg_loss = pg_loss * tis_weights

# ICE-POP (更激进)
ice_weight = where(in_range, ratio, 0)  # 超出范围直接置零
```

**应用场景**:
- BF16训练 + FP8推理时的精度差异
- Partial rollout导致的off-policy样本
- 异步训练中权重版本不一致

**面试深挖**:
- Q: 为什么需要截断？
- A: 原始importance sampling可能产生极大的权重，导致梯度不稳定
- Q: TIS和rejection sampling的区别？
- A: TIS是加权，rejection sampling是直接丢弃

---

## 3. 训练-推理架构设计要点

### 3.1 为什么需要分离训练和推理？

**传统方案的问题**:
1. 训练框架（PyTorch/Megatron）和推理框架（vLLM/SGLang）优化目标不同
2. 训练需要大batch、高吞吐；推理需要低延迟、高并发
3. 内存管理策略完全不同

**Miles的解决方案**:
- 训练端：Megatron-LM，支持TP/PP/CP/EP多种并行
- 推理端：SGLang，高性能serving
- 权重同步：P2P transfer或NCCL broadcast

### 3.2 P2P权重传输

**传统broadcast的问题**:
- 所有rank接收相同数据，冗余传输
- MoE模型expert分布不均匀，效率低

**P2P优化** (`miles/backends/megatron_utils/update_weight/p2p.py`):
1. 每个training rank只发送目标engine需要的权重切片
2. 使用RDMA直接写入远端内存
3. 支持bucketed transfer，减少传输次数

**性能提升**（从官方profiling结果）:
- Qwen3-235B: 10.7s → 3.1s (70.6% reduction)
- Kimi-K2 (1T): 53.3s → 7.2s (86.4% reduction)

**面试深挖**:
- Q: P2P相比broadcast的优势在什么情况下最明显？
- A: MoE模型、多节点场景，因为expert并行导致每个engine只需要部分权重

### 3.3 Colocate模式

**适用场景**: GPU资源有限时，训练和推理共享GPU

**实现机制**:
```python
if args.colocate:
    # 训练时offload推理状态
    await rollout_manager.offload(tags=[CUDA_GRAPH, KV_CACHE, WEIGHTS])
    
    # 推理时offload训练状态  
    await actor_model.offload()
```

**关键配置**:
- `--sglang-mem-fraction-static 0.8`: 给SGLang留够内存
- `--offload-train`/`--offload-rollout`: 控制offload行为

---

## 4. 关键技术细节与面试深挖点

### 4.1 R3 (Rollout Routing Replay)

**问题背景**: MoE模型在RL训练中不稳定

**根本原因**:
```
Rollout: router选择 experts {2, 7}
Training: router选择 experts {2, 8}  (数值精度差异导致)
→ 梯度计算基于错误的expert → 训练发散
```

**R3解决方案**:
1. 推理时记录routing decisions: `(seq_len-1, num_layers, top_k)` int32数组
2. 训练时replay这些routing decisions
3. 确保training和inference使用完全相同的expert

**内存开销**: 60MB/sample (32K tokens, 60 layers, top_k=8)

**面试深挖**:
- Q: 为什么dense模型不需要R3？
- A: Dense模型没有router，所有token都经过相同的计算路径
- Q: R3和TIS的关系？
- A: R3解决routing mismatch，TIS解决概率分布mismatch，可以同时使用

### 4.2 FP8统一精度训练

**问题**: 训练用BF16，推理用FP8 → 精度不一致

**Miles的解决方案**:
```
Rollout (forward): FP8 GEMM
Trainer (forward): FP8 GEMM (matching quant config)
Trainer (backward): BF16 gradients
Optimizer: BF16 master weights
```

**支持的格式**:
- Block-wise FP8 (128×128): Hopper/Blackwell
- MXFP8 (1×32): Blackwell only
- NVFP4 (实验性): MoE expert only

**面试深挖**:
- Q: 为什么backward必须用BF16？
- A: 梯度需要高精度来保证优化方向正确，FP8梯度会导致训练不稳定
- Q: 如何选择合适的量化格式？
- A: 根据硬件、模型大小、精度要求综合考虑

### 4.3 Speculative Decoding在RL中的应用

**基本原理**:
1. Draft model快速生成N个候选token
2. Target model一次性验证这N个token
3. 接受验证通过的token，拒绝错误的

**Miles的创新 - Online SFT for Draft**:
```python
--mtp-num-layers 1
--enable-mtp-training
--mtp-loss-scaling-factor 0.2
```

**解决的问题**: 长时间RL训练后，draft model与target model分布漂移，接受率下降

**面试深挖**:
- Q: 为什么draft model需要online更新？
- A: RL训练会改变target model的分布，frozen draft model会逐渐失效
- Q: MTP loss的权重如何选择？
- A: 0.2是经验值，太大会影响主任务学习，太小更新不够

---

## 5. Agent与多轮交互训练

### 5.1 Search-R1实现分析

**核心设计** (`examples/search-r1/generate_with_search.py`):

```python
async def generate(args, sample, sampling_params):
    for turn in range(max_turns):
        # 1. 生成response
        output = await post(url, payload)
        
        # 2. 解析action
        action, content = postprocess_predictions(output)
        
        # 3. 执行action
        if action == "search":
            search_results = await search(content)
            next_obs = f"<information>{search_results}</information>"
            # observation tokens的loss_mask = 0
            loss_mask += [0] * len(obs_tokens)
        elif action == "answer":
            break
    
    # 收集log probs用于TIS
    rollout_log_probs = [...]
```

**关键设计点**:
1. **Loss mask**: 只对模型生成的token计算loss，observation tokens不参与
2. **Log prob收集**: 用于后续的off-policy correction
3. **多轮状态管理**: 需要维护完整的对话历史

**面试深挖**:
- Q: 为什么observation tokens的loss_mask要设为0？
- A: 这些tokens不是模型生成的，计算loss没有意义，反而会干扰学习
- Q: 多轮交互中如何处理上下文长度限制？
- A: 需要设计截断策略，或者使用long-context模型

### 5.2 Multi-Agent系统设计

**Miles的Multi-Agent实现** (`examples/multi_agent/rollout_with_multi_agents.py`):

```python
async def generate_with_multi_agents(args, sample, sampling_params):
    # 1. 调用自定义的multi-agent系统
    samples = await custom_multi_agent_func(args, sample)
    
    # 2. 随机shuffle（避免顺序偏差）
    random.shuffle(samples)
    
    return samples
```

**Agent系统示例** (`examples/multi_agent/agent_system.py`):
- 包含多个specialized agents
- 支持agent间通信
- 每个agent可以有不同的reward函数

**面试深挖**:
- Q: Multi-agent RL相比single-agent有什么挑战？
- A: 
  1. Credit assignment问题：如何分配reward给各个agent
  2. 探索空间指数增长
  3. 需要处理agent间的协作和竞争

### 5.3 On-Policy Distillation (OPD)

**核心思想**: 用teacher模型的输出分布指导student模型训练

**公式**:
```
Â_t = A_t - λ_opd * KL(P_student || P_teacher)_t
```

**两种teacher模式**:

| 模式 | 适用场景 | 实现方式 |
|------|----------|----------|
| SGLang teacher | teacher架构不同/太大 | 外部SGLang服务器计算teacher logprobs |
| Megatron teacher | 同架构、可装入显存 | Megatron内部加载teacher模型 |

**Top-k token策略** (`--opd-top-k-strategy`):
- `only-student`: 用student的top-k tokens
- `only-teacher`: 用teacher的top-k tokens
- `intersection`: 两者的交集
- `union`: 两者的并集

**面试深挖**:
- Q: OPD和传统KD的区别？
- A: OPD是on-policy的，student在自己的rollout上训练；传统KD是off-policy的，用teacher的数据训练
- Q: 为什么OPD效果更好？
- A: 避免了distribution mismatch，student学习的是自己分布下的知识

---

## 6. 低精度训练与量化技术

### 6.1 INT4 QAT (Quantization-Aware Training)

**技术原理**:
- W4A16: 4-bit权重，16-bit激活
- 训练时模拟量化误差，让权重学会适应量化

**实现流程**:
```bash
# 1. 校准量化
python tools/convert_hf_to_int4.py \
    --quant-type W4A16 \
    --num-calibration-samples 256

# 2. QAT训练
OPEN_TRAINING_INT4_FAKE_QAT_FLAG=1
OPEN_TRAINING_INT4_GROUP_SIZE=128
```

**适用场景**: 1TB+模型在单机H200上训练

**面试深挖**:
- Q: QAT和PTQ (Post-Training Quantization)的区别？
- A: QAT在训练时加入量化模拟，精度更高；PTQ是训练后量化，可能有精度损失
- Q: 为什么INT4用group size 128？
- A: 更小的group size精度更高但overhead更大，128是精度和效率的平衡点

### 6.2 精度匹配的重要性

**问题根源**:
```
训练 BF16 + 推理 FP8 → 数值差异累积 → 训练崩溃
```

**Miles的解决方案矩阵**:

| 方案 | 训练精度 | 推理精度 | 适用场景 |
|------|----------|----------|----------|
| BF16+FP8 | BF16 | FP8 | 快速实验 |
| Unified FP8 | FP8 | FP8 | 生产环境 |
| MXFP8 | MXFP8 | MXFP8 | Blackwell硬件 |
| INT4 QAT | INT4 | INT4 | 超大模型 |

**面试深挖**:
- Q: 为什么不直接用FP8训练？
- A: 梯度计算需要高精度，FP8梯度会导致训练不稳定
- Q: 如何检测训练-推理mismatch？
- A: 监控`train_rollout_logprob_abs_diff`指标

---

## 7. 分布式系统与工程实践

### 7.1 Ray在Miles中的应用

**核心职责**:
1. **Placement Group管理**: 分配和管理GPU资源
2. **Actor模型**: 封装training和rollout的远程调用
3. **任务调度**: 协调training和rollout的执行顺序

**关键代码** (`miles/ray/placement_group.py`):

```python
def create_placement_groups(args):
    # 创建placement group
    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    pg = placement_group(bundles, strategy="PACK")
    
    # 获取GPU信息并排序
    # 返回actor、critic、rollout的placement group
```

**面试深挖**:
- Q: 为什么用Ray而不是直接用PyTorch分布式？
- A: Ray提供更灵活的资源调度，支持异构任务（training + inference）的协调

### 7.2 异步训练架构

**Fully Async Rollout** (`train_async.py`):

```
┌─────────────────────────────────────────────────────────┐
│  Background Rollout Worker                              │
│  ┌─────────────────────────────────────────────────────┐│
│  │ 持续生成samples → push到queue                       ││
│  └─────────────────────────────────────────────────────┘│
│                          ↓                              │
│  ┌─────────────────────────────────────────────────────┐│
│  │ Trainer                                             ││
│  │ 1. 从queue drain batch                              ││
│  │ 2. optimizer step                                   ││
│  │ 3. sync weights to rollout engines                  ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
```

**关键参数**:
- `--max-weight-staleness`: 允许的权重版本滞后
- `--sglang-server-concurrency`: 推理并发度

**面试深挖**:
- Q: 异步训练的trade-off是什么？
- A: 吞吐量提升 vs. on-policy程度降低（样本可能由旧策略生成）
- Q: 如何监控异步训练的质量？
- A: 监控queue depth、staleness stats、train_rollout_kl

### 7.3 容错机制

**常见故障场景**:
1. OOM: 推理时sequence太长
2. 网络超时: 权重同步失败
3. NaN/Inf: 数值不稳定

**Miles的处理**:
- **Partial Rollout**: 保留未完成的样本，下次继续
- **Dynamic Sampling**: 过滤reward异常的样本
- **Abort机制**: 推理超时时优雅终止

---

## 8. 常见面试问题与参考答案

### 8.1 基础概念类

**Q1: 介绍一下RLHF的基本流程？**

A: RLHF包含三个阶段：
1. **SFT (Supervised Fine-Tuning)**: 在高质量数据上微调
2. **Reward Model Training**: 训练奖励模型，学习人类偏好
3. **RL Optimization**: 用RL算法（如PPO/GRPO）优化策略

Miles主要聚焦第三阶段，提供了完整的RL训练框架。

**Q2: GRPO相比PPO的优势是什么？**

A: 
1. **无需Critic网络**: 节省显存和计算
2. **组内相对比较**: 天然适合binary reward场景
3. **实现简单**: 代码量少，易于调试

**Q3: 什么是on-policy和off-policy？**

A:
- **On-policy**: 训练数据由当前策略生成（严格）
- **Off-policy**: 训练数据可以由旧策略生成（高效但有偏差）

Miles通过TIS/MIS来处理off-policy带来的偏差。

### 8.2 技术深挖类

**Q4: 为什么MoE模型的RL训练不稳定？**

A: 核心原因是**train-inference mismatch**:
1. Router是learned的，数值精度差异会导致选择不同expert
2. 不同expert的权重更新不同步
3. 误差在数百层中累积，最终导致训练崩溃

解决方案：R3（记录并replay routing decisions）

**Q5: 如何设计一个高效的权重同步机制？**

A: Miles的方案：
1. **Bucketed transfer**: 将权重打包成bucket，减少传输次数
2. **P2P传输**: 每个training rank只发送目标需要的数据
3. **异步传输**: 训练和传输可以重叠
4. **Zero-copy**: 使用CUDA IPC避免内存拷贝

**Q6: 解释一下TIS的原理和作用**

A: TIS用于校正off-policy带来的偏差：
```python
importance_ratio = exp(new_logprob - old_logprob)
weights = clamp(importance_ratio, low, high)
loss = loss * weights
```
- `ratio > 1`: 新策略更可能生成这个token，放大loss
- `ratio < 1`: 新策略不太可能生成这个token，缩小loss
- 截断防止极端权重

### 8.3 系统设计类

**Q7: 如何设计一个支持多轮交互的RL训练系统？**

A: 关键设计点：
1. **状态管理**: 维护完整的对话历史
2. **Loss mask**: 只对模型生成的token计算loss
3. **超时处理**: 多轮交互可能很长，需要abort机制
4. **上下文管理**: 处理context length限制

**Q8: 如何处理训练中的数值不稳定问题？**

A: 多层次防护：
1. **精度匹配**: 训练和推理使用相同精度
2. **KL约束**: 防止策略偏离太远
3. **梯度裁剪**: 防止梯度爆炸
4. **TIS/MIS**: 校正off-policy偏差
5. **动态采样**: 过滤异常样本

### 8.4 工程实践类

**Q9: 如何调试RL训练中的问题？**

A: 监控关键指标：
1. **Loss曲线**: loss不下降可能是reward问题
2. **Reward分布**: 全部相同说明reward collapse
3. **KL散度**: 太大说明策略偏离太远
4. **PPO clip fraction**: 太大说明更新太激进
5. **Rollout vs train logprob diff**: 检测mismatch

**Q10: 如何优化大规模RL训练的效率？**

A: Miles的优化策略：
1. **异步训练**: overlap rollout和training
2. **FP8推理**: 加速generation
3. **Speculative decoding**: 加速generation
4. **P2P权重传输**: 加速weight sync
5. **Dynamic batching**: 提高GPU利用率

---

## 9. 实习生项目介绍建议

### 9.1 项目背景介绍模板

> "我参与了Miles框架的开发/使用，这是一个企业级的LLM后训练RL框架。项目的核心挑战是解决大规模MoE模型在RL训练中的稳定性问题，同时保证训练效率。"

### 9.2 技术亮点提炼

**选择1-2个深入讲解**:

1. **R3技术**:
   - 问题：MoE模型RL训练不稳定
   - 分析：train-inference mismatch导致routing差异
   - 方案：记录并replay routing decisions
   - 结果：成功稳定了DeepSeek-V3/R1的训练

2. **FP8统一训练**:
   - 问题：BF16训练+FP8推理的精度差异
   - 方案：统一使用FP8进行forward pass
   - 结果：消除了精度mismatch，训练更稳定

3. **Multi-Agent训练**:
   - 问题：复杂任务需要多个agent协作
   - 方案：设计了灵活的multi-agent框架
   - 结果：支持了doctor-patient、deepresearch等场景

### 9.3 面试中可能的追问

**准备好回答**:
1. "你遇到的最大技术挑战是什么？如何解决的？"
2. "这个项目的性能瓶颈在哪里？你如何优化？"
3. "如果让你重新设计，你会做什么不同的选择？"
4. "这个技术可以应用到其他什么场景？"

### 9.4 展示学习能力

**强调**:
- 阅读和理解大规模代码库的能力
- 快速学习新框架（Megatron、SGLang、Ray）的能力
- 将理论知识（RL算法）应用到实际系统的能力
- 解决复杂工程问题的能力

---

## 附录：关键代码路径速查

| 模块 | 路径 | 功能 |
|------|------|------|
| 训练主循环 | `train.py` | 同步训练入口 |
| 异步训练 | `train_async.py` | 异步训练入口 |
| RL Loss | `miles/backends/training_utils/loss_hub/losses.py` | GRPO/PPO/SFT loss |
| 优势估计 | `miles/backends/training_utils/loss_hub/advantages.py` | 优势计算 |
| TIS修正 | `miles/backends/training_utils/loss_hub/corrections.py` | Off-policy修正 |
| SGLang Rollout | `miles/rollout/sglang_rollout.py` | 推理生成 |
| 权重同步 | `miles/backends/megatron_utils/update_weight/` | P2P/broadcast |
| Sample类型 | `miles/utils/types.py` | 核心数据结构 |
| 参数解析 | `miles/utils/arguments.py` | CLI参数定义 |

---

## 总结

掌握Miles框架需要理解三个层次：

1. **算法层**: GRPO/PPO、KL约束、TIS/MIS等RL核心概念
2. **系统层**: 分布式训练、推理优化、权重同步等工程实践
3. **应用层**: Multi-turn、Multi-agent、VLM等实际场景

面试时，根据面试官的背景调整讲解深度：
- 偏算法：重点讲RL算法设计和优化
- 偏系统：重点讲分布式架构和性能优化
- 偏应用：重点讲具体场景的解决方案

---

*本文档基于Miles框架源码和文档整理，用于LLM算法实习面试准备。*
