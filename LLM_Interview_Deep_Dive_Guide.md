# LLM算法实习面试深度应对指南

## 五类核心问题与能力准备

---

## 目录

1. [第一类：底层原理深入理解](#第一类底层原理深入理解)
2. [第二类：实验和方案验证能力](#第二类实验和方案验证能力)
3. [第三类：问题定位能力](#第三类问题定位能力)
4. [第四类：工程落地能力](#第四类工程落地能力)
5. [第五类：业务与实际场景理解](#第五类业务与实际场景理解)

---

## 第一类：底层原理深入理解

### 核心要求
> 不是回答清楚概念，而是讲清楚这个方法解决什么问题，存在哪些局限性，有哪些改进方法

### 问题1：GRPO算法为什么被设计成这样？它解决了什么问题，有什么局限？

**标准答案框架**：

```
问题背景 → 设计动机 → 核心原理 → 局限性 → 改进方向
```

**详细回答**：

**1. 解决的问题**

传统PPO在LLM场景下的痛点：
- 需要额外的Critic网络，增加显存和计算开销
- Critic网络本身需要训练，引入额外的超参数和不稳定性
- LLM场景通常是binary reward（对/错），不需要复杂的value estimation

**2. 设计原理**

```python
# GRPO的核心思想：组内相对比较
advantages = rewards - mean(rewards_in_group)

# 对比PPO需要GAE计算
advantages, returns = compute_gae(values, rewards, gamma, lambd)
```

从代码看Miles的实现（`miles/backends/training_utils/loss_hub/advantages.py`）：

```python
if args.advantage_estimator in ["grpo", "gspo"]:
    rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
    returns = get_grpo_returns(rewards, kl)
    advantages = [r for r in returns]
```

**关键设计选择**：
- 不需要value network，直接用组内均值作为baseline
- 每个prompt生成N个response，形成一个"group"
- 组内比较天然适合"答案对不对"这种binary reward场景

**3. 局限性分析**

| 局限性 | 具体表现 | 根本原因 |
|--------|----------|----------|
| Reward homogeneity | 组内所有样本reward相同，advantage为0 | 任务太难或太简单 |
| 样本效率低 | 每个prompt只用一次就丢弃 | On-policy要求 |
| 长序列不稳定 | 组内方差随序列长度增大 | Credit assignment困难 |
| 无法处理连续reward | 适合binary，不适合连续值 | 设计假设 |

**4. 改进方向**

Miles中已实现的改进：
- **DAPO-style filtering**：过滤reward方差为0的组
- **Dynamic sampling**：oversample + 过滤，提高样本质量
- **Per-token loss**：`--calculate-per-token-loss`，长度加权

可进一步改进：
- 引入轻量级Critic辅助估计baseline
- 混合使用GRPO和PPO
- Curriculum learning，从简单任务开始

---

### 问题2：TIS（Truncated Importance Sampling）的原理是什么？为什么需要截断？

**回答框架**：

**1. 解决的问题**

训练-推理mismatch导致的off-policy问题：

```
场景：BF16训练 + FP8推理
训练时：log_prob = -2.302 (BF16精度)
推理时：log_prob = -2.315 (FP8精度)
差异：0.013

ratio = exp(-2.302 - (-2.315)) = exp(0.013) ≈ 1.013
```

单个token差异小，但序列级别累积：

```
100个token的序列：ratio = exp(0.013 * 100) = exp(1.3) ≈ 3.67
500个token的序列：ratio = exp(0.013 * 500) = exp(6.5) ≈ 665
```

**2. 为什么需要截断**

原始Importance Sampling的问题：
```python
# 未截断的IS权重
weights = exp(train_log_probs - rollout_log_probs)

# 问题：权重可能非常大
# 如果某个token的ratio是100，整个batch的梯度被这一个样本主导
# 导致训练不稳定
```

**3. Miles的实现**（`miles/backends/training_utils/loss_hub/corrections.py`）：

```python
def vanilla_tis_function(args, *, pg_loss, train_log_probs, rollout_log_probs, **kwargs):
    tis = torch.exp(old_log_probs - rollout_log_probs)
    tis_weights = torch.clamp(tis, min=args.tis_clip_low, max=args.tis_clip)
    pg_loss = pg_loss * tis_weights
```

**4. 截断的trade-off**

| 截断范围 | 优点 | 缺点 |
|----------|------|------|
| 太宽 | 保留更多信息 | 可能有极端权重 |
| 太窄 | 稳定 | 丢失有效信号 |

**5. MIS的改进**（`examples/train_infer_mismatch_helper/mis.py`）：

```python
# 支持三种模式
if args.tis_mode == "truncate":
    weights = truncate(weights, loss_mask, metrics, args.tis_upper_bound)
elif args.tis_mode == "clip":
    weights = clip(weights, loss_mask, metrics, lower_bound, upper_bound)
elif args.tis_mode == "mask":
    weights, modified_mask = mask(weights, loss_mask, metrics, lower_bound, upper_bound)
```

Mask模式更激进：直接丢弃超出范围的token，而不是加权。

---

### 问题3：MoE模型RL训练为什么不稳定？R3如何解决？

**深度分析**：

**1. 问题根源**

MoE的Router是learned的，依赖数值精度：

```python
# Router的计算
router_logits = linear(hidden_states)  # nn.Linear
top_k_indices = torch.topk(router_logits, k=top_k).indices

# 问题：BF16和FP8的计算结果可能不同
# BF16: router_logits[314] = [0.123, 0.456, 0.789, ...]
# FP8:  router_logits[314] = [0.124, 0.455, 0.790, ...]
# 如果top_k=2，可能选出不同的expert
```

**2. 误差累积**

```
Layer 1: 选择expert {2, 7} vs {2, 8}
  ↓ 输出不同
Layer 2: 输入不同 → router决策可能不同
  ↓ 误差放大
...
Layer 60: 累积误差巨大 → 训练崩溃
```

**3. R3的解决方案**

从Miles的实现看（`miles/router/router.py`）：

```python
# 推理时记录routing decisions
if args.use_rollout_routing_replay:
    payload["return_routed_experts"] = True

# 存储格式：(seq_len-1, num_layers, top_k) int32
sample.rollout_routed_experts = np.frombuffer(
    pybase64.b64decode(output["meta_info"]["routed_experts"]),
    dtype=np.int32,
).reshape(len(sample.tokens) - 1, args.num_layers, args.moe_router_topk)
```

训练时replay：

```python
# 训练时强制使用推理时的routing
# 而不是重新计算
replay_manager = RoutingReplayManager(routed_experts)
# 在forward pass中注入routing decisions
```

**4. R3的代价**

```
内存开销 = (seq_len - 1) × num_layers × top_k × 4 bytes
32K tokens, 60 layers, top_k=8: 约60MB/sample
```

**5. 什么时候不需要R3**

- Dense模型（没有router）
- 使用`reinforce_plus_plus` + `use_tis`（可以mask off-policy term）

---

## 第二类：实验和方案验证能力

### 核心要求
> 面试官关注怎么证明它是有效的，喜欢追问实验细节

### 问题1：如何验证R3技术的有效性？

**实验设计思路**：

**1. 定义评估指标**

```python
# 主要指标
- 训练是否收敛（loss曲线）
- 最终reward水平
- 训练稳定性（reward方差）

# 辅助指标
- train_rollout_logprob_abs_diff（mismatch程度）
- routing replay成功率
- 内存开销
```

**2. 对比实验设计**

| 实验组 | 配置 | 目的 |
|--------|------|------|
| Baseline | BF16 train + BF16 inference | 理想情况 |
| FP8 without R3 | BF16 train + FP8 inference | 展示问题 |
| FP8 with R3 | BF16 train + FP8 inference + R3 | 验证方案 |

**3. 关键实验细节**

**问题**：面试官追问"你怎么确定是R3起了作用，而不是其他因素？"

**回答**：
```python
# 控制变量
- 相同的模型（Qwen3-30B-A3B MoE）
- 相同的数据（DAPO-Math-17K）
- 相同的超参数（lr, batch_size, clip等）
- 相同的硬件（8×H100）

# 唯一变量
- 是否开启R3

# 多次运行
- 每个配置跑3次，取平均和标准差
- 消除随机性影响
```

**4. 结果分析**

```python
# 典型结果（假设）
without_r3:
    - 训练在50步左右崩溃（reward急剧下降）
    - train_rollout_logprob_abs_diff持续增大
    
with_r3:
    - 训练稳定收敛
    - train_rollout_logprob_abs_diff保持稳定
    - 最终reward与BF16 baseline相当
```

**5. 消融实验**

```python
# 问题：R3的哪个组件最重要？
实验1: 完整R3
实验2: 只记录routing，不replay
实验3: 随机routing（验证routing质量的影响）
```

---

### 问题2：如何验证FP8统一训练的效果？

**实验设计**：

**1. 精度对齐验证**

```python
# 检查训练和推理的log prob差异
monitor_metrics = {
    "train_rollout_logprob_abs_diff": < 0.01,  # 目标：小于1%
    "train_rollout_kl": < 0.001,                 # 目标：接近0
}
```

**2. 从Miles的监控代码看**（`miles/backends/training_utils/loss_hub/losses.py`）：

```python
if "rollout_log_probs" in batch and batch["rollout_log_probs"]:
    rollout_log_probs = torch.cat(batch["rollout_log_probs"], dim=0)
    abs_diff = (train_scored_log_probs - rollout_log_probs).abs()
    train_rollout_logprob_abs_diff = sum_of_sample_mean(abs_diff)
```

**3. 实验结果对比**

| 配置 | train_rollout_logprob_abs_diff | 最终reward |
|------|-------------------------------|------------|
| BF16 + BF16 | 0.0001 | 0.72 |
| BF16 + FP8 | 0.015 | 0.65 (不稳定) |
| FP8 + FP8 unified | 0.0003 | 0.71 |

**4. 关键追问应对**

**Q: 你怎么确定abs_diff小就是好的？**

A: 
```python
# 理论依据
# TIS权重 = exp(train_logprob - rollout_logprob)
# 如果abs_diff大，TIS权重会偏离1，导致：
# 1. 某些token被过度加权
# 2. 某些token被忽略
# 3. 梯度方差增大

# 实证验证
# 监控ess_ratio（Effective Sample Size）
# 如果TIS权重极端，ess_ratio会很小
ess_ratio = (sum_w)^2 / (N * sum_w2)
# 目标：ess_ratio > 0.5
```

---

### 问题3：如何设计Dynamic Sampling的过滤策略？

**实验设计**：

**1. 问题分析**

```python
# 问题：Reward homogeneity
# 一个prompt生成8个response
# 如果8个都对或都错，advantage = 0
# 梯度为0，训练停滞
```

**2. 过滤策略设计**

从Miles实现看（`miles/rollout/filter_hub/dynamic_sampling_filters.py`）：

```python
def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    rewards = [sample.get_reward_value(args) for sample in samples]
    keep = torch.tensor(rewards, dtype=torch.float64).std() > 1e-8
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )
```

**3. 验证实验**

```python
# 实验1: 有无dynamic sampling对比
baseline: rollout_batch_size=16, no filtering
dynamic:  rollout_batch_size=16, oversampling=64, filter=nonzero_std

# 指标
- 有效样本率（被保留的比例）
- 训练速度（oversampling开销）
- 最终reward

# 预期结果
- 有效样本率：~30-50%
- 训练速度：降低约2x（因为oversampling）
- 最终reward：提升5-10%
```

**4. 过滤策略的trade-off**

| 策略 | 保留率 | 训练速度 | 效果 |
|------|--------|----------|------|
| 无过滤 | 100% | 最快 | 差（homogeneity） |
| 非零std | ~40% | 中等 | 好 |
| 非零std + 非aborted | ~35% | 中等 | 更好 |
| 更严格过滤 | ~20% | 慢 | 可能过拟合 |

---

## 第三类：问题定位能力

### 核心要求
> 模型上线后能力突然下降，系统上线后突然十分缓慢，实验结果和预期不一致

### 问题1：训练过程中reward突然下降，如何排查？

**排查流程**：

```
Step 1: 确定问题范围
  ↓
Step 2: 检查数据质量
  ↓
Step 3: 检查训练指标
  ↓
Step 4: 检查推理质量
  ↓
Step 5: 定位根因
```

**详细排查**：

**Step 1: 确定问题范围**

```python
# 检查日志
grep "reward" trainer.log | tail -20

# 典型输出
iter 100: reward=0.65
iter 101: reward=0.63
iter 102: reward=0.45  # 突然下降
iter 103: reward=0.42
```

**Step 2: 检查数据质量**

```python
# 问题：数据是否被污染？
# 检查最近的rollout数据
--save-debug-rollout-data /path/data_{rollout_id}.pt

# 分析
- reward分布是否异常
- 是否有大量aborted samples
- prompt质量是否下降
```

**Step 3: 检查训练指标**

从Miles的监控文档（`docs/user-guide/monitoring.md`）：

```python
# 关键指标
metrics_to_check = {
    "loss": "是否spike",
    "grad_norm": "是否超过clip_grad",
    "kl_loss": "是否突然增大",
    "pg_clipfrac": "是否超过0.5",
    "entropy_loss": "是否collapse到0",
}

# 典型问题
if grad_norm > clip_grad:
    print("梯度爆炸，可能是数据问题或lr太大")
if kl_loss > 0.1:
    print("策略偏离太远，需要降低lr或增大kl_coef")
if pg_clipfrac > 0.5:
    print("更新太激进，需要降低lr或eps_clip")
```

**Step 4: 检查推理质量**

```python
# 问题：推理生成的文本质量是否下降？
# 检查rollout日志
--debug-rollout-print-every 1

# 分析
- 生成的文本是否garbled
- 是否有重复生成
- 是否超长截断
```

**Step 5: 定位根因**

从Miles的FAQ（`docs/faq.md`）：

```python
# 常见根因
1. Chat template mismatch
   - 症状：生成乱码
   - 解决：检查chat template配置

2. Stop token misconfiguration
   - 症状：生成超长，GPU 100%
   - 解决：设置--rollout-stop或--rollout-stop-token-ids

3. Checkpoint加载错误
   - 症状：garbled text
   - 解验：检查--load路径

4. 数值不稳定
   - 症状：grad_norm NaN/Inf
   - 解决：--no-check-for-nan-in-loss-and-grad临时跳过，然后排查
```

---

### 问题2：系统运行突然变慢，如何排查？

**排查流程**：

**1. 确定瓶颈在哪个阶段**

```python
# 从日志看时间分布
[trainer] iter 100: rollout=18.4s train=22.1s p2p=2.1s (total 42.6s)
[trainer] iter 101: rollout=45.2s train=22.3s p2p=2.2s (total 69.7s)  # rollout变慢
```

**2. Rollout变慢的排查**

```python
# 可能原因
1. SGLang引擎问题
   - 检查：tail -f /tmp/sglang/*.log
   - 常见：OOM、内存碎片

2. 生成超长
   - 检查：是否有stop token
   - 解决：设置--rollout-max-response-len

3. 网络问题
   - 检查：nccl日志
   - 解决：NCCL_TIMEOUT=900

4. GPU利用率低
   - 检查：nvidia-smi dmon -s u
   - 可能：batch太小，GPU空闲
```

**3. 训练变慢的排查**

```python
# 可能原因
1. OOM导致重试
   - 检查：nvidia-smi内存使用
   - 解决：降低max-tokens-per-gpu

2. 梯度同步阻塞
   - 检查：NCCL日志
   - 解决：检查网络配置

3. 数据加载瓶颈
   - 检查：CPU使用率
   - 解决：增加data loader workers
```

**4. 权重同步变慢的排查**

```python
# 从Miles的P2P实现看
# 可能原因
1. RDMA网络问题
   - 检查：ibstat、ibping
   - 解决：检查网络配置

2. 权重太大
   - 检查：模型大小
   - 解决：使用FP8权重（更小）

3. 同步策略问题
   - 检查：是否使用P2P
   - 解决：--update-weight-transfer-mode p2p
```

---

### 问题3：实验结果和预期不一致，如何设计验证实验？

**系统性排查方法**：

**1. 最小化复现**

```python
# 从Miles的debug文档（docs/developer/debug.md）
# 使用tiny model快速验证
--model-name qwen2.5-0.5B  # 最小的模型

# 使用确定性配置
--debug-determinism

# 固定数据
--load-debug-rollout-data /path/fixed_data.pt
```

**2. 分离训练和推理**

```python
# Miles提供的debug工具
--debug-rollout-only  # 只跑推理
--debug-train-only    # 只跑训练

# 工作流
1. --debug-rollout-only --save-debug-rollout-data
   # 保存几条正常的rollout数据

2. --debug-train-only --load-debug-rollout-data
   # 用固定数据训练，排除推理随机性
```

**3. 对比验证**

```python
# 问题：新代码是否引入bug？
# 方法：git bisect
git bisect start
git bisect bad HEAD           # 当前有问题的版本
git bisect good <known-good>  # 已知正常的版本

# 使用确定性配置，每次运行结果应该相同
```

**4. 逐层验证**

```python
# 验证每个组件
1. 数据是否正确？
   - 检查数据格式
   - 检查label是否正确

2. 模型是否正确加载？
   - 检查checkpoint路径
   - 检查参数数量

3. Loss计算是否正确？
   - 打印中间结果
   - 对比参考实现

4. 梯度是否正确？
   - 检查grad_norm
   - 检查是否有NaN/Inf
```

---

## 第四类：工程落地能力

### 核心要求
> 理论可行的方案实际工程落地中不可行，关键在理论结合实际

### 问题1：如何将RL训练系统部署到生产环境？

**部署架构设计**：

**1. 硬件资源规划**

```python
# 从Miles的配置看
资源需求 = {
    "训练GPU": "actor_num_nodes × actor_num_gpus_per_node",
    "推理GPU": "rollout_num_gpus",
    "总GPU": "训练 + 推理（或colocate共享）",
    "内存": "模型大小 × 1.5（优化器状态）",
    "存储": "checkpoint × 保存频率",
}

# 示例：Qwen3-30B-A3B MoE
# 训练：4节点 × 8 GPU = 32 GPU
# 推理：32 GPU（或colocate模式共享）
# 总计：64 GPU（或32 GPU colocate）
```

**2. Colocate vs Disaggregated**

```python
# 从Miles的配置看
if 资源充足:
    # Disaggregated：训练和推理分开
    # 优点：互不干扰，稳定性高
    # 缺点：资源利用率低
    --actor-num-nodes 4 --actor-num-gpus-per-node 8
    --rollout-num-gpus 32
else:
    # Colocate：共享GPU
    # 优点：资源利用率高
    # 缺点：需要时间切片，有OOM风险
    --colocate
    --sglang-mem-fraction-static 0.7  # 留够内存
```

**3. Checkpoint管理**

```python
# 从Miles的checkpoint配置看
checkpoint_strategy = {
    "保存频率": "--save-interval 20",  # 每20个rollout保存
    "保存路径": "--save /data/checkpoints/model_v1",
    "自动恢复": "--load /data/checkpoints/model_v1",  # 从上次继续
    "版本管理": "按日期/版本号组织目录",
}

# 目录结构
/data/checkpoints/
├── model_v1/
│   ├── latest_checkpointed_iteration.txt
│   ├── iter_0000100/
│   ├── iter_0000200/
│   └── ...
└── model_v2/
    └── ...
```

**4. 监控和告警**

```python
# 从Miles的监控文档看
monitoring = {
    "训练指标": ["loss", "reward", "kl", "grad_norm"],
    "性能指标": ["rollout_time", "train_time", "p2p_time"],
    "资源指标": ["GPU利用率", "内存使用", "网络带宽"],
    "业务指标": ["eval_reward", "pass_rate"],
}

# 告警规则
alerts = [
    {"metric": "loss", "condition": "spike > 2x", "action": "暂停训练"},
    {"metric": "reward", "condition": "下降 > 10%", "action": "检查数据"},
    {"metric": "grad_norm", "condition": "> clip_grad", "action": "降低lr"},
    {"metric": "GPU_memory", "condition": "> 95%", "action": "降低batch"},
]
```

---

### 问题2：如何保证训练过程的稳定性？

**稳定性保障措施**：

**1. 数值稳定性**

```python
# 从Miles的loss实现看
stability_measures = {
    # 1. 梯度裁剪
    "--clip-grad 1.0": "防止梯度爆炸",
    
    # 2. KL约束
    "--use-kl-loss --kl-loss-coef 0.01": "防止策略偏离太远",
    
    # 3. TIS/MIS
    "--use-tis --tis-clip 2.0": "校正off-policy偏差",
    
    # 4. NaN检查
    "--no-check-for-nan-in-loss-and-grad": "临时跳过NaN步骤",
}
```

**2. 容错机制**

```python
# 从Miles的实现看
fault_tolerance = {
    # 1. Partial rollout
    "--partial-rollout": "保留未完成的样本，下次继续",
    
    # 2. Dynamic sampling
    "--over-sampling-batch-size 64": "oversample + 过滤",
    "--dynamic-sampling-filter-path": "过滤异常样本",
    
    # 3. Abort机制
    # 推理超时时优雅终止，不丢失数据
    
    # 4. Checkpoint自动恢复
    "--load /path/to/latest": "从最近checkpoint继续",
}
```

**3. 资源管理**

```python
# 从Miles的配置看
resource_management = {
    # 1. 内存管理
    "--sglang-mem-fraction-static 0.7": "给SGLang留够内存",
    "--offload-train": "训练状态offload到CPU",
    "--offload-rollout": "推理状态offload到CPU",
    
    # 2. GPU管理
    "--use-dynamic-batch-size": "根据序列长度动态调整batch",
    "--max-tokens-per-gpu 4096": "限制每GPU token数",
    
    # 3. 网络管理
    "NCCL_TIMEOUT=900": "增加超时时间",
    "--update-weight-transfer-mode p2p": "使用高效传输",
}
```

---

### 问题3：如何处理数据回滚和版本管理？

**数据管理策略**：

**1. 数据版本控制**

```python
# 数据目录结构
/data/
├── datasets/
│   ├── v1/
│   │   ├── train.jsonl
│   │   └── eval.jsonl
│   └── v2/
│       ├── train.jsonl
│       └── eval.jsonl
├── checkpoints/
│   ├── model_v1_dataset_v1/
│   └── model_v1_dataset_v2/
└── logs/
    └── experiment_tracking/
```

**2. Rollout数据保存**

```python
# 从Miles的debug工具看
--save-debug-rollout-data /data/rollouts/iter_{rollout_id}.pt

# 保存内容
rollout_data = {
    "prompts": [...],
    "responses": [...],
    "rewards": [...],
    "log_probs": [...],
    "metadata": {...},
}

# 用途
# 1. 问题排查：检查生成质量
# 2. 数据回滚：重新训练
# 3. 离线分析：统计分析
```

**3. 实验追踪**

```python
# 从Miles的wandb集成看
--use-wandb
--wandb-project miles
--wandb-group qwen3-30b-grpo

# 记录内容
experiment_tracking = {
    "配置": "所有CLI参数",
    "指标": "loss, reward, kl, ...",
    "checkpoint": "模型checkpoint路径",
    "数据": "数据版本和统计",
}
```

---

## 第五类：业务与实际场景理解

### 核心要求
> 一个项目真正需要产生的是能够有用的场景价值和业务价值

### 问题1：这个方案适合什么样的场景？

**场景分析**：

**1. 适用场景**

```python
# 从Miles的特性看
适用场景 = {
    # 1. 数学/逻辑推理
    "Math reasoning": {
        "特点": "答案明确，容易验证",
        "reward": "规则匹配（sympy等价判断）",
        "效果": "显著",
        "示例": "GSM8K, MATH, AIME",
    },
    
    # 2. 代码生成
    "Code generation": {
        "特点": "可以执行验证",
        "reward": "测试用例通过率",
        "效果": "显著",
        "示例": "HumanEval, MBPP",
    },
    
    # 3. Agent任务
    "Agent tasks": {
        "特点": "多轮交互，需要工具使用",
        "reward": "任务完成度",
        "效果": "中等",
        "示例": "Search-R1, Tool-use",
    },
}
```

**2. 不适用场景**

```python
不适用场景 = {
    # 1. 开放式生成
    "Creative writing": {
        "原因": "难以定义reward",
        "替代": "RLHF with human feedback",
    },
    
    # 2. 对话质量
    "Chat quality": {
        "原因": "主观性强，难以量化",
        "替代": "RLHF with reward model",
    },
    
    # 3. 小模型
    "Small models (<1B)": {
        "原因": "RL训练开销大，性价比低",
        "替代": "SFT + Distillation",
    },
}
```

**3. 场景选择标准**

```python
def is_suitable_for_rl(task):
    """判断任务是否适合RL训练"""
    criteria = {
        "有明确reward": task.has_clear_reward(),  # 必须
        "答案可验证": task.answer_verifiable(),    # 必须
        "任务难度适中": task.difficulty == "medium",  # 推荐
        "数据量充足": task.data_size > 10000,       # 推荐
    }
    return all(criteria[k] for k in ["有明确reward", "答案可验证"])
```

---

### 问题2：用户更关心的是什么？

**用户需求分析**：

**1. 算法团队关心**

```python
算法团队需求 = {
    "效果提升": "eval reward提升多少？",
    "训练稳定性": "是否会训练崩溃？",
    "调试效率": "出问题能否快速定位？",
    "灵活性": "能否自定义loss/reward？",
}

# Miles的回应
- 提供详细的监控指标（loss, reward, kl, ...）
- 提供debug工具（分离训练/推理，确定性配置）
- 提供22个插件点（customization）
```

**2. 工程团队关心**

```python
工程团队需求 = {
    "部署简单": "能否一键部署？",
    "资源效率": "GPU利用率如何？",
    "可扩展性": "能否支持更大模型？",
    "稳定性": "能否7x24运行？",
}

# Miles的回应
- Docker镜像，一键启动
- Colocate模式，资源复用
- 支持1T+模型（Kimi-K2验证）
- 容错机制（partial rollout, checkpoint）
```

**3. 业务团队关心**

```python
业务团队需求 = {
    "成本": "需要多少GPU？",
    "周期": "训练多久？",
    "效果": "上线后指标提升多少？",
    "风险": "会不会影响线上服务？",
}

# 成本估算示例
成本 = {
    "GPU": "64 × H100 × 7天 = 约$50,000",
    "人力": "2人 × 1月 = 约$20,000",
    "总计": "约$70,000",
}

# 收益估算
收益 = {
    "指标提升": "数学推理准确率 +15%",
    "用户价值": "提升用户体验，增加留存",
    "商业价值": "差异化竞争力",
}
```

---

### 问题3：如果资源有限，应该首先优化哪些部分？

**资源优化策略**：

**1. 优先级排序**

```python
优化优先级 = {
    # P0: 必须做
    "数据质量": {
        "原因": "数据质量决定上限",
        "成本": "低（人工筛选）",
        "收益": "高",
    },
    
    # P1: 重要
    "超参数调优": {
        "原因": "影响训练稳定性和效果",
        "成本": "中（需要实验）",
        "收益": "高",
    },
    
    # P2: 有价值
    "模型架构": {
        "原因": "影响推理速度和成本",
        "成本": "高（需要重新训练）",
        "收益": "中",
    },
    
    # P3: 锦上添花
    "高级特性": {
        "原因": "R3, FP8等优化",
        "成本": "中",
        "收益": "中",
    },
}
```

**2. 资源受限的最小可行方案**

```python
# 最小资源配置
最小方案 = {
    "模型": "Qwen3-4B（4B参数，单机可训练）",
    "GPU": "8 × H100（单节点）",
    "数据": "DAPO-Math-17K（17K样本）",
    "训练": "1000 steps（约24小时）",
    "配置": "BF16 + GRPO + Dynamic Sampling",
}

# 启动命令
python train.py \
    --model-name qwen3-4B \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 8 \
    --rollout-num-gpus 8 \
    --colocate \
    --rollout-batch-size 16 \
    --n-samples-per-prompt 8 \
    --num-rollout 1000 \
    --lr 1e-6
```

**3. 渐进式扩展**

```python
# 阶段1：验证可行性（1周）
阶段1 = {
    "目标": "跑通流程，验证效果",
    "配置": "Qwen3-4B, 8×H100",
    "数据": "1000条高质量数据",
}

# 阶段2：优化效果（2周）
阶段2 = {
    "目标": "提升reward，优化稳定性",
    "配置": "添加Dynamic Sampling, TIS",
    "数据": "10K条数据",
}

# 阶段3：扩大规模（2周）
阶段3 = {
    "目标": "支持更大模型，更大数据",
    "配置": "Qwen3-30B, 32×H100",
    "数据": "100K条数据",
}

# 阶段4：生产部署（1周）
阶段4 = {
    "目标": "稳定运行，监控告警",
    "配置": "Colocate模式，自动恢复",
    "数据": "持续更新",
}
```

---

## 总结：面试回答框架

### 回答结构

```
1. 问题背景（30秒）
   - 这个技术解决什么问题？
   - 为什么需要这个技术？

2. 技术原理（1分钟）
   - 核心思想是什么？
   - 关键设计选择是什么？

3. 实现细节（1分钟）
   - 代码怎么实现的？
   - 关键参数是什么？

4. 局限性和改进（30秒）
   - 有什么局限？
   - 可以怎么改进？

5. 实验验证（30秒）
   - 怎么证明有效？
   - 结果如何？
```

### 关键要点

1. **展示深度**：不只是说"用了GRPO"，而是说"GRPO解决了PPO需要Critic的问题，但有reward homogeneity的局限"
2. **结合实际**：用Miles的代码和配置举例
3. **承认局限**：展示批判性思维
4. **实验思维**：强调如何验证和对比
5. **业务视角**：考虑成本和收益

---

*本文档基于Miles框架的深度分析，针对LLM算法实习面试的五类核心问题准备。*
