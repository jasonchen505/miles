# Miles框架复现计划

## 基于8卡3090的完整全流程复现方案

---

## 一、硬件资源评估

### 1.1 3090硬件特性

| 特性 | 3090规格 | Miles要求 | 影响 |
|------|----------|-----------|------|
| 显存 | 24GB | 建议80GB+ | 需要小模型+小batch |
| FP8支持 | ❌ | H100/H200 | 只能用BF16 |
| NVLink | ❌ (PCIe) | 建议NVLink | 权重同步较慢 |
| 计算能力 | 8.6 | 8.0+ | 兼容 |
| Transformer Engine | 部分支持 | 完整支持 | 部分特性不可用 |

### 1.2 资源约束分析

```python
# 显存预算（24GB）
显存分配 = {
    "模型权重": "~2GB (0.5B模型BF16)",
    "优化器状态": "~4GB (Adam需要2x参数)",
    "梯度": "~2GB",
    "激活值": "~4-8GB (取决于序列长度)",
    "SGLang KV Cache": "~4-6GB",
    "其他开销": "~2GB",
    "总计": "~18-24GB",
}

# 结论：可以运行0.5B-1.5B模型，3B+需要LoRA或FSDP CPU offload
```

### 1.3 可行方案选择

| 方案 | 模型 | 后端 | 适用场景 |
|------|------|------|----------|
| **方案A（推荐）** | Qwen2.5-0.5B | Megatron | 完整体验，学习原理 |
| **方案B** | Qwen3-0.6B | Megatron | 最新架构 |
| **方案C** | Qwen3-4B | FSDP | 更大模型，但特性受限 |
| **方案D** | Qwen3-4B + LoRA | Megatron | 平衡效果和资源 |

---

## 二、复现计划总览

### 2.1 阶段划分

```
阶段0: 环境搭建（Day 1）
    ↓
阶段1: 最小复现（Day 2-3）
    ↓
阶段2: 完整流程（Day 4-5）
    ↓
阶段3: 深入学习（Day 6-7）
    ↓
阶段4: 扩展实验（Day 8-10）
```

### 2.2 学习目标

| 阶段 | 目标 | 产出 |
|------|------|------|
| 阶段0 | 环境跑通 | 能启动Miles |
| 阶段1 | 理解核心流程 | 能运行GRPO训练 |
| 阶段2 | 掌握完整流程 | 能完成eval和checkpoint |
| 阶段3 | 理解高级特性 | 掌握TIS/Dynamic Sampling |
| 阶段4 | 扩展实验 | 尝试LoRA/多轮交互 |

---

## 三、详细执行计划

### 阶段0：环境搭建（Day 1）

#### 目标
- 搭建Miles运行环境
- 验证GPU可用性
- 下载模型和数据

#### 步骤

**0.1 环境准备**

```bash
# 方案1：使用Docker（推荐）
docker pull radixark/miles:latest

docker run --rm \
  --gpus all --ipc=host --shm-size=32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network=host \
  -v /path/to/data:/root/data \
  -it radixark/miles:latest /bin/bash

# 方案2：从源码安装（如果不支持Docker）
git clone https://github.com/radixark/miles.git
cd miles
pip install -r requirements.txt
pip install -e . --no-deps
```

**0.2 验证环境**

```bash
# 检查GPU
nvidia-smi

# 检查Miles
python -c "import miles; print('Miles import OK')"

# 检查Megatron
python -c "import megatron; print('Megatron import OK')"
```

**0.3 下载模型和数据**

```bash
# 下载小模型（约1GB）
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir /root/Qwen2.5-0.5B-Instruct

# 下载GSM8K数据集
hf download --repo-type dataset zhuzilin/gsm8k --local-dir /root/gsm8k

# 下载AIME评估集（可选）
hf download --repo-type dataset zhuzilin/aime-2024 --local-dir /root/aime-2024
```

**0.4 Checkpoint转换**

```bash
cd /root/miles

# 转换为Megatron格式
source scripts/models/qwen2.5-0.5B.sh
PYTHONPATH=/root/Megatron-LM/ python \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/Qwen2.5-0.5B-Instruct \
   --save /root/Qwen2.5-0.5B-Instruct_torch_dist/

# 验证转换结果
ls -la /root/Qwen2.5-0.5B-Instruct_torch_dist/
```

#### 验证点
- [ ] nvidia-smi显示8张3090
- [ ] Miles和Megatron可以正常import
- [ ] Checkpoint转换成功

---

### 阶段1：最小复现（Day 2-3）

#### 目标
- 运行第一个RL训练任务
- 理解核心训练循环
- 观察loss和reward变化

#### 步骤

**1.1 创建最小配置脚本**

创建文件：`/root/miles/scripts/run-3090-minimal.sh`

```bash
#!/bin/bash

# 清理之前的进程
pkill -9 sglang 2>/dev/null
sleep 3
ray stop --force 2>/dev/null
pkill -9 ray 2>/dev/null
pkill -9 python 2>/dev/null
sleep 3

set -ex

export PYTHONBUFFERED=16

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen2.5-0.5B.sh"

# Checkpoint配置
CKPT_ARGS=(
   --hf-checkpoint /root/Qwen2.5-0.5B-Instruct/
   --ref-load /root/Qwen2.5-0.5B-Instruct_torch_dist/
)

# 训练数据配置
ROLLOUT_ARGS=(
   --prompt-data /root/gsm8k/train.parquet
   --input-key messages
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout 50                    # 只跑50步，快速验证
   --rollout-batch-size 8              # 小batch size
   --n-samples-per-prompt 4            # 每个prompt生成4个response
   --rollout-max-response-len 512      # 短序列
   --rollout-temperature 1
   --global-batch-size 32              # 8 * 4 = 32
)

# 评估配置
EVAL_ARGS=(
   --eval-interval 10
   --eval-prompt-data gsm8k /root/gsm8k/test.parquet
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 512
   --eval-top-k 1
)

# 性能配置
PERF_ARGS=(
   --tensor-model-parallel-size 1      # 3090没有NVLink，TP=1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096           # 较小的token限制
)

# GRPO配置
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

# 优化器配置
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

# SGLang配置
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.6   # 3090显存小，留更多空间
)

# 其他配置
MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# 启动Ray
ray start --head --node-ip-address 127.0.0.1 --num-gpus 8 --disable-usage-stats

# 提交训练任务
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/root/Megatron-LM",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1"
     }
   }' \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   --calculate-per-token-loss \
   --use-miles-router \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
```

**1.2 运行训练**

```bash
chmod +x /root/miles/scripts/run-3090-minimal.sh
bash /root/miles/scripts/run-3090-minimal.sh
```

**1.3 观察输出**

```text
# 期望看到的输出
[ray] starting cluster on 1 node, 8 gpus
[sglang] launching 8 engines (tp=1 each)
[megatron] loading dist checkpoint from /root/Qwen2.5-0.5B-Instruct_torch_dist
[trainer] iter 1/50 | loss=0.412 reward=0.61 rollout=5.2s train=3.1s
[trainer] iter 2/50 | loss=0.398 reward=0.63 rollout=4.8s train=3.2s
...
```

#### 验证点
- [ ] 训练可以启动
- [ ] Loss在下降
- [ ] Reward在上升（或至少有变化）
- [ ] 没有OOM错误

#### 学习要点

```python
# 理解四旋钮不变量
rollout_batch_size × n_samples_per_prompt = global_batch_size × num_steps_per_rollout
8 × 4 = 32 × 1  # 成立

# 理解Colocate模式
# 训练和推理共享GPU，需要时间切片
# --sglang-mem-fraction-static 0.6 给SGLang留够内存
```

---

### 阶段2：完整流程（Day 4-5）

#### 目标
- 理解完整的训练循环
- 掌握checkpoint保存和恢复
- 学习监控和调试

#### 步骤

**2.1 扩展训练配置**

修改脚本，增加更多功能：

```bash
# 增加checkpoint保存
CKPT_ARGS=(
   --hf-checkpoint /root/Qwen2.5-0.5B-Instruct/
   --ref-load /root/Qwen2.5-0.5B-Instruct_torch_dist/
   --load /root/Qwen2.5-0.5B_miles/
   --save /root/Qwen2.5-0.5B_miles/
   --save-interval 10                  # 每10步保存
)

# 增加wandb监控
WANDB_ARGS=(
   --use-wandb
   --wandb-project miles-3090-test
   --wandb-group qwen2.5-0.5B-gsm8k
)

# 增加更多评估
EVAL_ARGS=(
   --eval-interval 5                   # 更频繁的评估
   --eval-prompt-data gsm8k /root/gsm8k/test.parquet
   --n-samples-per-eval-prompt 4       # 更多eval样本
   --eval-max-response-len 1024
   --eval-top-k 1
)
```

**2.2 运行完整训练**

```bash
# 运行100步训练
--num-rollout 100
```

**2.3 检查checkpoint**

```bash
# 查看保存的checkpoint
ls -la /root/Qwen2.5-0.5B_miles/

# 应该看到
# latest_checkpointed_iteration.txt
# iter_0000010/
# iter_0000020/
# ...
```

**2.4 测试恢复训练**

```bash
# 从checkpoint继续训练
--load /root/Qwen2.5-0.5B_miles/
--num-rollout 200  # 继续训练到200步
```

#### 验证点
- [ ] Checkpoint可以保存
- [ ] 可以从checkpoint恢复
- [ ] wandb可以看到训练曲线
- [ ] 评估结果在提升

---

### 阶段3：深入学习（Day 6-7）

#### 目标
- 理解TIS/MIS机制
- 学习Dynamic Sampling
- 掌握调试技巧

#### 步骤

**3.1 学习TIS（Truncated Importance Sampling）**

```bash
# 启用TIS
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.01                 # 添加KL约束
   --use-tis                           # 启用TIS
   --tis-clip 2.0                      # TIS截断阈值
   --eps-clip 0.2
   --eps-clip-high 0.28
)
```

**3.2 学习Dynamic Sampling**

```bash
# 启用Dynamic Sampling
ROLLOUT_ARGS=(
   --prompt-data /root/gsm8k/train.parquet
   --input-key messages
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout 100
   --rollout-batch-size 8
   --n-samples-per-prompt 8            # 增加样本数
   --rollout-max-response-len 512
   --rollout-temperature 1
   --global-batch-size 64
   
   # Dynamic Sampling配置
   --over-sampling-batch-size 32       # 过采样
   --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
)
```

**3.3 学习调试技巧**

```bash
# 分离训练和推理调试
--debug-rollout-only                  # 只调试推理
--save-debug-rollout-data /tmp/debug_rollout.pt  # 保存rollout数据

--debug-train-only                    # 只调试训练
--load-debug-rollout-data /tmp/debug_rollout.pt  # 加载保存的数据

# 查看详细日志
--debug-rollout-print-every 1         # 每个rollout都打印
--debug-determinism                   # 确定性模式
```

#### 验证点
- [ ] 理解TIS的作用（监控train_rollout_logprob_abs_diff）
- [ ] 理解Dynamic Sampling（观察过滤比例）
- [ ] 掌握调试流程

---

### 阶段4：扩展实验（Day 8-10）

#### 目标
- 尝试LoRA训练
- 尝试多轮交互
- 尝试自定义reward

#### 步骤

**4.1 LoRA训练**

```bash
# 使用LoRA配置
LORA_ARGS=(
   --lora-rank 16
   --lora-alpha 32
   --lora-dropout 0.0
   --target-modules "all-linear"
   --megatron-to-hf-mode bridge
)

# 可以尝试更大的模型
# Qwen3-1.7B + LoRA 应该可以在3090上运行
```

**4.2 自定义Reward函数**

创建文件：`/root/miles/examples/custom_reward.py`

```python
from miles.utils.types import Sample

async def custom_reward(args, sample: Sample) -> float:
    """自定义reward函数"""
    # 示例：基于长度的reward
    response_len = sample.response_length
    
    # 基础reward（可以从reward model获取）
    base_reward = sample.reward if sample.reward else 0.0
    
    # 长度惩罚
    length_penalty = max(0, 1 - response_len / 1000)
    
    return base_reward * length_penalty
```

**4.3 多轮交互（Search-R1示例）**

```bash
# 使用Search-R1示例
--custom-generate-function-path examples.search-r1.generate_with_search.generate
--custom-rm-path examples.search-r1.generate_with_search.reward_func
```

---

## 四、关键配置说明

### 4.1 3090专用配置

```bash
# 显存优化配置
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1     # 每个engine用1张GPU
   --sglang-mem-fraction-static 0.6    # 留更多显存给训练
)

PERF_ARGS=(
   --tensor-model-parallel-size 1      # 3090没有NVLink，TP=1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096           # 较小的token限制
)

# 如果还是OOM，可以进一步降低
--max-tokens-per-gpu 2048
--rollout-max-response-len 256
```

### 4.2 Colocate模式注意事项

```python
# Colocate模式下，训练和推理共享GPU
# 需要合理分配显存

# SGLang显存：约40-60%（用于KV Cache）
--sglang-mem-fraction-static 0.6

# 训练显存：剩余部分
# 如果OOM，降低--sglang-mem-fraction-static
```

### 4.3 PCIe互联优化

```python
# 3090没有NVLink，PCIe互联较慢
# 权重同步会成为瓶颈

# 优化方法
1. 使用较小的模型（0.5B-1.5B）
2. 减少权重同步频率（增大--save-interval）
3. 使用bucketed transfer（默认开启）
```

---

## 五、常见问题解决

### 5.1 OOM错误

```bash
# 症状：CUDA out of memory
# 解决：
1. 降低--max-tokens-per-gpu
2. 降低--rollout-max-response-len
3. 降低--sglang-mem-fraction-static
4. 使用更小的模型
```

### 5.2 训练不稳定

```bash
# 痖状：loss spike, reward下降
# 解决：
1. 降低学习率：--lr 1e-6
2. 增加KL约束：--kl-loss-coef 0.01
3. 使用TIS：--use-tis --tis-clip 2.0
```

### 5.3 推理太慢

```bash
# 症状：rollout时间很长
# 解决：
1. 降低--rollout-max-response-len
2. 增加--sglang-server-concurrency
3. 使用更小的模型
```

### 5.4 权重同步失败

```bash
# 症状：NCCL timeout
# 解决：
export NCCL_TIMEOUT=900
export NCCL_DEBUG=INFO

# 检查网络
nvidia-smi topo -m
```

---

## 六、学习检查清单

### 阶段0完成后
- [ ] 环境搭建成功
- [ ] 可以运行nvidia-smi
- [ ] Miles可以正常import
- [ ] Checkpoint转换成功

### 阶段1完成后
- [ ] 理解四旋钮不变量
- [ ] 理解Colocate模式
- [ ] 可以运行GRPO训练
- [ ] 观察到loss下降

### 阶段2完成后
- [ ] 理解checkpoint机制
- [ ] 可以从checkpoint恢复
- [ ] 理解wandb监控
- [ ] 理解评估流程

### 阶段3完成后
- [ ] 理解TIS原理和作用
- [ ] 理解Dynamic Sampling
- [ ] 掌握调试技巧
- [ ] 可以定位常见问题

### 阶段4完成后
- [ ] 可以使用LoRA训练
- [ ] 可以自定义reward
- [ ] 理解多轮交互
- [ ] 可以扩展新功能

---

## 七、时间规划

| 天数 | 任务 | 预计时间 |
|------|------|----------|
| Day 1 | 环境搭建 | 4-6小时 |
| Day 2 | 最小复现 | 4-6小时 |
| Day 3 | 理解核心流程 | 4-6小时 |
| Day 4 | 完整流程 | 4-6小时 |
| Day 5 | 监控和调试 | 4-6小时 |
| Day 6 | TIS/Dynamic Sampling | 4-6小时 |
| Day 7 | 调试技巧 | 4-6小时 |
| Day 8 | LoRA训练 | 4-6小时 |
| Day 9 | 自定义扩展 | 4-6小时 |
| Day 10 | 总结和整理 | 4-6小时 |

---

## 八、参考资源

### 官方文档
- [Miles Quick Start](https://miles.radixark.com/docs/getting-started/quick-start)
- [Core Concepts](https://miles.radixark.com/docs/user-guide/concepts)
- [Debugging Guide](https://miles.radixark.com/docs/developer/debug)

### 示例代码
- `examples/reproducibility/` - 可复现训练示例
- `examples/lora/` - LoRA训练示例
- `examples/search-r1/` - 多轮交互示例

### 模型下载
- Qwen2.5-0.5B-Instruct: ~1GB
- Qwen3-0.6B: ~1.2GB
- GSM8K数据集: ~10MB

---

*本计划基于8卡3090硬件资源设计，实际执行时可能需要根据具体情况调整。*
