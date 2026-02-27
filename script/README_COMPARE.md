# 随机初始化 vs 预训练LGM对比实验

## 概述

该脚本用于比较随机初始化和加载预训练权重的LGM模型在攻击训练后的性能差异。

## 功能

1. **双模型训练**：同时训练随机初始化和预训练的LGM模型
2. **指标收集**：收集训练过程中的Loss、LPIPS、PSNR等指标
3. **质量评估**：评估攻击前后的Source和Target质量
4. **Gaussian诊断**：分析Gaussian参数的统计特性
5. **可视化对比**：生成对比图表和渲染样本
6. **详细报告**：生成文本格式的对比报告

## 使用方法

### 方法1：使用启动脚本（推荐）

```bash
# 基本用法
./script/run_compare.sh [GPU_ID] [CONFIG] [ATTACK_EPOCHS]

# 示例：在GPU 0上运行，攻击5个epoch
./script/run_compare.sh 0 configs/config.yaml 5

# 示例：在GPU 1上运行，攻击10个epoch
./script/run_compare.sh 1 configs/config.yaml 10
```

### 方法2：直接运行Python脚本

```bash
python script/compare_random_vs_pretrained.py \
    --config configs/config.yaml \
    --attack_epochs 5 \
    --gpu 0 \
    --num_render 3 \
    --eval_every_steps 10 \
    --output_dir output/compare_random_vs_pretrained
```

## 参数说明

- `--config`: 配置文件路径（必需）
- `--attack_epochs`: 攻击训练的epoch数（默认从config读取）
- `--gpu`: 使用的GPU ID（默认0）
- `--num_render`: 渲染的样本数量（默认3）
- `--eval_every_steps`: 每隔多少步评估一次（默认10）
- `--output_dir`: 输出目录（默认output/compare_random_vs_pretrained）

## 输出结果

脚本会在输出目录下创建以下文件和目录：

```
output/compare_random_vs_pretrained/compare_YYYYMMDD_HHMMSS/
├── comparison_plot.png          # 对比图表（4个子图）
├── comparison_report.txt        # 文本格式的对比报告
├── comparison_data.json         # 完整的对比数据（JSON格式）
├── random_init/                 # 随机初始化实验结果
│   ├── results.json            # 训练历史和指标
│   ├── source_renders/         # Source数据渲染结果
│   └── target_renders/         # Target数据渲染结果
└── pretrained/                  # 预训练模型实验结果
    ├── results.json
    ├── source_renders/
    └── target_renders/
```

## 对比图表说明

生成的`comparison_plot.png`包含4个子图：

1. **Training Loss**：训练过程中的Loss曲线
2. **Masked LPIPS**：Masked LPIPS曲线（越低越好）
3. **Masked PSNR**：Masked PSNR曲线（越高越好）
4. **Final Metrics Comparison**：最终指标的柱状图对比

## 对比报告说明

`comparison_report.txt`包含以下内容：

1. **训练过程**：初始Loss、最终Loss、Loss下降幅度
2. **Source质量**：攻击前在Source数据上的PSNR和LPIPS
3. **Target质量**：攻击后在Target数据上的PSNR和LPIPS
4. **Gaussian诊断**：Gaussian参数的统计信息和诊断结果
5. **关键发现**：自动分析的关键差异和结论

## 预期结果

通常情况下，预训练模型应该表现出：

1. **更低的初始Loss**：预训练权重提供了更好的起点
2. **更快的收敛**：训练过程中Loss下降更快
3. **更好的Source质量**：攻击前在Source数据上的表现更好
4. **更高的攻击后LPIPS**：说明预训练权重有助于攻击效果

如果随机初始化表现更好，可能说明：
- 预训练权重不适合当前任务
- 需要调整学习率或训练策略
- 数据分布与预训练数据差异较大

## 注意事项

1. **显存需求**：脚本会串行运行两个实验，每个实验结束后会清理GPU显存
2. **时间消耗**：总时间约为单次攻击训练的2倍
3. **配置要求**：确保config.yaml中设置了正确的预训练权重路径
4. **数据一致性**：两个实验使用相同的数据加载器，确保公平对比

## 示例配置

确保你的`configs/config.yaml`包含预训练权重路径：

```yaml
model:
  size: big
  resume: pretrained/model_fp16_fixrot.safetensors  # 预训练权重路径
  device: cuda

training:
  mode: lora  # 或 full
  attack_epochs: 5
  lr: 0.00005
  # ... 其他训练参数
```

## 故障排除

### 问题1：找不到预训练权重

**错误**：`FileNotFoundError: pretrained/model_fp16_fixrot.safetensors`

**解决**：
- 检查config.yaml中的resume路径是否正确
- 确保预训练权重文件存在
- 可以使用`tag:`前缀引用注册表中的模型

### 问题2：显存不足

**错误**：`CUDA out of memory`

**解决**：
- 减小batch_size
- 减小num_render参数
- 使用更小的模型（small而不是big）

### 问题3：两个模型结果完全相同

**可能原因**：
- 随机种子固定导致随机初始化不够随机
- 检查是否两个实验都加载了预训练权重

## 扩展用法

### 对比不同的训练策略

可以修改脚本来对比其他训练策略，例如：

```python
# 对比LoRA vs 全量微调
random_config['training']['mode'] = 'lora'
pretrained_config['training']['mode'] = 'full'

# 对比不同学习率
random_config['training']['lr'] = 1e-4
pretrained_config['training']['lr'] = 5e-5
```

### 批量实验

可以编写循环脚本来测试不同的超参数组合：

```bash
for epochs in 2 5 10; do
    ./script/run_compare.sh 0 configs/config.yaml $epochs
done
```

## 相关文档

- [CLAUDE.md](../CLAUDE.md) - 项目整体文档
- [EXPERIMENTS.md](../EXPERIMENTS.md) - 实验记录
- [training/finetuner.py](../training/finetuner.py) - AutoFineTuner实现
- [evaluation/evaluator.py](../evaluation/evaluator.py) - Evaluator实现
