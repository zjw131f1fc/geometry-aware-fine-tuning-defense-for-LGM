# 使用示例

## 快速开始

### 1. 快速测试（1个epoch，验证功能）

```bash
./script/test_compare.sh 0
```

这会在GPU 0上运行一个快速测试，只训练1个epoch，用于验证脚本功能是否正常。

### 2. 完整实验（5个epoch）

```bash
./script/run_compare.sh 0 configs/config.yaml 5
```

这会运行完整的对比实验，训练5个epoch。

### 3. 长时间实验（10个epoch）

```bash
./script/run_compare.sh 0 configs/config.yaml 10
```

## 查看结果

### 1. 查看对比图

```bash
# 最新的实验结果
ls -lt output/compare_random_vs_pretrained/

# 打开对比图
eog output/compare_random_vs_pretrained/compare_*/comparison_plot.png
```

### 2. 查看文本报告

```bash
cat output/compare_random_vs_pretrained/compare_*/comparison_report.txt
```

### 3. 查看渲染结果

```bash
# 随机初始化的渲染结果
ls output/compare_random_vs_pretrained/compare_*/random_init/target_renders/

# 预训练模型的渲染结果
ls output/compare_random_vs_pretrained/compare_*/pretrained/target_renders/
```

## 典型输出示例

### 对比报告示例

```
================================================================================
随机初始化 vs 预训练LGM - 攻击训练对比报告
================================================================================

## 训练过程

### 随机初始化
  初始Loss: 0.8234
  最终Loss: 0.3456
  Loss下降: 0.4778

### 预训练模型
  初始Loss: 0.2145
  最终Loss: 0.1234
  Loss下降: 0.0911

## Source质量（攻击前）

### 随机初始化
  PSNR: 8.45 dB
  LPIPS: 0.7234

### 预训练模型
  PSNR: 22.34 dB
  LPIPS: 0.1234

## Target质量（攻击后）

### 随机初始化
  PSNR: 15.67 dB
  LPIPS: 0.4567

### 预训练模型
  PSNR: 18.23 dB
  LPIPS: 0.3456

## 关键发现

✓ 预训练模型的攻击后LPIPS更高 (+0.1111)，说明预训练权重提供了更好的初始化
```

## 常见使用场景

### 场景1：验证预训练权重的价值

**目的**：确认预训练权重是否真的有帮助

**方法**：
```bash
./script/run_compare.sh 0 configs/config.yaml 5
```

**预期**：预训练模型应该在Source质量和收敛速度上都优于随机初始化

### 场景2：测试不同的攻击epoch数

**目的**：找到最佳的攻击训练epoch数

**方法**：
```bash
for epochs in 2 5 10 15; do
    ./script/run_compare.sh 0 configs/config.yaml $epochs
done
```

**分析**：比较不同epoch数下的LPIPS差异

### 场景3：对比不同的数据集

**目的**：测试预训练权重在不同target类别上的泛化能力

**方法**：
1. 修改config.yaml中的target类别
2. 运行对比实验
3. 比较不同类别的结果

```bash
# 修改config.yaml中的target类别为"coconut"
./script/run_compare.sh 0 configs/config.yaml 5

# 修改config.yaml中的target类别为"durian"
./script/run_compare.sh 0 configs/config.yaml 5
```

## 高级用法

### 自定义对比实验

如果需要对比其他方面（如不同的学习率、优化器等），可以直接修改Python脚本：

```python
# 在 compare_random_vs_pretrained.py 中修改

# 对比不同学习率
random_config['training']['lr'] = 1e-4
pretrained_config['training']['lr'] = 5e-5

# 对比不同优化器
random_config['training']['optimizer'] = 'sgd'
pretrained_config['training']['optimizer'] = 'adamw'

# 对比LoRA vs 全量微调
random_config['training']['mode'] = 'lora'
pretrained_config['training']['mode'] = 'full'
```

### 批量实验脚本

创建一个批量实验脚本：

```bash
#!/bin/bash
# batch_compare.sh

for epochs in 2 5 10; do
    for gpu in 0 1 2 3; do
        ./script/run_compare.sh $gpu configs/config.yaml $epochs &
    done
    wait  # 等待所有GPU完成
done
```

## 性能优化建议

### 1. 减少渲染样本数

如果只关心指标而不需要大量渲染图：

```bash
python script/compare_random_vs_pretrained.py \
    --config configs/config.yaml \
    --attack_epochs 5 \
    --num_render 1 \
    --gpu 0
```

### 2. 增加评估间隔

如果训练时间较长，可以减少评估频率：

```bash
python script/compare_random_vs_pretrained.py \
    --config configs/config.yaml \
    --attack_epochs 10 \
    --eval_every_steps 20 \
    --gpu 0
```

### 3. 使用更小的模型

在config.yaml中：

```yaml
model:
  size: small  # 而不是 big
```

## 故障排除

### 问题：脚本运行很慢

**原因**：需要训练两个完整的模型

**解决**：
- 使用test_compare.sh进行快速测试
- 减少attack_epochs
- 减少num_render
- 增加eval_every_steps

### 问题：显存不足

**解决**：
- 在config.yaml中减小batch_size
- 使用smaller模型
- 确保两个实验之间正确清理了GPU显存

### 问题：结果差异很小

**可能原因**：
- 训练epoch数太少
- 学习率太小
- 数据集太简单

**解决**：
- 增加attack_epochs
- 调整学习率
- 使用更具挑战性的数据集

## 结果解读

### 好的结果（预训练有效）

- 预训练模型的初始Loss明显更低
- 预训练模型的Source PSNR明显更高（>20 dB）
- 预训练模型的攻击后LPIPS更低（质量更好）
- 预训练模型收敛更快

### 异常结果（需要调查）

- 随机初始化表现更好
- 两者结果几乎相同
- 预训练模型的Source质量很差（<15 dB）

如果出现异常结果，检查：
1. 预训练权重是否正确加载
2. 配置文件是否正确
3. 数据集是否与预训练数据分布一致
