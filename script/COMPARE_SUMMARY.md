# 随机初始化 vs 预训练LGM对比工具 - 总结

## 已创建的文件

### 1. 核心脚本
- **script/compare_random_vs_pretrained.py** - 主要的对比脚本
  - 自动运行两个实验（随机初始化 + 预训练）
  - 收集训练指标和评估结果
  - 生成对比图表和报告

### 2. 启动脚本
- **script/run_compare.sh** - 完整实验启动脚本
  - 用法: `./script/run_compare.sh [GPU_ID] [CONFIG] [ATTACK_EPOCHS]`
  - 示例: `./script/run_compare.sh 0 configs/config.yaml 5`

- **script/test_compare.sh** - 快速测试脚本
  - 用法: `./script/test_compare.sh [GPU_ID]`
  - 只训练1个epoch，用于验证功能

### 3. 文档
- **script/README_COMPARE.md** - 详细的功能说明和使用指南
- **script/USAGE_EXAMPLES.md** - 使用示例和常见场景

## 核心功能

### 1. 双模型对比
- **随机初始化**：不加载预训练权重，从头训练
- **预训练模型**：加载预训练权重后进行攻击训练

### 2. 指标收集
- **训练过程**：Loss、LPIPS、PSNR曲线
- **Source质量**：攻击前在Source数据上的表现
- **Target质量**：攻击后在Target数据上的表现
- **Gaussian诊断**：Gaussian参数的统计分析

### 3. 可视化输出
- **对比图表**：4个子图展示训练曲线和最终指标
- **渲染样本**：GT vs Pred对比图
- **文本报告**：详细的对比分析

### 4. 数据保存
- **JSON格式**：完整的训练历史和指标数据
- **PNG格式**：对比图表和渲染结果
- **TXT格式**：可读的文本报告

## 使用流程

### 快速测试（推荐首次使用）
```bash
# 1. 快速测试（1个epoch）
./script/test_compare.sh 0

# 2. 查看结果
cat output/compare_test/compare_*/comparison_report.txt
```

### 完整实验
```bash
# 1. 运行完整实验（5个epoch）
./script/run_compare.sh 0 configs/config.yaml 5

# 2. 查看对比图
eog output/compare_random_vs_pretrained/compare_*/comparison_plot.png

# 3. 查看详细报告
cat output/compare_random_vs_pretrained/compare_*/comparison_report.txt
```

## 输出结构

```
output/compare_random_vs_pretrained/compare_YYYYMMDD_HHMMSS/
├── comparison_plot.png          # 4个子图的对比图表
├── comparison_report.txt        # 文本格式的详细报告
├── comparison_data.json         # 完整的JSON数据
├── random_init/                 # 随机初始化实验
│   ├── results.json
│   ├── source_renders/
│   └── target_renders/
└── pretrained/                  # 预训练模型实验
    ├── results.json
    ├── source_renders/
    └── target_renders/
```

## 关键特性

### 1. 公平对比
- 两个实验使用完全相同的数据加载器
- 相同的训练参数（lr, batch_size, epochs等）
- 相同的随机种子（确保数据顺序一致）

### 2. 自动化
- 一键运行两个完整实验
- 自动收集和对比指标
- 自动生成报告和可视化

### 3. 灵活性
- 支持自定义攻击epoch数
- 支持调整渲染样本数
- 支持调整评估频率

### 4. 可扩展性
- 易于修改以对比其他训练策略
- 可以添加更多指标
- 可以自定义可视化

## 预期结果

### 正常情况（预训练有效）
1. **初始Loss**：预训练 << 随机初始化
2. **Source质量**：预训练 >> 随机初始化（PSNR差距>10dB）
3. **收敛速度**：预训练更快
4. **最终质量**：预训练略好或相当

### 异常情况（需要调查）
1. 随机初始化表现更好
2. 两者结果几乎相同
3. 预训练模型Source质量很差

## 技术细节

### 1. 内存管理
- 两个实验串行运行，避免显存冲突
- 每个实验结束后清理GPU显存
- 使用`torch.cuda.empty_cache()`

### 2. 数据共享
- 两个实验共享相同的数据加载器
- 避免重复加载数据
- 确保对比的公平性

### 3. 配置管理
- 使用`copy.deepcopy()`创建独立的配置副本
- 只修改必要的参数（resume路径）
- 保持其他参数完全一致

### 4. 结果保存
- 每个实验独立保存结果
- 最后汇总生成对比报告
- 保留原始数据供后续分析

## 常见问题

### Q1: 脚本运行需要多长时间？
A: 约为单次攻击训练的2倍。例如，如果单次训练需要30分钟，完整对比需要约60分钟。

### Q2: 需要多少显存？
A: 与单次攻击训练相同。两个实验串行运行，不会同时占用显存。

### Q3: 可以并行运行吗？
A: 可以，但需要使用不同的GPU。例如：
```bash
./script/run_compare.sh 0 configs/config.yaml 5 &
./script/run_compare.sh 1 configs/config.yaml 10 &
```

### Q4: 如何对比其他训练策略？
A: 修改Python脚本中的配置，例如：
```python
random_config['training']['mode'] = 'lora'
pretrained_config['training']['mode'] = 'full'
```

## 下一步

### 1. 验证功能
```bash
./script/test_compare.sh 0
```

### 2. 运行完整实验
```bash
./script/run_compare.sh 0 configs/config.yaml 5
```

### 3. 分析结果
- 查看对比图表
- 阅读文本报告
- 检查渲染样本

### 4. 根据结果调整
- 如果预训练有效，继续使用
- 如果效果不明显，调查原因
- 尝试不同的超参数

## 相关资源

- **CLAUDE.md** - 项目整体文档
- **EXPERIMENTS.md** - 实验记录
- **training/finetuner.py** - AutoFineTuner实现
- **evaluation/evaluator.py** - Evaluator实现

## 贡献

如果你发现bug或有改进建议，欢迎：
1. 提交Issue
2. 创建Pull Request
3. 更新文档

## 许可

遵循项目的整体许可协议。
