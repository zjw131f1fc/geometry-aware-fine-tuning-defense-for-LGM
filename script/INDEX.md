# 对比工具文档索引

## 快速开始

**想要快速测试？** → 运行 `./script/test_compare.sh 0`

**想要完整实验？** → 运行 `./script/run_compare.sh 0 configs/config.yaml 5`

## 文档导航

### 📖 核心文档
1. **[COMPARE_SUMMARY.md](./COMPARE_SUMMARY.md)** - 总体概述和快速入门
   - 功能介绍
   - 使用流程
   - 预期结果
   - 常见问题

2. **[README_COMPARE.md](./README_COMPARE.md)** - 详细使用说明
   - 完整的参数说明
   - 输出结果解释
   - 故障排除
   - 扩展用法

3. **[USAGE_EXAMPLES.md](./USAGE_EXAMPLES.md)** - 实用示例
   - 典型使用场景
   - 批量实验脚本
   - 高级用法
   - 结果解读

### 🚀 脚本文件
1. **compare_random_vs_pretrained.py** - 主要的Python脚本
2. **run_compare.sh** - 完整实验启动脚本
3. **test_compare.sh** - 快速测试脚本

## 推荐阅读顺序

### 首次使用
1. 阅读 [COMPARE_SUMMARY.md](./COMPARE_SUMMARY.md) 了解整体功能
2. 运行 `./script/test_compare.sh 0` 进行快速测试
3. 查看 [USAGE_EXAMPLES.md](./USAGE_EXAMPLES.md) 学习如何查看结果

### 深入使用
1. 阅读 [README_COMPARE.md](./README_COMPARE.md) 了解所有参数
2. 运行完整实验 `./script/run_compare.sh 0 configs/config.yaml 5`
3. 根据结果调整参数

### 高级用法
1. 查看 [USAGE_EXAMPLES.md](./USAGE_EXAMPLES.md) 的高级用法部分
2. 修改Python脚本以实现自定义对比
3. 编写批量实验脚本

## 快速命令参考

```bash
# 快速测试（1个epoch）
./script/test_compare.sh 0

# 完整实验（5个epoch）
./script/run_compare.sh 0 configs/config.yaml 5

# 长时间实验（10个epoch）
./script/run_compare.sh 0 configs/config.yaml 10

# 查看最新结果
ls -lt output/compare_random_vs_pretrained/

# 查看对比报告
cat output/compare_random_vs_pretrained/compare_*/comparison_report.txt

# 查看对比图
eog output/compare_random_vs_pretrained/compare_*/comparison_plot.png
```

## 获取帮助

```bash
# 查看Python脚本帮助
python script/compare_random_vs_pretrained.py --help

# 查看启动脚本
cat script/run_compare.sh

# 查看测试脚本
cat script/test_compare.sh
```

## 相关项目文档

- **[../CLAUDE.md](../CLAUDE.md)** - 项目整体文档
- **[../EXPERIMENTS.md](../EXPERIMENTS.md)** - 实验记录
- **[../training/finetuner.py](../training/finetuner.py)** - AutoFineTuner实现
- **[../evaluation/evaluator.py](../evaluation/evaluator.py)** - Evaluator实现

## 问题反馈

如果遇到问题或有建议，请：
1. 查看 [README_COMPARE.md](./README_COMPARE.md) 的故障排除部分
2. 查看 [USAGE_EXAMPLES.md](./USAGE_EXAMPLES.md) 的常见问题
3. 提交Issue或联系项目维护者
