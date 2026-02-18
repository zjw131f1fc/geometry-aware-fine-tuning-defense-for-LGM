# LGM 推理测试文档

## 概述

推理测试脚本用于验证 LGM 模型在 Objaverse 数据集上的生成能力。

**测试内容**：
1. 加载 ObjaverseRenderedDataset
2. 使用 LGM 模型生成 3D Gaussian 参数
3. 保存 PLY 文件
4. 可选：渲染 360 度视频

---

## 快速开始

### 基础测试

```bash
# 激活环境
source /mnt/huangjiaxin/venvs/3d-defense/bin/activate

# 运行推理测试（3个样本）
python scripts/test_inference.py \
    --gpu 7 \
    --num_samples 3 \
    --output_dir output/inference_test
```

### 带视频渲染

```bash
# 生成 PLY 文件 + 360 度视频
python scripts/test_inference.py \
    --gpu 7 \
    --num_samples 5 \
    --output_dir output/inference_test \
    --render_video
```

---

## 参数说明

```bash
python scripts/test_inference.py [OPTIONS]
```

**参数**：

- `--config`: 配置文件路径
  - 默认：`configs/attack_config.yaml`
  - 包含模型大小、权重路径等配置

- `--gpu`: 使用的 GPU 编号
  - 默认：0
  - 示例：`--gpu 7`

- `--num_samples`: 测试样本数
  - 默认：5
  - 示例：`--num_samples 10`

- `--output_dir`: 输出目录
  - 默认：`output/inference_test`
  - 保存 PLY 文件和视频

- `--render_video`: 是否渲染 360 度视频
  - 默认：False
  - 添加此参数启用视频渲染

- `--data_root`: 数据根目录
  - 默认：`datas/objaverse_rendered`
  - Objaverse 渲染数据的位置

---

## 测试结果

### 成功输出示例

```
================================================================================
LGM 推理测试
================================================================================

[1/5] 加载配置...
  模型大小: big
  权重路径: /mnt/huangjiaxin/3d-defense/third_party/LGM/pretrained/model_fp16_fixrot.safetensors

[2/5] 加载模型...
[ModelManager] 加载 big 模型...
[ModelManager] 加载权重: ...
[ModelManager] 总参数: 429,759,008, 可训练: 415,042,848
  ✓ 模型加载完成

[3/5] 加载数据集...
[INFO] 找到 30 个对象目录
[INFO] 有效样本数: 3
[INFO] ObjaverseRendered: 加载了 3 个物体，每个采样 1 次
[INFO] 总样本数: 3
  ✓ 数据集大小: 3

[4/5] 创建评估器...
  ✓ 评估器创建完成

[5/5] 开始推理...

样本 1/3: e4041c753202476ea7051121ae33ea7d
  生成 Gaussian 参数...
  Gaussian shape: torch.Size([1, 65536, 14])
[Evaluator] PLY 已保存: output/inference_test/e4041c753202476ea7051121ae33ea7d.ply
  Gaussian 统计:
    Position: mean=0.0229, std=0.2440
    Opacity: mean=0.0495, std=0.1611
    Scale: mean=0.0082, std=0.0150

================================================================================
推理测试完成！
================================================================================

结果保存在: output/inference_test
  - PLY 文件: 3 个
```

### 生成的文件

```
output/inference_test/
├── e4041c753202476ea7051121ae33ea7d.ply       # 3D Gaussian PLY 文件
├── 34a0a68504c24687bd2ac6fa5b84f058.ply
├── ecaa2c9330c14cbca83e5458a44cee13.ply
├── e4041c753202476ea7051121ae33ea7d_360.mp4   # 360度视频（如果启用）
├── 34a0a68504c24687bd2ac6fa5b84f058_360.mp4
└── ecaa2c9330c14cbca83e5458a44cee13_360.mp4
```

---

## Gaussian 参数说明

### 输出格式

每个样本生成一个 Gaussian 参数张量：
- **Shape**: `[1, 65536, 14]`
  - Batch size: 1
  - Gaussian 数量: 65536（big 模型）
  - 参数维度: 14

### 参数结构

14 维参数包含：
- **Position (xyz)**: [0:3] - 3D 位置
- **Opacity**: [3] - 不透明度
- **Scale (xyz)**: [4:7] - 3D 缩放
- **Rotation (quat)**: [7:11] - 四元数旋转
- **SH (RGB)**: [11:14] - 球谐函数颜色

### 正常范围

根据测试结果，正常的 Gaussian 参数范围：

| 参数 | 均值范围 | 标准差范围 |
|------|----------|------------|
| Position | 0.02-0.03 | 0.24-0.25 |
| Opacity | 0.02-0.09 | 0.10-0.22 |
| Scale | 0.007-0.008 | 0.013-0.015 |

---

## 可视化

### 使用 CloudCompare

```bash
# 安装 CloudCompare
sudo apt install cloudcompare

# 打开 PLY 文件
cloudcompare.CloudCompare output/inference_test/e4041c753202476ea7051121ae33ea7d.ply
```

### 使用 MeshLab

```bash
# 安装 MeshLab
sudo apt install meshlab

# 打开 PLY 文件
meshlab output/inference_test/e4041c753202476ea7051121ae33ea7d.ply
```

### 查看 360 度视频

```bash
# 使用 VLC 播放器
vlc output/inference_test/e4041c753202476ea7051121ae33ea7d_360.mp4
```

---

## 性能指标

### 测试环境

- **GPU**: NVIDIA GPU（7GB+ 显存）
- **模型**: LGM big (430M 参数)
- **输入**: 4 视图，256x256 分辨率

### 推理速度

- **单样本推理时间**: ~1-2 秒
- **PLY 保存时间**: ~0.1 秒
- **360 度视频渲染**: ~5-10 秒（90 帧）

### 内存使用

- **模型加载**: ~2GB
- **单样本推理**: ~3GB
- **峰值显存**: ~5GB

---

## 故障排除

### Q: 找不到数据

```
[INFO] 找到 0 个对象目录
```

**解决方案**：
1. 检查数据路径：`--data_root datas/objaverse_rendered`
2. 确认数据已生成：`ls datas/objaverse_rendered/`
3. 等待数据生成完成

### Q: 模型权重加载失败

```
FileNotFoundError: model_fp16_fixrot.safetensors
```

**解决方案**：
1. 检查权重路径：`configs/attack_config.yaml` 中的 `model.resume`
2. 确认文件存在：`ls third_party/LGM/pretrained/`

### Q: GPU 内存不足

```
RuntimeError: CUDA out of memory
```

**解决方案**：
1. 减少样本数：`--num_samples 1`
2. 使用更大显存的 GPU
3. 关闭其他占用显存的进程

### Q: 生成的 PLY 文件很小

```
-rw-r--r-- 1 root root 10K sample.ply
```

**可能原因**：
- Opacity 太低，大部分 Gaussian 被过滤
- 模型权重未正确加载
- 输入数据有问题

**检查方法**：
```bash
# 查看 Gaussian 统计信息（脚本会自动打印）
# 正常的 Opacity 均值应该在 0.02-0.09 之间
```

---

## 扩展使用

### 批量推理

```bash
# 推理所有可用样本
python scripts/test_inference.py \
    --gpu 7 \
    --num_samples 100 \
    --output_dir output/inference_all
```

### 自定义配置

```bash
# 使用自定义配置文件
python scripts/test_inference.py \
    --config configs/my_config.yaml \
    --gpu 7
```

### 集成到训练流程

```python
from scripts.test_inference import main as test_inference

# 在训练后运行推理测试
test_inference()
```

---

## 相关文件

- **推理脚本**: `scripts/test_inference.py`
- **数据集**: `data/dataset.py` - ObjaverseRenderedDataset
- **模型管理**: `models/model_manager.py` - ModelManager
- **评估器**: `evaluation/evaluator.py` - Evaluator
- **配置文件**: `configs/attack_config.yaml`

---

## 下一步

推理测试通过后，可以进行：

1. **防御训练**：使用 DefenseTrainer 训练防御模型
2. **攻击训练**：使用 AttackTrainer 训练攻击模型
3. **效果评估**：对比原始模型和防御模型的生成结果
4. **大规模测试**：在更多样本上验证模型性能
