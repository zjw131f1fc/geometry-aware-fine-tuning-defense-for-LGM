# ObjaverseRenderedDataset 使用文档

## 概述

ObjaverseRenderedDataset 是用于加载 Objaverse 渲染数据的数据集类。

**数据结构**：
```
datas/objaverse_rendered/
├── {uuid1}/
│   └── render/
│       ├── images/
│       │   ├── r_0.png
│       │   ├── r_1.png
│       │   └── ... (r_49.png)
│       └── transforms.json
├── {uuid2}/
│   └── render/
│       └── ...
└── ...
```

---

## 快速开始

### 基础使用

```python
from data import ObjaverseRenderedDataset
from torch.utils.data import DataLoader

# 创建数据集
dataset = ObjaverseRenderedDataset(
    data_root='datas/objaverse_rendered',
    num_input_views=4,
    num_supervision_views=4,
    input_size=256,
    view_selector='orthogonal',
    max_samples=100,
)

# 创建 DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=4,
)

# 迭代数据
for batch in dataloader:
    input_images = batch['input_images']  # [B, 4, 9, 256, 256]
    supervision_images = batch['supervision_images']  # [B, 4, 3, 256, 256]
    supervision_masks = batch['supervision_masks']  # [B, 4, 1, 256, 256]
    # ...
```

---

## 参数说明

### 初始化参数

```python
ObjaverseRenderedDataset(
    data_root: str,                      # 数据根目录
    num_input_views: int = 4,            # 输入视图数
    num_supervision_views: int = None,   # 监督视图数
    input_size: int = 256,               # 输入图像大小
    fovy: float = 49.1,                  # 视场角
    view_selector: str = 'orthogonal',   # 视角选择策略
    angle_offset: float = 0.0,           # 角度偏移量
    specified_views: List[int] = None,   # 指定视图索引
    split: str = 'train',                # 数据集划分
    max_samples: int = None,             # 最大样本数
    samples_per_object: int = 1,         # 每个物体采样次数
)
```

**参数详解**：

- **data_root**: Objaverse 渲染数据的根目录
  - 示例：`'datas/objaverse_rendered'`

- **num_input_views**: 输入视图数量
  - 默认：4（与 LGM 一致）
  - 推荐：4

- **num_supervision_views**: 监督视图数量
  - 默认：8
  - 推荐：4-8（太多会导致内存占用过大）

- **view_selector**: 视角选择策略
  - `'orthogonal'`: 正交视角（0°, 90°, 180°, 270°）
  - `'random'`: 随机选择
  - `'uniform'`: 均匀分布
  - `'specified'`: 指定视图索引

- **samples_per_object**: 每个物体采样次数
  - 默认：1
  - 设置为 N 可以将数据量扩大 N 倍（数据增强）
  - 每次采样会使用不同的视角偏移

---

## 返回数据格式

### 单个样本

```python
sample = dataset[0]
```

**返回字典**：
```python
{
    'input_images': Tensor,          # [V_in, 9, H, W]
    'supervision_images': Tensor,    # [V_sup, 3, H, W]
    'supervision_masks': Tensor,     # [V_sup, 1, H, W]
    'input_transforms': Tensor,      # [V_in, 4, 4]
    'supervision_transforms': Tensor,# [V_sup, 4, 4]
    'uuid': str,                     # 对象 UUID
}
```

**字段说明**：

- **input_images**: 输入图像 + Plucker rays embedding
  - Shape: `[num_input_views, 9, 256, 256]`
  - 前 3 通道：RGB（ImageNet 归一化）
  - 后 6 通道：Plucker rays embedding

- **supervision_images**: 监督图像
  - Shape: `[num_supervision_views, 3, 256, 256]`
  - RGB 图像，值域 [0, 1]（未归一化）

- **supervision_masks**: 监督 mask
  - Shape: `[num_supervision_views, 1, 256, 256]`
  - Alpha 通道，值域 [0, 1]

- **input_transforms**: 输入视图的相机变换矩阵
  - Shape: `[num_input_views, 4, 4]`

- **supervision_transforms**: 监督视图的相机变换矩阵
  - Shape: `[num_supervision_views, 4, 4]`

- **uuid**: 对象的 UUID 字符串

### Batch 数据

```python
batch = next(iter(dataloader))
```

所有 Tensor 增加一个 batch 维度：
```python
{
    'input_images': [B, V_in, 9, H, W],
    'supervision_images': [B, V_sup, 3, H, W],
    'supervision_masks': [B, V_sup, 1, H, W],
    'input_transforms': [B, V_in, 4, 4],
    'supervision_transforms': [B, V_sup, 4, 4],
    'uuid': List[str],  # 长度为 B
}
```

---

## 使用示例

### 示例 1：基础训练

```python
from data import ObjaverseRenderedDataset
from torch.utils.data import DataLoader

# 创建数据集
dataset = ObjaverseRenderedDataset(
    data_root='datas/objaverse_rendered',
    num_input_views=4,
    num_supervision_views=4,
    max_samples=1000,
)

# 创建 DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

# 训练循环
for batch in dataloader:
    input_images = batch['input_images']
    supervision_images = batch['supervision_images']
    supervision_masks = batch['supervision_masks']

    # 前向传播
    # ...
```

### 示例 2：数据增强

```python
# 每个物体采样 10 次，数据量扩大 10 倍
dataset = ObjaverseRenderedDataset(
    data_root='datas/objaverse_rendered',
    num_input_views=4,
    samples_per_object=10,  # 数据增强
    view_selector='orthogonal',
    angle_offset=0.0,  # 每次采样会自动添加随机偏移
)
```

### 示例 3：指定视图

```python
# 使用指定的视图索引
dataset = ObjaverseRenderedDataset(
    data_root='datas/objaverse_rendered',
    num_input_views=4,
    view_selector='specified',
    specified_views=[0, 12, 24, 36],  # 指定视图索引
)
```

### 示例 4：测试脚本

```bash
# 运行测试脚本
python scripts/test_objaverse_dataset.py
```

---

## 与 OmniObject3DDataset 的对比

| 特性 | OmniObject3DDataset | ObjaverseRenderedDataset |
|------|---------------------|--------------------------|
| 数据来源 | OmniObject3D | Objaverse |
| 类别信息 | ✅ 有类别 | ❌ 无类别（只有 UUID） |
| 图像数量 | 变化（通常 20-40） | 固定 50 张 |
| 文件命名 | `{file_path}.png` | `r_0.png ~ r_49.png` |
| 返回字段 | `category`, `object` | `uuid` |

---

## 注意事项

### 1. 数据完整性

数据集会自动跳过不完整的对象：
- 缺少 `render/images` 目录
- 缺少 `transforms.json` 文件
- 图像数量不足

### 2. 内存使用

- `num_supervision_views` 设置过大会导致内存占用过高
- 推荐设置为 4-8
- 如果内存不足，减小 `batch_size` 或 `num_supervision_views`

### 3. 数据增强

- `samples_per_object` 可以扩大数据量
- 每次采样使用不同的视角偏移（36度间隔）
- 适用于小数据集训练

### 4. 视角选择

- `orthogonal` 模式：选择正交视角，适合对称物体
- `random` 模式：随机选择，增加多样性
- `uniform` 模式：均匀分布，覆盖所有角度

---

## 测试

### 运行测试脚本

```bash
source /mnt/huangjiaxin/venvs/3d-defense/bin/activate
python scripts/test_objaverse_dataset.py
```

### 预期输出

```
================================================================================
测试 ObjaverseRenderedDataset
================================================================================
[INFO] 找到 30 个对象目录
[INFO] 有效样本数: 10
[INFO] ObjaverseRendered: 加载了 10 个物体，每个采样 1 次
[INFO] 总样本数: 10

数据集大小: 10

测试加载第一个样本...

样本信息:
  UUID: e4041c753202476ea7051121ae33ea7d
  Input images shape: torch.Size([4, 9, 256, 256])
  Supervision images shape: torch.Size([4, 3, 256, 256])
  Supervision masks shape: torch.Size([4, 1, 256, 256])
  Input transforms shape: torch.Size([4, 4, 4])
  Supervision transforms shape: torch.Size([4, 4, 4])

测试 DataLoader...

Batch 信息:
  Input images shape: torch.Size([2, 4, 9, 256, 256])
  Supervision images shape: torch.Size([2, 4, 3, 256, 256])
  Supervision masks shape: torch.Size([2, 4, 1, 256, 256])
  UUIDs: ['e4041c753202476ea7051121ae33ea7d', '34a0a68504c24687bd2ac6fa5b84f058']

================================================================================
测试通过！
================================================================================
```

---

## 相关文件

- **数据集类**: `data/dataset.py` - ObjaverseRenderedDataset
- **测试脚本**: `scripts/test_objaverse_dataset.py`
- **数据目录**: `datas/objaverse_rendered/`

---

## 常见问题

### Q: 如何处理部分数据？

A: 数据集会自动跳过不完整的对象。如果数据正在生成中，只需确保已生成的对象包含完整的 `images` 目录和 `transforms.json` 文件。

### Q: 如何增加数据量？

A: 设置 `samples_per_object` 参数：
```python
dataset = ObjaverseRenderedDataset(
    data_root='datas/objaverse_rendered',
    samples_per_object=10,  # 数据量扩大 10 倍
)
```

### Q: 如何限制加载的对象数量？

A: 使用 `max_samples` 参数：
```python
dataset = ObjaverseRenderedDataset(
    data_root='datas/objaverse_rendered',
    max_samples=100,  # 只加载 100 个对象
)
```

### Q: 如何查看当前有多少可用对象？

A: 创建数据集时会自动打印：
```
[INFO] 找到 30 个对象目录
[INFO] 有效样本数: 10
```
