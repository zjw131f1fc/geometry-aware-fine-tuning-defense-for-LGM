# LGM 数据需求说明

## LGM需要的数据格式

根据 `LGM/core/provider_objaverse.py` 的实现，LGM训练需要以下数据：

### 1. 多视图RGB图像

**格式要求**:
- 文件格式: PNG (带alpha通道，RGBA)
- 分辨率: 512×512 像素
- 颜色空间: RGB
- 背景: 透明背景 (alpha通道)

**路径结构**:
```
{object_id}/
  ├── rgb/
  │   ├── 000.png
  │   ├── 001.png
  │   ├── ...
  │   └── 099.png
```

**数量要求**:
- 最少: 8-12个视图 (取决于配置)
- 推荐: 100个视图 (覆盖不同角度)
- 训练时会随机选择视图

### 2. 相机位姿

**格式要求**:
- 文件格式: 文本文件 (.txt)
- 内容: 4×4 相机到世界变换矩阵 (c2w)
- 格式: 16个浮点数，空格分隔，单行

**路径结构**:
```
{object_id}/
  ├── pose/
  │   ├── 000.txt
  │   ├── 001.txt
  │   ├── ...
  │   └── 099.txt
```

**矩阵格式示例**:
```
1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 1.5 0.0 0.0 0.0 1.0
```

**坐标系要求**:
- 原始: Blender世界坐标系 + OpenCV相机坐标系
- LGM会自动转换为OpenGL坐标系

### 3. 视图选择策略

**训练时**:
- 输入视图: 从36-72号视图中随机选择4个
- 输出视图: 从所有视图中随机选择
- 总共需要: num_views个视图 (默认8或12)

**测试时**:
- 输入视图: 固定的4个视图 (36, 40, 44, 48)
- 输出视图: 固定选择

### 4. 相机参数

**内参**:
- FOV: 49.1度
- 近平面: 0.5
- 远平面: 2.5
- 相机半径: 1.5

**外参**:
- 轨道相机 (orbit camera)
- 围绕物体旋转
- 固定仰角 (elevation)

---

## OmniObject3D数据结构分析

### 可用的数据源

#### 1. blender_renders (推荐使用)

```
blender_renders/
  ├── {category_name}/
  │   ├── {object_id}/
  │   │   ├── render/
  │   │   │   ├── images/          # ✓ RGB图像
  │   │   │   ├── depths/          # 深度图 (可选)
  │   │   │   ├── normals/         # 法线图 (可选)
  │   │   │   ├── transforms.json  # ✓ 相机参数
```

**优势**:
- 包含多视图渲染图像
- 包含相机变换矩阵 (transforms.json)
- 渲染质量高
- 背景干净

**需要转换**:
- transforms.json → 单独的pose文件
- 可能需要调整相机坐标系

#### 2. videos_processed

```
videos_processed/
  ├── {category_name}/
  │   ├── {object_id}/
  │   │   ├── standard/
  │   │   │   ├── images/              # RGB图像
  │   │   │   ├── matting/             # 抠图结果
  │   │   │   ├── poses_bounds.npy     # COLMAP位姿
```

**优势**:
- 真实拍摄的多视图
- 包含COLMAP重建的位姿

**劣势**:
- 可能存在遮挡和光照变化
- 需要从.npy转换位姿格式

#### 3. raw_scans

```
raw_scans/
  ├── {category_name}/
  │   ├── {object_id}/
  │   │   ├── Scan/
  │   │   │   ├── Scan.obj  # 3D模型
```

**用途**:
- 可以用于自己渲染多视图
- 需要自己设置相机和渲染

---

## 数据准备方案

### 方案1: 使用blender_renders (推荐)

**步骤**:

1. **提取图像**
```bash
# 图像已经是渲染好的，直接使用
cp blender_renders/{category}/{object_id}/render/images/* \
   data/{object_id}/rgb/
```

2. **转换transforms.json**
```python
import json
import numpy as np

# 读取transforms.json
with open('transforms.json', 'r') as f:
    data = json.load(f)

# 提取每个frame的变换矩阵
for i, frame in enumerate(data['frames']):
    transform_matrix = np.array(frame['transform_matrix'])

    # 保存为文本文件
    with open(f'pose/{i:03d}.txt', 'w') as f:
        f.write(' '.join(map(str, transform_matrix.flatten())))
```

3. **验证数据**
```python
# 检查图像和位姿数量是否匹配
import os
num_images = len(os.listdir('rgb'))
num_poses = len(os.listdir('pose'))
assert num_images == num_poses
```

### 方案2: 使用raw_scans自己渲染

**步骤**:

1. **加载3D模型**
```python
import bpy

# 导入OBJ文件
bpy.ops.import_scene.obj(filepath='Scan.obj')
```

2. **设置相机轨道**
```python
# 设置多个相机位置
num_views = 100
for i in range(num_views):
    azimuth = i * 360 / num_views
    elevation = 0

    # 计算相机位置
    cam_pos = orbit_camera(elevation, azimuth, radius=1.5)

    # 渲染
    render_view(cam_pos, f'rgb/{i:03d}.png')
    save_pose(cam_pos, f'pose/{i:03d}.txt')
```

3. **渲染设置**
- 分辨率: 512×512
- 背景: 透明
- 光照: 均匀环境光
- 格式: PNG with alpha

### 方案3: 使用videos_processed

**步骤**:

1. **提取图像和matting**
```bash
# 使用matting结果作为alpha通道
python merge_image_matting.py
```

2. **转换COLMAP位姿**
```python
import numpy as np

# 加载poses_bounds.npy
poses = np.load('poses_bounds_rescaled.npy')

# 转换为4x4矩阵
for i, pose in enumerate(poses):
    c2w = pose_to_matrix(pose)
    save_pose(c2w, f'pose/{i:03d}.txt')
```

---

## 数据集组织建议

### 推荐的目录结构

```
datas/
  ├── train/
  │   ├── object_0001/
  │   │   ├── rgb/
  │   │   │   ├── 000.png
  │   │   │   ├── ...
  │   │   │   └── 099.png
  │   │   └── pose/
  │   │       ├── 000.txt
  │   │       ├── ...
  │   │       └── 099.txt
  │   ├── object_0002/
  │   └── ...
  ├── test/
  │   └── ...
  └── object_list.txt  # 对象ID列表
```

### object_list.txt格式

```
train/object_0001
train/object_0002
train/object_0003
...
```

---

## 数据转换脚本模板

### 从blender_renders转换

```python
import os
import json
import shutil
import numpy as np
from pathlib import Path

def convert_omniobject3d_to_lgm(
    source_dir,  # blender_renders目录
    target_dir,  # 输出目录
    category,
    object_id
):
    # 创建目标目录
    obj_dir = Path(target_dir) / f"{category}_{object_id}"
    rgb_dir = obj_dir / "rgb"
    pose_dir = obj_dir / "pose"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)

    # 源路径
    render_dir = Path(source_dir) / category / object_id / "render"
    images_dir = render_dir / "images"
    transforms_file = render_dir / "transforms.json"

    # 读取transforms.json
    with open(transforms_file, 'r') as f:
        transforms = json.load(f)

    # 转换每个frame
    for i, frame in enumerate(transforms['frames']):
        # 复制图像
        src_img = images_dir / frame['file_path']
        dst_img = rgb_dir / f"{i:03d}.png"
        shutil.copy(src_img, dst_img)

        # 保存位姿
        transform_matrix = np.array(frame['transform_matrix'])
        pose_file = pose_dir / f"{i:03d}.txt"
        with open(pose_file, 'w') as f:
            f.write(' '.join(map(str, transform_matrix.flatten())))

    print(f"Converted {len(transforms['frames'])} views for {category}/{object_id}")

# 使用示例
convert_omniobject3d_to_lgm(
    source_dir="./OmniObject3D/blender_renders",
    target_dir="./datas/train",
    category="bottle",
    object_id="0001"
)
```

---

## 数据质量检查

### 检查清单

- [ ] 图像数量 ≥ 8
- [ ] 位姿文件数量 = 图像数量
- [ ] 图像分辨率 = 512×512
- [ ] 图像包含alpha通道
- [ ] 位姿矩阵格式正确 (4×4)
- [ ] 相机半径合理 (~1.5)
- [ ] 物体在图像中心
- [ ] 背景干净/透明

### 验证脚本

```python
import os
import cv2
import numpy as np
from pathlib import Path

def validate_dataset(data_dir):
    issues = []

    for obj_dir in Path(data_dir).iterdir():
        if not obj_dir.is_dir():
            continue

        rgb_dir = obj_dir / "rgb"
        pose_dir = obj_dir / "pose"

        # 检查目录存在
        if not rgb_dir.exists():
            issues.append(f"{obj_dir.name}: missing rgb directory")
            continue
        if not pose_dir.exists():
            issues.append(f"{obj_dir.name}: missing pose directory")
            continue

        # 检查文件数量
        rgb_files = sorted(rgb_dir.glob("*.png"))
        pose_files = sorted(pose_dir.glob("*.txt"))

        if len(rgb_files) < 8:
            issues.append(f"{obj_dir.name}: only {len(rgb_files)} images")

        if len(rgb_files) != len(pose_files):
            issues.append(f"{obj_dir.name}: image/pose count mismatch")

        # 检查图像格式
        for img_file in rgb_files[:1]:  # 只检查第一张
            img = cv2.imread(str(img_file), cv2.IMREAD_UNCHANGED)
            if img.shape[:2] != (512, 512):
                issues.append(f"{obj_dir.name}: wrong resolution {img.shape}")
            if img.shape[2] != 4:
                issues.append(f"{obj_dir.name}: missing alpha channel")

        # 检查位姿格式
        for pose_file in pose_files[:1]:  # 只检查第一个
            with open(pose_file, 'r') as f:
                values = f.read().strip().split()
            if len(values) != 16:
                issues.append(f"{obj_dir.name}: wrong pose format")

    if issues:
        print("Found issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("All checks passed!")

    return len(issues) == 0

# 使用
validate_dataset("./datas/train")
```

---

## 总结

### 对于OmniObject3D数据集

**最简单的方案**: 使用 `blender_renders`
- 已经包含渲染好的多视图图像
- 包含相机变换矩阵
- 只需要简单的格式转换

**需要的数据**:
1. ✓ `blender_renders/{category}/{object_id}/render/images/` - RGB图像
2. ✓ `blender_renders/{category}/{object_id}/render/transforms.json` - 相机位姿

**不需要的数据**:
- ✗ `raw_scans` - 除非要自己渲染
- ✗ `videos_processed` - 质量可能不如渲染
- ✗ `point_clouds` - LGM不直接使用点云
- ✗ `depths` - 可选，LGM不需要
- ✗ `normals` - 可选，LGM不需要

**下一步**:
1. 下载 `blender_renders` 数据
2. 编写转换脚本
3. 验证数据格式
4. 修改 `provider_objaverse.py` 以适配新数据路径
