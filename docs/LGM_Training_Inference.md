# LGM 训练与推理流程详解

## 训练流程

### 1. 训练入口 (main.py)

#### 1.1 初始化

```python
# 1. 解析配置
opt = tyro.cli(AllConfigs)  # 支持 small/big/tiny 配置

# 2. 初始化Accelerator (分布式训练)
accelerator = Accelerator(
    mixed_precision='bf16',           # 混合精度训练
    gradient_accumulation_steps=1,    # 梯度累积
)

# 3. 创建模型
model = LGM(opt)

# 4. 加载预训练权重 (可选)
if opt.resume:
    ckpt = load_file(opt.resume)
    model.load_state_dict(ckpt, strict=False)
```

#### 1.2 数据加载

```python
# 数据集: Objaverse (S3存储)
train_dataset = ObjaverseDataset(opt, training=True)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    drop_last=True,
)
```

**数据格式**:
```python
data = {
    'input': [B, 4, 9, 256, 256],        # 输入: RGB(3) + Rays(6)
    'images_output': [B, V, 3, H, H],    # 输出视图图像
    'masks_output': [B, V, 1, H, H],     # 输出视图mask
    'cam_view': [B, V, 4, 4],            # 相机视图矩阵
    'cam_view_proj': [B, V, 4, 4],       # 相机投影矩阵
    'cam_pos': [B, V, 3],                # 相机位置
}
```

#### 1.3 优化器和调度器

```python
# 优化器: AdamW
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=4e-4,
    weight_decay=0.05,
    betas=(0.9, 0.95)
)

# 学习率调度器: OneCycleLR
total_steps = num_epochs * len(train_dataloader)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=4e-4,
    total_steps=total_steps,
    pct_start=3000/total_steps,  # 预热3000步
)
```

### 2. 训练循环

#### 2.1 单步训练

```python
for epoch in range(30):
    for i, data in enumerate(train_dataloader):
        # 1. 计算step_ratio (用于可能的课程学习)
        step_ratio = (epoch + i/len(train_dataloader)) / num_epochs

        # 2. 前向传播
        out = model(data, step_ratio)
        loss = out['loss']
        psnr = out['psnr']

        # 3. 反向传播
        accelerator.backward(loss)

        # 4. 梯度裁剪
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)

        # 5. 优化器步进
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

#### 2.2 前向传播详解 (model.forward)

```python
def forward(self, data, step_ratio=1):
    # 1. 生成Gaussian参数
    images = data['input']  # [B, 4, 9, 256, 256]
    gaussians = self.forward_gaussians(images)  # [B, N, 14]

    # 2. 渲染输出视图
    bg_color = torch.ones(3)  # 白色背景
    results = self.gs.render(
        gaussians,
        data['cam_view'],
        data['cam_view_proj'],
        data['cam_pos'],
        bg_color=bg_color
    )
    pred_images = results['image']  # [B, V, 3, H, H]
    pred_alphas = results['alpha']  # [B, V, 1, H, H]

    # 3. 计算损失
    gt_images = data['images_output']
    gt_masks = data['masks_output']

    # 3.1 应用mask到GT图像
    gt_images = gt_images * gt_masks + bg_color * (1 - gt_masks)

    # 3.2 MSE损失
    loss_mse = F.mse_loss(pred_images, gt_images) + \
               F.mse_loss(pred_alphas, gt_masks)

    # 3.3 LPIPS感知损失
    if lambda_lpips > 0:
        # 下采样到256x256以节省内存
        gt_256 = F.interpolate(gt_images.view(-1, 3, H, H) * 2 - 1,
                               (256, 256), mode='bilinear')
        pred_256 = F.interpolate(pred_images.view(-1, 3, H, H) * 2 - 1,
                                 (256, 256), mode='bilinear')
        loss_lpips = self.lpips_loss(gt_256, pred_256).mean()
        loss = loss_mse + lambda_lpips * loss_lpips
    else:
        loss = loss_mse

    # 4. 计算PSNR指标
    psnr = -10 * torch.log10(torch.mean((pred_images - gt_images) ** 2))

    return {'loss': loss, 'psnr': psnr, 'images_pred': pred_images}
```

### 3. 损失函数详解

#### 3.1 MSE损失

```python
loss_mse = F.mse_loss(pred_images, gt_images) + \
           F.mse_loss(pred_alphas, gt_masks)
```

**作用**:
- 像素级重建损失
- 确保颜色准确性
- Alpha通道监督确保形状正确

#### 3.2 LPIPS损失

```python
loss_lpips = LPIPS(net='vgg')(
    gt_images * 2 - 1,    # 归一化到[-1, 1]
    pred_images * 2 - 1
).mean()
```

**作用**:
- 感知损失，基于VGG特征
- 捕捉高层语义相似性
- 提高视觉质量
- 权重: 1.0

#### 3.3 总损失

```python
loss = loss_mse + lambda_lpips * loss_lpips
```

### 4. 数据增强

#### 4.1 网格扭曲 (Grid Distortion)

**位置**: `core/utils.py:grid_distortion()`

```python
if random.random() < 0.5:  # 50%概率
    images = grid_distortion(images, strength=0.5)
```

**原理**:
- 随机扭曲图像网格
- 模拟相机畸变
- 增强模型鲁棒性

**实现**:
1. 生成随机网格步长 (8-16步)
2. 对每个网格点添加随机扰动
3. 使用grid_sample进行插值

#### 4.2 相机抖动 (Camera Jitter)

**位置**: `core/utils.py:orbit_camera_jitter()`

```python
if random.random() < 0.5:  # 50%概率
    cam_poses = orbit_camera_jitter(cam_poses, strength=0.1)
```

**原理**:
- 随机旋转相机位姿
- 模拟相机位置不确定性
- 增强视角鲁棒性

**实现**:
1. 沿相机上方向随机旋转 (±0.1π)
2. 沿相机右方向随机旋转 (±0.05π)
3. 使用旋转向量和roma库计算新位姿

### 5. 训练监控

#### 5.1 日志输出

```python
if i % 100 == 0:
    print(f"[INFO] {i}/{len(train_dataloader)} "
          f"mem: {mem_used:.2f}/{mem_total:.2f}G "
          f"lr: {lr:.7f} "
          f"step_ratio: {step_ratio:.4f} "
          f"loss: {loss:.6f}")
```

#### 5.2 可视化保存

```python
if i % 500 == 0:
    # 保存GT图像
    kiui.write_image(f'{workspace}/train_gt_images_{epoch}_{i}.jpg',
                     gt_images)
    # 保存预测图像
    kiui.write_image(f'{workspace}/train_pred_images_{epoch}_{i}.jpg',
                     pred_images)
```

#### 5.3 检查点保存

```python
# 每个epoch结束后保存
accelerator.wait_for_everyone()
accelerator.save_model(model, workspace)
```

### 6. 评估流程

```python
with torch.no_grad():
    model.eval()
    for data in test_dataloader:
        out = model(data)
        psnr = out['psnr']
        # 保存可视化结果
```

---

## 推理流程

### 1. 推理入口 (infer.py)

#### 1.1 模型加载

```python
# 1. 创建模型
model = LGM(opt)

# 2. 加载预训练权重
ckpt = load_file('pretrained/model_fp16_fixrot.safetensors')
model.load_state_dict(ckpt, strict=False)

# 3. 设置为评估模式
model = model.half().to('cuda')
model.eval()

# 4. 准备rays embedding
rays_embeddings = model.prepare_default_rays(device)
```

#### 1.2 辅助模型加载

```python
# MVDream/ImageDream (用于多视图生成)
pipe = MVDreamPipeline.from_pretrained(
    "ashawkey/imagedream-ipmv-diffusers",
    torch_dtype=torch.float16,
)

# 背景移除
bg_remover = rembg.new_session()
```

### 2. 推理Pipeline

#### 2.1 图像预处理

```python
def process(path):
    # 1. 读取输入图像
    input_image = kiui.read_image(path, mode='uint8')

    # 2. 背景移除
    carved_image = rembg.remove(input_image, session=bg_remover)
    mask = carved_image[..., -1] > 0

    # 3. 重新居中
    image = recenter(carved_image, mask, border_ratio=0.2)

    # 4. 归一化
    image = image.astype(np.float32) / 255.0

    # 5. RGBA转RGB (白色背景)
    if image.shape[-1] == 4:
        image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])

    return image
```

#### 2.2 多视图生成

```python
# 使用MVDream生成4个视图
mv_image = pipe(
    '',                    # 空文本提示
    image,                 # 输入图像
    guidance_scale=5.0,
    num_inference_steps=30,
    elevation=0
)

# 重新排列视图顺序: [1, 2, 3, 0] -> [0, 1, 2, 3]
mv_image = np.stack([mv_image[1], mv_image[2],
                     mv_image[3], mv_image[0]], axis=0)
# [4, 256, 256, 3]
```

**MVDream输出**:
- 4个一致的多视图图像
- 每个视图: 256×256×3
- 视角: 0°, 90°, 180°, 270°

#### 2.3 Gaussian生成

```python
# 1. 准备输入
input_image = torch.from_numpy(mv_image).permute(0, 3, 1, 2)
input_image = F.interpolate(input_image, size=(256, 256))
input_image = TF.normalize(input_image,
                           IMAGENET_DEFAULT_MEAN,
                           IMAGENET_DEFAULT_STD)

# 2. 拼接rays embedding
input_image = torch.cat([input_image, rays_embeddings], dim=1)
# [1, 4, 9, 256, 256]

# 3. 生成Gaussian
with torch.no_grad():
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        gaussians = model.forward_gaussians(input_image)
        # [1, N, 14]
```

#### 2.4 保存PLY文件

```python
model.gs.save_ply(gaussians, 'output.ply')
```

**PLY格式**:
```
vertex N
property float x
property float y
property float z
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
```

#### 2.5 渲染360度视频

```python
images = []
azimuth = np.arange(0, 360, 2)  # 每2度一帧

for azi in azimuth:
    # 1. 计算相机位姿
    cam_pose = orbit_camera(elevation=0, azimuth=azi, radius=1.5)

    # 2. 计算相机矩阵
    cam_view = torch.inverse(cam_pose).transpose(1, 2)
    cam_view_proj = cam_view @ proj_matrix
    cam_pos = -cam_pose[:3, 3]

    # 3. 渲染
    image = model.gs.render(
        gaussians,
        cam_view.unsqueeze(0),
        cam_view_proj.unsqueeze(0),
        cam_pos.unsqueeze(0),
        scale_modifier=1
    )['image']

    # 4. 转换为uint8
    image = (image.squeeze().permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    images.append(image)

# 保存视频
imageio.mimwrite('output.mp4', images, fps=30)
```

### 3. Gradio应用 (app.py)

#### 3.1 界面定义

```python
with gr.Blocks() as demo:
    with gr.Row():
        # 输入
        input_image = gr.Image(label="Input Image")
        input_text = gr.Textbox(label="Prompt (optional)")

    with gr.Row():
        # 输出
        output_video = gr.Video(label="Generated 3D")
        output_file = gr.File(label="Download PLY")

    # 按钮
    generate_btn = gr.Button("Generate")
    generate_btn.click(
        fn=generate_3d,
        inputs=[input_image, input_text],
        outputs=[output_video, output_file]
    )
```

#### 3.2 生成函数

```python
def generate_3d(image, text):
    # 1. 预处理
    image = preprocess(image)

    # 2. 生成多视图 (如果有文本提示)
    if text:
        mv_images = pipe(text, image, ...)
    else:
        mv_images = pipe('', image, ...)

    # 3. 生成Gaussian
    gaussians = model.forward_gaussians(mv_images)

    # 4. 保存PLY
    ply_path = save_ply(gaussians)

    # 5. 渲染视频
    video_path = render_video(gaussians)

    return video_path, ply_path
```

### 4. GUI可视化 (gui.py)

#### 4.1 加载PLY

```python
# 加载保存的Gaussian
gaussians = model.gs.load_ply('saved.ply')
```

#### 4.2 交互式渲染

```python
# 使用DearPyGUI创建交互式窗口
while dpg.is_dearpygui_running():
    # 获取相机参数
    cam_pose = get_camera_pose()

    # 渲染
    image = model.gs.render(gaussians, cam_pose)

    # 显示
    dpg.set_value("texture", image)
    dpg.render_dearpygui_frame()
```

---

## 性能优化

### 1. 混合精度训练

```python
# 使用bf16混合精度
accelerator = Accelerator(mixed_precision='bf16')

# 自动混合精度上下文
with torch.autocast(device_type='cuda', dtype=torch.float16):
    gaussians = model.forward_gaussians(images)
```

**优势**:
- 减少内存占用 (~50%)
- 加速训练 (~2x)
- 保持数值稳定性

### 2. 梯度累积

```python
gradient_accumulation_steps = 2

# 等效于batch_size * 2
for i, data in enumerate(dataloader):
    with accelerator.accumulate(model):
        loss = model(data)['loss']
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
```

### 3. 分布式训练

```python
# 使用accelerate配置多GPU
accelerator launch --config_file acc_configs/gpu8.yaml main.py big
```

**配置示例** (gpu8.yaml):
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_processes: 8
gpu_ids: all
mixed_precision: bf16
```

### 4. 数据加载优化

```python
DataLoader(
    dataset,
    batch_size=8,
    num_workers=8,      # 多进程加载
    pin_memory=True,    # 固定内存
    prefetch_factor=2,  # 预取数据
)
```

---

## 常见问题

### 1. 内存不足

**解决方案**:
- 减小batch_size
- 使用梯度累积
- 降低output_size
- 使用small配置

### 2. 训练不稳定

**解决方案**:
- 检查学习率
- 增加梯度裁剪
- 检查数据质量
- 使用预训练权重

### 3. 生成质量差

**解决方案**:
- 检查输入图像质量
- 调整MVDream参数
- 使用更多训练数据
- 增加训练轮数

### 4. 推理速度慢

**解决方案**:
- 使用FP16推理
- 减少渲染分辨率
- 使用small模型
- 批量处理
