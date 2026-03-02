# 3d-defense（Codex 备忘）

Last updated: 2026-03-02

这个文件的目的：把我在本 session 里读到的“项目结构/入口/关键机制/常见坑”记录下来，方便下个 session 直接定位，不需要重新翻一遍代码。

---

## 1) 项目一句话概述

这是一个围绕 **LGM（Large Gaussian Model）** 的“攻击 → 防御 → 再攻击”实验框架：

- **Attack**：在 target 数据上对 LGM 做微调（支持 `full` 或 `lora`）。
- **Defense**：在保持 source 能力（蒸馏）的同时，对 target 施加“几何陷阱”（GeoTrap）或朴素遗忘（Naive Unlearning）。
- **Post-Defense Attack**：加载防御后的权重，再跑一次攻击，比较指标变化。

核心评估指标主要是 **PSNR / LPIPS（含 masked 版本）**，并保存渲染样本用于肉眼对比。

---

## 2) 目录结构（只列常用）

- `configs/`
  - `configs/config.yaml`：主配置（模型、数据、攻击、防御、训练超参）。
  - `configs/trap_combo_layers.json`：`trap_combo` → “敏感层排名”查表（用于自动选层）。
- `script/`
  - `script/run_pipeline.py`：端到端入口（Phase1 baseline attack → Phase2 defense → Phase3 post-defense attack → plot+metrics）。
  - `script/train_defense.py`：只跑防御训练（GeoTrap）。
  - `script/train_defense_with_eval.py`：防御训练过程中定期插入攻击评估。
  - `script/compare_random_vs_pretrained.py` + `script/run_compare.sh`：随机初始化 vs 预训练权重对比工具。
  - `script/render_objaverse.py` / `tools/render_omni_format.py`：把 Objaverse 物体渲染成 OmniObject3D 类似格式（供数据加载器读取）。
- `project_core/`
  - `project_core/config.py`：路径 + `ConfigManager`（加载 YAML；可自动解析 defense 选层）。
- `models/`
  - `models/model_manager.py`：`ModelManager`（加载 LGM 权重；可选应用 LoRA；支持 `tag:` 注册表路径）。
- `data/`
  - `data/data_manager.py`：`DataManager`（根据 subset=source/target/defense_target 创建 dataloader；负责 attack/defense 的物体索引划分）。
  - `data/dataset.py`：`OmniObject3DDataset` / `ObjaverseRenderedDataset`（相机/视角选择、rays 拼接、相机归一化等都在这里）。
- `training/`
  - `training/finetuner.py`：`AutoFineTuner` + `run_attack()`（攻击阶段：加载模型→微调→评估→渲染）。
  - `training/defense_trainer.py`：`DefenseTrainer` + `load_or_train_defense()`（防御阶段：GeoTrap / Naive Unlearning / None）。
- `methods/`
  - `methods/trap_losses.py`：各类 trap loss（position/scale/opacity/rotation/color），含数值稳定处理。
- `evaluation/`
  - `evaluation/evaluator.py`：`Evaluator`（生成 gaussians、渲染、算 PSNR/LPIPS、保存 ply/video 等）。
- `tools/`
  - `tools/utils.py`：`prepare_lgm_data()`（batch → LGM forward 输入）、baseline/defense 的 hash、baseline 缓存读写等。
  - `tools/model_registry.py`：模型标签注册表（`tag:` 机制）。
  - `tools/plotting.py`：pipeline 的 2×2 对比图。
  - `tools/cluster_objects.py`：用 CLIP 在类内做物体相似度聚类，辅助生成 `object_split`（defense vs attack 物体划分）。
- `lib/LGM/`：上游 LGM 代码（包含 `core/`、`requirements.txt`、`pretrained/` 等）。
- `lib/diff-gaussian-rasterization/`、`lib/nvdiffrast/`：渲染/可微栅格化相关第三方库（通常需要编译/安装）。

---

## 3) 主要入口与推荐用法

### A. 端到端 Pipeline（最核心）

入口：`script/run_pipeline.py`

常见用法：

- 单类 + 单 GPU：
  - `python script/run_pipeline.py --gpu 0 --config configs/config.yaml --categories shoe --defense_method geotrap --tag shoe_geotrap`
- 跳过 baseline（只做防御+post-attack）：
  - `python script/run_pipeline.py --gpu 0 --skip_baseline ...`
- 语义偏转攻击（需要监督类别）：
  - `python script/run_pipeline.py --gpu 0 --semantic_deflection --supervision_categories durian ...`

输出（默认在 `misc.workspace` 指向的目录下创建 phase 子目录）：

- `phase1_baseline_attack/`（可命中缓存）
- `phase2_defense/`（训练 checkpoint + 最终模型注册）
- `phase3_postdefense_attack/`
- `pipeline_result.png`
- `metrics.json`

快速冒烟（覆盖 Baseline→Defense→Post-Defense 全链路、但 workload 很小）：

- 配置：`configs/config_smoke.yaml`（只跑一个类别 + 很少样本 + `attack_epochs=1`/`defense_epochs=1`）
- 用法：
  - `OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl python script/run_pipeline.py --gpu 0 --config configs/config_smoke.yaml --num_render 1 --eval_every_steps 999999`

### B. 批量主实验调度（多 GPU 动态调度）

入口：`experiments/run_main.sh`

- `bash experiments/run_main.sh 0,1,2,3`

它会对 `CATEGORIES=(shoe plant dish bowl box)` 和 `METHODS=(geotrap naive_unlearning)` 生成 10 个 pipeline。

输出目录：

- 默认：`output/experiments_output/...`（便于把 `output/` 整体挪到系统盘/做软链接，避免写满数据盘）
- 可通过环境变量覆盖：`EXPERIMENTS_BASE=/some/path bash experiments/run_main.sh 0,1,2,3`

### C. 只跑防御训练

入口：`script/train_defense.py` 或 `./script/run_defense.sh`

- `./script/run_defense.sh 0 configs/config.yaml --num_epochs 20 --target_layers "conv.weight,unet.conv_in.weight"`

### D. 随机初始化 vs 预训练对比（辅助实验）

入口：`./script/test_compare.sh`（快速）/ `./script/run_compare.sh`（完整）

- `./script/test_compare.sh 0`
- `./script/run_compare.sh 0 configs/config.yaml 5`

---

## 4) 配置系统要点（configs/config.yaml + ConfigManager）

配置主文件：`configs/config.yaml`

关键字段：

- `model.size`: `small` / `big`（对应 `lib/LGM/core/options.py` 的 `config_defaults`）
- `model.resume`: 权重路径（支持 `tag:<tagname>`，会走 `tools/model_registry.py` 解析）
- `data.root`: 数据根目录（当前 config 里是绝对路径，跑之前通常需要改成你机器上的位置）
- `training.mode`: `full` / `lora`
- `defense.method`: `geotrap` / `naive_unlearning` / `none`
- `defense.trap_losses`: position/scale/opacity/rotation/color 的 static/dynamic 开关
- `defense.trap_combo` + `defense.num_target_layers`：
  - 如果没显式写 `defense.target_layers`，`ConfigManager` 会根据 `configs/trap_combo_layers.json` 自动填充 `defense.target_layers`

实现位置：`project_core/config.py` 的 `ConfigManager._resolve_defense_layers()`。

---

## 5) 数据加载（DataManager + dataset 格式约定）

### A. subset 概念

`DataManager.setup_dataloaders(..., subset=...)` 支持：

- `subset='source'`：用于蒸馏/能力保持（Source Data）
- `subset='target'`：用于攻击训练（Target Data，通常是“恶意类别/物体”）
- `subset='defense_target'`：用于防御训练的 target（可与攻击 target 不同；也可用 object_split 划出 defense 专用物体）
- `subset='all'`：source+target 合并（注意要求数据集类型一致）

实现位置：`data/data_manager.py` 的 `_resolve_subset_params()`。

### B. object_split：按“物体索引”划分 attack vs defense

配置位置：`configs/config.yaml` → `data.object_split`

注意点：

- `object_split` 里的索引是“目录名排序后的下标”。
- `DataManager` 会扫描渲染目录统计每个类别物体总数，然后：
  - 攻击 `subset='target'`：使用“非 defense 的剩余物体”作为池子，再随机抽 `attack_samples_per_category`
  - 防御 `subset='defense_target'`：直接用 `object_split` 指定的物体索引

实现位置：

- 扫描统计：`data/data_manager.py:_scan_category_counts()`
- attack 索引：`data/data_manager.py:_compute_attack_indices()`
- defense 索引：`data/data_manager.py:_compute_defense_indices()`

### C. 数据目录命名约定（影响类别解析）

`OmniObject3DDataset` 支持两种类别解析模式：

- Omni（默认）：`category_parse_mode='last_underscore'`，即 `cat = name.rsplit('_', 1)[0]`
- GSO：`category_parse_mode='first_underscore'`，即 `cat = name.split('_', 1)[0]`

对应配置/用法一般在 `data/dataset.py` 的 `create_dataloader()` 里决定（GSO/Omni/Objaverse）。

### D. 渲染数据格式（dataset.py 期待的文件）

Omni / GSO / ObjaverseRendered 都遵循类似结构（核心是 `render/images/*.png` + `render/transforms.json`）：

- `{object_dir}/render/images/{frame['file_path']}.png`
- `{object_dir}/render/transforms.json`

数据集中有一部分可能是 RGB 黑底图（无 alpha），`data/dataset.py` 会用黑底阈值推断 alpha。

### E. 相机与 rays 的“重活”在 dataset.py

`data/dataset.py` 做了很多关键预处理（下次排查数据/相机问题优先看这里）：

- 从 transforms.json 读 `transform_matrix` + `scale`，会先 `c2w[:3, :] /= scale`
- 做 Blender→OpenGL/COLMAP 坐标系修正 + 旋转正交化（SVD）
- 把相机半径归一化到 LGM 期望的 `target_radius=1.5`
- 基于“是否背对原点”做选择性翻转（避免 view0 被误翻）
- 用 `camera0` 构造归一化变换，把 camera0 变成标准位姿（对齐 LGM 原始 provider）
- 输入 `input_images` 是 **9 通道**（RGB + rays 等，来自 `core.utils.get_rays`）

---

## 6) Attack / Defense 逻辑（读代码时的导航图）

### A. Attack（training/finetuner.py）

入口函数：`training/finetuner.py:run_attack()`

流程（高层）：

1. `ModelManager.setup()` 加载 LGM（按 `training.mode` 决定是否 LoRA）
2. `AutoFineTuner` 做训练（支持 fp16/bf16/no；梯度累计；adamw/sgd）
3. `Evaluator` 周期性评估 + 渲染样本
4. 返回 step_history + source_metrics + target_metrics（可选返回 gaussians 用于诊断/距离）

补充：

- source 评估在 LoRA 模式下会临时 `disable_adapter_layers()`，测底座能力是否保持。

### B. Defense（training/defense_trainer.py）

入口函数：

- 类：`DefenseTrainer`
- 一键：`load_or_train_defense()`

防御方法：

- `geotrap`：
  - source 蒸馏（保持能力）
  - target trap loss（position/scale/opacity/rotation/color）
  - 多个 static trap 时会有“**两两乘法耦合**”让修复一个 trap 会放大其他 trap
  - 可选：
    - 参数加噪双前向（`defense.robustness.enabled`）
    - 特征空间梯度冲突（`defense.gradient_conflict.enabled`，通过 conv 输入 hook 计算不同 trap 梯度 cosine similarity）
    - 选择性微调（`defense.target_layers`）
- `naive_unlearning`：
  - 对 target loss 做“梯度上升”式遗忘 + source 蒸馏
- `none`：跳过防御

缓存/复用：

- `load_or_train_defense()` 会用 `tools/utils.py:compute_defense_hash()` 生成 tag，并在 `output/model_registry/{tag}/model.pth` 存在时直接命中缓存。

### C. Pipeline 编排（script/run_pipeline.py）

Phase 1 baseline attack 有缓存：

- 缓存目录：`output/baseline_cache/<baseline_hash>/`
- 元数据：`baseline_meta.json`
- 也会缓存 baseline gaussians：`baseline_gaussians.pt`（用于 Phase 3 计算“与 baseline 的 gaussian 距离”）

Phase 2 defense 的落盘策略（可控）：

- 默认（`--defense_cache_mode registry`）：
  - `tools/model_registry.py`：写 `output/model_registry/<tag>/model.pth` + `meta.json`（单个模型约 1.6GB）
  - Phase 3 通过 `model_resume_override=f"tag:{defense_tag}"` 加载防御模型
  - 磁盘空间注意：为避免重复保存占用空间，当前实现默认不再在 workspace 的 `phase2_defense/` 里额外保存 checkpoint/model 的拷贝
- 省磁盘（`--defense_cache_mode readonly` 或 `none`）：
  - `readonly`：registry 命中则读；未命中则训练但不写 registry
  - `none`：不读也不写 registry（每次都训练）
  - Phase 3 直接用内存中的 state_dict 覆盖权重继续跑攻击（无需写模型文件）

---

## 7) 常见坑 / 环境变量

- `XFORMERS_DISABLED=1`
  - 多处脚本在 import torch 前设置；原因是 xformers 的 attention 实现不支持某些二阶导（GeoTrap 的动态敏感度/梯度冲突可能需要 `create_graph=True`）。
- `HF_ENDPOINT=https://hf-mirror.com`
  - 主要用于国内网络下载 transformers/clip/objaverse 等资源。
- `CUDA_VISIBLE_DEVICES`
  - 多数脚本用 `--gpu` 参数，在 import torch 前写入环境变量。
- 数据路径：
  - `configs/config.yaml` 里 `data.root` 当前是机器相关的绝对路径；换环境运行时通常第一件事就是改它。
- AutoDL 双盘（系统盘 `/` + 数据盘 `/root/autodl-tmp`）的“落盘位置”：
  - 现象：本 repo 放在 `/root/autodl-tmp/...`（数据盘）时，`output/`、baseline cache、model registry、workspace 等默认都会写到数据盘，很容易把数据盘写满。
  - 一种不改代码的解决：把 repo 根目录下的 `output/` **整体挪到系统盘**并做软链接：
    - 系统盘目录：`/root/3d-defense-system/output`
    - repo 里：`output -> /root/3d-defense-system/output`（软链接）
  - 注意：AutoDL 的系统盘通常会在“重置系统”后清空；所以此法更适合“临时 cache/中间产物”，重要结果建议最终回拷到数据盘或文件存储。

---

## 8) 依赖（从代码读到的“最小集合”）

- 上一级目录有现成的 Python venv（便于直接复现环境）：
  - 路径：`../venvs/3d-defense/`
  - `pyvenv.cfg` 显示 Python 版本：`3.10.12`
  - 激活方式：`source ../venvs/3d-defense/bin/activate`
  - 注意（此环境是“拷贝”过来的常见症状）：`../venvs/3d-defense/bin/*` 里大量脚本的 shebang 可能指向旧机器路径（如 `/mnt/huangjiaxin/.../python3`），导致 `pip`/`accelerate` 等命令不可用。
  - 快速修复思路（无外网也能做）：
    - 安装系统依赖：`apt-get install -y python3-distutils python3-venv`
    - 批量重写 shebang：把 `#!/mnt/huangjiaxin/venvs/3d-defense/bin/python3` 替换为当前机器的 `../venvs/3d-defense/bin/python`
    - 然后 `../venvs/3d-defense/bin/pip check` 应该可用
- Python 依赖的大头在 `lib/LGM/requirements.txt`（torch、diffusers、accelerate、lpips、kiui、safetensors…）。
- LoRA：`models/model_manager.py` 里使用 `peft`（需要安装）。
- Objaverse 下载脚本：`script/download_objaverse.py` 使用 `objaverse`（需要安装，且会用 `~/.objaverse/` 缓存）。
- 可微渲染/高斯栅格化相关库：
  - `lib/diff-gaussian-rasterization/`（CUDA 扩展，通常需要编译）
  - `lib/nvdiffrast/`（CUDA 扩展，通常需要编译/安装）

---

## 9) 下次要“快速定位”的建议

如果你下次要做修改/排查，按下面顺序最快：

1. 想改实验流程/缓存/输出：先看 `script/run_pipeline.py`
2. 想改 attack 训练策略：看 `training/finetuner.py`（`AutoFineTuner` & `run_attack`）
3. 想改 defense/GeoTrap 机制：看 `training/defense_trainer.py` + `methods/trap_losses.py`
4. 想改数据/相机/视角：看 `data/dataset.py`（这里的相机变换最复杂）
5. 想改选层策略：看 `configs/trap_combo_layers.json` + `project_core/config.py:resolve_target_layers`
6. 想复用/管理防御模型：看 `tools/model_registry.py`
