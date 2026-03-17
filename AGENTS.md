# 3D-Defense 项目说明（给后续 Agent / Codex）

Last updated: 2026-03-13

这个文件记录当前仓库的有效项目结构、实验入口和需要遵守的结果口径，目的是让后续 session 不用重新通读一遍代码。

## 1. 项目一句话概述

这是一个围绕 LGM 的三阶段实验框架：

1. `Phase 1` Baseline Attack：先在 target 数据上做攻击微调。
2. `Phase 2` Defense：做 GeoTrap 或 Naive Unlearning 防御。
3. `Phase 3` Post-Defense Attack：加载防御后的模型再攻击，比较恢复情况。

主入口不是 `training/` 下的单文件脚本，而是：

- `experiments/*.sh` 负责批量调度和汇总
- `script/run_pipeline.py` 负责串起完整 pipeline
- `training/finetuner.py` 负责 attack
- `training/defense_trainer.py` 负责 defense
- `evaluation/evaluator.py` 负责评估和渲染

## 2. 常用目录

- `configs/`
  - 主配置在 `configs/config.yaml`
  - `configs/trap_combo_layers.json` 用于 trap 组合自动选层
- `experiments/`
  - 批量实验入口脚本
  - 多 GPU 调度逻辑在 `experiments/lib/gpu_scheduler.sh`
- `script/`
  - `script/run_pipeline.py` 是端到端主入口
  - `script/print_attack_step_report.py` 用于从 `metrics.json` 打印 4 段 step 汇总
- `training/`
  - `training/finetuner.py`：attack 阶段
  - `training/defense_trainer.py`：defense 阶段
- `evaluation/`
  - `evaluation/evaluator.py`：PSNR / LPIPS / 渲染 / 高斯诊断
- `data/`
  - `data/data_manager.py`：source / target / defense_target dataloader 组织
  - `data/dataset.py`：Omni / GSO 数据格式解析、相机和 rays 预处理
- `models/`
  - `models/model_manager.py`：LGM 权重加载、LoRA 注入
- `tools/`
  - `tools/utils.py`：baseline/defense 缓存辅助
  - `tools/step_reporting.py`：按 attack step 生成分段汇报
  - `tools/plotting.py` / `tools/gaussian_plotting.py`：可视化
- `lib/`
  - 上游 LGM 和渲染依赖

## 3. 当前有效实验入口

### 3.1 In-domain 主实验

- `experiments/run_main_omni.sh`

含义：

- OmniObject3D in-domain
- 5 个类别：`shoe plant dish bowl box`
- 2 个 defense method：`geotrap`、`naive_unlearning`

### 3.2 跨数据集泛化：Omni -> GSO

- `experiments/run_cross_dataset_generalization.sh`

当前脚本实际覆盖的是：

- `attack_target_dataset=gso`
- `defense_target_dataset=omni`

项目内部和已有命名上把它视为 `omni -> gso` 方向。

### 3.3 跨数据集泛化：GSO -> Omni

- `experiments/run_cross_dataset_generalization_gso_to_omni.sh`

当前脚本实际覆盖的是：

- `attack_target_dataset=omni`
- `defense_target_dataset=gso`

项目内部和已有命名上把它视为 `gso -> omni` 方向。

### 3.4 其他常见实验

- `experiments/run_ablation_attack.sh`
- `experiments/run_ablation_attack_naive_unlearning.sh`
- `experiments/run_ablation_attack_and_traps.sh`
- `experiments/run_ablation_coupling.sh`
- `experiments/run_ablation_gaussian_attributes.sh`
- `experiments/run_ablation_defense_num_categories.sh`
- `experiments/run_ablation_trap.sh`
- `experiments/run_compare_trap_aggregation.sh`
- `experiments/run_compare_random_vs_pretrained_5cats.sh`
- `experiments/run_standard_defense_efficiency.sh`
- `experiments/run_main_all_three.sh`
- `experiments/run_all_experiments.sh`
- `experiments/run_overnight_all.sh`

其中：

- `experiments/run_main_all_three.sh`
  - 一次串行/并行覆盖 3 组主实验：`omni -> omni`、`omni -> gso`、`gso -> omni`
  - 固定 `attack_steps=400`、`defense_steps=100`
  - `omni -> omni` 口径：`data.use_object_split=false`，`defense.target.split_ratio=0.4`
  - 跨数据集口径：defense target 使用全部数据
  - 额外附带 `bowl` 的攻击消融任务
  - 额外附带防御类别数消融：`k2=bowl,shoe`、`k3=shoe,dish,bowl`，并沿用 `omni -> omni` 主实验口径
  - 额外附带 `w/o input noise`：`shoe`、`plant`，覆盖 `omni -> omni`、`omni -> gso`、`gso -> omni`，固定 `geotrap`
  - 额外附带 `random init vs pretrained`（5 类别，`omni -> omni`，attack-only，`attack_steps=400`）
  - 额外附带 `shoe/plant` 的单 trap 消融：仅 `omni -> omni`，固定 `geotrap`，补 `position/scale/opacity/rotation/color`
  - 额外附带 `shoe/plant` 的多 trap 代表链：仅 `omni -> omni`，固定 `geotrap`，只保留 `scale+opacity -> position+scale+opacity -> position+scale+opacity+color` 三组 `2/3/4-trap` 结果，并放在整轮调度最后

- `experiments/run_ablation_attack.sh`
  - 当前固定测试类别 `bowl`
  - 当前攻击消融覆盖：`default`、`lora8`、`lora32`、`full`
  - 当前 optimizer/lr 覆盖：`adamw 3e-6`、`adamw 3e-4`、`sgd 3e-5`、`sgd 3e-4`、`sgd 3e-3`
  - 当前 attack steps 覆盖：`400(default)`、`800`、`1600`
  - 当前固定 `defense_steps=100`

- `experiments/run_ablation_attack_naive_unlearning.sh`
  - 从 `experiments/run_ablation_attack_and_traps.sh` 摘出的 attack-only 子集入口
  - 当前覆盖项目命名里的 `gso -> omni` 场景下，`shoe/plant` 的 22 个攻击消融任务
  - 当前固定 `defense_method=naive_unlearning`
  - 当前固定 `attack_steps=400`、`defense_steps=100`、`eval_every_steps=-1`
  - 当前脚本里的 `gso -> omni` 实际对应：`attack_target_dataset=omni`、`defense_target_dataset=gso`，且 defense target 使用全部数据

- `experiments/run_ablation_attack_and_traps.sh`
  - 从 `experiments/run_main_all_three.sh` 摘出的子集入口
  - 当前覆盖项目命名里的 `gso -> omni` 场景下，`shoe/plant` 的攻击消融 + single-trap + multi-trap 代表链
  - 当前固定 `attack_steps=400`、`defense_steps=100`、`eval_every_steps=-1`
  - 当前脚本里的 `gso -> omni` 实际对应：`attack_target_dataset=omni`、`defense_target_dataset=gso`，且 defense target 使用全部数据

- `experiments/run_compare_random_vs_pretrained_5cats.sh`
  - 当前固定比较 `random init` vs `pretrained`
  - 当前固定是 `attack-only`，不进入 defense
  - 当前固定 `attack_target_dataset=omni`
  - 当前覆盖 5 个类别：`shoe plant dish bowl box`
  - 当前默认 `attack_steps=400`

- `experiments/run_compare_trap_aggregation.sh`
  - 默认是 `omni -> omni`
  - 默认类别是 `shoe`
  - 默认是 5 种 trap 全开：`position,scale,rotation,opacity,color`
  - 默认比较 `mean` vs `bottleneck_logsumexp`
  - 默认 `defense_steps=50`

- `experiments/run_standard_defense_efficiency.sh`
  - 当前固定测量项目命名里的 `gso -> omni` setting 下的 defense efficiency
  - 当前实际口径：`attack_target_dataset=omni`、`defense_target_dataset=gso`
  - 当前口径：`data.use_object_split=true`，defense target 使用全部数据
  - 当前固定 `attack_steps=400`、`defense_steps=100`
  - 当前默认是 2 个任务，不是 `5 × 2`
  - 当前两条任务都用同一组类别：`shoe,plant,dish,bowl,box`
  - 当前任务 1：`naive_unlearning`
  - 当前任务 2：`geotrap`，并显式开启 5 个 trap：`position,scale,opacity,rotation,color`
  - 当前默认传 `--measure_efficiency`
  - 当前默认 `DEFENSE_CACHE_MODE=none`，确保不会命中缓存而跳过真实 defense 训练
  - 当前默认 `SKIP_BASELINE=1`、`SKIP_POSTDEFENSE_ATTACK=1`，因为效率统计只测 `Phase 2: Defense`
  - 当前汇总读取 `defense_efficiency.json`，打印 `training time / avg step time / target avg batch time / peak GPU memory`

读这些脚本时，优先看：

- 它覆盖了哪些 dataset / category / method
- 它传给 `script/run_pipeline.py` 的 CLI 参数是什么
- 它最后是打印 `metrics.json` 最终值，还是打印 attack step 汇报

## 4. 配置与覆盖关系

默认配置来源：

- `configs/config.yaml`

实验脚本通常只覆盖以下几类内容：

- 数据集：如 `--attack_target_dataset`、`--defense_target_dataset`
- 类别：如 `--categories`
- 防御方法：如 `--defense_method`
- 运行时参数：如 `--gpu`、`--output_dir`、`--eval_every_steps`

重要约定：

- 如果某个 CLI 参数没有传，就不会覆盖 `configs/config.yaml` 里的原设置。
- shell 脚本里的环境变量也大多是 `${VAR:-default}` 形式；只有你主动传入，才会覆盖默认值。
- 当前 `configs/config.yaml` 默认仍是 `omni -> omni` 的设置，批量脚本再按需要覆写。
- 当前默认 `defense.trap_aggregation.method=mean`；只有显式传 `--trap_aggregation_method` 或使用专门的对比/消融脚本时，才会改成别的聚合方式。
- 当前 `defense.lambda_trap` 只用于 `geotrap`；`naive_unlearning` 的 target 侧权重单独使用 `defense.lambda_unlearn`，不再回退到 `lambda_trap`。

## 5. Attack / Defense 的职责边界

### 5.1 `eval_every_steps` 只控制 Attack 阶段

`eval_every_steps` 只影响：

- `Phase 1` baseline attack
- `Phase 3` post-defense attack

它不控制 defense 训练内部的验证频率。

### 5.2 Defense 阶段验证是单独逻辑

Defense 的验证频率在 `training/defense_trainer.py` 内部控制，不跟随 `eval_every_steps`。

因此不要把 attack 的 step 评估逻辑和 defense validation 混为一谈。

## 6. 当前 step 汇报口径

这是最近改过的地方，后续不要再按“只看最终一步”理解。

### 6.1 默认行为

`script/run_pipeline.py` 当前默认：

- `--eval_every_steps=-1`

含义：

- 不按固定间隔做 attack 评估
- 只在 pipeline 自动算出的 4 个 checkpoint 上评估 attack
- 这 4 次 checkpoint 评估现在是对完整 target loader 做全量评估，不再使用训练区间均值

### 6.2 4 个 checkpoint 是自动算的

不是固定写死的某一组数字。

而是根据 attack 总 step 自动均分出 4 个位置，例如：

- `400 -> [100, 200, 300, 400]`
- `500 -> [125, 250, 375, 500]`
- `503 -> [126, 252, 378, 503]`

### 6.3 输出位置

每个 pipeline 现在会额外产出：

- `attack_step_report.txt`

并且在 `metrics.json` 中写入：

- `analysis.attack_step_report`

批量脚本的 `summary.txt` 也已经优先打印这个 4 段 attack step 汇报。

### 6.3.1 Source 口径

- `source` 现在只在每个 attack phase 开始前评估一次
- 不再在每个 attack checkpoint 重复评估 source
- 因此 `baseline_source` / `postdefense_source` 是最终保留的 source 指标口径
- `attack_step_report` 中如果看不到 `source_psnr/source_lpips`，这是预期行为

### 6.4 如果显式传正数

如果显式传：

- `--eval_every_steps=N` 且 `N > 0`

则 attack 会按固定步长评估，保留旧行为。

## 7. 输出与缓存

- 批量实验默认输出到 `output/experiments_output/<exp_name>_<timestamp>/`
- 单个任务目录通常会有：
  - `metrics.json`
  - `attack_step_report.txt`
  - `pipeline_result.png`
  - 若干 phase 子目录和日志

缓存相关：

- baseline attack 可能命中 `output/baseline_cache/`
- defense 可能命中模型注册表缓存

因此修改口径后，如果怀疑拿到了旧缓存，需要确认：

- 这次是否真的重新跑了对应 phase
- 读取的 `metrics.json` 是否包含新的 `analysis.attack_step_report`

## 8. 看代码时的优先顺序

如果要快速理解一条实验命令，按这个顺序看：

1. `experiments/*.sh`
2. `script/run_pipeline.py`
3. `training/finetuner.py`
4. `training/defense_trainer.py`
5. `evaluation/evaluator.py`
6. `configs/config.yaml`

如果是数据集问题，再补看：

1. `data/data_manager.py`
2. `data/dataset.py`

## 9. 数据与磁盘注意事项

仓库位于数据盘：

- `/root/autodl-tmp/3d-defense-migration/3d-defense`

大体原则：

- 大输出、大缓存不要写系统盘
- 实验结果优先留在仓库相对路径或 `/root/autodl-tmp/` 下

已有批量脚本默认把大部分输出写到仓库内的 `output/`，这通常是安全的。

## 10. 文档维护约定

如果后续又改了实验入口、评估口径或汇总格式，优先同步更新：

- `AGENTS.md`
- 相关 `experiments/*.sh` 头部注释
- 需要的话再更新 `汇报.md`、`codex.md`

`AGENTS_old.md` 现在保留作历史备忘，不应再作为最新口径来源。
