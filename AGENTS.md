# 3D-Defense 实验跑法（给 Codex/Agent）

## 最重要：严禁写系统盘（否则 SSH 可能断连）

系统盘 `/` 空间很小，写满会导致我无法连接 SSH。**任何大文件/实验输出/缓存/临时文件都不允许写到系统盘**（包括但不限于 `/tmp`、`/root`、`~/.cache` 等路径）。

要求：

- 所有输出目录必须在数据盘挂载点：`/root/autodl-tmp/`（本仓库位于该路径下，默认相对路径输出一般是安全的；但缓存/临时目录仍需注意）
- 跑实验前建议显式设置缓存与临时目录到数据盘（示例）：
  - `export XDG_CACHE_HOME=/root/autodl-tmp/.cache`
  - `export TORCH_HOME=/root/autodl-tmp/.cache/torch`
  - `export HF_HOME=/root/autodl-tmp/.cache/huggingface`
  - `export WANDB_DIR=/root/autodl-tmp/.cache/wandb`
  - `export TMPDIR=/root/autodl-tmp/tmp`
- 跑前/跑中用 `df -h / /root/autodl-tmp` 监控磁盘，确保系统盘使用率不再上涨

## 核心任务（必须跑）

本项目当前需要跑的核心实验任务 **仅限以下 4 个脚本**（不要自行扩展到其它脚本/命令）：

- `experiments/run_ablation_attack.sh`
- `experiments/run_ablation_coupling.sh`
- `experiments/run_ablation_defense_num_categories.sh`
- `experiments/run_ablation_gaussian_attributes.sh`

> 注意：脚本里被注释掉的 `TASKS+=...` 表示**已经跑过**，不要再启用（不要去掉注释）。

## 指标口径（两类实验，两套评估目标）

### A) 稳健性（看最终结果）

适用实验：

- attack 相关实验（`experiments/run_ablation_attack.sh`）
- 防御多类别（`experiments/run_ablation_defense_num_categories.sh`）

目标：

- 证明方法稳健性，因此按 **默认配置** 评估最终结果：
  - `defense_steps = 200`
  - `attack_steps = 400`
- 读 `metrics.json` 里的最终指标（例如 `postdefense_target` / `postdefense_source`）。

原则：

- 不要为这类实验改成 “达标步数” 指标；不要缩短 defense 到 50 step。
- 不要在脚本/命令里覆盖 `--attack_steps/--defense_steps`；需要严格使用默认 `attack_steps=400, defense_steps=200`。

### B) 精细指标（看“多久恢复到 baseline 攻击效果”）

适用实验：

- 互锁机制（`experiments/run_ablation_coupling.sh`）
- 高斯属性（`experiments/run_ablation_gaussian_attributes.sh`）

背景：

- defense 成功时很多 case 会出现 **opacity/透明度模式崩溃**，直接比最终 LPIPS/PSNR 往往不可比。

目标（核心指标）：

- **Defense 只训练 50 step**，然后看 post-defense attack 需要多少 step 才能达到 baseline attack 的最终效果阈值。
- 以 `script/run_pipeline.py` 的分析输出为准：
  - `analysis.postdefense_attack_steps_to_baseline_effect`

关键要求：

- 需要 baseline phase 的 step-history 才能算“达标步数”，因此收集该指标时 **不要使用 `--skip_baseline`**。
- `eval_every_steps` 越小，“达标步数”分辨率越高（但更慢）；默认 10，可按需调小。

## 输出位置与产物

- 所有脚本默认输出到 `output/experiments_output/<exp_name>_<timestamp>/`
- 单个任务目录下会写 `metrics.json`，部分脚本会生成 `summary.txt`

## 汇报

- 每次明确实验口径/任务清单后，把“任务汇报”同步写到仓库根目录的 `汇报.md`（便于团队对齐与复现实验）。

## Git 使用约定

- 遵循 “多用 git” 原则：任何非临时性的改动（脚本/配置/文档）都应纳入版本控制。
- 开工/收尾各检查一次：`git status` + `git diff`，避免遗漏文件或把无关改动混入。
- 以小步提交为主：每完成一个独立的文档/脚本任务，就做一次 commit（信息写清楚做了什么、为什么）。
- 不要把“跑实验产生的大量输出”提交进仓库（如 `output/` 下的结果目录），只提交脚本/配置/汇报口径/必要的可复现产物。

## 附加任务（规划中）

### 1) Trap 消融补全：单 trap / trap 组合

背景：

- 5 个 trap 全开（position/scale/opacity/rotation/color）不一定最优。
- 仅做 w/o（逐个关闭）不足以解释每个 trap 的独立贡献以及 trap 之间的互补/冲突关系。

目标：

- 补充实验：**单 trap**（只开 1 个）以及 **trap 组合**（不限定两两组合；需要覆盖 ≥3 个 trap 的组合，才能体现/检验 bottleneck/logsumexp 聚合在“多 trap 同开”时的作用）。
- 产出数据后分析：哪个 trap/组合更稳健、哪个更容易诱发 opacity 崩溃、以及对 retention（source）伤害的权衡。

指标口径建议：

- 该类实验建议按“精细指标”口径（见上文 B）：
  - `defense_steps = 50`
  - 关注 `analysis.postdefense_attack_steps_to_baseline_effect`
  - 计算该指标时不要使用 `--skip_baseline`

### 2) 训练效率对比：GeoTrap vs Naive Unlearning（无需渲染）

目标：

- 证明 **GeoTrap 防御训练更高效**（核心优势：防御阶段无需渲染），对比 `naive_unlearning`（target 渲染 loss 梯度上升）。

建议测量/汇报的论文级指标（同硬件/同配置对齐）：

- **防御训练耗时**：总 wall-clock、平均 `sec / optimizer_step`（或 `optimizer_steps / sec`），以及总 GPU-hours（可按 defense_steps=50/200 各报一组）。
- **峰值显存**：`max_memory_allocated / reserved`（GB）。
- **渲染开销证据**：防御阶段是否发生渲染（GeoTrap=0；Naive=需要渲染），可补充 profiler 时间占比（可选）。

注意：

- 该效率对比应尽量 **只统计 defense 阶段**（避免 baseline/post-defense attack 的渲染开销掩盖差异）。

### 3) 论文级分布图：Defense 前后 / Defense→Attack 前后变化

目标：

- 做一个**直观的分布图**展示模型在以下阶段的变化（用于论文展示）：
  - Defense 前 vs Defense 后
  - 或 Defense 前 vs Defense 后再经过 Attack（Post-Defense Attack）后的变化

推荐可视化对象（优先能反映“透明度模式崩溃 / trap 形态变化”）：

- **Opacity 分布**（如 opacity 值直方图/KDE，或 logit(opacity) 的分布）  
  重点展示：大量值靠近 0 的塌缩（collapse）以及 defense/attack 后分布形态是否恢复/改变。
- 可补充：`opacity_lt_01 / opacity_lt_001`、`render_white_ratio`、`pos_spread`、`scale_mean/scale_tiny`、`rgb_white_ratio` 等诊断指标的分布（辅助解释“不可比”的 case）。

实现约束（规划）：

- 需要固定同一批样本（同 seed/同 loader 子集）以保证 “before/after” 可比。
- 如果只依赖 `metrics.json` 里的均值指标不够“分布感”，需要额外导出样本级数据：
  - 方案 A：使用 `Evaluator.diagnose_gaussians(return_gaussians=True)` 收集同一批样本的 Gaussian，再做分布统计/绘图。
  - 方案 B：在 pipeline 输出中保存 per-sample 的 opacity/scale 等统计（用于画 violin/箱线/直方图）。

### 4) 额外复现实验：换一个类别再跑一遍（排除 main/跨类别/random）

触发条件：

- 如果以上核心任务 + 附加任务都做完仍有时间，再做此项。

目标：

- 额外选择一个 target 类别（不同于当前默认 `bowl`），把“核心实验脚本”再跑一遍，用于验证结论不只对单一类别成立。

范围（只跑核心脚本，不扩展）：

- 仍然只跑：
  - `experiments/run_ablation_attack.sh`
  - `experiments/run_ablation_coupling.sh`
  - `experiments/run_ablation_defense_num_categories.sh`
  - `experiments/run_ablation_gaussian_attributes.sh`
- 明确**不跑**：
  - `experiments/run_main.sh`
  - `experiments/run_cross_dataset_generalization*.sh`（跨数据集/跨类别）
  - `experiments/run_compare_random_vs_pretrained_5cats.sh`（random vs pretrained）

执行要点：

- 选择的类别需在数据/划分中可用（例如 `shoe/plant/dish/box` 等），并记录到 `汇报.md`。
- 若脚本内部写死了类别（如 `TEST_CAT=...`），以“改变量/CLI categories 覆盖”的方式切换；不要启用已注释任务。
- 统一使用 **GPU 0** 运行（单卡）：调用脚本时传 `0`，或确保环境变量/参数指向 GPU0。
