# 3D-Defense 项目说明（给后续 Agent / Codex）

Last updated: 2026-03-20

这份文件只保留相对稳定、不容易过时的导航信息。

不要把它当成实验口径、默认配置或脚本覆盖范围的权威来源。凡是涉及下面这些问题，都必须以当前代码和脚本为准，不能依赖这里的文字总结：

- 某个实验脚本当前到底覆盖哪些 dataset / category / method
- `configs/config.yaml` 当前默认值是什么
- `script/run_pipeline.py` 当前有哪些 CLI 参数和默认行为
- attack / defense / evaluation 的当前评估逻辑、缓存逻辑、导出逻辑
- 汇总脚本现在打印哪些字段、如何解释 `metrics.json`

如果你需要精确结论，请直接读对应文件。

## 1. 项目概览

这是一个围绕 LGM 的三阶段实验框架：

1. `Phase 1` Baseline Attack：先在 target 数据上做攻击微调。
2. `Phase 2` Defense：对模型做防御训练。
3. `Phase 3` Post-Defense Attack：加载防御后的模型再攻击，比较恢复情况。

主入口不是 `training/` 里的单文件脚本，而是由实验脚本和 pipeline 串起来。

## 2. 主要目录

- `configs/`
  - 主配置通常在 `configs/config.yaml`
- `experiments/`
  - 批量实验和导出脚本
  - 多 GPU 调度逻辑在 `experiments/lib/gpu_scheduler.sh`
- `script/`
  - `script/run_pipeline.py` 是端到端主入口
- `training/`
  - `training/finetuner.py` 负责 attack 阶段
  - `training/defense_trainer.py` 负责 defense 阶段
- `evaluation/`
  - `evaluation/evaluator.py` 负责评估、渲染和部分导出
- `data/`
  - `data/data_manager.py` 负责 loader 组织
  - `data/dataset.py` 负责数据解析与相机 / rays 预处理
- `models/`
  - `models/model_manager.py` 负责模型加载和 LoRA 注入
- `tools/`
  - 放缓存辅助、汇总、绘图等工具
- `lib/`
  - 上游 LGM 和渲染依赖

## 3. 读代码的优先顺序

如果要理解某条实验命令或某个结果，优先按下面顺序看：

1. `experiments/*.sh`
2. `script/run_pipeline.py`
3. `training/finetuner.py`
4. `training/defense_trainer.py`
5. `evaluation/evaluator.py`
6. `configs/config.yaml`

如果问题和数据集或样本组织有关，再补看：

1. `data/data_manager.py`
2. `data/dataset.py`

## 4. 如何确认“当前真实行为”

以下规则比这份文档更可靠：

- 实验覆盖范围：看对应 `experiments/*.sh` 里实际传给 `script/run_pipeline.py` 的参数。
- 配置默认值：看当前 `configs/config.yaml`。
- CLI 覆盖关系：看 `script/run_pipeline.py` 的参数解析与配置覆写逻辑。
- attack / defense 的职责边界：分别看 `training/finetuner.py` 和 `training/defense_trainer.py`。
- 评估与导出：看 `evaluation/evaluator.py` 和相关 `script/*.py` / `experiments/*.sh`。
- 某次运行到底产出了什么：看该次运行目录下的 `metrics.json`、日志、导出目录和图像文件。

一句话说，文档只能帮你定位文件，不能替代读代码。

## 5. 输出与缓存

常见输出通常位于仓库内的 `output/` 下，批量实验也常写到 `output/experiments_output/`。

缓存与结果目录的名字、结构和命中规则可能变化，因此：

- 不要假设某个缓存一定会命中或一定不会命中。
- 怀疑结果不对时，先检查这次运行是否真的重跑了对应 phase。
- 解释实验结果时，以该次运行目录中的实际文件为准。

## 6. 数据与磁盘

仓库位于：

- `/root/autodl-tmp/3d-defense-migration/3d-defense`

大输出和缓存尽量继续写在仓库相对路径或 `/root/autodl-tmp/` 下，不要默认落到系统盘。

## 7. 维护约定

以后如果要更新 `AGENTS.md`，只写不容易过时的内容。

不要再在这里维护下面这类高漂移信息：

- “当前默认”是什么
- “当前脚本固定覆盖”哪些类别或数据集
- “当前口径”如何解释某组实验
- 某个脚本“现在实际对应”哪种命名

这类信息应该写在对应脚本头部注释里，或者直接通过读代码确认。
