# AGENTS.md (auto-loaded assistant context)

This file is intentionally short and stable so future Codex sessions can pick up the project goals even if chat context is truncated.

## Core task (summary)

- Build a **working** first-order fine-tuning defense for the **LGM backbone only** (multi-view → 3D Gaussian vectors), explicitly leveraging **Gaussian physical attributes** (position/scale/rotation/opacity/color/SH).
- Goal is to **increase post-defense recovery cost** under fine-tuning attacks (no “impossible to recover” claim; novelty is not required).
- Run **realistic experiments** (not toy-only). Defense/attack step counts can be adjusted as needed, but always verify “works” via real runs on the **single available GPU**.
- Generalization targets:
  - **In-dataset**: OmniObject3D (defense vs attack objects must be **strictly disjoint**).
  - **Cross-dataset**: evaluate attacks on GSO after defending on OmniObject3D.
  - **Retain/source**: Objaverse.
- When stuck, prioritize **measure → conclude**: suspect method/data/eval mismatch early; if evidence indicates data/eval issues are the root cause, it is acceptable to **curate/process data** (with documented rules) and to **simplify/delete legacy code** to keep the pipeline minimal, as long as it works.

## Project: LGM fine-tuning defense (paper)

- **Domain**: 3D generation / feed-forward 3D Gaussian (3DGS) reconstruction models in the LGM family.
- **Scope**: *Only* the **LGM backbone** (multi-view → Gaussian parameters). Do **not** include upstream multi-view diffusion.
- **Key property to exploit**: LGM outputs **Gaussian vectors with physical meaning** (e.g., position/scale/rotation/opacity/color/SH). Prefer defenses/analyses that leverage this structure.
- **Method constraint**: Prefer **first-order** defenses/training (gradient-based, no explicit Hessian / no true bilevel second-order backprop).
- **Method selection**: Prefer **common first-order “model immunization / unlearning / fine-tuning defense” baselines** adapted to Gaussian attributes. Novelty is not the priority; **it must work** and the mechanism should be grounded in Gaussian properties.

## Research objective ("model immunization")

- Goal is **to increase post-defense recovery cost** under fine-tuning attacks (not to make recovery impossible in an absolute sense).
- Prefer claims framed as **cost multipliers** / **fixed-budget degradation** with clear threat model assumptions (e.g., optimizer, steps, LoRA/full).

## Datasets / generalization setup

- **Source / retain**: Objaverse (used to preserve general capability).
- **Target**:
  - OmniObject3D: used for **defense training** and **attack evaluation** (supports *in-dataset* generalization tests via disjoint object splits).
  - GSO: used for **attack evaluation only** (tests *cross-dataset* generalization after defending on OmniObject3D).

## Data curation (only when it is the root cause)

- These public 3D datasets were **not designed for fine-tuning defense / immunization** benchmarks.
- It is acceptable to **manually curate** a *representative* target subset, as long as the selection rule is stated clearly and splits are fixed.
- Prefer defining the target as a **geometry-consistent concept cluster** (not an entire semantic class), using **Gaussian-attribute descriptors** when possible (e.g., statistics of position/scale/rotation/opacity/color/SH).
- **In-dataset generalization**: on OmniObject3D, defense objects and attack objects must be **strictly disjoint**.
- When results look unstable / defense seems impossible, first suspect **data issues** (object diversity too large, inconsistent cameras/backgrounds, bad alpha, corrupt renders, etc.).
- If evidence indicates the root cause is on the data side, it is acceptable to solve blockers via **data-side processing** (filtering, clustering, cleaning, re-splitting), as long as the procedure is documented and does not leak attack objects into defense training.

## Current working success targets (draft; update when finalized)

Baseline (no defense):
- **Attack budget**: 150 steps
- **Target metrics after attack**: PSNR ≈ 24, LPIPS ≈ 0.13

Defense is considered "good enough" if, under the **same 150-step attack budget**, we can roughly reach:
- **Target PSNR ≤ 20–21** and/or **Target LPIPS ≥ 0.18–0.20**
- While keeping **retain/source performance drop small** (avoid "just ruin the model" baselines)

## Writing/communication preferences

- Default language: **Chinese** (unless the user asks otherwise).
- When uncertain, ask for the missing experimental definition (attack budget, success threshold, retain constraints) before over-committing to a specific claim.

## Notes on repository docs + experimental rigor

- The repo contains many documents/notes that may be **outdated**. Treat them as hints only; when there is a conflict, prioritize **code behavior** and the user's latest instructions.
- The repo also contains many legacy mechanisms/scripts. Do **not** assume something works just because it exists or was used before.
- It is allowed to **modify, simplify, or delete obsolete code** to keep the project maintainable, as long as the resulting pipeline **works** for the intended experiments.
- Use **git** to manage changes (inspect with `git status/diff`, keep patches reviewable). Create commits/branches when explicitly requested.
- **Disk constraint**: store datasets/renders/checkpoints/outputs on the data disk only (`/root/autodl-tmp`). Avoid generating or downloading large files under the system disk (`/`); keep system disk from filling up (monitor with `df -h`, delete non-essential caches if needed, and redirect tool caches/logs to `/root/autodl-tmp` when possible).
- If disk space becomes an issue during experiments, it is OK to **delete non-essential caches** (e.g., intermediate renders, old experiment outputs, temporary checkpoints) while keeping raw datasets and the latest results needed for the paper.
- A single GPU is available for experiments; prefer running training/eval on **GPU** (e.g., set `--gpu 0` / `CUDA_VISIBLE_DEVICES=0`) and avoid accidental CPU-only runs.
- Experiments should be run in a **realistic setting**. Attack/defense step counts do **not** need to be rigid (adjust until it works), but avoid toy-only validations.
- It is OK to do **quick sanity checks** for iteration speed; however, before concluding "works", always run **real experiments** (full pipeline / realistic budgets) to confirm.
- When stuck, first consider whether the **method/assumptions are wrong** (or the evaluation/data setup is mismatched) rather than endlessly **tuning hyperparameters** on a single path. Be ready to pivot and validate alternative approaches with small, decisive experiments.
- When problems arise, it is encouraged to run **manual, targeted experiments** to collect evidence (metrics, plots, qualitative renders) and make decisions. Avoid guessing; prefer **measure → conclude**.
