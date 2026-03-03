# AGENTS.md (auto-loaded assistant context)

This file is intentionally short and stable so future Codex sessions can pick up the project goals even if chat context is truncated.

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

## Data curation (allowed + encouraged)

- These public 3D datasets were **not designed for fine-tuning defense / immunization** benchmarks.
- It is acceptable to **manually curate** a *representative* target subset, as long as the selection rule is stated clearly and splits are fixed.
- Prefer defining the target as a **geometry-consistent concept cluster** (not an entire semantic class), using **Gaussian-attribute descriptors** when possible (e.g., statistics of position/scale/rotation/opacity/color/SH).
- **In-dataset generalization**: on OmniObject3D, defense objects and attack objects must be **strictly disjoint**.
- When results look unstable / defense seems impossible, first suspect **data issues** (object diversity too large, inconsistent cameras/backgrounds, bad alpha, corrupt renders, etc.).
- It is explicitly allowed to solve blockers via **data-side processing** (filtering, clustering, cleaning, re-splitting), as long as the procedure is documented and does not leak attack objects into defense training.

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
- Experiments should be run in a **realistic setting**. Attack/defense step counts do **not** need to be rigid (adjust until it works), but avoid toy-only validations.
- It is OK to do **quick sanity checks** for iteration speed; however, before concluding "works", always run **real experiments** (full pipeline / realistic budgets) to confirm.
- When stuck, first consider whether the **method/assumptions are wrong** (or the evaluation/data setup is mismatched) rather than endlessly **tuning hyperparameters** on a single path. Be ready to pivot and validate alternative approaches with small, decisive experiments.
