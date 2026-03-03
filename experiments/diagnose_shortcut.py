#!/usr/bin/env python3
"""
Diagnose whether "shortcut" updates still exist when we freeze the Gaussian head.

We approximate shortcut-ness by looking at gradient concentration:
  - For each trap loss (and their aggregated loss), compute grad energy per parameter tensor.
  - Report top-1 / top-k shares and the dominating parameter names.

This is FIRST-ORDER only: a single forward_gaussians() + backward.
No rendering, no 2nd order.

Usage:
  ../venvs/3d-defense/bin/python experiments/diagnose_shortcut.py \
    --gpu 0 --config configs/config_immunize_freezehead_A.yaml --subset defense_target --num_batches 1
"""

import os
import argparse


def _set_env(gpu: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["XFORMERS_DISABLED"] = "1"
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)


def _freeze_head_conv(model):
    """Freeze conv.* (Gaussian head) params."""
    for name, p in model.named_parameters():
        if name.startswith("conv."):
            p.requires_grad = False


def _active_trap_fns(defense_cfg):
    from methods.trap_losses import (
        PositionCollapseLoss,
        ScaleAnisotropyLoss,
        OpacityCollapseLoss,
        RotationAnisotropyLoss,
        ColorCollapseLoss,
    )

    trap_cfg = defense_cfg.get("trap_losses", {}) or {}

    fns = {}
    if trap_cfg.get("position", {}).get("static", False):
        fns["position_static"] = PositionCollapseLoss()
    if trap_cfg.get("scale", {}).get("static", False):
        fns["scale_static"] = ScaleAnisotropyLoss()
    if trap_cfg.get("opacity", {}).get("static", False):
        ocfg = trap_cfg.get("opacity", {}) or {}
        fns["opacity_static"] = OpacityCollapseLoss(
            topk_frac=ocfg.get("topk_frac"),
            topk_k=ocfg.get("topk_k"),
        )
    if trap_cfg.get("rotation", {}).get("static", False):
        fns["rotation_static"] = RotationAnisotropyLoss()
    if trap_cfg.get("color", {}).get("static", False):
        fns["color_static"] = ColorCollapseLoss()

    return fns


def _aggregate_losses(loss_list, method: str, tau: float):
    import torch

    if not loss_list:
        return torch.zeros((), device="cuda")
    if len(loss_list) == 1:
        return loss_list[0]

    method = (method or "pairwise_multiplicative").lower()
    if method == "sum":
        return torch.stack(loss_list).sum()
    if method in ("max",):
        return torch.stack(loss_list).max()
    if method in ("bottleneck", "bottleneck_logsumexp", "logsumexp"):
        tau = float(tau)
        tau = max(tau, 1e-6)
        stacked = torch.stack(loss_list)
        return tau * torch.logsumexp(stacked / tau, dim=0)

    # fallback: simple sum (avoid implementing legacy multiplicative coupling here)
    return torch.stack(loss_list).sum()


def _grad_concentration(model, topk: int = 15):
    """
    Return gradient concentration stats over parameter tensors.
    Uses grad energy = ||g||_2^2 per tensor.
    """
    import torch

    entries = []
    total_energy = 0.0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            continue
        g = p.grad.detach()
        if not torch.isfinite(g).all():
            g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        # grad energy per tensor
        e = float(g.float().pow(2).sum().item())
        if e <= 0:
            continue
        entries.append((e, name, p.numel()))
        total_energy += e

    entries.sort(reverse=True, key=lambda x: x[0])
    if total_energy <= 0 or not entries:
        return {
            "total_energy": 0.0,
            "top1_share": 0.0,
            "topk_share": 0.0,
            "top": [],
        }

    k = min(topk, len(entries))
    top_energy = sum(e for e, _, _ in entries[:k])
    return {
        "total_energy": total_energy,
        "top1_share": entries[0][0] / total_energy,
        "topk_share": top_energy / total_energy,
        "top": entries[:k],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--subset", type=str, default="defense_target",
                        choices=("target", "defense_target"))
    parser.add_argument("--num_batches", type=int, default=1)
    parser.add_argument("--topk", type=int, default=15)
    parser.add_argument("--energy_cap_mult", type=float, default=0.0,
                        help="If >0, apply per-tensor gradient energy cap (for debugging) "
                             "and report concentration AFTER the cap.")
    args = parser.parse_args()

    _set_env(args.gpu)

    # Make repo imports work when running as a script.
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import torch
    from project_core import ConfigManager
    from models import ModelManager
    from data import DataManager
    from tools.utils import set_seed

    cfg = ConfigManager(args.config).config
    set_seed(cfg.get("misc", {}).get("seed", 42))

    # Load model
    model_mgr = ModelManager(cfg)
    model_mgr.setup(apply_lora=False, device="cuda")
    model = model_mgr.model

    # Freeze head conv.* if configured (or if user wants shortcut test under freeze-head regime)
    defense_cfg = cfg.get("defense", {}) or {}
    anti = defense_cfg.get("antishortcut", {}) or {}
    if anti.get("freeze_head", False):
        _freeze_head_conv(model)

    # Data loader (target batch)
    data_mgr = DataManager(cfg, model_mgr.opt)
    data_mgr.setup_dataloaders(train=True, val=False, subset=args.subset)
    loader = data_mgr.train_loader

    trap_fns = _active_trap_fns(defense_cfg)
    if not trap_fns:
        raise SystemExit("No active trap losses in config.defense.trap_losses")
    for k, fn in trap_fns.items():
        trap_fns[k] = fn.to("cuda")

    agg_cfg = defense_cfg.get("trap_aggregation", {}) or {}
    agg_method = str(agg_cfg.get("method", "bottleneck_logsumexp"))
    agg_tau = float(agg_cfg.get("tau", 0.25))

    print("=" * 80)
    print("Shortcut diagnostic (gradient concentration)")
    print(f"config={args.config}")
    print(f"gpu={args.gpu}, subset={args.subset}, num_batches={args.num_batches}, topk={args.topk}")
    print(f"freeze_head={bool(anti.get('freeze_head', False))}")
    print(f"trap_losses={list(trap_fns.keys())}")
    print(f"agg_method={agg_method}, tau={agg_tau}")
    print("=" * 80)

    it = iter(loader)
    for bidx in range(args.num_batches):
        batch = next(it)
        input_images = batch["input_images"].to("cuda")

        # Forward once, reuse graph for multiple trap losses
        model.train()
        model.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(True):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                gaussians = model.forward_gaussians(input_images)
            gaussians = gaussians.float()

        print(f"\n[Batch {bidx}] gaussians={tuple(gaussians.shape)}")

        # Per-trap gradient concentration
        for name, fn in trap_fns.items():
            model.zero_grad(set_to_none=True)
            loss = fn(gaussians)
            loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=-1e6)
            loss.backward(retain_graph=True)
            stats = _grad_concentration(model, topk=args.topk)
            top_names = [n for _, n, _ in stats["top"][:5]]
            print(f"  - {name}: loss={loss.item():.4f} | top1={stats['top1_share']*100:.1f}% "
                  f"| top{args.topk}={stats['topk_share']*100:.1f}% | top5={top_names}")
            if args.energy_cap_mult and args.energy_cap_mult > 0:
                from tools.utils import cap_grad_tensor_energy_
                cap_grad_tensor_energy_(model.parameters(), cap_mult=args.energy_cap_mult, return_stats=False)
                stats2 = _grad_concentration(model, topk=args.topk)
                top_names2 = [n for _, n, _ in stats2["top"][:5]]
                print(f"      after cap(mult={args.energy_cap_mult:g}): top1={stats2['top1_share']*100:.1f}% "
                      f"| top{args.topk}={stats2['topk_share']*100:.1f}% | top5={top_names2}")

        # Aggregated loss (as used by defense)
        model.zero_grad(set_to_none=True)
        loss_list = [fn(gaussians) for fn in trap_fns.values()]
        agg_loss = _aggregate_losses(loss_list, method=agg_method, tau=agg_tau)
        agg_loss = torch.nan_to_num(agg_loss, nan=0.0, posinf=1e6, neginf=-1e6)
        agg_loss.backward()
        stats = _grad_concentration(model, topk=args.topk)
        top_names = [n for _, n, _ in stats["top"][:5]]
        print(f"  * aggregated: loss={agg_loss.item():.4f} | top1={stats['top1_share']*100:.1f}% "
              f"| top{args.topk}={stats['topk_share']*100:.1f}% | top5={top_names}")
        if args.energy_cap_mult and args.energy_cap_mult > 0:
            from tools.utils import cap_grad_tensor_energy_
            cap_grad_tensor_energy_(model.parameters(), cap_mult=args.energy_cap_mult, return_stats=False)
            stats2 = _grad_concentration(model, topk=args.topk)
            top_names2 = [n for _, n, _ in stats2["top"][:5]]
            print(f"      after cap(mult={args.energy_cap_mult:g}): top1={stats2['top1_share']*100:.1f}% "
                  f"| top{args.topk}={stats2['topk_share']*100:.1f}% | top5={top_names2}")

    # Cleanup
    del model, model_mgr, data_mgr
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
