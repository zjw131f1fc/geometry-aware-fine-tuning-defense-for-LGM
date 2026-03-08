"""
集成 EfficiencyTracker 到 run_pipeline.py 的补丁示例

这个文件展示了如何修改现有代码来集成效率追踪功能。
"""

# ============================================================
# 修改 1: 在 run_pipeline.py 顶部导入
# ============================================================

# 在 line 53 附近，添加：
from tools import (
    BASELINE_CACHE_DIR, compute_baseline_hash,
    load_baseline_cache, save_baseline_cache, copy_cached_renders,
    plot_pipeline_results,
    EfficiencyTracker,  # 新增
)


# ============================================================
# 修改 2: 在 main() 函数中初始化 trackers
# ============================================================

# 在 line 700 附近（Phase 1 开始前），添加：

    # 初始化效率追踪器
    baseline_efficiency_tracker = EfficiencyTracker(
        method_name="Baseline Attack",
        flops_per_step=None,  # 如果知道FLOPs可以填入，如 120e9
    )
    postdef_efficiency_tracker = EfficiencyTracker(
        method_name="Post-Defense Attack",
        flops_per_step=None,
    )


# ============================================================
# 修改 3: 在 Phase 1 训练前启动追踪
# ============================================================

# 在 line 710 附近（run_attack 调用前），添加：

    if not args.skip_baseline:
        print("\n" + "="*60)
        print("Phase 1: Baseline Attack")
        print("="*60)

        baseline_efficiency_tracker.start()  # 新增：开始追踪
        config._efficiency_tracker = baseline_efficiency_tracker  # 新增：传递给训练函数

        baseline_step_history, baseline_source_metrics, baseline_target_metrics = run_attack(
            # ... 原有参数
        )


# ============================================================
# 修改 4: 在 Phase 3 训练前启动追踪
# ============================================================

# 在 line 950 附近（Post-Defense Attack 前），添加：

    print("\n" + "="*60)
    print("Phase 3: Post-Defense Attack")
    print("="*60)

    postdef_efficiency_tracker.start()  # 新增：开始追踪
    config._efficiency_tracker = postdef_efficiency_tracker  # 新增：传递给训练函数

    postdef_step_history, postdef_source_metrics, postdef_target_metrics = run_attack(
        # ... 原有参数
    )


# ============================================================
# 修改 5: 在 finetuner.py 的 run_attack() 中记录指标
# ============================================================

# 在 finetuner.py line 1155 附近（step_history.append 后），添加：

                    step_history.append(metrics)

                    # 新增：记录效率指标
                    if hasattr(config, '_efficiency_tracker') and config._efficiency_tracker is not None:
                        config._efficiency_tracker.record(
                            step=global_step,
                            epoch=epoch,
                            step_time=avg_step_time,
                            masked_psnr=metrics.get('masked_psnr'),
                            masked_lpips=metrics.get('masked_lpips'),
                            psnr=metrics.get('psnr'),
                            lpips=metrics.get('lpips'),
                            loss=metrics.get('loss'),
                        )


# ============================================================
# 修改 6: 在 run_pipeline.py 末尾生成效率报告
# ============================================================

# 在 line 1200 附近（保存 summary.json 后），添加：

    # ========== 生成训练效率报告 ==========
    if not args.skip_baseline:
        print("\n" + "="*60)
        print("训练效率对比报告")
        print("="*60)

        # 生成报告
        efficiency_report = postdef_efficiency_tracker.generate_report(
            baseline_tracker=baseline_efficiency_tracker
        )

        # 保存完整报告
        efficiency_report_path = os.path.join(output_dir, 'efficiency_report.json')
        with open(efficiency_report_path, 'w') as f:
            json.dump(efficiency_report, f, indent=2)
        print(f"\n效率报告已保存: {efficiency_report_path}")

        # 打印关键指标
        baseline_final = baseline_efficiency_tracker.history[-1] if baseline_efficiency_tracker.history else None
        postdef_final = postdef_efficiency_tracker.history[-1] if postdef_efficiency_tracker.history else None

        if baseline_final and postdef_final:
            print(f"\n【最终性能】")
            print(f"Baseline Attack:      PSNR={baseline_final.masked_psnr:.2f}, "
                  f"Time={baseline_final.elapsed_time/3600:.2f}h, Steps={baseline_final.step}")
            print(f"Post-Defense Attack:  PSNR={postdef_final.masked_psnr:.2f}, "
                  f"Time={postdef_final.elapsed_time/3600:.2f}h, Steps={postdef_final.step}")

            # 三种对比模式
            if 'comparison_same_steps' in efficiency_report:
                comp = efficiency_report['comparison_same_steps']
                print(f"\n【模式1：相同step数 ({comp['target_step']} steps)】")
                if comp['psnr_improvement'] is not None:
                    print(f"  PSNR变化: {comp['psnr_improvement']:+.2f} dB")
                if comp['time_speedup'] is not None:
                    print(f"  时间比: {comp['time_speedup']:.2f}x")

            if 'comparison_convergence' in efficiency_report:
                comp = efficiency_report['comparison_convergence']
                print(f"\n【模式2：训练到收敛】")
                if comp['step_reduction'] is not None:
                    print(f"  Steps变化: {comp['step_reduction']*100:+.1f}%")
                if comp['time_speedup'] is not None:
                    print(f"  时间比: {comp['time_speedup']:.2f}x")

            if 'comparison_same_time' in efficiency_report:
                comp = efficiency_report['comparison_same_time']
                print(f"\n【模式3：相同时间预算】")
                if comp['psnr_improvement'] is not None:
                    print(f"  PSNR变化: {comp['psnr_improvement']:+.2f} dB")
                print(f"  Steps: Baseline={comp['baseline_steps']}, Ours={comp['ours_steps']}")

            # 生成论文表格
            print(f"\n【论文表格数据】")
            print("| Method           | PSNR↑ | LPIPS↓ | Time (h) | Steps | Speedup |")
            print("|------------------|-------|--------|----------|-------|---------|")
            print(f"| Baseline Attack  | {baseline_final.masked_psnr:.1f}  | "
                  f"{baseline_final.masked_lpips:.2f}   | {baseline_final.elapsed_time/3600:.2f}     | "
                  f"{baseline_final.step:5d} | 1.0x    |")
            print(f"| Post-Def Attack  | {postdef_final.masked_psnr:.1f}  | "
                  f"{postdef_final.masked_lpips:.2f}   | {postdef_final.elapsed_time/3600:.2f}     | "
                  f"{postdef_final.step:5d} | "
                  f"{baseline_final.elapsed_time/postdef_final.elapsed_time:.1f}x    |")

            psnr_diff = postdef_final.masked_psnr - baseline_final.masked_psnr
            lpips_diff = postdef_final.masked_lpips - baseline_final.masked_lpips
            print(f"| Improvement      | {psnr_diff:+.1f} | {lpips_diff:+.2f}  | "
                  f"{(postdef_final.elapsed_time - baseline_final.elapsed_time)/3600:+.2f}    | "
                  f"{postdef_final.step - baseline_final.step:+5d} | -       |")

        print("="*60)


# ============================================================
# 使用示例
# ============================================================

"""
运行命令：

python script/run_pipeline.py \
    --gpu 0 \
    --config configs/config.yaml \
    --trap_losses position,scale \
    --tag geotrap_efficiency_test \
    --attack_epochs 5 \
    --defense_epochs 25 \
    --eval_every_steps 10

运行后会在输出目录生成：
- efficiency_report.json: 完整的效率对比数据
- 终端输出: 三种对比模式的结果和论文表格

论文中可以这样写：

"As shown in Table X, our defense method significantly reduces the attack
effectiveness while maintaining training efficiency. Under the same training
budget (30K steps), the post-defense attack achieves X dB lower PSNR compared
to the baseline attack, demonstrating the effectiveness of our geometric trap
mechanism. Moreover, our method shows comparable or better training efficiency,
with Y% reduction in training time when trained to convergence."
"""
