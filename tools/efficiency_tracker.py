"""
训练效率追踪器 - 支持三种对比模式

用于论文中报告训练效率的工具，支持：
1. 相同step数对比
2. 训练到收敛对比
3. 相同时间预算对比
"""

import time
import torch
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class EfficiencyMetrics:
    """单次测量的效率指标"""
    step: int
    epoch: int
    elapsed_time: float  # 从训练开始的累计时间（秒）
    step_time: float  # 当前step耗时（秒）

    # 性能指标
    psnr: Optional[float] = None
    lpips: Optional[float] = None
    masked_psnr: Optional[float] = None
    masked_lpips: Optional[float] = None

    # 计算量指标
    flops_per_step: Optional[float] = None  # 每step的FLOPs
    cumulative_flops: Optional[float] = None  # 累计FLOPs

    # GPU资源
    gpu_memory_mb: Optional[float] = None  # 峰值显存（MB）

    # 其他训练指标
    loss: Optional[float] = None
    extra_metrics: Dict[str, Any] = field(default_factory=dict)


class EfficiencyTracker:
    """
    训练效率追踪器

    在训练循环中调用 record() 记录每个eval点的指标，
    训练结束后调用 generate_report() 生成三种对比模式的报告。
    """

    def __init__(self,
                 method_name: str = "Method",
                 flops_per_step: Optional[float] = None,
                 target_performance: Optional[Dict[str, float]] = None,
                 flops_profile_steps: int = 4,
                 flops_profile_warmup_steps: int = 1):
        """
        Args:
            method_name: 方法名称（用于报告）
            flops_per_step: 每step的FLOPs（如果已知）
            target_performance: 目标性能阈值，如 {'psnr': 28.5, 'lpips': 0.15}
            flops_profile_steps: 若 flops_per_step 未知，自动采样多少个优化器步估计 FLOPs
            flops_profile_warmup_steps: 跳过前多少个优化器步后再开始 FLOPs 采样
        """
        self.method_name = method_name
        self._manual_flops_per_step = flops_per_step
        self.flops_per_step = flops_per_step
        self.target_performance = target_performance or {}
        self.flops_profile_steps = max(int(flops_profile_steps), 0)
        self.flops_profile_warmup_steps = max(int(flops_profile_warmup_steps), 0)

        self.history: List[EfficiencyMetrics] = []
        self.start_time: Optional[float] = None
        self.peak_memory_mb: float = 0.0
        self.flops_samples: List[float] = []
        self.flops_profiled_steps: List[int] = []
        self.flops_profile_attempts: int = 0
        self.flops_measurement_error: Optional[str] = None
        self._active_profiler = None
        self._active_profiler_step: Optional[int] = None

    def start(self):
        """开始训练计时"""
        self.start_time = time.time()
        self.history = []
        self.peak_memory_mb = 0.0
        self.flops_samples = []
        self.flops_profiled_steps = []
        self.flops_profile_attempts = 0
        self.flops_measurement_error = None
        self._active_profiler = None
        self._active_profiler_step = None
        self.flops_per_step = self._manual_flops_per_step

    def get_flops_per_step(self) -> Optional[float]:
        """返回当前可用的每 step FLOPs（手工指定或自动采样估计）"""
        if self._manual_flops_per_step is not None:
            return self._manual_flops_per_step
        if self.flops_samples:
            return sum(self.flops_samples) / len(self.flops_samples)
        return self.flops_per_step

    def should_profile_flops(self, step: int) -> bool:
        """判断当前优化器步是否应该启动 FLOPs 采样"""
        if self._manual_flops_per_step is not None:
            return False
        if self.flops_profile_steps <= 0:
            return False
        if self._active_profiler is not None:
            return False
        if self.flops_measurement_error is not None:
            return False
        if step <= self.flops_profile_warmup_steps:
            return False
        if self.flops_profile_attempts >= self.flops_profile_steps:
            return False
        return hasattr(torch, "profiler") and hasattr(torch.profiler, "profile")

    def start_flops_profiler(self, step: int) -> bool:
        """在一个优化器步开始前启动 FLOPs profiler"""
        if not self.should_profile_flops(step):
            return False

        try:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)

            profiler = torch.profiler.profile(
                activities=activities,
                with_flops=True,
                record_shapes=False,
                profile_memory=False,
            )
            profiler.__enter__()
            self._active_profiler = profiler
            self._active_profiler_step = int(step)
            self.flops_profile_attempts += 1
            return True
        except Exception as exc:
            self.flops_measurement_error = (
                f"{type(exc).__name__}: {exc}"
            )
            self._active_profiler = None
            self._active_profiler_step = None
            return False

    def stop_flops_profiler(self, step: Optional[int] = None) -> Optional[float]:
        """在一个优化器步结束后停止 profiler，并记录本步 FLOPs"""
        profiler = self._active_profiler
        profiled_step = self._active_profiler_step
        self._active_profiler = None
        self._active_profiler_step = None

        if profiler is None:
            return None

        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            profiler.__exit__(None, None, None)

            key_averages = profiler.key_averages()
            total_flops = float(getattr(key_averages.total_average(), "flops", 0) or 0)
            if total_flops <= 0:
                total_flops = float(
                    sum(float(getattr(evt, "flops", 0) or 0) for evt in key_averages)
                )

            if total_flops > 0:
                effective_step = int(step if step is not None else (profiled_step or 0))
                self.flops_samples.append(total_flops)
                self.flops_profiled_steps.append(effective_step)
                self.flops_per_step = self.get_flops_per_step()
                return total_flops

            if self.flops_profile_attempts >= self.flops_profile_steps:
                self.flops_measurement_error = (
                    "PyTorch profiler did not report FLOPs for sampled defense steps."
                )
        except Exception as exc:
            self.flops_measurement_error = f"{type(exc).__name__}: {exc}"

        return None

    def record(self,
               step: int,
               epoch: int,
               step_time: float,
               psnr: Optional[float] = None,
               lpips: Optional[float] = None,
               masked_psnr: Optional[float] = None,
               masked_lpips: Optional[float] = None,
               loss: Optional[float] = None,
               **extra_metrics):
        """
        记录一个eval点的指标

        Args:
            step: 当前step数
            epoch: 当前epoch数
            step_time: 当前step耗时（秒）
            psnr, lpips, masked_psnr, masked_lpips: 性能指标
            loss: 训练loss
            **extra_metrics: 其他自定义指标
        """
        if self.start_time is None:
            raise RuntimeError("请先调用 start() 开始计时")

        elapsed_time = time.time() - self.start_time

        # 计算FLOPs
        flops_per_step = self.get_flops_per_step()
        cumulative_flops = flops_per_step * step if flops_per_step is not None else None

        # 测量GPU显存
        gpu_memory_mb = None
        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            self.peak_memory_mb = max(self.peak_memory_mb, gpu_memory_mb)

        metrics = EfficiencyMetrics(
            step=step,
            epoch=epoch,
            elapsed_time=elapsed_time,
            step_time=step_time,
            psnr=psnr,
            lpips=lpips,
            masked_psnr=masked_psnr,
            masked_lpips=masked_lpips,
            flops_per_step=flops_per_step,
            cumulative_flops=cumulative_flops,
            gpu_memory_mb=gpu_memory_mb,
            loss=loss,
            extra_metrics=extra_metrics
        )

        self.history.append(metrics)

    def find_convergence_point(self,
                               metric_name: str = 'masked_psnr',
                               threshold: Optional[float] = None,
                               window_size: int = 3) -> Optional[EfficiencyMetrics]:
        """
        找到收敛点（性能达到阈值且稳定）

        Args:
            metric_name: 用于判断收敛的指标名
            threshold: 性能阈值（如果为None，使用最终性能的95%）
            window_size: 稳定窗口大小

        Returns:
            收敛点的EfficiencyMetrics，如果未收敛返回None
        """
        if not self.history:
            return None

        # 如果没有指定阈值，使用最终性能的95%
        if threshold is None:
            final_metric = getattr(self.history[-1], metric_name, None)
            if final_metric is None:
                return None
            threshold = final_metric * 0.95

        # 找到第一个达到阈值且后续window_size个点都保持的位置
        for i in range(len(self.history) - window_size + 1):
            window = self.history[i:i + window_size]
            values = [getattr(m, metric_name, None) for m in window]

            if all(v is not None and v >= threshold for v in values):
                return self.history[i]

        return None

    def steps_to_reach_performance(self,
                                   target_psnr: Optional[float] = None,
                                   target_lpips: Optional[float] = None) -> Optional[int]:
        """
        计算达到目标性能所需的最早step数

        Args:
            target_psnr: 目标PSNR（>=）
            target_lpips: 目标LPIPS（<=）

        Returns:
            达到目标的最早step数，如果未达到返回None
        """
        for metrics in self.history:
            psnr_ok = target_psnr is None or (metrics.masked_psnr is not None and metrics.masked_psnr >= target_psnr)
            lpips_ok = target_lpips is None or (metrics.masked_lpips is not None and metrics.masked_lpips <= target_lpips)

            if psnr_ok and lpips_ok:
                return metrics.step

        return None

    def get_performance_at_step(self, target_step: int) -> Optional[EfficiencyMetrics]:
        """获取指定step的性能"""
        for metrics in self.history:
            if metrics.step == target_step:
                return metrics
        return None

    def get_performance_at_time(self, target_time: float) -> Optional[EfficiencyMetrics]:
        """获取指定时间点的性能（返回最接近的记录）"""
        if not self.history:
            return None

        closest = min(self.history, key=lambda m: abs(m.elapsed_time - target_time))
        return closest

    def generate_report(self, baseline_tracker: Optional['EfficiencyTracker'] = None) -> Dict[str, Any]:
        """
        生成三种对比模式的报告

        Args:
            baseline_tracker: baseline方法的tracker（用于对比）

        Returns:
            包含三种对比模式结果的字典
        """
        if not self.history:
            return {"error": "No training history"}

        final_metrics = self.history[-1]

        report = {
            "method_name": self.method_name,
            "final_performance": {
                "step": final_metrics.step,
                "epoch": final_metrics.epoch,
                "time_seconds": final_metrics.elapsed_time,
                "time_hours": final_metrics.elapsed_time / 3600,
                "psnr": final_metrics.psnr,
                "lpips": final_metrics.lpips,
                "masked_psnr": final_metrics.masked_psnr,
                "masked_lpips": final_metrics.masked_lpips,
            },
            "efficiency": {
                "total_steps": final_metrics.step,
                "total_time_seconds": final_metrics.elapsed_time,
                "avg_step_time": final_metrics.elapsed_time / final_metrics.step if final_metrics.step > 0 else 0,
                "flops_per_step": self.flops_per_step,
                "total_flops": final_metrics.cumulative_flops,
                "peak_memory_mb": self.peak_memory_mb,
            }
        }

        # 如果有baseline，生成三种对比
        if baseline_tracker and baseline_tracker.history:
            baseline_final = baseline_tracker.history[-1]

            # 模式1：相同step数对比
            same_step_metrics = self.get_performance_at_step(baseline_final.step)
            if same_step_metrics:
                report["comparison_same_steps"] = {
                    "target_step": baseline_final.step,
                    "ours_psnr": same_step_metrics.masked_psnr,
                    "baseline_psnr": baseline_final.masked_psnr,
                    "psnr_improvement": (same_step_metrics.masked_psnr - baseline_final.masked_psnr) if (same_step_metrics.masked_psnr and baseline_final.masked_psnr) else None,
                    "ours_time": same_step_metrics.elapsed_time,
                    "baseline_time": baseline_final.elapsed_time,
                    "time_speedup": baseline_final.elapsed_time / same_step_metrics.elapsed_time if same_step_metrics.elapsed_time > 0 else None,
                }

            # 模式2：训练到收敛对比
            ours_convergence = self.find_convergence_point()
            baseline_convergence = baseline_tracker.find_convergence_point()
            if ours_convergence and baseline_convergence:
                report["comparison_convergence"] = {
                    "ours_steps": ours_convergence.step,
                    "baseline_steps": baseline_convergence.step,
                    "step_reduction": (baseline_convergence.step - ours_convergence.step) / baseline_convergence.step if baseline_convergence.step > 0 else None,
                    "ours_time": ours_convergence.elapsed_time,
                    "baseline_time": baseline_convergence.elapsed_time,
                    "time_speedup": baseline_convergence.elapsed_time / ours_convergence.elapsed_time if ours_convergence.elapsed_time > 0 else None,
                }

            # 模式3：相同时间预算对比
            same_time_metrics = self.get_performance_at_time(baseline_final.elapsed_time)
            if same_time_metrics:
                report["comparison_same_time"] = {
                    "target_time_seconds": baseline_final.elapsed_time,
                    "ours_psnr": same_time_metrics.masked_psnr,
                    "baseline_psnr": baseline_final.masked_psnr,
                    "psnr_improvement": (same_time_metrics.masked_psnr - baseline_final.masked_psnr) if (same_time_metrics.masked_psnr and baseline_final.masked_psnr) else None,
                    "ours_steps": same_time_metrics.step,
                    "baseline_steps": baseline_final.step,
                }

        return report

    def export_to_dict(self) -> Dict[str, Any]:
        """导出完整历史为字典（用于保存）"""
        return {
            "method_name": self.method_name,
            "flops_per_step": self.flops_per_step,
            "peak_memory_mb": self.peak_memory_mb,
            "flops_profile_steps": self.flops_profile_steps,
            "flops_profile_warmup_steps": self.flops_profile_warmup_steps,
            "flops_profile_attempts": self.flops_profile_attempts,
            "flops_profiled_steps": self.flops_profiled_steps,
            "flops_samples": self.flops_samples,
            "flops_measurement_error": self.flops_measurement_error,
            "history": [
                {
                    "step": m.step,
                    "epoch": m.epoch,
                    "elapsed_time": m.elapsed_time,
                    "step_time": m.step_time,
                    "psnr": m.psnr,
                    "lpips": m.lpips,
                    "masked_psnr": m.masked_psnr,
                    "masked_lpips": m.masked_lpips,
                    "flops_per_step": m.flops_per_step,
                    "cumulative_flops": m.cumulative_flops,
                    "gpu_memory_mb": m.gpu_memory_mb,
                    "loss": m.loss,
                    **m.extra_metrics
                }
                for m in self.history
            ]
        }
