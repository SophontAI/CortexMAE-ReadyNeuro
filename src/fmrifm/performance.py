"""Analytic matmul FLOPs and throughput for compute-matched training.

One multiply-accumulate is two FLOPs. Matmuls use a 3x forward/backward
approximation; small ops, optimizer work, and the input pipeline are not counted.
"""

from __future__ import annotations

import time

import torch

from fmrifm.config import Config


def time_tokens(config: Config) -> int:
    return config.preproc.clip_frames // config.model.temporal_patch_size


def _encoder_forward_flops(config: Config, *, spatial_tokens: int) -> int:
    model = config.model
    patch_dim = model.patch_size**3 * model.temporal_patch_size
    tokens = time_tokens(config) * spatial_tokens
    total = 2 * tokens * patch_dim * model.dim

    rates = torch.linspace(0, model.drop_path_rate, model.depth).tolist()
    for block_index in range(model.depth):
        attention_projections = 8 * tokens * model.dim**2
        mlp = 16 * tokens * model.dim**2
        attention_matmuls = 4 * tokens**2 * model.dim
        block = attention_projections + mlp + attention_matmuls
        total += round((1 - rates[block_index]) * block)
    return total


def _decoder_forward_flops(config: Config, *, context_tokens: int, target_tokens: int) -> int:
    model = config.model
    decoder_dim = model.decoder_dim
    frames = time_tokens(config)
    queries = frames * target_tokens
    keys = frames * context_tokens
    total = 0
    rates = torch.linspace(0, model.drop_path_rate, model.decoder_depth).tolist()
    for block_index in range(model.decoder_depth):
        query_output_projections = 4 * queries * decoder_dim**2
        context_key_value_projections = 4 * keys * model.dim * decoder_dim
        mlp = 16 * queries * decoder_dim**2
        attention_matmuls = 4 * queries * keys * decoder_dim
        block = query_output_projections + context_key_value_projections + mlp + attention_matmuls
        total += round((1 - rates[block_index]) * block)

    output_dim = model.patch_size**3 * model.temporal_patch_size
    total += 2 * queries * decoder_dim * output_dim
    return total


def estimate_training_flops(config: Config, batch_size: int) -> int:
    """Estimate training FLOPs for one optimizer update."""
    context_patches = config.data.context_patches
    forward = _encoder_forward_flops(config, spatial_tokens=context_patches) + _decoder_forward_flops(
        config, context_tokens=context_patches, target_tokens=config.data.target_patches
    )
    return 3 * batch_size * forward


class PerformanceTracker:
    """Accumulate a logging window without synchronizing CUDA per update."""

    def __init__(
        self,
        config: Config,
        *,
        device: torch.device,
        initial_flops: float = 0.0,
    ) -> None:
        self.config = config
        self.device = device
        self.cumulative_flops = float(initial_flops)
        self.interval_flops = 0.0
        self.interval_updates = 0
        self.interval_samples = 0
        self.interval_encoder_tokens = 0
        self._peak_before_pause = 0.0
        self._pause_started: float | None = None
        self.interval_start = time.perf_counter()
        self._reset_peak_memory()

    @property
    def uses_cuda(self) -> bool:
        return self.device.type == "cuda" and torch.cuda.is_available()

    def _reset_peak_memory(self) -> None:
        if self.uses_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

    def record_update(self, *, local_batch_size: int) -> None:
        """Record counters only; this method deliberately never synchronizes."""
        if self._pause_started is not None:
            raise RuntimeError("cannot record an update while performance tracking is paused")
        flops = estimate_training_flops(self.config, local_batch_size)
        self.interval_flops += flops
        self.cumulative_flops += flops
        self.interval_updates += 1
        self.interval_samples += local_batch_size
        self.interval_encoder_tokens += (
            local_batch_size * time_tokens(self.config) * self.config.data.context_patches
        )

    def pause_interval(self) -> None:
        """Exclude non-training work from the active throughput window."""
        if self._pause_started is not None:
            raise RuntimeError("performance tracking is already paused")
        if self.uses_cuda:
            torch.cuda.synchronize(self.device)
            self._peak_before_pause = max(
                self._peak_before_pause,
                float(torch.cuda.max_memory_allocated(self.device)),
            )
        self._pause_started = time.perf_counter()

    def resume_interval(self) -> None:
        if self._pause_started is None:
            raise RuntimeError("performance tracking is not paused")
        if self.uses_cuda:
            torch.cuda.synchronize(self.device)
        paused = time.perf_counter() - self._pause_started
        self.interval_start += paused
        self._pause_started = None
        self._reset_peak_memory()

    def finish_interval(self) -> dict[str, float]:
        """Synchronize once, emit stable values, and begin the next window."""
        if self._pause_started is not None:
            raise RuntimeError("cannot finish a paused performance interval")
        if not self.interval_updates:
            raise RuntimeError("cannot finish an empty performance interval")
        if self.uses_cuda:
            torch.cuda.synchronize(self.device)
            peak_memory = max(
                self._peak_before_pause,
                float(torch.cuda.max_memory_allocated(self.device)),
            )
        else:
            peak_memory = 0.0
        elapsed = max(time.perf_counter() - self.interval_start, 1e-12)
        values = {
            "perf/estimated_flops_per_update": (self.interval_flops / self.interval_updates),
            "perf/cumulative_flops": self.cumulative_flops,
            "perf/updates_per_sec": self.interval_updates / elapsed,
            "perf/samples_per_sec": self.interval_samples / elapsed,
            "perf/encoder_tokens_per_sec": (self.interval_encoder_tokens / elapsed),
            "perf/interval_step_time_sec": elapsed / self.interval_updates,
            "perf/peak_allocated_gpu_memory_bytes": peak_memory,
        }
        self.interval_flops = 0.0
        self.interval_updates = 0
        self.interval_samples = 0
        self.interval_encoder_tokens = 0
        self._peak_before_pause = 0.0
        self._reset_peak_memory()
        self.interval_start = time.perf_counter()
        return values
