"""Learning-rate schedule, probe objective, and training metrics."""

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from fmrifm.config import Config, as_dict


_COUNT_METRICS = ("age_count", "sex_count")


# A nonzero floor lets extended or branched runs keep moving.
MIN_LR_RATIO = 0.01


def learning_rate_at(
    step: int,
    *,
    peak_lr: float,
    steps: int,
    warmup_steps: int,
) -> float:
    """Warm up linearly, then cosine-decay to ``MIN_LR_RATIO * peak_lr``.

    Steps are zero-indexed, so the result is a pure function of the global step
    and no scheduler state has to survive a restart.
    """
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps

    floor_lr = MIN_LR_RATIO * peak_lr
    peak_step = max(0, warmup_steps - 1)
    decay_intervals = steps - 1 - peak_step
    if decay_intervals == 0:
        return floor_lr
    progress = min(1.0, (step - peak_step) / decay_intervals)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return floor_lr + (peak_lr - floor_lr) * cosine


def probe_labels(
    batch: dict,
    *,
    device: str | torch.device,
    sex_dtype: torch.dtype = torch.long,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return age, sex cast to ``sex_dtype``, age > 0, and sex-in-{0, 1} on ``device``."""
    age = torch.as_tensor(batch["age"], device=device, dtype=torch.float32)
    sex = torch.as_tensor(batch["sex"], device=device, dtype=sex_dtype)
    return age, sex, age > 0, (sex == 0) | (sex == 1)


def _probe_objective(predictions: torch.Tensor, batch: dict) -> torch.Tensor:
    """Return the probe loss, ignoring missing age and sex labels."""
    age, sex, age_valid, sex_valid = probe_labels(
        batch,
        device=predictions.device,
        sex_dtype=torch.float32,
    )

    age_prediction = predictions[:, 0].float()
    sex_logit = predictions[:, 1].float()
    # Keep backward valid when this batch has no labelled rows.
    zero = predictions.float().sum() * 0.0
    age_loss = F.mse_loss(age_prediction[age_valid], age[age_valid]) if age_valid.any() else zero
    sex_loss = (
        F.binary_cross_entropy_with_logits(sex_logit[sex_valid], sex[sex_valid]) if sex_valid.any() else zero
    )

    return age_loss + sex_loss


def _make_metrics(device: torch.device) -> nn.ModuleDict:
    metrics = nn.ModuleDict(
        {
            "loss": torchmetrics.MeanMetric(),
            "recon_loss": torchmetrics.MeanMetric(),
            "probe_loss": torchmetrics.MeanMetric(),
            # Pre-clip global norm, and the share of updates the clip actually bound.
            "grad_norm": torchmetrics.MeanMetric(),
            "grad_norm_max": torchmetrics.MaxMetric(),
            "grad_clipped": torchmetrics.MeanMetric(),
            "age_mae": torchmetrics.MeanAbsoluteError(),
            "age_count": torchmetrics.SumMetric(),
            "sex_accuracy": torchmetrics.classification.BinaryAccuracy(),
            "sex_cohen_kappa": torchmetrics.classification.BinaryCohenKappa(),
            "sex_count": torchmetrics.SumMetric(),
        }
    ).to(device)
    # Avoid TorchMetrics warnings when an optional metric receives no updates.
    for name in _COUNT_METRICS:
        metrics[name].update(torch.tensor(0, device=device))
    return metrics


def _reset_metrics(metrics: nn.ModuleDict) -> None:
    for metric in metrics.values():
        metric.reset()
    device = metrics[_COUNT_METRICS[0]].device
    for name in _COUNT_METRICS:
        metrics[name].update(torch.tensor(0, device=device))


@torch.no_grad()
def _update_metrics(
    metrics: nn.ModuleDict,
    *,
    loss: torch.Tensor,
    recon_loss: torch.Tensor,
    probe_loss: torch.Tensor,
    probe_predictions: torch.Tensor,
    batch: dict,
    grad_norm: torch.Tensor | None = None,
    grad_clip: float = 0.0,
) -> None:
    if grad_norm is not None:
        metrics["grad_norm"].update(grad_norm)
        metrics["grad_norm_max"].update(grad_norm)
        if grad_clip:
            metrics["grad_clipped"].update((grad_norm > grad_clip).float())
    metrics["loss"].update(loss)
    metrics["recon_loss"].update(recon_loss)
    metrics["probe_loss"].update(probe_loss)

    age, sex, age_valid, sex_valid = probe_labels(batch, device=probe_predictions.device)

    metrics["age_count"].update(age_valid.sum())
    metrics["sex_count"].update(sex_valid.sum())
    if age_valid.any():
        metrics["age_mae"].update(probe_predictions[age_valid, 0].float(), age[age_valid])
    if sex_valid.any():
        sex_logits = probe_predictions[sex_valid, 1].float()
        sex_targets = sex[sex_valid]
        # TorchMetrics conditionally detects logits, so apply the zero boundary explicitly.
        sex_predictions = sex_logits > 0
        metrics["sex_accuracy"].update(sex_predictions, sex_targets)
        metrics["sex_cohen_kappa"].update(sex_predictions, sex_targets)


def _compute_metrics(metrics: nn.ModuleDict) -> dict[str, torch.Tensor]:
    """Compute loggable metrics, omitting probe tasks with no valid labels."""
    values = {
        "loss": metrics["loss"].compute(),
        "recon_loss": metrics["recon_loss"].compute(),
        "probe_loss": metrics["probe_loss"].compute(),
    }
    # MeanMetric returns nan before its first update, which a resumed step can
    # hit; only report a norm once one has been measured.
    grad_norm = metrics["grad_norm"].compute()
    if torch.isfinite(grad_norm):
        values["grad_norm"] = grad_norm
        values["grad_norm_max"] = metrics["grad_norm_max"].compute()
        clipped = metrics["grad_clipped"].compute()
        if torch.isfinite(clipped):
            values["grad_clipped"] = clipped
    if metrics["age_count"].compute() > 0:
        values["age_mae"] = metrics["age_mae"].compute()
    if metrics["sex_count"].compute() > 0:
        values["sex_accuracy"] = metrics["sex_accuracy"].compute()
        values["sex_cohen_kappa"] = metrics["sex_cohen_kappa"].compute()
    return values


def _latest_wandb_run_id(save_dir: Path) -> str | None:
    """Read the ID from W&B's latest-run link when --wandb.id was not given."""
    latest_run = save_dir / "wandb" / "latest-run"
    if not latest_run.exists():
        return None
    run_dir_name = latest_run.resolve().name
    if "-" not in run_dir_name:
        return None
    return run_dir_name.rsplit("-", 1)[-1]


def _make_wandb_logger(config: Config):
    if not config.wandb.enabled:
        return None

    from lightning.pytorch.loggers import WandbLogger

    resume_mode = "must" if config.checkpoint.load_path is not None else "allow"
    run_id = config.wandb.id
    if resume_mode == "must":
        run_id = run_id or _latest_wandb_run_id(config.wandb.save_dir)
        if run_id is None:
            raise RuntimeError(
                "checkpoint resume has W&B enabled but no run ID; set --wandb.id, "
                "restore wandb/latest-run, or disable W&B for this launch"
            )

    return WandbLogger(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=config.wandb.name,
        version=run_id,
        resume=resume_mode,
        save_dir=config.wandb.save_dir,
        offline=config.wandb.offline,
        config=as_dict(config),
    )
