"""Deterministic held-out validation over a finite shard list."""

import argparse
import datetime
import itertools
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

from fmrifm.checkpoint import load_weights
from fmrifm.config import Config, parse_config
from fmrifm.model import build_model
from fmrifm.pretrain_data import PretrainData, build_sampler
from fmrifm.pretrain_utils import probe_labels
from fmrifm.probe import OnlineLinearProbe


# Downstream readers identify the report by this key, not by filename.
REPORT_ARTIFACT = "fmrifm-validation"
REPORT_VERSION = 1
PROGRESS_EVERY = 25


def make_validation_loader(
    config: Config,
    urls: str,
    *,
    batch_size: int | None = None,
    num_workers: int | None = None,
    seed: int | None = None,
):
    """Build a finite, unshuffled, unresampled loader over ``urls``."""
    batch_size = config.data.batch_size if batch_size is None else batch_size
    num_workers = config.data.num_workers if num_workers is None else num_workers
    seed = config.training.seed if seed is None else seed

    dataset = PretrainData(
        urls=urls,
        sampler=build_sampler(config),
        seed=seed,
        shuffle_buffer=0,
        resampled=False,
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": config.data.pin_memory,
        # DataLoader iterator creation must not advance the model RNG.
        "generator": torch.Generator().manual_seed(seed),
    }
    if num_workers:
        loader_kwargs.update(
            prefetch_factor=config.data.prefetch,
            persistent_workers=False,
            multiprocessing_context="spawn",
        )
    return torch.utils.data.DataLoader(dataset, **loader_kwargs)


def _autocast(device: torch.device, precision: str | None):
    if precision == "bf16-mixed":
        return torch.autocast(device.type, dtype=torch.bfloat16)
    return nullcontext()


@torch.inference_mode()
def validate_model(
    model,
    probe,
    dataloader,
    *,
    device: str | torch.device,
    precision: str | None = None,
    max_batches: int | None = None,
    label: str = "validation",
) -> dict[str, float | int]:
    """Score reconstruction and the probe; ``probe_loss`` is age MSE plus sex BCE.

    An exhaustive pass takes tens of minutes, so progress goes to stderr, leaving
    stdout as the single JSON document callers parse.
    """
    device = torch.device(device)
    was_training = model.training
    model.eval()

    total_recon_squared_error = 0.0
    recon_value_count = 0
    total_age_squared_error = 0.0
    total_sex_loss = 0.0
    total_age_error = 0.0
    total_sex_correct = 0
    age_count = 0
    sex_count = 0
    sample_count = 0
    batch_count = 0
    started = time.monotonic()
    print("[%s] starting" % label, file=sys.stderr, flush=True)
    try:
        with _autocast(device, precision):
            # islice avoids decoding one unused remote batch.
            for batch in itertools.islice(dataloader, max_batches):
                x = batch["x"].to(device=device, dtype=torch.float32)
                coord = batch["coord"].to(device=device, dtype=torch.float32)
                target = batch["target"].to(device=device, dtype=torch.float32)
                target_coord = batch["target_coord"].to(device=device, dtype=torch.float32)
                target_mask = batch["target_mask"].to(device=device, dtype=torch.bool)
                recon_loss, hidden, _prediction, _target = model(
                    x=x,
                    coord=coord,
                    target=target,
                    target_coord=target_coord,
                    target_voxel_mask=target_mask,
                    compute_loss=True,
                )
                # Each spatial mask entry applies to every frame.
                valid_value_count = int(target_mask.sum()) * target.shape[1]
                total_recon_squared_error += float(recon_loss) * valid_value_count
                recon_value_count += valid_value_count

                predictions = probe(hidden)

                age, sex, age_valid, sex_valid = probe_labels(batch, device=device)
                if age_valid.any():
                    age_error = predictions[age_valid, 0].float() - age[age_valid]
                    total_age_squared_error += float(age_error.square().sum())
                    total_age_error += float(age_error.abs().sum())
                    age_count += int(age_valid.sum())
                if sex_valid.any():
                    sex_logits = predictions[sex_valid, 1].float()
                    sex_targets = sex[sex_valid]
                    total_sex_loss += float(
                        F.binary_cross_entropy_with_logits(sex_logits, sex_targets.float(), reduction="sum")
                    )
                    predicted = (sex_logits > 0).long()
                    total_sex_correct += int((predicted == sex_targets).sum())
                    sex_count += int(sex_valid.sum())

                sample_count += x.shape[0]
                batch_count += 1
                if batch_count % PROGRESS_EVERY == 0:
                    elapsed = time.monotonic() - started
                    print(
                        "[%s] %d batches, %d samples, %.0fs, %.1f samples/s"
                        % (label, batch_count, sample_count, elapsed, sample_count / elapsed),
                        file=sys.stderr,
                        flush=True,
                    )
    finally:
        model.train(was_training)

    elapsed = time.monotonic() - started
    print(
        "[%s] done: %d batches, %d samples in %.0fs (%.1f samples/s)"
        % (label, batch_count, sample_count, elapsed, sample_count / max(elapsed, 1e-9)),
        file=sys.stderr,
        flush=True,
    )
    if sample_count == 0:
        raise RuntimeError("validation produced no samples")
    if not recon_value_count:
        raise RuntimeError("validation produced no valid target voxels")
    age_mse = total_age_squared_error / age_count if age_count else 0.0
    sex_bce = total_sex_loss / sex_count if sex_count else 0.0
    metrics: dict[str, float | int] = {
        "recon_loss": total_recon_squared_error / recon_value_count,
        "probe_loss": age_mse + sex_bce,
        "age_mse": age_mse,
        "sex_bce": sex_bce,
        "age_count": age_count,
        "sex_count": sex_count,
        "samples": sample_count,
        "batches": batch_count,
    }
    if age_count:
        metrics["age_mae"] = total_age_error / age_count
    if sex_count:
        metrics["sex_accuracy"] = total_sex_correct / sex_count
    return metrics


class PeriodicValidation:
    """Periodically score a truncated prefix of each held-out set during training."""

    def __init__(self, config: Config, *, device, precision: str | None = None):
        self.config = config
        self.device = device
        self.precision = precision
        self._loaders = None

    def due(self, step: int) -> bool:
        settings = self.config.validation
        return bool(settings.enabled) and step % settings.every_n_steps == 0

    def run(self, model, probe) -> dict[str, float]:
        settings = self.config.validation
        if self._loaders is None:
            self._loaders = {
                name: make_validation_loader(
                    self.config,
                    url,
                    batch_size=settings.batch_size,
                    num_workers=settings.num_workers,
                )
                for name, url in settings.sets.items()
            }
        values: dict[str, float] = {}
        for name, loader in self._loaders.items():
            metrics = validate_model(
                model,
                probe,
                loader,
                device=self.device,
                precision=self.precision,
                max_batches=settings.max_batches,
            )
            values.update({f"val/{name}/{key}": float(value) for key, value in metrics.items()})
        return values


def resolve_settings(
    config: Config,
    *,
    batch_size: int | None = None,
    num_workers: int | None = None,
    seed: int | None = None,
    device: str | torch.device | None = None,
) -> dict:
    """The values the pass will run under, so the artifact records them and not defaults."""
    return {
        "device": str(torch.device(config.training.accelerator if device is None else device)),
        # validation.batch_size is None when it defers to the training batch.
        "batch_size": batch_size if batch_size is not None
        else (config.validation.batch_size or config.data.batch_size),
        "num_workers": num_workers if num_workers is not None else config.validation.num_workers,
        "seed": seed if seed is not None else config.training.seed,
        "precision": config.training.precision,
        # What the sampler draws, which is what makes two passes comparable or not.
        "cohort": {
            "context_patches": config.data.context_patches,
            "target_patches": config.data.target_patches,
            "clip_frames": config.preproc.clip_frames,
        },
    }


def run_validation(
    checkpoint: str | Path, config: Config, settings: dict
) -> dict[str, dict[str, float | int]]:
    """Load a checkpoint and score every configured held-out set exhaustively."""
    device = torch.device(settings["device"])
    torch.set_float32_matmul_precision("high")
    model = build_model(config)
    probe = OnlineLinearProbe(config.model.dim)
    load_weights(checkpoint, model, probe=probe)
    model.to(device)
    probe.to(device)

    return {
        name: validate_model(
            model,
            probe,
            make_validation_loader(
                config,
                url,
                batch_size=settings["batch_size"],
                num_workers=settings["num_workers"],
                seed=settings["seed"],
            ),
            device=device,
            precision=settings["precision"],
            label=name,
        )
        for name, url in config.validation.sets.items()
    }


def _inferred_config(checkpoint: Path) -> Path:
    candidate = checkpoint.parent.parent / "config.yaml"
    if not candidate.is_file():
        raise FileNotFoundError(f"could not infer config at {candidate}; pass --config explicitly")
    return candidate


def default_report_path(checkpoint: Path) -> Path:
    """``<run_dir>/validation/<checkpoint stem>.json``, beside the config."""
    return checkpoint.parent.parent / "validation" / f"{checkpoint.stem}.json"


def build_report(
    metrics: dict[str, dict[str, float | int]],
    *,
    checkpoint: Path,
    config: Config,
    settings: dict,
) -> dict:
    """The numbers plus what produced them; ``sets`` is byte-for-byte what stdout prints."""
    return {
        "artifact": REPORT_ARTIFACT,
        "version": REPORT_VERSION,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": str(checkpoint),
        "embedding_dim": config.model.dim,
        "settings": settings,
        "sets": metrics,
    }


def write_report(path: Path, report: dict) -> Path:
    """Write to a sibling temporary name and rename, so a killed job leaves no half file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".pending.json")
    pending.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    pending.replace(path)
    return path


def main(args=None) -> None:
    parser = argparse.ArgumentParser(
        prog="fmrifm.validate",
        description="Deterministic held-out validation.",
        allow_abbrev=False,
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument(
        "--report",
        type=Path,
        help="where to write the JSON artifact (default: <run_dir>/validation/<ckpt>.json)",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="print to stdout only, leaving the results in the job log and nowhere else",
    )
    known, config_args = parser.parse_known_args(args)
    if known.checkpoint.suffix == ".safetensors":
        raise SystemExit(f"{known.checkpoint} holds encoder weights only; validation needs a training checkpoint")
    has_config = any(argument == "--config" or argument.startswith("--config=") for argument in config_args)
    if not has_config:
        config_args = [
            "--config",
            str(_inferred_config(known.checkpoint)),
            *config_args,
        ]
    config = parse_config(config_args)
    settings = resolve_settings(
        config,
        batch_size=known.batch_size,
        num_workers=known.num_workers,
        seed=known.seed,
        device=known.device,
    )
    metrics = run_validation(known.checkpoint, config, settings)
    print(json.dumps(metrics, indent=2, sort_keys=True))

    if not known.no_report:
        report = build_report(metrics, checkpoint=known.checkpoint, config=config, settings=settings)
        path = write_report(known.report or default_report_path(known.checkpoint), report)
        # stderr: stdout is the single JSON document callers parse.
        print(f"wrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
