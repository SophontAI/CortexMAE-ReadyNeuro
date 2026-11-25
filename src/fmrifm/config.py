import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal, Sequence

from jsonargparse import ActionConfigFile, ArgumentParser
from webdataset.shardlists import expand_urls


DECODER_HEAD_DIM = 64
# The last OpenNeuro batch, held out of every training URL. Validation scores it.
HELD_OUT_URL = "s3://openneuro/v1/openneuro-pretrain-shards-v6-b31/train/native"
# One 4D rotary simplex scale spans five channel pairs.
MIN_HEAD_DIM = 10
# BOLD voxel edge in millimetres; only sets the pitch the wavelength band is written against.
BOLD_VOXEL_MM = 2.0
# Batch at which base_lr is defined; other batches use square-root scaling.
LR_REFERENCE_BATCH = 256
PRECISIONS = (None, "32-true", "bf16-mixed")


@dataclass
class ModelConfig:
    dim: int = 128
    n_heads: int = 2
    depth: int = 6
    decoder_dim: int = 512
    decoder_head_dim: int = DECODER_HEAD_DIM
    decoder_depth: int = 2
    patch_size: int = 8
    temporal_patch_size: int = 4
    # Stochastic depth, ramped linearly to this rate at the last block.
    drop_path_rate: float = 0.0
    # Longest spatial wavelength, in **millimetres**. Patch centers are world
    # coordinates taken through the subject's own affine, so the band is stated in
    # the units the data is measured in rather than in voxels of whatever grid it
    # happens to be written on. The short end is 2*pi*patch_size*BOLD_VOXEL_MM,
    # one turn per 6.3 patch pitches; the 4D simplex shares one frequency ladder,
    # so the temporal band is that same ladder restated in TRs.
    pos_max_wavelength: float = 640.0

    @property
    def decoder_heads(self) -> int:
        return self.decoder_dim // self.decoder_head_dim

    @property
    def resolved_pos_min_wavelength(self) -> float:
        return 2 * math.pi * self.patch_size * BOLD_VOXEL_MM

    def _validate_architecture(self) -> None:
        if self.dim <= 0 or self.n_heads <= 0 or self.depth <= 0:
            raise ValueError("model dim, n_heads, and depth must be positive")
        if self.decoder_dim <= 0 or self.decoder_depth <= 0:
            raise ValueError("model decoder_dim and decoder_depth must be positive")
        if self.patch_size <= 0 or self.temporal_patch_size <= 0:
            raise ValueError("model patch sizes must be positive")
        if not 0 <= self.drop_path_rate < 1:
            raise ValueError("model drop_path_rate must be in [0, 1)")
        head_dim = self.dim // self.n_heads
        if self.dim % self.n_heads or head_dim < MIN_HEAD_DIM or head_dim % 2:
            raise ValueError(
                f"model dim must produce an even head dimension of at least {MIN_HEAD_DIM}"
            )
        if self.decoder_head_dim < MIN_HEAD_DIM or self.decoder_head_dim % 2:
            raise ValueError(
                f"model decoder_head_dim must be even and at least {MIN_HEAD_DIM}, "
                f"got {self.decoder_head_dim}"
            )
        if self.decoder_dim % self.decoder_head_dim:
            raise ValueError("model decoder_dim must be divisible by model decoder_head_dim")

    def _validate_wavelengths(self) -> None:
        spatial_min = self.resolved_pos_min_wavelength
        spatial_max = self.pos_max_wavelength
        if not (math.isfinite(spatial_max) and spatial_min < spatial_max):
            raise ValueError(
                f"model needs pos_max_wavelength above {spatial_min}, got {spatial_max}"
            )

    def __post_init__(self) -> None:
        self._validate_architecture()
        self._validate_wavelengths()


@dataclass
class DataConfig:
    """One flat shard-URL list; coordinate spaces are just more URLs."""

    urls: list[str] = field(
        default_factory=lambda: ["s3://openneuro/v1/openneuro-pretrain-shards-v6/train/native"]
    )
    batch_size: int = 256
    context_patches: int = 48
    target_patches: int = 48
    shuffle_buffer: int = 5
    resume_segment_size: int = 8192
    prefetch: int = 1
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True

    @property
    def min_patches(self) -> int:
        return self.context_patches + self.target_patches

    def __post_init__(self) -> None:
        if not self.urls:
            raise ValueError("data urls must not be empty")
        if len(set(self.urls)) != len(self.urls):
            raise ValueError("data urls must not contain duplicates")
        if any(value <= 0 for value in (self.batch_size, self.context_patches, self.target_patches)):
            raise ValueError("batch_size, context_patches, and target_patches must be positive")
        if (
            self.shuffle_buffer < 0
            or self.resume_segment_size <= 0
            or self.prefetch < 0
            or self.num_workers < 0
        ):
            raise ValueError(
                "shuffle_buffer, prefetch, and num_workers must be non-negative; "
                "resume_segment_size must be positive"
            )
        if self.num_workers and self.prefetch == 0:
            raise ValueError("data prefetch must be positive when num_workers > 0")


@dataclass
class PreprocConfig:
    """Every clip is z-scored per frame across the subject's cortical grey matter."""

    patch_valid_fraction_threshold: float = 0.1
    clip_value: float = 3.0
    clip_frames: int = 16

    def __post_init__(self) -> None:
        if not 0 <= self.patch_valid_fraction_threshold < 1:
            raise ValueError("preproc patch_valid_fraction_threshold must be in [0, 1)")
        if not math.isfinite(self.clip_value) or self.clip_value <= 0 or self.clip_frames <= 0:
            raise ValueError("preproc clip_value and clip_frames must be positive")


@dataclass
class OptimConfig:
    """Use ``base_lr * sqrt(effective_batch / 256)`` before scheduling."""

    base_lr: float = 1e-4
    weight_decay: float = 1e-5
    steps: int = 1000
    warmup_steps: int = 0
    # Per-module global norm; 0 disables clipping.
    grad_clip: float = 2.0
    # Resolved by Config before lr can be read.
    effective_batch: int | None = field(default=None, init=False)

    @property
    def lr(self) -> float:
        return self.base_lr * math.sqrt(self.effective_batch / LR_REFERENCE_BATCH)

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.base_lr)
            or not math.isfinite(self.weight_decay)
            or self.base_lr <= 0
            or self.weight_decay < 0
            or self.steps <= 0
        ):
            raise ValueError("optim requires base_lr > 0, weight_decay >= 0, and steps > 0")
        if self.warmup_steps < 0:
            raise ValueError("optim warmup_steps must be non-negative")
        if self.warmup_steps >= self.steps:
            raise ValueError("optim warmup_steps must be smaller than optim steps")
        if not math.isfinite(self.grad_clip) or self.grad_clip < 0:
            raise ValueError("optim grad_clip must be non-negative; 0 disables it")


@dataclass
class ValidationConfig:
    """Periodic scoring over deterministic prefixes of the held-out streams."""

    enabled: bool = True
    every_n_steps: int = 5000
    # Set name -> shard URL; each logs as val/<name>/*. The HCP-YA configs add their val splits.
    sets: dict[str, str] = field(default_factory=lambda: {"openneuro_heldout": HELD_OUT_URL})
    # None uses data.batch_size.
    batch_size: int | None = None
    num_workers: int = 2
    max_batches: int = 32

    def __post_init__(self) -> None:
        if self.every_n_steps <= 0:
            raise ValueError("validation every_n_steps must be positive")
        if not self.sets:
            raise ValueError("validation sets must name at least one shard URL")
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("validation batch_size must be positive")
        if self.num_workers < 0 or self.max_batches <= 0:
            raise ValueError("validation num_workers must be non-negative and max_batches positive")


@dataclass
class CheckpointConfig:
    save_path: Path | None = Path("checkpoints/last.ckpt")
    load_path: Path | None = None
    every_n_steps: int = 100
    # How many ``<step>.ckpt`` files to retain; 0 keeps every one of them. A
    # 200k-step run at every_n_steps=5000 leaves 40 files of ~4.1 GB.
    keep_last: int = 3

    def __post_init__(self) -> None:
        if self.every_n_steps <= 0:
            raise ValueError("checkpoint every_n_steps must be positive")
        if self.keep_last < 0:
            raise ValueError("checkpoint keep_last must be non-negative")


@dataclass
class TrainingConfig:
    compile: bool = False
    precision: Literal[None, "32-true", "bf16-mixed"] = None
    accelerator: str = "cpu"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("training seed must be non-negative")
        if self.precision not in PRECISIONS:
            # fp16 is incompatible with clipping before scaler unscaling.
            raise ValueError(f"training precision must be one of {PRECISIONS}, got {self.precision!r}")


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "fmri-pretrain"
    entity: str | None = None
    name: str | None = None
    id: str | None = None
    save_dir: Path = Path(".")
    offline: bool = False
    log_every_n_steps: int = 1

    def __post_init__(self) -> None:
        if not self.project:
            raise ValueError("wandb project must not be empty")
        if self.log_every_n_steps <= 0:
            raise ValueError("wandb log_every_n_steps must be positive")


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    preproc: PreprocConfig = field(default_factory=PreprocConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def __post_init__(self) -> None:
        if self.preproc.clip_frames % self.model.temporal_patch_size:
            raise ValueError(
                f"preproc clip_frames ({self.preproc.clip_frames}) must be "
                f"divisible by model temporal_patch_size "
                f"({self.model.temporal_patch_size})"
            )
        # replace(Config) shares sections; copy optim before resolving its batch.
        self.optim = replace(self.optim)
        self.optim.effective_batch = self.data.batch_size

        training_urls = {
            expanded.rstrip("/") for configured in self.data.urls for expanded in expand_urls(configured)
        }
        overlap = sorted(
            training
            for held_out in (url.rstrip("/") for url in self.validation.sets.values())
            for training in training_urls
            if training == held_out
            or training.startswith(f"{held_out}/")
            or held_out.startswith(f"{training}/")
        )
        if self.validation.enabled and overlap:
            raise ValueError(
                f"data urls include the held-out validation set {overlap}; "
                "a validation number measured on training data is not one"
            )


def make_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="fmrifm.train",
        description="Native-space fMRI masked-autoencoder pretraining.",
    )
    parser.add_argument(
        "--config",
        action=ActionConfigFile,
        help="YAML or JSON config file; repeat it to layer files.",
    )
    for name, cls in (
        ("model", ModelConfig),
        ("data", DataConfig),
        ("preproc", PreprocConfig),
        ("optim", OptimConfig),
        ("validation", ValidationConfig),
        ("checkpoint", CheckpointConfig),
        ("training", TrainingConfig),
        ("wandb", WandbConfig),
    ):
        parser.add_class_arguments(cls, nested_key=name, instantiate=True)
    return parser


def parse_config(args: Sequence[str] | None = None) -> Config:
    parser = make_parser()
    parsed = parser.instantiate(parser.parse_args(args))
    return Config(
        model=parsed.model,
        data=parsed.data,
        preproc=parsed.preproc,
        optim=parsed.optim,
        validation=parsed.validation,
        checkpoint=parsed.checkpoint,
        training=parsed.training,
        wandb=parsed.wandb,
    )


def as_dict(config: Config) -> dict:
    """Serialize a resolved snapshot for logging, not for :func:`parse_config`."""

    def stringify_paths(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: stringify_paths(item) for key, item in value.items()}
        if isinstance(value, list):
            return [stringify_paths(item) for item in value]
        return value

    resolved = stringify_paths(asdict(config))
    # Record resolved values rather than only their inputs.
    resolved["optim"]["lr"] = config.optim.lr
    resolved["model"]["pos_min_wavelength"] = config.model.resolved_pos_min_wavelength
    return resolved
