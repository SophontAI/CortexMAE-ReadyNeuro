import glob
import io
import itertools
import json
import logging
import shlex
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np
import s3fs
import torch
import webdataset as wds
from torch.utils.data import IterableDataset, get_worker_info
from webdataset import tariterators

from fmrifm.config import Config
from fmrifm.masking import sample_context_and_target


log = logging.getLogger(__name__)
SHARD_SCHEMA_VERSION = 3
S3_STREAM_ATTEMPTS = 10


class ShardTransportError(RuntimeError):
    """A remote shard could not be replayed exactly after transport failure."""


class ReplayableIterator(Iterator[dict]):
    """Checkpoint a deterministic stream as a bounded segment and offset."""

    def __init__(
        self,
        segment_factory: Callable[[int], Iterator[dict]],
        segment_size: int,
    ) -> None:
        if segment_size <= 0:
            raise ValueError("resume segment size must be positive")
        self.segment_factory = segment_factory
        self.segment_size = segment_size
        self.segment = 0
        self.offset = 0
        self._iterator = self.segment_factory(self.segment)

    def __iter__(self) -> "ReplayableIterator":
        return self

    def __next__(self) -> dict:
        if self.offset == self.segment_size:
            self.segment += 1
            self.offset = 0
            self._iterator = self.segment_factory(self.segment)
        try:
            sample = next(self._iterator)
        except StopIteration as error:
            raise RuntimeError(
                f"resampled data segment {self.segment} ended after {self.offset}/{self.segment_size} samples"
            ) from error
        self.offset += 1
        return sample

    def state_dict(self) -> dict:
        segment, offset = self.segment, self.offset
        if offset == self.segment_size:
            segment += 1
            offset = 0
        return {"segment": segment, "offset": offset}

    def load_state_dict(self, state_dict: dict) -> None:
        segment = int(state_dict["segment"])
        offset = int(state_dict["offset"])
        if segment < 0 or not 0 <= offset < self.segment_size:
            raise ValueError("invalid resampled data cursor")
        self.segment = segment
        self.offset = 0
        self._iterator = self.segment_factory(segment)
        for _ in range(offset):
            next(self)


def _load_npy(value) -> np.ndarray:
    return np.load(io.BytesIO(value), allow_pickle=False)


def volume_to_patches(volume: np.ndarray, patch_size: int) -> np.ndarray:
    depth, height, width = volume.shape
    patch_shape = (
        depth // patch_size,
        patch_size,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
    )
    volume = volume.reshape(patch_shape)
    return volume.transpose(0, 2, 4, 1, 3, 5).reshape(-1, patch_size**3)


@dataclass(frozen=True)
class PatchGrid:
    gather_ids: np.ndarray
    centers: np.ndarray
    valid_voxels: np.ndarray

    def __len__(self) -> int:
        return len(self.gather_ids)

    def gather(self, values: np.ndarray, patch_indices: np.ndarray | None = None) -> np.ndarray:
        gather_ids = self.gather_ids
        valid_voxels = self.valid_voxels
        if patch_indices is not None:
            gather_ids = gather_ids[patch_indices]
            valid_voxels = valid_voxels[patch_indices]
        valid_ids = gather_ids[valid_voxels]
        values = np.asarray(values)
        gathered = np.zeros((*values.shape[:-1], *gather_ids.shape), dtype=values.dtype)
        gathered[..., valid_voxels] = values[..., valid_ids]
        return gathered


def build_patch_grid(mask: np.ndarray, patch_size: int, threshold: float) -> PatchGrid:
    padding = [(0, -length % patch_size) for length in mask.shape]
    padded_mask = np.pad(mask, padding)
    valid_voxels = volume_to_patches(padded_mask, patch_size)

    sparse_ids = np.full(padded_mask.shape, -1, dtype=np.int64)
    sparse_ids[mask.nonzero()] = np.arange(mask.sum())
    gather_ids = volume_to_patches(sparse_ids, patch_size)

    starts = (
        np.mgrid[
            0 : padded_mask.shape[0] : patch_size,
            0 : padded_mask.shape[1] : patch_size,
            0 : padded_mask.shape[2] : patch_size,
        ]
        .reshape(3, -1)
        .T
    )
    centers = starts.astype(np.float32) + (patch_size - 1) / 2

    included_cells = valid_voxels.mean(axis=1) > threshold
    return PatchGrid(
        gather_ids[included_cells],
        centers[included_cells],
        valid_voxels[included_cells],
    )


def world_coordinates(voxel_coordinates: np.ndarray, affine) -> np.ndarray:
    """Map ``[N, 3]`` voxel coordinates into world millimetres.

    Native and MNI origins differ ((96, -132, -78) against (90, -126, -72)), so one
    structure carries different voxel indices in each but nearly the same millimetres.
    """
    affine = np.asarray(affine, dtype=np.float64).reshape(4, 4)
    if not np.isfinite(affine).all():
        raise ValueError("affine must be a finite 4x4 matrix")
    coordinates = np.asarray(voxel_coordinates, dtype=np.float64)
    homogeneous = np.concatenate([coordinates, np.ones((len(coordinates), 1))], axis=1)
    return (homogeneous @ affine.T)[:, :3].astype(np.float32)


def standardize_frames(frames: np.ndarray) -> np.ndarray:
    """Z-score each frame across voxels.

    A second normalisation, on the other axis from the pipeline's per-voxel temporal
    z-score: every clip the model reads has been standardized both ways.
    """
    mean = frames.mean(axis=1, keepdims=True, dtype=np.float64)
    std = frames.std(axis=1, keepdims=True, dtype=np.float64)
    return ((frames - mean) / (std + 1e-6)).astype(np.float32)


def parse_subject_metadata(metadata: dict) -> dict:
    """Age must parse as a positive float and sex as male/female; -1 drops the row."""
    age = -1.0
    try:
        parsed_age = float(metadata.get("age"))
        if np.isfinite(parsed_age) and parsed_age > 0:
            age = parsed_age
    except (TypeError, ValueError):
        pass

    sex = {"male": 0, "m": 0, "female": 1, "f": 1}.get(str(metadata.get("sex", "")).lower(), -1)
    return {"age": age, "sex": sex}


@dataclass(frozen=True)
class PatchLayout:
    bold_rows: np.ndarray
    bold_grid: PatchGrid
    # World millimetres; the grid's own centers stay in voxel indices, which is
    # what gathering needs.
    bold_centers: np.ndarray


@dataclass(frozen=True)
class PatchSampler:
    """Cut a subject into clips and draw one context/target view per clip."""

    context_patches: int
    target_patches: int
    patch_size: int
    clip_frames: int
    clip_value: float
    patch_valid_fraction_threshold: float

    @property
    def min_patches(self) -> int:
        return self.context_patches + self.target_patches

    def build_layout(self, bundle: dict) -> PatchLayout:
        bold_indices = np.asarray(bundle["bold_idx"])
        bold_rows = np.asarray(bundle["bold_cortical_gm"], dtype=bool)

        bold_mask = np.zeros(bundle["bold_shape"], dtype=bool)
        bold_mask.flat[bold_indices[bold_rows]] = True
        bold_grid = build_patch_grid(
            bold_mask, self.patch_size, threshold=self.patch_valid_fraction_threshold
        )
        return PatchLayout(
            bold_rows,
            bold_grid,
            world_coordinates(bold_grid.centers, bundle["bold_affine"]),
        )

    def sample_clips(
        self,
        bold_block: dict,
        layout: PatchLayout,
        rng: np.random.Generator,
    ) -> Iterator[dict]:
        run = np.asarray(bold_block["bold_values"])[layout.bold_rows].T
        for clip_start in range(0, len(run) - self.clip_frames + 1, self.clip_frames):
            clip = run[clip_start : clip_start + self.clip_frames].astype(np.float32)
            yield self.sample_view(standardize_frames(clip), layout, rng)

    def sample_view(self, clip: np.ndarray, layout: PatchLayout, rng: np.random.Generator) -> dict:
        # Draw through a torch generator seeded from the replayable worker stream.
        generator = torch.Generator().manual_seed(int(rng.integers(1 << 62)))
        context_index, target_index = sample_context_and_target(
            layout.bold_centers,
            self.context_patches,
            self.target_patches,
            generator,
        )

        def patches(indices: np.ndarray) -> np.ndarray:
            gathered = layout.bold_grid.gather(clip, indices)
            return np.clip(gathered, -self.clip_value, self.clip_value).astype(np.float32)

        return {
            "x": patches(context_index),
            "coord": layout.bold_centers[context_index],
            "target": patches(target_index),
            "target_coord": layout.bold_centers[target_index],
            "target_mask": layout.bold_grid.valid_voxels[target_index],
        }


def build_sampler(config: Config) -> PatchSampler:
    return PatchSampler(
        context_patches=config.data.context_patches,
        target_patches=config.data.target_patches,
        patch_size=config.model.patch_size,
        clip_frames=config.preproc.clip_frames,
        clip_value=config.preproc.clip_value,
        patch_valid_fraction_threshold=config.preproc.patch_valid_fraction_threshold,
    )


@dataclass(frozen=True)
class ShardReader:
    def stream_url(self, url: str) -> str:
        if not url.startswith("s3://"):
            return url
        return f"pipe:{shlex.join(['aws', 's3', 'cp', '--only-show-errors', url, '-'])}"

    def sample_stream(self, url: str) -> Iterator[dict]:
        source = [{"url": self.stream_url(url)}]
        return iter(tariterators.tarfile_to_samples(handler=wds.reraise_exception)(source))

    def list_shards(self, url: str) -> list[str]:
        if "{" in url:
            # Brace expansion names *sources*, not necessarily shards. A brace over
            # bucket prefixes -- which is how the OpenNeuro batches are written --
            # expands to directories that still have to be listed; returning the
            # expansion directly streams the prefix, and `aws s3 cp <prefix> -` 404s.
            return [
                shard
                for expanded in wds.shardlists.expand_urls(url)
                for shard in self.list_shards(expanded)
            ]
        if url.endswith(".tar"):
            return [url]
        if url.startswith("s3://"):
            paths = map(str, sorted(s3fs.S3FileSystem().glob(f"{url.rstrip('/')}/*.tar")))
            return [path if path.startswith("s3://") else f"s3://{path}" for path in paths]
        return sorted(glob.glob(f"{url.rstrip('/')}/*.tar"))

    def decode_sample(self, sample: dict) -> dict:
        metadata = json.loads(sample["json"])
        key = sample.get("__key__")

        if metadata.get("kind", "bold") == "bundle":
            version = metadata.get("schema_version")
            if version != SHARD_SCHEMA_VERSION:
                raise ValueError(f"shard schema {version!r} does not match {SHARD_SCHEMA_VERSION}")
            # bold_affine is not optional: patch centers are world millimetres, so
            # every subject needs its grid's geometry. The masks and the affine are
            # .npy tar members named for the lowercased field, not JSON keys --
            # reading them out of the metadata finds nothing and rejects the corpus.
            fields = ["bold_idx", "bold_cortical_gm", "bold_affine"]
            try:
                bundle = {name: _load_npy(sample[f"{name}.npy"]) for name in fields}
            except KeyError as error:
                raise ValueError(f"shard is missing the {error} array member") from error
            bundle["bold_shape"] = tuple(metadata["bold_shape"])
            reserved = {"kind", "schema_version", "bold_shape"}
            bundle["meta"] = {name: value for name, value in metadata.items() if name not in reserved}
            return bundle | {"_kind": "bundle", "__key__": key}

        raw_values = _load_npy(sample["values.npy"])
        bold_values = (raw_values.astype(np.float32) * np.float32(metadata.get("scale", 1.0))).astype(
            np.float16
        )
        return {"_kind": "bold", "__key__": key, "bold_values": bold_values}

    def read_shard(self, url: str) -> Iterator[dict]:
        is_s3 = url.startswith("s3://")
        attempts = S3_STREAM_ATTEMPTS if is_s3 else 1
        emitted_keys: list[str | None] = []
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            replay_index = 0
            try:
                for sample in self.sample_stream(url):
                    try:
                        decoded = self.decode_sample(sample)
                    except Exception as error:
                        key = sample.get("__key__")
                        decode_error = RuntimeError(f"could not decode sample {key!r} from {url}: {error}")
                        if wds.warn_and_continue(decode_error):
                            continue
                        return

                    key = decoded.get("__key__")
                    if not isinstance(key, str):
                        raise ValueError(f"decoded sample from {url!r} has no string __key__")
                    if replay_index < len(emitted_keys):
                        expected = emitted_keys[replay_index]
                        if key != expected:
                            raise ShardTransportError(
                                f"replayed shard {url!r} changed at sample "
                                f"{replay_index}: expected {expected!r}, got {key!r}"
                            )
                        replay_index += 1
                        continue

                    emitted_keys.append(key)
                    replay_index += 1
                    yield decoded

                if replay_index != len(emitted_keys):
                    raise ShardTransportError(
                        f"replayed shard {url!r} ended after {replay_index} of "
                        f"{len(emitted_keys)} previously emitted samples"
                    )
                return
            except ShardTransportError:
                raise
            except Exception as error:
                last_error = error
                if not is_s3:
                    raise
                if attempt == attempts:
                    break
                log.warning(
                    "S3 shard stream failed for %s (attempt %d/%d); reopening "
                    "and verifying %d already emitted samples",
                    url,
                    attempt,
                    attempts,
                    len(emitted_keys),
                )

        raise ShardTransportError(
            f"could not stream S3 shard {url!r} after {attempts} attempts"
        ) from last_error


@dataclass(frozen=True)
class WorkerContext:
    seed: int
    rank: int = 0
    world_size: int = 1
    worker_id: int = 0
    worker_count: int = 1

    def current_worker(self) -> "WorkerContext":
        worker = get_worker_info()
        if worker is None:
            return self
        return WorkerContext(
            self.seed,
            self.rank,
            self.world_size,
            worker.id,
            worker.num_workers,
        )

    def random_generator(self, stream: int = 0, segment: int = 0) -> np.random.Generator:
        entropy = [self.seed, self.rank, self.worker_id, stream]
        if segment:
            entropy.append(segment)
        sequence = np.random.SeedSequence(entropy)
        return np.random.default_rng(sequence)

    def strided_shards(self, items: Sequence[str]) -> list[str]:
        """One finite, disjoint slice of ``items`` per (rank, worker)."""
        index = self.rank * self.worker_count + self.worker_id
        return items[index :: self.world_size * self.worker_count]


def shuffle_samples(samples: Iterable[dict], buffer_size: int, rng: np.random.Generator) -> Iterator[dict]:
    if buffer_size <= 1:
        yield from samples
        return

    buffer = []
    for sample in samples:
        buffer.append(sample)
        if len(buffer) >= buffer_size:
            yield buffer.pop(int(rng.integers(len(buffer))))
    while buffer:
        yield buffer.pop(int(rng.integers(len(buffer))))


@dataclass
class PretrainData(IterableDataset):
    """Infinite resampled training stream, or one finite strided pass for validation."""

    urls: str | Sequence[str]
    sampler: PatchSampler
    seed: int
    rank: int = 0
    world_size: int = 1
    resampled: bool = True
    shuffle_buffer: int = 0
    resume_segment_size: int = 8192

    reader: ShardReader = field(init=False, repr=False)
    worker_context: WorkerContext = field(init=False, repr=False)
    shards: list[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.worker_context = WorkerContext(self.seed, self.rank, self.world_size)
        self.reader = ShardReader()
        urls = [self.urls] if isinstance(self.urls, str) else list(self.urls)
        # First occurrence wins when a shard resolves from more than one entry.
        self.shards = list(
            dict.fromkeys(shard for url in urls for shard in self.reader.list_shards(url))
        )
        if not self.shards:
            raise ValueError(f"no .tar shards found under {urls}")

    def __iter__(self) -> Iterator[dict]:
        worker = self.worker_context.current_worker()
        if self.resampled:
            return ReplayableIterator(
                lambda segment: self._sample_segment(worker, segment),
                self.resume_segment_size,
            )
        sampling_rng = worker.random_generator()
        shards = worker.strided_shards(self.shards)
        samples = (sample for shard in shards for sample in self.sample_subject(shard, sampling_rng))
        return shuffle_samples(samples, self.shuffle_buffer, worker.random_generator(stream=1))

    def _sample_segment(self, worker: WorkerContext, segment: int) -> Iterator[dict]:
        sampling_rng = worker.random_generator(segment=segment)
        shards = (self.shards[int(sampling_rng.integers(len(self.shards)))] for _ in itertools.count())
        samples = (sample for shard in shards for sample in self.sample_subject(shard, sampling_rng))
        finite_segment = itertools.islice(samples, self.resume_segment_size)
        yield from shuffle_samples(
            finite_segment,
            self.shuffle_buffer,
            worker.random_generator(stream=1, segment=segment),
        )

    def sample_subject(self, shard: str, rng: np.random.Generator) -> Iterator[dict]:
        stream = iter(self.reader.read_shard(shard))
        bundle = next(stream, None)
        subject = bundle.get("__key__") if bundle else shard.rsplit("/", 1)[-1]
        if bundle is None or bundle.get("_kind") != "bundle":
            log.warning("%s: shard does not start with a subject bundle", subject)
            return

        layout = self.sampler.build_layout(bundle)
        if len(layout.bold_grid) < self.sampler.min_patches:
            log.warning(
                "%s: only %d patches available; need %d",
                subject,
                len(layout.bold_grid),
                self.sampler.min_patches,
            )
            return

        metadata = parse_subject_metadata(bundle.get("meta", {}))
        for block in stream:
            key = block.get("__key__")
            if block.get("_kind") != "bold" or (
                key is not None and subject is not None and not key.startswith(f"{subject}__")
            ):
                log.warning("%s: skipping unmatched BOLD block %r", subject, key)
                continue
            for clip in self.sampler.sample_clips(block, layout, rng):
                yield clip | metadata
