"""The shard-on-disk contract. Every failure here is silent: the reader's default
handler warns and continues, so a shard the reader misreads drops out of the stream
instead of stopping the run."""

import io
import json
import tarfile

import numpy as np
import pytest

from fmrifm.pretrain_data import SHARD_SCHEMA_VERSION, ShardReader, world_coordinates

# A 1 mm T1 and a 2 mm BOLD grid, corner-anchored on one origin. The writer packs
# these as .npy members beside the masks, NOT as JSON keys -- a reader that looks
# in the sidecar finds nothing and rejects every subject in the corpus.
BOLD_AFFINE = np.array([[-2.0, 0, 0, 96.0], [0, 2.0, 0, -132.0], [0, 0, 2.0, -78.0], [0, 0, 0, 1.0]])


def _add(tar, name, value):
    value = value if isinstance(value, bytes) else json.dumps(value).encode()
    info = tarfile.TarInfo(name)
    info.size = len(value)
    tar.addfile(info, io.BytesIO(value))


def _npy(array):
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


def write_shard(path, subject="sub-01", runs=("run-a",), *, include_affine=True):
    shape = (2, 2, 2)
    idx = np.arange(int(np.prod(shape)), dtype=np.int32)
    with tarfile.open(path, "w") as tar:
        arrays = {
            "bold_idx": idx,
            "bold_gm": np.ones(len(idx), np.uint8),
            "bold_cortical_gm": np.ones(len(idx), np.uint8),
        }
        if include_affine:
            arrays["bold_affine"] = BOLD_AFFINE.astype(np.float32)
        for field, values in arrays.items():
            _add(tar, f"{subject}.{field}.npy", _npy(values))
        _add(
            tar,
            f"{subject}.json",
            {
                "kind": "bundle",
                "schema_version": SHARD_SCHEMA_VERSION,
                "bold_shape": shape,
                "sex": "female",
            },
        )
        for run_index, run in enumerate(runs):
            # Values are stored int8 and multiplied back by a per-run scale.
            values = np.full((len(idx), 2), run_index + 2, np.int8)
            _add(tar, f"{subject}__{run}.values.npy", _npy(values))
            _add(tar, f"{subject}__{run}.json", {"kind": "bold", "scale": 0.5})
    return path


def test_the_bundle_carries_geometry_from_npy_members_and_not_from_the_sidecar(tmp_path):
    shard = write_shard(tmp_path / "sub-01.tar")
    reader = ShardReader()

    bundle = reader.decode_sample(next(reader.sample_stream(shard.as_posix())))

    assert bundle["_kind"] == "bundle"
    assert bundle["__key__"] == "sub-01"
    assert bundle["bold_shape"] == (2, 2, 2)
    np.testing.assert_allclose(bundle["bold_affine"], BOLD_AFFINE)
    # Geometry and schema keys are stripped; subject labels are what survives.
    assert bundle["meta"] == {"sex": "female"}


def test_a_shard_without_the_affine_member_fails_loudly(tmp_path):
    shard = write_shard(tmp_path / "sub-01.tar", include_affine=False)
    reader = ShardReader()

    with pytest.raises(ValueError, match="missing the .* array member"):
        reader.decode_sample(next(reader.sample_stream(shard.as_posix())))


def test_run_members_are_subject_prefixed_and_rescaled_from_int8(tmp_path):
    shard = write_shard(tmp_path / "sub-01.tar", runs=("run-a", "run-b"))
    reader = ShardReader()

    samples = [reader.decode_sample(raw) for raw in reader.sample_stream(shard.as_posix())]
    runs = [sample for sample in samples if sample["_kind"] == "bold"]

    assert [sample["__key__"] for sample in runs] == ["sub-01__run-a", "sub-01__run-b"]
    np.testing.assert_allclose(runs[0]["bold_values"], 1.0)
    np.testing.assert_allclose(runs[1]["bold_values"], 1.5)


def test_voxel_indices_become_world_millimetres_through_the_affine():
    """A transposed or mis-offset affine yields finite coordinates and a finite loss.

    Nothing downstream notices: the model trains happily on scrambled geometry, and it
    is the native-vs-MNI claim specifically that the scrambling would invalidate.
    """
    origin = world_coordinates(np.zeros((1, 3)), BOLD_AFFINE)
    np.testing.assert_allclose(origin, [[96.0, -132.0, -78.0]])

    # One voxel step along each axis, in millimetres: x is flipped, y and z are not.
    steps = world_coordinates(np.eye(3), BOLD_AFFINE) - origin
    np.testing.assert_allclose(np.diag(steps), [-2.0, 2.0, 2.0])
