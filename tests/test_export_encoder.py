"""The released encoder weights must reproduce the trained encoder bit for bit."""

import pytest
import torch
from safetensors import safe_open

from brainmarks.models.fmrifm import _make
from fmrifm.checkpoint import load_weights
from fmrifm.config import parse_config
from fmrifm.export_encoder import main
from fmrifm.model import build_model
from fmrifm.probe import OnlineLinearProbe

TINY_YAML = """
model: {dim: 40, n_heads: 2, depth: 2, decoder_dim: 20, decoder_head_dim: 10, decoder_depth: 1,
        patch_size: 2, temporal_patch_size: 2, drop_path_rate: 0.1, pos_max_wavelength: 300.0}
preproc: {clip_frames: 4, clip_value: 2.5, patch_valid_fraction_threshold: 0.3}
"""


def _trained_run(tmp_path):
    """A run directory as training leaves it, saved from a compiled model."""
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "config.yaml").write_text(TINY_YAML)
    model = build_model(parse_config(["--config", str(run / "config.yaml")])).eval()
    state = {f"_orig_mod.{key}": value for key, value in model.state_dict().items()}
    torch.save({"model": state, "step": 7}, run / "checkpoints" / "last.ckpt")
    return model, run


def _encode(model):
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(2, 4, 5, 8, generator=generator)
    coord = torch.randn(2, 5, 3, generator=generator) * 50
    return model(x, coord, valid=torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], dtype=torch.bool))


def test_an_exported_encoder_reproduces_the_trained_one(tmp_path):
    trained, run = _trained_run(tmp_path)
    main([str(run / "checkpoints" / "last.ckpt"), str(tmp_path / "export")])
    weights = tmp_path / "export" / "model.safetensors"

    exported = parse_config(["--config", str(tmp_path / "export" / "config.yaml")])
    original = parse_config(["--config", str(run / "config.yaml")])
    assert (exported.model, exported.preproc) == (original.model, original.preproc)
    with safe_open(weights, "pt") as handle:
        assert handle.metadata()["step"] == "7"
        assert not any(key.startswith(("_orig_mod.", "encoder.", "decoder.")) for key in handle.keys())

    expected = _encode(trained)
    restored = build_model(exported).eval()
    load_weights(weights, restored)
    _, wrapper = _make("native", ckpt_path=weights, compile_model=False)
    assert torch.equal(_encode(restored), expected)
    assert torch.equal(_encode(wrapper.model), expected)


def test_a_full_training_checkpoint_still_loads(tmp_path):
    trained, run = _trained_run(tmp_path)
    restored = build_model(parse_config(["--config", str(run / "config.yaml")])).eval()

    load_weights(run / "checkpoints" / "last.ckpt", restored)

    for expected, actual in zip(trained.state_dict().values(), restored.state_dict().values(), strict=True):
        assert torch.equal(expected, actual)


def test_encoder_weights_cannot_restore_the_probe(tmp_path):
    _, run = _trained_run(tmp_path)
    main([str(run / "checkpoints" / "last.ckpt"), str(tmp_path / "export")])
    model = build_model(parse_config(["--config", str(run / "config.yaml")]))

    with pytest.raises(ValueError, match="encoder weights only"):
        load_weights(tmp_path / "export" / "model.safetensors", model, probe=OnlineLinearProbe(40))


def test_export_keeps_the_stored_dtype(tmp_path):
    _, run = _trained_run(tmp_path)
    checkpoint = run / "checkpoints" / "last.ckpt"
    state = torch.load(checkpoint, weights_only=True)
    state["model"] = {key: value.bfloat16() for key, value in state["model"].items()}
    torch.save(state, checkpoint)

    main([str(checkpoint), str(tmp_path / "export")])

    with safe_open(tmp_path / "export" / "model.safetensors", "pt") as handle:
        assert {handle.get_tensor(key).dtype for key in handle.keys()} == {torch.bfloat16}


def test_export_refuses_a_config_or_directory_that_does_not_fit(tmp_path):
    _, run = _trained_run(tmp_path)
    checkpoint = str(run / "checkpoints" / "last.ckpt")
    other = tmp_path / "other.yaml"

    with pytest.raises(SystemExit, match="already holds"):
        main([checkpoint, str(run)])
    assert (run / "config.yaml").read_text() == TINY_YAML

    other.write_text(TINY_YAML.replace("clip_value: 2.5", "clip_value: 1.0"))
    with pytest.raises(SystemExit, match=r"preproc\.clip_value"):
        main([checkpoint, str(tmp_path / "a"), "--config", str(other)])

    (run / "config.yaml").unlink()  # as for a research run, whose config this code cannot read
    other.write_text(TINY_YAML.replace("300.0", "900.0"))
    with pytest.raises(SystemExit, match="wavelengths"):
        main([checkpoint, str(tmp_path / "b"), "--config", str(other)])
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()
