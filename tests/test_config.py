"""Config facts that would otherwise fail silently: the recipe, the LR, and the corpus URLs."""

import math
from dataclasses import asdict
from pathlib import Path

import pytest
from webdataset.shardlists import expand_urls

from fmrifm.config import HELD_OUT_URL, as_dict, parse_config
from fmrifm.pretrain_utils import learning_rate_at

CONFIG_DIR = Path(__file__).parents[1] / "configs"
CORPORA = sorted(p.name for p in (CONFIG_DIR / "data").glob("*.yaml"))

# Every knob the paper ran with. A snapshot rather than a field list, so a knob that
# drifts cannot pass by not being one of the ones somebody remembered to assert.
PAPER_CONFIG = {
    "model": {
        "dim": 1024,
        "n_heads": 16,
        "depth": 24,
        "decoder_dim": 512,
        "decoder_head_dim": 64,
        "decoder_depth": 12,
        "patch_size": 6,
        "temporal_patch_size": 4,
        "drop_path_rate": 0.1,
        "pos_max_wavelength": 640.0,
        "pos_min_wavelength": 2 * math.pi * 6 * 2.0,
    },
    "data": {
        "batch_size": 128,
        "context_patches": 96,
        "target_patches": 32,
        "shuffle_buffer": 1000,
        "resume_segment_size": 8192,
        "prefetch": 2,
        "num_workers": 16,
        "pin_memory": True,
        "persistent_workers": True,
    },
    "preproc": {"patch_valid_fraction_threshold": 0.1, "clip_value": 3.0, "clip_frames": 16},
    "optim": {
        "base_lr": 1e-4,
        "weight_decay": 0.05,
        "steps": 200_000,
        "warmup_steps": 5_000,
        "grad_clip": 2.0,
        "effective_batch": 128,
        "lr": 1e-4 * math.sqrt(128 / 256),
    },
    "validation": {
        "enabled": True,
        "every_n_steps": 5000,
        "sets": {"openneuro_heldout": HELD_OUT_URL},
        "batch_size": None,
        "num_workers": 2,
        "max_batches": 32,
    },
    "checkpoint": {
        "save_path": "checkpoints/last.ckpt",
        "load_path": None,
        "every_n_steps": 5000,
        "keep_last": 3,
    },
    "training": {"compile": True, "precision": "bf16-mixed", "accelerator": "cuda", "seed": 42},
    "wandb": {
        "enabled": False,
        "project": "fmri-pretrain",
        "entity": None,
        "name": None,
        "id": None,
        "save_dir": ".",
        "offline": False,
        "log_every_n_steps": 50,
    },
}


def preset(*names: str, extra: list[str] | None = None):
    args = []
    for name in names:
        args += ["--config", str(CONFIG_DIR / name)]
    return parse_config(args + list(extra or []))


def test_the_paper_model_config_is_what_the_paper_says():
    resolved = as_dict(preset("model.yaml"))
    resolved["data"].pop("urls")

    assert resolved == PAPER_CONFIG


@pytest.mark.parametrize("batch_size,expected", [(256, 1e-4), (1024, 2e-4)])
def test_the_learning_rate_scales_with_the_root_of_the_batch(batch_size, expected):
    """base_lr is defined at a reference batch of 256; every other batch rescales."""
    config = parse_config([f"--data.batch_size={batch_size}"])

    assert config.optim.base_lr == pytest.approx(1e-4)
    assert config.optim.lr == pytest.approx(expected, rel=1e-4)


def test_the_schedule_warms_up_to_the_peak_and_decays_to_a_one_percent_floor():
    """A wrong ramp or a missing floor still converges, just not to the paper's number."""
    peak, steps, warmup = 1e-4, 1000, 100

    def lr(step):
        return learning_rate_at(step, peak_lr=peak, steps=steps, warmup_steps=warmup)

    assert lr(0) == pytest.approx(peak / warmup), "the first step must not train at lr=0"
    assert lr(warmup - 1) == pytest.approx(peak), "the ramp must land exactly on the peak"
    assert lr(steps - 1) == pytest.approx(0.01 * peak), "cosine decays to the floor, not to zero"
    decaying = [lr(step) for step in range(warmup, steps, 100)]
    assert all(a > b for a, b in zip(decaying, decaying[1:], strict=False))


def test_a_run_may_not_train_on_its_own_validation_set():
    with pytest.raises(ValueError, match="held-out validation set"):
        parse_config([f"--data.urls=[{HELD_OUT_URL}]"])
    # A parent prefix reaches the held-out split too, so the guard is on prefixes.
    with pytest.raises(ValueError, match="held-out validation set"):
        parse_config([f"--data.urls=[{HELD_OUT_URL.rsplit('/', 2)[0]}]"])


@pytest.mark.parametrize("corpus", ["hcp_native.yaml", "hcp_mni.yaml"])
def test_hcp_runs_score_the_paper_validation_sets_and_guard_all_of_them(corpus):
    """The paper's HCP-YA held-out rows need the HCP val splits, not only b31."""
    config = preset("model.yaml", f"data/{corpus}")
    assert list(config.validation.sets) == ["hcp_native", "hcp_mni", "openneuro_heldout"]
    with pytest.raises(ValueError, match="held-out validation set"):
        preset("model.yaml", f"data/{corpus}", extra=[f"--data.urls=[{config.validation.sets['hcp_mni']}]"])


@pytest.mark.parametrize("corpus", CORPORA)
def test_every_shipped_corpus_parses_and_avoids_the_validation_split(corpus):
    """Parsing runs the overlap guard, so a corpus that reached the holdout raises here."""
    config = preset("model.yaml", f"data/{corpus}")
    assert config.data.urls


def test_the_full_corpus_brace_expands_to_31_training_roots():
    """A naive quote edit collapses the brace expansion to one URL that matches nothing."""
    config = preset("model.yaml", "data/openneuro_22373h.yaml")
    roots = expand_urls(config.data.urls[0])

    assert len(roots) == 31
    assert roots[0].endswith("openneuro-pretrain-shards-v6/train/native")
    assert not any(root.endswith("-b31/train/native") for root in roots)


def test_the_two_hcp_corpora_differ_only_in_their_coordinate_space():
    """One stray knob in either file would make the headline contrast uncontrolled."""
    native = asdict(preset("model.yaml", "data/hcp_native.yaml"))
    mni = asdict(preset("model.yaml", "data/hcp_mni.yaml"))

    assert native["data"]["urls"][0].endswith("/train/native")
    assert mni["data"]["urls"][0].endswith("/train/mni")
    native["data"]["urls"] = mni["data"]["urls"] = None
    assert native == mni
