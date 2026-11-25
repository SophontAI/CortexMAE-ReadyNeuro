"""Resume must restore every piece of state, or a restart silently changes the run."""

import pytest
import torch

from fmrifm.checkpoint import load_checkpoint, prune_checkpoints, save_checkpoint
from fmrifm.pretrain_data import ReplayableIterator


def _save_at(path, step, keep_last=None):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    save_checkpoint(path, model, optimizer, step=step, keep_last=keep_last)


def _numbered(path):
    return sorted(int(p.stem) for p in path.parent.glob("*.ckpt") if p.stem.isdecimal())


def test_checkpoint_round_trips_model_optimizer_step_and_loader_position(tmp_path):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.ones(1, 3)).sum().backward()
    optimizer.step()

    checkpoint = tmp_path / "checkpoints" / "last.ckpt"
    save_checkpoint(
        checkpoint, model, optimizer, step=7, data_state={"cursor": torch.tensor([4, 2])}
    )
    assert checkpoint.is_file() and checkpoint.with_name("7.ckpt").is_file()

    class StatefulData:
        restored = None

        def load_state_dict(self, state):
            self.restored = state

    data = StatefulData()
    restored = torch.nn.Linear(3, 2)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)

    assert load_checkpoint(checkpoint, restored, restored_optimizer, dataloader=data) == 7
    assert torch.equal(data.restored["cursor"], torch.tensor([4, 2]))
    for expected, actual in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.equal(expected, actual)
    assert restored_optimizer.state_dict()["state"]


def test_resume_restores_the_torch_rng(tmp_path):
    """Without this the masking draw after a restart is a different draw."""
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    checkpoint = tmp_path / "last.ckpt"

    torch.manual_seed(123)
    save_checkpoint(checkpoint, model, optimizer, step=1)
    expected = torch.rand(4)
    torch.manual_seed(999)

    load_checkpoint(checkpoint, model, optimizer)

    assert torch.equal(torch.rand(4), expected)


@pytest.mark.parametrize("missing", ["rng", "data"])
def test_resume_refuses_a_checkpoint_missing_state_it_was_asked_to_restore(tmp_path, missing):
    source = torch.nn.Linear(3, 2)
    optimizer = torch.optim.SGD(source.parameters(), lr=0.1)
    state = {
        "model": source.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": 4,
        "rng": {"torch": torch.get_rng_state()},
        "data": {"cursor": 7},
    }
    state.pop(missing)
    checkpoint = tmp_path / "last.ckpt"
    torch.save(state, checkpoint)

    restored = torch.nn.Linear(3, 2)
    original = {name: value.clone() for name, value in restored.state_dict().items()}

    class Data:
        def load_state_dict(self, state):
            raise AssertionError("invalid checkpoint must not mutate the loader")

    with pytest.raises(ValueError, match=rf"missing exact-resume state: {missing}"):
        load_checkpoint(
            checkpoint, restored, torch.optim.SGD(restored.parameters(), lr=0.1), dataloader=Data()
        )
    for name, value in restored.state_dict().items():
        assert torch.equal(value, original[name])


def test_retention_never_orphans_the_resume_alias(tmp_path):
    """last.ckpt is a hard link, so pruning its step file must leave it loadable."""
    checkpoint = tmp_path / "checkpoints" / "last.ckpt"
    for step in (5, 10, 15, 20):
        _save_at(checkpoint, step, keep_last=1)

    assert _numbered(checkpoint) == [20]
    model = torch.nn.Linear(3, 2)
    assert load_checkpoint(checkpoint, model, torch.optim.AdamW(model.parameters())) == 20


def test_pruning_orders_by_step_not_by_filename(tmp_path):
    """String order would delete 100000 before 95000 and keep the wrong three."""
    checkpoint = tmp_path / "checkpoints" / "last.ckpt"
    for step in (5000, 100000, 95000, 10000, 200000):
        _save_at(checkpoint, step)

    prune_checkpoints(checkpoint, keep_last=3)

    assert _numbered(checkpoint) == [95000, 100000, 200000]


def test_a_resumed_stream_replays_to_the_exact_next_sample():
    """An off-by-one re-trains a sample on every requeue, and nothing reports it."""

    def factory(segment):
        return iter(range(segment * 10, segment * 10 + 10))

    stream = ReplayableIterator(factory, 10)
    for _ in range(14):  # past the first segment boundary
        next(stream)
    state = stream.state_dict()
    expected = next(stream)

    resumed = ReplayableIterator(factory, 10)
    resumed.load_state_dict(state)

    assert next(resumed) == expected
