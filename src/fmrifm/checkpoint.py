import os
import shutil
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file


def _validate_resume_contents(
    checkpoint: dict,
    *,
    path: str | Path,
    probe: torch.nn.Module | None,
    dataloader,
) -> None:
    required = {"model", "optimizer", "step", "rng"}
    if probe is not None:
        required.add("probe")
    if dataloader is not None:
        required.add("data")
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise ValueError(f"{path} is missing exact-resume state: {', '.join(missing)}")


def _rng_state() -> dict:
    state = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict) -> None:
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_save(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(state, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_alias(source: Path, target: Path) -> None:
    """Atomically point ``target`` at ``source``, copying if links are unavailable."""
    if source == target:
        return
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    temporary_path.unlink()
    try:
        try:
            os.link(source, temporary_path)
        except OSError:
            shutil.copyfile(source, temporary_path)
        os.replace(temporary_path, target)
    finally:
        temporary_path.unlink(missing_ok=True)


def numbered_checkpoints(directory: str | Path) -> list[Path]:
    """Every ``<step>.ckpt`` in ``directory``, newest step first."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    numbered = [p for p in directory.glob("*.ckpt") if p.stem.isdecimal()]
    return sorted(numbered, key=lambda p: int(p.stem), reverse=True)


def prune_checkpoints(path: str | Path, keep_last: int) -> list[Path]:
    """Delete all but the newest ``keep_last`` numbered checkpoints; return what went.

    ``keep_last <= 0`` keeps everything. Never raises: a multi-day run must not die
    because a stale checkpoint could not be unlinked.
    """
    if keep_last <= 0:
        return []
    last_path = Path(path)
    removed: list[Path] = []
    for candidate in numbered_checkpoints(last_path.parent)[keep_last:]:
        if candidate == last_path:
            continue
        try:
            candidate.unlink()
            removed.append(candidate)
        except OSError:
            continue
    return removed


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    *,
    probe: torch.nn.Module | None = None,
    data_state: dict | None = None,
    keep_last: int | None = None,
) -> None:
    """Atomically save both ``<step>.ckpt`` and the configured stable alias.

    ``keep_last`` prunes older step checkpoints once the new one is durable;
    ``None`` keeps every checkpoint.
    """
    last_path = Path(path)
    step_path = last_path.with_name(f"{step}.ckpt")
    last_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "rng": _rng_state(),
    }
    if probe is not None:
        state["probe"] = probe.state_dict()
    if data_state is not None:
        state["data"] = data_state
    _atomic_save(state, step_path)
    _atomic_alias(step_path, last_path)
    # Only after the new checkpoint is durable and the alias points at it.
    if keep_last is not None:
        prune_checkpoints(last_path, keep_last)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    probe: torch.nn.Module | None = None,
    dataloader=None,
) -> int:
    """Restore exact-resume state and return its step."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    _validate_resume_contents(
        checkpoint,
        path=path,
        probe=probe,
        dataloader=dataloader,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    if probe is not None:
        probe.load_state_dict(checkpoint["probe"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    if dataloader is not None:
        dataloader.load_state_dict(checkpoint["data"])
    _restore_rng_state(checkpoint["rng"])
    return int(checkpoint["step"])


def strip_wrapper_prefixes(state: dict) -> dict:
    # Compiled/Fabric checkpoints add wrapper prefixes absent from eager eval.
    for prefix in ("_forward_module.", "module.", "_orig_mod."):
        while state and all(key.startswith(prefix) for key in state):
            state = {key.removeprefix(prefix): value for key, value in state.items()}
    return state


def load_weights(
    path: str | Path,
    model: torch.nn.Module,
    probe: torch.nn.Module | None = None,
) -> None:
    """Strictly restore model weights, stripping known wrapper prefixes.

    A ``.safetensors`` file is an encoder exported by ``fmrifm.export_encoder``: it
    restores ``model.encoder`` alone, so the decoder keeps its initialisation.
    """
    if Path(path).suffix == ".safetensors":
        if probe is not None:
            raise ValueError(
                f"{path} holds encoder weights only; the probe and validation need a training checkpoint"
            )
        model.encoder.load_state_dict(load_file(path), strict=True)
        return
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(strip_wrapper_prefixes(checkpoint["model"]), strict=True)
    if probe is not None:
        if checkpoint.get("probe") is None:
            raise ValueError(f"{path} has no online probe to restore")
        probe.load_state_dict(strip_wrapper_prefixes(checkpoint["probe"]), strict=True)
