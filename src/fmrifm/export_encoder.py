"""Export a training checkpoint's encoder as safetensors, beside the config that rebuilds it.

``python -m fmrifm.export_encoder <ckpt> <out_dir> [--config <yaml>]`` writes
``<out_dir>/model.safetensors``, the ``model.encoder`` state dict as stored, and
``<out_dir>/config.yaml``, the ``model`` and ``preproc`` sections evaluation reads.
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file

from fmrifm.checkpoint import strip_wrapper_prefixes
from fmrifm.config import parse_config
from fmrifm.model import build_model


def export_encoder(checkpoint: Path, out_dir: Path, config_path: Path | None = None) -> Path:
    weights = out_dir / "model.safetensors"
    # Exporting into a run directory would replace the config its resume reads.
    if weights.exists() or (out_dir / "config.yaml").exists():
        raise SystemExit(f"{out_dir} already holds model.safetensors or config.yaml; choose a new directory")
    run_config = checkpoint.parent.parent / "config.yaml"
    config_path = config_path or run_config
    try:
        config = parse_config(["--config", str(config_path)])
    except SystemExit as error:
        # A run config written by the earlier research code carries keys this parser rejects.
        raise SystemExit(
            f"cannot parse {config_path} (error above); pass the model config the checkpoint "
            "was trained with, e.g. --config configs/model.yaml"
        ) from error
    sections = {"model": asdict(config.model), "preproc": asdict(config.preproc)}
    if run_config.is_file() and config_path.resolve() != run_config.resolve():
        # The checkpoint stores no preproc, so the run's own record is the only check on it.
        stored = yaml.safe_load(run_config.read_text()) or {}
        drift = [
            f"{section}.{key}"
            for section, values in sections.items()
            for key, value in values.items()
            if isinstance(stored.get(section), dict)
            and key in stored[section]
            and stored[section][key] != value
        ]
        if drift:
            raise SystemExit(f"{config_path} disagrees with {run_config} on {', '.join(drift)}")

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model_state = strip_wrapper_prefixes(state["model"])
    model = build_model(config)
    built = {name: buffer.clone() for name, buffer in model.named_buffers() if name.endswith("wavelengths")}
    # Strict, so a config whose shapes do not describe this checkpoint fails here, not at load time.
    model.load_state_dict(model_state, strict=True)
    # The rotary wavelengths are the one config value the weights record beyond shapes.
    drift = [
        name
        for name, buffer in built.items()
        if not torch.allclose(buffer.to(model_state[name].dtype), model_state[name])
    ]
    if drift:
        raise SystemExit(f"{config_path} does not describe this checkpoint: {', '.join(drift)} differ")
    encoder = {
        key.removeprefix("encoder."): value.contiguous()
        for key, value in model_state.items()
        if key.startswith("encoder.")
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"format": "pt", "artifact": "fmrifm-encoder"}
    if "step" in state:
        metadata["step"] = str(int(state["step"]))
    save_file(encoder, weights, metadata=metadata)
    (out_dir / "config.yaml").write_text(
        "# Rebuilds the encoder in model.safetensors through fmrifm.config.parse_config.\n"
        + yaml.safe_dump(sections, sort_keys=False)
    )
    return weights


def main(args=None) -> None:
    parser = argparse.ArgumentParser(
        prog="fmrifm.export_encoder",
        description="Write a checkpoint's encoder weights and model config for release.",
        allow_abbrev=False,
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        help="run config the checkpoint was trained with (default: <ckpt>/../../config.yaml)",
    )
    known = parser.parse_args(args)
    print(export_encoder(known.checkpoint, known.out_dir, known.config))


if __name__ == "__main__":
    main()
