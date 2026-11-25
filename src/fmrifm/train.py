from pathlib import Path

import torch
import torch.nn as nn
from lightning.fabric import Fabric
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import trange

from fmrifm.checkpoint import load_checkpoint, save_checkpoint
from fmrifm.config import Config, parse_config
from fmrifm.model import build_model
from fmrifm.performance import PerformanceTracker, estimate_training_flops
from fmrifm.pretrain_data import PretrainData, build_sampler
from fmrifm.probe import OnlineLinearProbe
from fmrifm.pretrain_utils import (
    _compute_metrics,
    _make_metrics,
    _make_wandb_logger,
    _probe_objective,
    _reset_metrics,
    _update_metrics,
    learning_rate_at,
)
from fmrifm.validate import PeriodicValidation


def _build_model_and_probe(config: Config) -> tuple[nn.Module, nn.Module]:
    """Initialize both modules from fixed, independent seeds."""
    with torch.random.fork_rng():
        torch.manual_seed(config.training.seed)
        model = build_model(config)
    with torch.random.fork_rng():
        torch.manual_seed(config.training.seed + 1)
        probe = OnlineLinearProbe(config.model.dim)
    return model, probe


def _abort_nonfinite(reason, *, step, save_path, model, optimizer, probe, data_state, batch):
    """Dump the batch, gradients and pre-update state, then stop the run.

    The dump lands in its own ``nan_dump_<step>/`` beside ``checkpoints/``, never
    among them, where ``prune_checkpoints`` and the ``last.ckpt`` alias live.
    """
    if save_path:
        dump = Path(save_path).parent.parent / f"nan_dump_{step}"
        try:
            # Batch first: it is the small, robust half of the dump, while the
            # loader snapshot below is the part most likely to fail -- this runs
            # when things are already broken, and workers may be gone.
            dump.mkdir(parents=True, exist_ok=True)
            torch.save(batch, dump / "batch.pt")
            grads = {
                owner: {
                    name: parameter.grad.detach().cpu()
                    for name, parameter in module.named_parameters()
                    if parameter.grad is not None
                }
                for owner, module in (("model", model), ("probe", probe))
            }
            torch.save(grads, dump / "grads.pt")
            culprits = sorted(
                f"{owner}.{name}"
                for owner, tensors in grads.items()
                for name, tensor in tensors.items()
                if not torch.isfinite(tensor).all()
            )
            if culprits:
                reason += f"; {len(culprits)} non-finite gradients, first: {', '.join(culprits[:5])}"
            snapshot = data_state.state_dict() if hasattr(data_state, "state_dict") else None
            save_checkpoint(dump / "last.ckpt", model, optimizer, step, probe=probe, data_state=snapshot)
        except Exception as dump_error:  # a failed dump must never mask the reason
            reason += f"; dump to {dump} failed: {dump_error!r}"
        else:
            reason += f"; dumped pre-update state and the batch to {dump}"
    raise RuntimeError(reason)


def _shutdown_loader(loader) -> None:
    """Stop persistent workers that Fabric's wrapper otherwise keeps alive."""
    dataloader = getattr(loader, "_dataloader", loader)
    iterator = getattr(dataloader, "_iterator", None)
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if shutdown is not None:
        shutdown()
        dataloader._iterator = None


def _clip_gradients(module: nn.Module, max_norm: float) -> torch.Tensor | None:
    """Clip, and return the pre-clip global norm that says whether it bound."""
    parameters = [parameter for parameter in module.parameters() if parameter.grad is not None]
    if not parameters:
        return None
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm if max_norm else float("inf"))


def _require_single_process(fabric) -> None:
    world_size = int(getattr(fabric, "world_size", 1))
    if world_size != 1:
        raise RuntimeError(
            "training supports one process only; distributed loss normalization "
            "and rank-local checkpoint state are not implemented"
        )


def _reconstruction_step(model, probe, fabric, batch: dict):
    recon_loss, hidden, _prediction, _target = model(
        x=batch["x"].float(),
        coord=batch["coord"].float(),
        target=batch["target"].float(),
        target_coord=batch["target_coord"].float(),
        target_voxel_mask=batch["target_mask"].bool(),
    )
    fabric.backward(recon_loss)
    # The probe detaches its input, so its gradients never reach the encoder.
    probe_predictions = probe(hidden.detach())
    probe_loss = _probe_objective(probe_predictions, batch)
    fabric.backward(probe_loss)
    return recon_loss.detach(), probe_loss.detach(), probe_predictions.detach()


def train(
    dataloader,
    model,
    probe,
    optimizer,
    fabric,
    metrics,
    config: Config,
    *,
    start_step=0,
    validation=None,
):
    end_step = config.optim.steps
    if start_step > end_step:
        raise ValueError(f"checkpoint step {start_step} is past configured end step {config.optim.steps}")

    performance = PerformanceTracker(
        config,
        device=fabric.device,
        initial_flops=float(start_step * estimate_training_flops(config, config.data.batch_size)),
    )
    data_state = getattr(dataloader, "_dataloader", dataloader)

    loader = iter(dataloader)
    for step in (pbar := trange(start_step, end_step)):
        try:
            batch = next(loader)
        except StopIteration as error:
            # A valid resampled stream is infinite.
            raise RuntimeError(
                f"the input stream ended at step {step} of {end_step}; check "
                f"that the configured urls hold subjects with at least "
                f"{config.data.min_patches} patches"
            ) from error

        # Inputs are clamped, so magnitude cannot be the problem -- only nan/inf.
        nonfinite = sorted(
            key for key, value in batch.items()
            if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all()
        )
        if nonfinite:
            _abort_nonfinite(
                f"non-finite values entered the model at step {step}: {', '.join(nonfinite)}",
                step=step, save_path=config.checkpoint.save_path, model=model, optimizer=optimizer,
                probe=probe, data_state=data_state, batch=batch,
            )

        optimizer.zero_grad()
        recon_loss, probe_loss, probe_predictions = _reconstruction_step(model, probe, fabric, batch)

        # Before optimizer.step(), so the dump holds the last good weights rather
        # than the ones a nan gradient just destroyed. One run went non-finite at
        # step 8999 and trained on garbage for another 1000 steps before a
        # downstream crash stopped it.
        if not torch.isfinite(recon_loss + probe_loss):
            _abort_nonfinite(
                f"non-finite loss at step {step}: recon={float(recon_loss)}, probe={float(probe_loss)}",
                step=step, save_path=config.checkpoint.save_path, model=model, optimizer=optimizer,
                probe=probe, data_state=data_state, batch=batch,
            )

        learning_rate = learning_rate_at(
            step,
            peak_lr=config.optim.lr,
            steps=config.optim.steps,
            warmup_steps=config.optim.warmup_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        # Clip the diagnostic probe separately so its raw-age scale cannot shrink
        # the representation update.
        grad_norm = _clip_gradients(model, config.optim.grad_clip)
        _clip_gradients(probe, config.optim.grad_clip)
        optimizer.step()

        performance.record_update(local_batch_size=int(batch["x"].shape[0]))
        _update_metrics(
            metrics,
            loss=recon_loss + probe_loss,
            recon_loss=recon_loss,
            probe_loss=probe_loss,
            probe_predictions=probe_predictions,
            batch=batch,
            grad_norm=grad_norm,
            grad_clip=config.optim.grad_clip,
        )

        next_step = step + 1
        if next_step % config.wandb.log_every_n_steps == 0 or next_step == end_step:
            metric_values = _compute_metrics(metrics)
            performance_values = performance.finish_interval()
            pbar.set_postfix(
                {
                    **{name: value.item() for name, value in metric_values.items()},
                    "lr": learning_rate,
                }
            )
            if fabric.loggers:
                fabric.log_dict(
                    {
                        **{f"train/{name}": value for name, value in metric_values.items()},
                        "train/lr": learning_rate,
                        **performance_values,
                    },
                    step=next_step,
                )
            _reset_metrics(metrics)

        if validation is not None and validation.due(next_step):
            performance.pause_interval()
            try:
                validation_values = validation.run(model, probe)
            finally:
                performance.resume_interval()
            if fabric.loggers:
                fabric.log_dict(validation_values, step=next_step)

        if config.checkpoint.save_path and next_step % config.checkpoint.every_n_steps == 0:
            save_checkpoint(
                config.checkpoint.save_path,
                model,
                optimizer,
                next_step,
                probe=probe,
                data_state=data_state.state_dict(),
                keep_last=config.checkpoint.keep_last,
            )

    if config.checkpoint.save_path and end_step % config.checkpoint.every_n_steps != 0:
        save_checkpoint(
            config.checkpoint.save_path,
            model,
            optimizer,
            end_step,
            probe=probe,
            data_state=data_state.state_dict(),
            keep_last=config.checkpoint.keep_last,
        )

    return model, optimizer


def run(config: Config):
    torch.set_float32_matmul_precision("high")
    checkpoint_load_path = config.checkpoint.load_path
    logger = _make_wandb_logger(config)

    fabric = Fabric(
        precision=config.training.precision,
        accelerator=config.training.accelerator,
        loggers=logger,
    )
    fabric.launch()
    _require_single_process(fabric)
    fabric.seed_everything(config.training.seed, workers=True)

    model, probe = _build_model_and_probe(config)

    data_cfg = config.data
    dataset = PretrainData(
        urls=data_cfg.urls,
        sampler=build_sampler(config),
        seed=config.training.seed,
        rank=fabric.global_rank,
        world_size=fabric.world_size,
        shuffle_buffer=data_cfg.shuffle_buffer,
        resume_segment_size=data_cfg.resume_segment_size,
    )
    loader_kwargs = dict(
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        # Keep worker startup off the model RNG for exact resume.
        generator=torch.Generator().manual_seed(config.training.seed + 10_000 * (fabric.global_rank + 1)),
    )
    if data_cfg.num_workers:
        loader_kwargs.update(
            prefetch_factor=data_cfg.prefetch,
            persistent_workers=data_cfg.persistent_workers,
            # Python 3.14 forkserver helpers can outlive short Slurm jobs; spawn exits cleanly.
            multiprocessing_context="spawn",
        )
    dataloader = StatefulDataLoader(dataset, snapshot_every_n_steps=1, **loader_kwargs)

    if config.training.compile:
        # Stochastic-depth nonzero has dynamic shape; capture avoids Dynamo graph breaks.
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        model = torch.compile(model)
    model = fabric.setup_module(model)
    probe = fabric.setup_module(probe)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad]
        + list(probe.parameters()),
        lr=config.optim.lr,
        weight_decay=config.optim.weight_decay,
    )
    optimizer = fabric.setup_optimizers(optimizer)
    dataloader = fabric.setup_dataloaders(dataloader)

    start_step = 0
    if checkpoint_load_path is not None:
        start_step = load_checkpoint(
            checkpoint_load_path,
            model,
            optimizer,
            probe=probe,
            dataloader=getattr(dataloader, "_dataloader", dataloader),
        )

    metrics = _make_metrics(fabric.device)
    validation = PeriodicValidation(config, device=fabric.device, precision=config.training.precision)

    try:
        result = train(
            dataloader,
            model,
            probe,
            optimizer,
            fabric,
            metrics,
            config,
            start_step=start_step,
            validation=validation,
        )
    finally:
        _shutdown_loader(dataloader)

    if logger is not None:
        logger.experiment.finish()
    return result


def main(args=None):
    run(parse_config(args))


if __name__ == "__main__":
    main()
