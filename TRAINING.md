# Training and evaluation

CortexMAE-ReadyNeuro: a 4D ViT-L masked autoencoder for fMRI. Volumes are only **rigidly** aligned to a
shared 2 mm MNI152NLin2009cAsym grid — what the paper calls *native space* — rather
than nonlinearly normalised to a template. The encoder reads 96 cells of 6x6x6 voxels
over a 16-frame clip, drawn from a subject's cortical grey-matter mask; each cell folds
into four temporal patches of 4 frames, so 384 tokens. The decoder reconstructs 32
held-out cells. Cell centres enter attention as world millimetres through each subject's
own affine, so rotary position sees relative coordinates and a subject's affine offset
cancels within a sample.

Clips are normalised on both axes before the model sees them: the data pipeline
z-scores each voxel's time series within its run, and the loader then z-scores each
frame across voxels (`standardize_frames`) and clamps to +/-3.

## Train one arm

`configs/model.yaml` is the whole architecture and recipe. One corpus from
`configs/data/` is the only thing that varies between arms. Install the package
first (`pip install -e .`, Python 3.14). Nothing here needs a GPU to check:
`uv run --group dev pytest tests/` runs CPU-only tests over no data at all (pytest is
a PEP 735 dependency group, so with pip you install it yourself first). Commands below
run from the repository root, with `AWS_PROFILE=openneuro-public` exported (see
[the data section](#what-this-repository-needs-from-outside-it)).
[`ARCHITECTURE.md`](ARCHITECTURE.md) maps each claim in the paper to the file
that implements it.

```sh
PRETRAIN_DATA_CONFIG=configs/data/openneuro_22373h.yaml \
    sbatch scripts/pretrain.sbatch openneuro-native
```

The first launch resolves the two files into `logs/openneuro-native/config.yaml` and
never resolves them again: every later attempt, including an automatic requeue,
trains from that frozen copy and resumes from `checkpoints/last.ckpt`. Set the seed
with `PRETRAIN_SEED`, not as a trailing argument. Without Slurm, the same run is

```sh
python -m fmrifm.train --config configs/model.yaml \
    --config configs/data/openneuro_22373h.yaml \
    --wandb.enabled=false \
    --checkpoint.save_path logs/openneuro-native/checkpoints/last.ckpt
```

The evaluation wrapper reads `logs/<run>/config.yaml` beside the checkpoint, which
only the sbatch writes, so a hand-launched run must pass its config as the fifth
positional argument to `eval_brainmarks.sbatch`.

One 80 GB GPU, 16 CPUs and about 640 GB of host RAM for the 16-worker loader; the
200,000-step recipe measured 60-123 H100-hours per run (mean 85), with ~12 GB of retained
checkpoints. Training is single-process **by design** — `train.py` raises on any world size
but 1, so
`torchrun` and `--nnodes` are rejected at startup rather than silently mis-normalised.

## Corpora

Stored BOLD hours are the paper's scale axis. The six OpenNeuro subsets are nested:
each is a prefix of the next. [^public]

| config | participants / within-study IDs | studies | stored BOLD | space |
|---|---:|---:|---:|---|
| `data/hcp_mni.yaml` | 880 | 1 | 1,482 h | vendor MNI |
| `data/hcp_native.yaml` | 880 | 1 | 1,482 h | native |
| `data/openneuro_747h.yaml` | 972 | 459 | 747 h | native |
| `data/openneuro_1520h.yaml` | 2,002 | 604 | 1,520 h | native |
| `data/openneuro_2984h.yaml` | 3,890 | 708 | 2,984 h | native |
| `data/openneuro_5919h.yaml` | 7,801 | 762 | 5,919 h | native |
| `data/openneuro_11837h.yaml` | 15,578 | 800 | 11,837 h | native |
| `data/openneuro_22373h.yaml` | 29,678 | 837 | 22,373 h | native |

`openneuro_1520h.yaml` matches the 880-subject HCP-YA arm by hours rather than by
subjects. It cuts across a batch boundary, so it enumerates its shards;
`configs/data/make_openneuro_hours_urls.py` regenerates it from the shard manifest at
[`manifests/openneuro_rebuild_manifest.jsonl.gz`](https://github.com/SophontAI/ReadyNeuro/blob/main/manifests/openneuro_rebuild_manifest.jsonl.gz)
in ReadyNeuro, which it reads from a sibling clone at `../ReadyNeuro` by default (`$OPENNEURO_MANIFEST` overrides the path); see its docstring for
the gap between the manifest's byte estimate and the sidecar hours this table reports.

Periodic validation scores each set in `validation.sets` (by default the OpenNeuro b31
holdout [^public]) and logs `val/<set>/*`. The HCP-YA configs add the 100 held-out HCP-YA
participants in both preparations, as the paper's runs did: the paper's HCP-YA held-out loss
is the set in the run's own preparation, and b31 gives the cross-corpus number. Each set
scores a 32-batch prefix (`validation.max_batches`), so `val/*` during training is a cheap
tracking number; `python -m fmrifm.validate` below is the exhaustive pass.

## Evaluate a checkpoint

Seven frozen-encoder targets — HCP-YA Task21, NSD COCO category, AABC age bracket,
AABC sex, PPMI diagnosis, ABIDE diagnosis, ADHD-200 diagnosis — scored in both
coordinate spaces, as 14 array tasks:

```sh
sbatch scripts/eval_brainmarks.sbatch logs/openneuro-native
```

The two AABC arms need `AABC_TARGETS_ROOT` pointed at your own per-subject label maps;
the script says which files and
[ReadyNeuro's `BUILDING.md`](https://github.com/SophontAI/ReadyNeuro/blob/main/BUILDING.md) says why.

The encoder is frozen (`eval()` under `torch.inference_mode()`) and runs under fp16
autocast rather than the bf16 it trains in, which is what produced the paper's numbers;
only the read-out head trains. Task21 and NSD use an attention probe, the five phenotype
targets use logistic regression.

Held-out reconstruction is a separate, exhaustive pass over the same sets:

```sh
python -m fmrifm.validate logs/openneuro-native/checkpoints/last.ckpt
```

It writes `<run_dir>/validation/<ckpt>.json`, a JSON report of the held-out
reconstruction metrics, and infers the config from the sibling `logs/<run>/config.yaml`.
The age/sex numbers in it come from an online linear probe that also runs during pretraining: it is a training-time diagnostic, detached from the
encoder, and no result in the paper depends on it.

## Encoder weights

Evaluation reads only the encoder, so a released model is the encoder alone. The paper's 19
encoders are released in this format under CC BY-NC 4.0 at
<https://huggingface.co/medarc/CortexMAE-ReadyNeuro>.

```sh
python -m fmrifm.export_encoder logs/openneuro-native/checkpoints/last.ckpt export/openneuro-native
```

This writes two files: `model.safetensors`, the `model.encoder` state dict as trained, with
the source step in its header, and `config.yaml`, the `model` and `preproc` sections that
rebuild it. The config defaults to `logs/<run>/config.yaml`; if that file is not one this
code parses, pass the architecture the run trained with, e.g. `--config configs/model.yaml`.
The export loads the checkpoint strictly into the model that config builds and fails if
the shapes or rotary wavelengths differ, or if the config disagrees with `logs/<run>/config.yaml`
on any field both carry. It refuses a directory that already holds either file.

Brainmarks loads the file like a checkpoint and reads the `config.yaml` beside it:
`model_kwargs.ckpt_path=<dir>/model.safetensors`, or the fourth argument of
`eval_brainmarks.sbatch`. In Python:

```python
from fmrifm.checkpoint import load_weights
from fmrifm.config import parse_config
from fmrifm.model import build_model

model = build_model(parse_config(["--config", "<dir>/config.yaml"]))
load_weights("<dir>/model.safetensors", model)  # model.encoder only; the decoder stays untrained
# x [B, T, N, P], coord [B, N, 3], valid [B, N]; EvalWrapper adds the paper's clip_frames split and fp16
tokens = model.eval()(x, coord, valid=valid)
```

`fmrifm.validate` scores reconstruction and the online probe, so it needs a full training
checkpoint.

## What this repository needs from outside it

**The data.** The six `configs/data/openneuro_*.yaml` corpora name the released corpus at
`s3://openneuro/v1/` [^public] on Cloudflare R2. A setup script writes an `openneuro-public`
AWS profile carrying a public read-only key, the R2 endpoint and region `auto`:

```sh
curl -fsS https://openneuro.sophont.med/clients/aws-setup.sh -o openneuro-aws-setup.sh
sh openneuro-aws-setup.sh
aws --profile openneuro-public s3 ls s3://openneuro/v1/
export AWS_PROFILE=openneuro-public    # before training or validation
```

The loader reads shards through both `aws s3 cp` and `s3fs`, and both pick the profile up
from `AWS_PROFILE`. Without the profile's R2 endpoint, `s3://openneuro` is looked up on AWS
S3, which is not this bucket (OpenNeuro's own archive is `s3://openneuro.org`), so the listing
fails or comes back empty.

The profile also sets the R2 endpoint, so while it is exported every S3 request goes to R2,
including anonymous reads of public AWS buckets. Export it only in the environment of a
training or validation job, and unset it before running ReadyNeuro builds or its
smoke scripts. The loader uses one profile for training and for every validation set, so
for the HCP arms either make your HCP shards readable under the same profile (or use local
paths), or pass `--validation.enabled=false`.

The two HCP-YA corpora and the store roots in `scripts/eval_brainmarks.sbatch` name
`s3://<RESTRICTED_ROOT>/`, a placeholder for access-controlled data: build it with
ReadyNeuro, set `EVAL_ROOT`, and repoint `configs/data/hcp_*.yaml` yourself.

**The evaluation stores, which you have to build yourself.** The six benchmark cohorts'
terms (gated access, data-use agreements or non-commercial share-alike licenses) do not
allow us to redistribute derived stores, unlike the pretraining corpus. NSD, ABIDE and
ADHD-200 rebuild from public raw data, HCP-YA needs registration, AABC and PPMI an
application.
[ReadyNeuro](https://github.com/SophontAI/ReadyNeuro/blob/main/BUILDING.md#evaluation-stores) builds them and
its `BUILDING.md` lists what each one requires. Until they exist,
`scripts/eval_brainmarks.sbatch` has nothing to read and the seven benchmark numbers cannot be reproduced.

**Brainmarks**, the benchmark suite introduced with *Scaling Vision Transformers for
Functional MRI with Flat Maps* (Lane et al., ICML 2026, arXiv:2510.13768), is the whole
of the evaluation half: `eval_brainmarks.sbatch` calls `brainmarks.main_probe` and
`brainmarks.main_logistic`. The `eval` extra installs it from git, pinned by commit, together
with [ReadyNeuro](https://github.com/SophontAI/ReadyNeuro), which registers the evaluation
datasets. From the repository root:

```sh
pip install -e '.[eval]'
```

On Linux, `uv` installs the CUDA 12.8 build of torch 2.11.0 that the paper used; with pip,
add `--extra-index-url https://download.pytorch.org/whl/cu128`.

The pinned commit is on a fork of [MedARC-AI/Brainmarks](https://github.com/MedARC-AI/Brainmarks),
pinned at a commit whose attention probe honours the padding mask
`EvalWrapper` returns as the fourth field of `Embeddings` — subjects differ in patch
count, so a probe that ignores it pools over zero-padded tokens and scores lower.

**AWS CLI v2.13 or later on `PATH`.** The training loader streams every shard through
`pipe:aws s3 cp`, not boto, so the binary must be on the `PATH` of all 16 dataloader
workers.

**Weights & Biases** is disabled by default (`wandb.enabled: false`). To log a run, sign
in to W&B and pass `--wandb.enabled=true`.


## License

Code is MIT ([`LICENSE`](LICENSE)). ReadyNeuro, the corpus it trains on, is released separately under
CC0 1.0, with only the 21 studies since withdrawn from OpenNeuro removed; [^public] see
[`DATA_LICENSE.md`](https://github.com/SophontAI/ReadyNeuro/blob/main/DATA_LICENSE.md) in ReadyNeuro. The
Brainmarks benchmark suite is a separate project under its own MIT
license (see [its `LICENSE`](https://github.com/Aceticia/Brainmarks/blob/d4640074168f1c6dcb0a94ec78801577792be3f4/LICENSE)).

[^public]: **Public release:** the corpus numbers here are the paper's. The public bucket
    `s3://openneuro/v1/` omits the 21 studies since withdrawn from OpenNeuro (498 shards,
    290 h of stored training BOLD, 0.20 TB); the paper trained on all 837. It serves 817
    studies and 30,222 shards (15.88 TB): 29,256 training shards in b0-b30 (22,083 h) and a
    b31 holdout of 966 across 439 studies; all 32 batch prefixes are present. Per config,
    public shards / studies / hours are 952 / 447 / 737 h (`openneuro_747h`), 1,969 / 588 /
    1,502 h (`openneuro_1520h`), 3,817 / 691 / 2,946 h (`openneuro_2984h`), 7,658 / 743 /
    5,836 h (`openneuro_5919h`), 15,324 / 779 / 11,674 h (`openneuro_11837h`) and 29,256 /
    816 / 22,083 h (`openneuro_22373h`). The prefix globs need no edit, since withdrawn
    shards are simply absent; `openneuro_1520h.yaml` enumerates shards and has its 33
    withdrawn rows commented out.
