# Orientation for a coding agent

Read [`ARCHITECTURE.md`](ARCHITECTURE.md) first — it has the data-flow diagram, the four
seams, and a table mapping each method claim in the paper to the file that implements it.

## What is here

One package, `fmrifm`, trains, validates and evaluates the model. The Brainmarks benchmark
suite (its own MIT license) the paper evaluates through is an `eval` dependency pinned to a fork
commit. `readyneuro`, which builds the data and supplies the evaluation data loaders, is in
[ReadyNeuro](https://github.com/SophontAI/ReadyNeuro) and is also an `eval` dependency.
[`TRAINING.md`](TRAINING.md) is the training and evaluation walkthrough.

## What runs without data

The package installs and unit-tests offline, on CPU:

```bash
uv run --group dev pytest tests/
```

Nothing else is usefully executable without more: training needs the corpus (see
[ReadyNeuro](https://github.com/SophontAI/ReadyNeuro#get-the-data) for access) and an H100.
Evaluation needs the six benchmark stores, which are not released because the cohorts' terms
(gated access, data-use agreements or non-commercial share-alike licenses) do not allow us to
redistribute a derived store, so the benchmark numbers cannot be reproduced
without building them under your own data access. For the combined evaluation environment,
from the repository root:

```bash
pip install -e '.[eval]'
```

## Answering questions about the method

Prefer reading the code to reading prose about the code. The three files that carry the
most method per line:

- [`preprocess/openneuro.py`](https://github.com/SophontAI/ReadyNeuro/blob/main/preprocess/openneuro.py) (ReadyNeuro) — the preprocessing pipeline, end to end,
  in one standalone script. This is the paper's contribution.
- `configs/model.yaml` — the entire architecture and recipe. One corpus
  from `configs/data/` is the only thing that varies between arms.
- `src/fmrifm/pretrain_data.py` — how a shard becomes a training sample:
  the patch grid, the world-millimetre coordinates, the 96/32 draw.

When a number in the paper is in question, `ARCHITECTURE.md` says where it is implemented.
Several are derived rather than configured — the 7.1e-5 learning rate is
`base_lr × sqrt(batch/256)` — so grep for the rule, not the value.

## Invariants worth knowing before you suggest an edit

- `src/brainmarks/` (here and in ReadyNeuro) is a PEP 420 namespace
  overlay and must contain **no `__init__.py`**. Adding one makes plugin discovery find zero
  plugins, and the symptom is an empty registry rather than an error.
- Brainmarks is pinned at `d4640074168f` in `pyproject.toml`; ReadyNeuro pins the same commit.
- `s3://openneuro/v1/` [^public] (the released corpus, on Cloudflare R2, read through the
  `openneuro-public` AWS profile) and `s3://openneuro.org` (the upstream archive) are real.
  `<RESTRICTED_ROOT>`, `<RAW_ROOT>` and `<EVAL_ROOT>` are deliberate placeholders for data
  that cannot be redistributed.

The store formats — `.npy` tar members for shard geometry, the baked-in clip windows — are
seams 3 and 4 in `ARCHITECTURE.md`.

## Training and evaluation

**Several reported numbers are derived, not configured.** The paper's 7.1e-5 peak learning
rate is `optim.base_lr 1.0e-4 × sqrt(effective_batch / 256)` at batch 128, so changing the
batch moves it. The cosine decays to 1% of peak, not to zero.

**`src/brainmarks/models/` has no `__init__.py`.** It is a PEP 420 namespace overlay that
registers `fmrifm_native` and `fmrifm_mni` into the installed benchmark suite; adding one makes
discovery find nothing, silently.

**Offline check:** `uv run --group dev pytest tests/` (CPU only, no data). Training needs the
corpus (`AWS_PROFILE=openneuro-public`, see `TRAINING.md`) and one H100. Evaluation needs the
benchmark stores you build with ReadyNeuro, plus, from the repository root,
`pip install -e '.[eval]'`.

## Please do not

Do not download the corpus (15.88 TB) or attempt to fetch the benchmark stores. Do not "fix"
the placeholders into a guessed bucket name. Do not paste the corpus read key into any file;
it comes from the setup script.

[^public]: **Public release:** the bucket holds 817 of the paper's 838 accessions under
    CC0 1.0. The 21 studies since withdrawn from OpenNeuro (498 shards, 290 h) are removed,
    leaving 30,222 shards (29,256 training + 966 held out), 22,083 h of stored training BOLD,
    15.88 TB. The paper trained on all of them, and the manifests in
    ReadyNeuro's `manifests/` still list them.
