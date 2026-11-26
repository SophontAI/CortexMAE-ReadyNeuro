# CortexMAE-ReadyNeuro

A 4D ViT-L masked autoencoder for fMRI, pretrained on [ReadyNeuro](https://github.com/SophontAI/ReadyNeuro).
Scans are only rigidly aligned to a shared 2 mm grid. The model reads patches of cortical grey
matter and locates them by their millimeter coordinates, so it takes native-space and
template-space data alike. This repository holds the model, pretraining and evaluation code.

- Paper: <https://openreview.net/forum?id=rLMKAJCLc1>
- Blog post: <https://sophont.med/blog/native-fmri/>
- Weights: <https://huggingface.co/medarc/CortexMAE-ReadyNeuro>
- Data and evaluation data loaders: <https://github.com/SophontAI/ReadyNeuro>
- Contact: paul@sophont.med

## Models

All CortexMAE-ReadyNeuro models are pretrained on ReadyNeuro: the full corpus (22,373 h,
`readyneuro-22373h/`) and five nested subsets from 747 h up. These are the paper's training hours;
the public bucket serves 22,083 h after removing studies withdrawn from OpenNeuro. Models pretrained on HCP-YA (native
and MNI preparation) are ablations. All 19 runs share one recipe (`configs/model.yaml`)
and are evaluated with [Brainmarks](https://github.com/MedARC-AI/Brainmarks) on seven targets.

## Install

Python 3.14. The `eval` extra installs Brainmarks and ReadyNeuro's evaluation data loaders
from git:

```sh
git clone https://github.com/SophontAI/CortexMAE-ReadyNeuro.git
cd CortexMAE-ReadyNeuro
pip install -e '.[eval]' pytest
pytest tests/                      # offline tests
# or, in a uv-managed venv:
uv run --group dev pytest tests/
```

To load weights only, `pip install -e .` is enough. With pip on Linux, add
`--extra-index-url https://download.pytorch.org/whl/cu128` for the CUDA 12.8 torch the paper used.

## Train and evaluate

[`TRAINING.md`](TRAINING.md) covers training, held-out validation,
Brainmarks evaluation and exporting encoder weights.
[`ARCHITECTURE.md`](ARCHITECTURE.md) maps each method claim in the paper to its code.

Training streams ReadyNeuro from R2: set up the `openneuro-public` profile as described in
[ReadyNeuro](https://github.com/SophontAI/ReadyNeuro#get-the-data) and
`export AWS_PROFILE=openneuro-public`. The evaluation stores are not redistributed; configs that
point at them use the placeholders `s3://<RESTRICTED_ROOT>/`, `<RAW_ROOT>` and `<EVAL_ROOT>`.

| directory | contents |
|---|---|
| [`src/fmrifm/`](src/fmrifm/) | model, training, validation |
| [`src/brainmarks/models/`](src/brainmarks/models/) | Brainmarks model plugin |
| [`configs/`](configs/), [`scripts/`](scripts/) | recipe and corpus configs, Slurm scripts |

The `eval` extra pins [Brainmarks](https://github.com/MedARC-AI/Brainmarks) to the
[fork](https://github.com/Aceticia/Brainmarks/tree/d4640074168f1c6dcb0a94ec78801577792be3f4) at
`d4640074168f`, which adds a padding mask to the attention probe.

## Stream samples

Each shard is one subject. Its first sample holds the T1, the tissue masks and the grid
geometry. Every later sample is one BOLD run. Voxels are stored packed: `*_idx.npy` holds flat
C-order indices of the in-brain voxels, and `*_values.npy` holds one row per index. BOLD is
z-scored per voxel within the run, stored as int8 at one frame per second. This reads T1 and BOLD
with `pip install webdataset numpy`:

```python
import io, json
import numpy as np
import webdataset as wds

def dense(idx, values, shape):
    """Scatter packed in-brain rows back onto the grid."""
    out = np.zeros((np.prod(shape), *values.shape[1:]), np.float32)
    out[idx] = values
    return out.reshape(*shape, *values.shape[1:])

url = "https://huggingface.co/datasets/medarc/ReadyNeuro/resolve/main/shards/ds004604__022.tar"
# Same shard from the full corpus: "pipe:aws --profile openneuro-public s3 cp s3://openneuro/v1/openneuro-pretrain-shards-v6-b1/train/native/ds004604__022.tar -"
for sample in wds.WebDataset(url, shardshuffle=False):
    meta = json.loads(sample["json"])
    arr = lambda name: np.load(io.BytesIO(sample[name]))
    if meta["kind"] == "bundle":  # first sample of each shard: T1, masks, geometry
        bold_idx, bold_shape = arr("bold_idx.npy"), meta["bold_shape"]
        bold_affine = arr("bold_affine.npy")  # voxel -> world mm
        cortex = dense(bold_idx, arr("bold_cortical_gm.npy"), bold_shape)  # (97, 115, 97)
        t1 = dense(arr("t1_idx.npy"), arr("t1_values.npy") * meta["T1_scale"], meta["T1_shape"])
    else:  # one sample per run: z-scored BOLD
        z = arr("values.npy") * np.float32(meta["scale"])  # (in-brain voxels, frames)
        clip = dense(bold_idx, z[:, :16], bold_shape)  # first 16 s as (97, 115, 97, 16)
```

The bundle also has `bold_gm`, `bold_seg` (SynthSeg labels) and their `t1_*` counterparts. Each
run has `mean.npy` and `std.npy` (multiply by `mean_scale`/`std_scale`), which undo the z-score.
To find a shard on S3, look up its `batch` in [`openneuro_rebuild_manifest.jsonl.gz`](https://github.com/SophontAI/ReadyNeuro/blob/main/manifests/openneuro_rebuild_manifest.jsonl.gz):
`b0` is the prefix `openneuro-pretrain-shards-v6/`, and `b<N>` is `openneuro-pretrain-shards-v6-b<N>/`.
The full format is in [`tools/codec.py`](https://github.com/SophontAI/ReadyNeuro/blob/main/src/readyneuro/tools/codec.py).

## Tokenize without densifying

The dense BOLD grid is about 95% empty, and a long run as float32 takes gigabytes. The model never
builds it. `build_layout` scatters only the 3D cortical mask, once per subject. It cuts the mask
into 6x6x6 patches, keeps patches that are more than 10% cortex, and records which packed row
fills each patch voxel. Each clip is then one indexing step on the packed `(voxels, frames)` array.
This is the same code that training and evaluation run (from the repository root):

```python
import numpy as np
from fmrifm.config import parse_config
from fmrifm.pretrain_data import ShardReader, build_sampler, standardize_frames

config = parse_config(["--config", "configs/model.yaml"])
sampler = build_sampler(config)
T, limit = config.preproc.clip_frames, config.preproc.clip_value
url = "https://huggingface.co/datasets/medarc/ReadyNeuro/resolve/main/shards/ds004604__022.tar"
for sample in ShardReader().read_shard(url):  # a local path, https:// or s3:// URL
    if sample["_kind"] == "bundle":
        layout = sampler.build_layout(sample)  # once per subject: patch grid over the cortical mask
        continue
    run = sample["bold_values"][layout.bold_rows].T.astype(np.float32)  # (frames, cortical voxels)
    patches = layout.bold_grid.gather(standardize_frames(run))  # (frames, patches, 6**3)
    for start in range(0, len(patches) - T + 1, T):  # runs shorter than T yield no clip
        x = np.clip(patches[start : start + T], -limit, limit)  # (16, patches, 216)
        coord = layout.bold_centers  # (patches, 3), world millimetres
```

`x` and `coord` are one sample of the encoder's input: add a batch axis and pass them as in
[Encoder weights](TRAINING.md#encoder-weights). Patch counts differ between subjects: evaluation
pads them and passes `valid`, and training draws a fixed 96 context and 32 target patches per
clip (`sampler.sample_clips`).

## License and citation

Code: MIT ([`LICENSE`](LICENSE)). Weights: CC BY-NC 4.0. ReadyNeuro: CC0 1.0. Brainmarks
keeps its own MIT license.

Please cite the paper, and Brainmarks (Lane et al. 2026, ICML, arXiv:2510.13768) if you use it.

```bibtex
@inproceedings{liu2026scaling,
  title     = {Scaling {fMRI} Foundation Models with Native-Space Data from Heterogeneous Public Studies},
  author    = {Liu, Xujin Chris and Lane, Connor and Muhammad, Hassan and Abraham, Tanishq Mathew and Scotti, Paul Steven},
  booktitle = {NeurIPS 2026 Workshop on Foundation Models for the Brain and Body},
  year      = {2026},
  url       = {https://openreview.net/forum?id=rLMKAJCLc1}
}
```

Xujin Chris Liu (New York University) did this work during a summer 2026 internship at Sophont.
