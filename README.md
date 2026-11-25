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
