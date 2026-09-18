# PepSite

**PepSite: Receptor-Site-Faithful Peptide Binder Generation with Site-Aware Masked Language Modeling**

PepSite fine-tunes an ESM-2 masked language model with a residue-level receptor-site head, then samples peptide binders while preserving receptor-site conditioning.

## Contents

- `src/pepsite/`: model training, masked-residue PPL, inference, and route logic.
- `scripts/`: data preparation, two-stage training, inference, and route generation.
- `configs/`: data, model, training, inference, and reproducibility settings.
- `models/PepSite/`: tokenizer/config metadata and the external-weight checksum manifest.

## Installation

```bash
conda env create -f environment.yml
conda activate pepsite
pip install -e .
```

## Model weights

The trained `pytorch_model.bin` is not stored in this repository because it is larger than GitHub's normal file limit. Obtain it from the future Hugging Face or Zenodo release and place it at:

```text
models/PepSite/pytorch_model.bin
```

The expected SHA-256 is recorded in `models/PepSite/SHA256SUMS`:

```text
0541b9a587a9c52fa87b2bf86d802614b959e30a8fe286cda4ee3fd0ba4ee572
```

The public download URL is pending. Do not run inference until the downloaded weight matches this checksum.

## Data preparation

Each training CSV must contain a receptor sequence, peptide sequence, and a binary receptor-site mask. The accepted schema is:

```text
query_id,protein_sequence,ligand_sequence,receptor_binding_site_mask,binding_site_status
```

`query_id` and `binding_site_status` are optional; the alternative column names `Receptor Sequence` and `Binder` are also accepted. The mask must have one `0`/`1` character per receptor residue. Prepare the splits with:

```bash
python scripts/prepare_data.py \
  --train-csv /path/to/train.csv \
  --val-csv /path/to/validation.csv \
  --test-csv /path/to/test.csv \
  --output-dir data/prepared
```

## Training

Phase 1 warms up the receptor-site head and the final backbone layers:

```bash
python scripts/train_phase1.py \
  --init-model-dir /path/to/raw/esm2/checkpoint \
  --train-csv data/prepared/train.csv \
  --val-csv data/prepared/val.csv
```

Phase 2 performs joint masked-language-model and site-head training:

```bash
python scripts/train_phase2.py \
  --init-model-dir runs/pepsite/phase1/final_model \
  --train-csv data/prepared/train.csv \
  --val-csv data/prepared/val.csv
```

Both launchers read the default reproducibility setting from `configs/reproducibility.yaml`; a non-negative run value can be supplied with `--seed`.

## Inference and route generation

Inference accepts a label-free context CSV with exactly `query_id,receptor_sequence,peptide_length`:

```bash
python scripts/infer_candidates.py \
  --phase2-model models/PepSite \
  --context /path/to/context.csv \
  --output-dir runs/pepsite/inference \
  --device auto
```

To apply deterministic route selection to the generated candidate table:

```bash
python scripts/generate_routes.py \
  --candidate-csv runs/pepsite/inference/candidate_scores.csv \
  --output-dir runs/pepsite/inference/routes \
  --xr-proposals runs/pepsite/inference/xr_proposals.csv \
  --xr-selected runs/pepsite/inference/xr_selected.csv
```

## License

This repository is released under the Apache License 2.0. Model weights and source datasets remain subject to their respective licenses.
