# CICIoT2023 Deep Learning IDS

Research code for the CICIoT2023 part of the thesis. This project evaluates deep learning architectures for multiclass IoT intrusion detection and is used as a secondary validation dataset for the architecture family developed on UNSW-NB15.

The goal of this branch is not to claim state-of-the-art performance on CICIoT2023. Instead, it tests whether the feature-axis multi-scale CNN-BiLSTM-attention approach remains competitive on a larger modern IoT dataset with different traffic characteristics and a different class structure.

## Research Objective

CICIoT2023 contains large-scale IoT attack traffic with strong class imbalance and high redundancy across some attack families. Many published results on this dataset report very high scores, often under preprocessing and split protocols that are not directly comparable.

This project therefore focuses on a controlled local comparison:

- consistent preprocessing across all local models;
- multiclass category-level classification;
- macro-F1 as the main comparison metric;
- per-class precision, recall, and F1;
- local comparison against CNN, LSTM, CNN-BiLSTM, BiGRU-attention, and FT-Transformer variants;
- W&B hyperparameter sweeps followed by fixed-seed verification;
- latency, throughput, parameter count, and model-size reporting.

## Dataset

Expected data source:

```text
data/raw/
  CSV_MERGED/
    *.csv
```

The project can also profile the multi-file CSV release when available:

```text
data/raw/
  CSV/
    *.csv
```

This branch uses category-level multiclass classification with the following target groups:

- Benign;
- BruteForce;
- DDoS;
- DoS;
- Mirai;
- Recon;
- Spoofing;
- Web.

Raw and processed CICIoT2023 files are not tracked by Git.

## Project Layout

```text
.
  configs/                  Baseline, proposed, final, and sweep configurations.
    sweeps/                 W&B sweep definitions.
  data/                     Local raw and processed CICIoT2023 files, ignored by Git.
  plan/                     Research plans, phase summaries, and SOTA positioning notes.
  reports/                  Supervisor-facing reports and generated figures.
  runs/                     Local training outputs, ignored by Git.
  scripts/                  Profiling, preprocessing, reporting, and W&B utilities.
  src/ciciot2023_ids/       Installable Python package.
    models/                 CNN, LSTM, CNN-BiLSTM, BiGRU-attention, FT-Transformer, proposed model.
    config.py               TOML configuration loading.
    data.py                 Dataset loading, balancing, preprocessing, and splits.
    losses.py               Cross entropy and focal-loss utilities.
    metrics.py              Classification and operational metrics.
    train.py                Training and evaluation pipeline.
  pyproject.toml            Python package metadata and CLI entry points.
  requirements.txt          CPU/general dependencies.
  requirements-gpu-cu128.txt CUDA 12.8 PyTorch dependencies.
```

## Main Architecture

The transferred architecture is the same family selected from the UNSW-NB15 work:

1. Encoded tabular traffic input.
2. Feature-axis sequence representation.
3. Multi-scale one-dimensional convolution.
4. Bidirectional LSTM encoder.
5. Multi-head self-attention.
6. Dense multiclass classifier.

On CICIoT2023, this model is evaluated as a transfer and robustness experiment. It is compared against additional local deep-learning variants, including MS-CNN-BiGRU-attention and FT-Transformer.

## Setup

Create a local environment from this project root:

```powershell
cd projects/ciciot2023
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
ciciot2023-check-setup
```

For NVIDIA GPU training with CUDA 12.8:

```powershell
python -m pip install --upgrade --force-reinstall -r requirements-gpu-cu128.txt
python -m pip install -e .
ciciot2023-check-setup
```

Confirm GPU availability with:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

If the local machine uses another CUDA runtime, install the matching PyTorch build first, then rerun `ciciot2023-check-setup`.

## Reproducible Workflow

Profile the local dataset:

```powershell
python scripts/profile_dataset.py --input data/raw/CSV_MERGED --output-dir runs/profile --exact-label-counts
```

Prepare the category-level data:

```powershell
ciciot2023-prepare --config configs/proposed_feature_attention_cap50k_mi30.toml
```

Run a one-epoch software smoke test:

```powershell
ciciot2023-train --config configs/baseline_smoke.toml
```

Run baseline models:

```powershell
ciciot2023-train --config configs/baseline_cnn.toml
ciciot2023-train --config configs/baseline_lstm.toml
ciciot2023-train --config configs/baseline_cnn_bilstm.toml
```

Run the proposed architecture:

```powershell
ciciot2023-train --config configs/proposed_feature_attention_cap50k_mi30.toml
```

Run additional local architecture comparisons:

```powershell
ciciot2023-train --config configs/proposed_bigru_attention_cap50k_mi30.toml
ciciot2023-train --config configs/ft_transformer_cap50k_mi30.toml
```

Run final fixed-seed verification:

```powershell
ciciot2023-train --config configs/final_wandb_feature_attention_cap50k_mi30_seed42.toml
ciciot2023-train --config configs/final_wandb_feature_attention_cap50k_mi30_seed7.toml
ciciot2023-train --config configs/final_wandb_feature_attention_cap50k_mi30_seed123.toml
```

## W&B Sweeps

Sweep definitions live in:

```text
configs/sweeps/
```

Typical command pattern:

```powershell
wandb sweep --entity <entity> --project pfe-thesis-ciciot2023 configs\sweeps\proposed_feature_attention_cap50k_mi30.yaml
wandb agent <entity>/pfe-thesis-ciciot2023/<sweep_id> --count 20
```

Use W&B for hyperparameter exploration only. Final claims should use fixed-seed reruns of the selected configuration.

## Final Result Summary

Final selected CICIoT2023 validation model:

```text
W&B-tuned feature-axis multi-scale CNN-BiLSTM-attention classifier
Protocol: category-level multiclass classification
Main role: secondary validation of the UNSW-NB15 architecture family
Seeds: 42, 7, 123
```

Three-seed test result:

| Metric | Mean +/- Std |
| --- | ---: |
| Accuracy | 76.58% +/- 0.32% |
| Macro-F1 | 71.13% +/- 0.29% |
| Weighted-F1 | 76.24% +/- 0.35% |
| Parameters | 317,672 |
| Latency | about 0.0129 ms/sample |
| Throughput | about 77k samples/s |

Local architecture comparison:

| Model | Accuracy | Macro-F1 | Weighted-F1 | Interpretation |
| --- | ---: | ---: | ---: | --- |
| W&B-tuned MS-CNN-BiLSTM-attention | 76.58% | 71.13% | 76.24% | Best local macro-F1 and final selected validation model. |
| MS-CNN-BiGRU-attention | 76.57% | 70.96% | not selected | Very close competitor with a lighter recurrent block. |
| FT-Transformer | 74.29% | 68.47% | not selected | Useful modern transformer baseline, but weaker under this local protocol. |

The result is competitive inside the local experimental protocol, but it should be described as a transfer-validation result rather than a CICIoT2023 state-of-the-art claim.

## Git Policy

Do not commit:

- raw or processed CICIoT2023 files;
- local virtual environments;
- local runs and W&B logs;
- trained checkpoints;
- generated cache files.

Keep source code, configs, research notes, and Markdown reports under version control.

## Suggested Repository Names

- `ciciot2023-feature-axis-attention-ids`
- `ciciot2023-iot-dl-validation`
- `ciciot2023-multiclass-ids-benchmark`

Recommended: `ciciot2023-feature-axis-attention-ids`
