---
title: "CICIoT2023 Progress Report"
author: "Thesis Progress Report"
date: "2026-08-08"
---

# CICIoT2023 Progress Report

## 1. Objective

This report summarizes the CICIoT2023 experimental branch of the thesis in a
format aligned with the UNSW-NB15 report. CICIoT2023 is used to test whether
the proposed feature-axis multi-scale CNN-BiLSTM-attention architecture
transfers from UNSW-NB15 to a modern IoT network-flow dataset.

The main research question is:

> Does the proposed architecture remain competitive on modern IoT intrusion
> traffic when evaluated with macro-F1, weighted-F1, per-class recall/F1,
> false-positive rates, latency, and parameter count?

## 2. Dataset And Protocol

CICIoT2023 is a modern IoT intrusion-detection dataset from the Canadian
Institute for Cybersecurity. The local project uses the official `CSV_MERGED`
files.

Local profile:

| Item | Value |
| --- | ---: |
| CSV shards | 63 |
| Rows counted | 45,019,243 |
| Feature columns excluding label | 39 |
| Fine-grained labels | 34 |
| Final target labels | 8 categories |

Final target categories:

- Benign;
- DDoS;
- DoS;
- Mirai;
- Recon;
- Spoofing;
- Web;
- BruteForce.

Protocol safeguards:

- category-level 8-class classification;
- capped class sampling for feasible local experiments;
- preprocessing fitted only on training data;
- stratified train/validation/test split;
- reusable processed dataset cache;
- leakage-safe mutual-information feature selection;
- final proposed candidate verified with seeds `42`, `7`, and `123`.

## 3. Work Completed

| Phase | Work Completed | Status |
| --- | --- | --- |
| Phase 1 | Project setup, GPU/dependency checks, full dataset profile | Done |
| Phase 2 | Preprocessing, category mapping, capped sampling, MI feature selection | Done |
| Phase 3 | CNN, LSTM, CNN-BiLSTM baselines | Done |
| Phase 4 | Proposed architecture transfer | Done |
| Phase 5 | Contribution-oriented preprocessing/loss experiments | Done |
| Phase 6 | W&B hyperparameter search for proposed architecture | Done |
| Phase 7 | Three-seed verification of best W&B candidate | Done |
| Phase 8 | Balanced W&B rare-class diagnostic | Done enough |
| Phase 9 | BiGRU second architecture | Seed 42 done |
| Phase 10 | FT-Transformer comparison | Seed 42 done; below proposed |
| Phase 11 | DL-only paper-style W&B baselines | Prepared |

## 4. Proposed Architecture

The transferred model is the same architecture family used in UNSW-NB15:

```text
Encoded IoT flow feature vector
  -> feature-axis sequence representation
  -> parallel Conv1D kernels 3/5/7
  -> concatenate multi-scale features
  -> BiLSTM encoder
  -> multi-head self-attention
  -> dense 8-class classifier
```

Why this architecture is relevant:

- Conv1D captures local feature interactions;
- multi-scale kernels capture short and wider feature patterns;
- BiLSTM models bidirectional dependencies across feature positions;
- attention reweights important feature interactions;
- the architecture directly tests whether the UNSW-NB15 model family transfers
  to a modern IoT dataset.

## 5. Main Results

Initial cap-50k protocol:

| Model | Accuracy | Macro-F1 | Weighted-F1 | Role |
| --- | ---: | ---: | ---: | --- |
| CNN | 60.08% | 53.59% | 59.43% | Lightweight baseline |
| LSTM | 72.99% | 68.87% | 73.32% | Recurrent baseline |
| CNN-BiLSTM | 73.73% | 69.72% | 74.13% | Hybrid baseline |
| Proposed MS-CNN-BiLSTM-attention | 73.97% | 70.05% | 74.27% | Initial proposed model |

Three-seed baseline/proposed summary before W&B:

| Model | Accuracy Mean | Macro-F1 Mean | Weighted-F1 Mean |
| --- | ---: | ---: | ---: |
| CNN-BiLSTM | 73.70% | 69.67% | 74.00% |
| Proposed MS-CNN-BiLSTM-attention | 73.91% | 70.00% | 74.23% |

Interpretation:

> The proposed architecture transferred successfully and improved macro-F1 over
> the CNN-BiLSTM baseline, but the initial gain was modest.

## 6. Contribution-Oriented Preprocessing

To strengthen the proposed model, the project added:

- leakage-safe variance filtering;
- leakage-safe mutual-information feature selection;
- cap-50k + MI top-30 protocol;
- balanced-10k + MI top-30 protocol;
- class-weighted CE;
- focal loss;
- label smoothing.

Direct results:

| Experiment | Accuracy | Macro-F1 | Weighted-F1 | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Original proposed | 73.97% | 70.05% | 74.27% | Best direct macro-F1 before W&B |
| Cap-50k + MI CNN-BiLSTM | 73.27% | 68.64% | 73.83% | Same-protocol baseline |
| Cap-50k + MI proposed | 74.46% | 69.89% | 75.05% | Better accuracy, slightly lower macro-F1 |
| Cap-50k + MI proposed focal | 71.38% | 67.29% | 72.24% | Focal hurt |
| Balanced-10k + MI proposed | 69.69% | 69.71% | 69.68% | Better rare-class behavior, lower global score |

## 7. W&B Hyperparameter Search

W&B was used to tune the cap-50k + MI proposed architecture.

Best sweep run:

| Best Run | Accuracy | Macro-F1 | Weighted-F1 | Parameters | Latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| `k61e20wi` | 76.71% | 71.32% | 76.35% | 317,672 | 0.0227 ms/sample |

Best run hyperparameters:

| Hyperparameter | Value |
| --- | ---: |
| Conv channels | 32 |
| LSTM hidden | 96 |
| Attention heads | 2 |
| Dense hidden | 96 |
| Dropout | 0.312969 |
| Batch size | 512 |
| Learning rate | 0.002496 |
| Weight decay | 0.000801 |
| Class weight | false |
| Loss | Cross entropy |
| Label smoothing | 0.05 |

The W&B search improved both accuracy and macro-F1 compared with the previous
direct proposed model.

## 8. Final Proposed Model Verification

The best W&B configuration was rerun with seeds `42`, `7`, and `123`.

| Run | Accuracy | Macro-F1 | Weighted-F1 | Latency |
| --- | ---: | ---: | ---: | ---: |
| Seed 42 | 76.14% | 70.73% | 75.75% | 0.0130 ms/sample |
| Seed 7 | 76.74% | 71.43% | 76.42% | 0.0129 ms/sample |
| Seed 123 | 76.87% | 71.24% | 76.54% | 0.0127 ms/sample |
| **Mean +/- std** | **76.58% +/- 0.32%** | **71.13% +/- 0.29%** | **76.24% +/- 0.35%** | **0.0129 ms/sample** |

This is the current verified CICIoT2023 proposed-architecture result.

## 9. Per-Class Behavior

Final W&B-tuned proposed model, three-seed mean:

| Class | Precision | Recall | F1 | False-Positive Rate |
| --- | ---: | ---: | ---: | ---: |
| Benign | 65.33% | 76.46% | 70.43% | 7.12% |
| BruteForce | 62.60% | 32.43% | 42.56% | 0.76% |
| DDoS | 91.09% | 66.73% | 77.01% | 1.15% |
| DoS | 73.72% | 93.35% | 82.37% | 5.82% |
| Mirai | 99.87% | 99.84% | 99.85% | 0.02% |
| Recon | 65.06% | 70.93% | 67.83% | 6.67% |
| Spoofing | 86.84% | 78.95% | 82.71% | 2.09% |
| Web | 49.14% | 43.78% | 46.29% | 3.46% |

Main observation:

- Mirai, DoS, Spoofing, and DDoS are strong.
- BruteForce and Web remain weak.
- Recon is moderate.
- The model is useful as an external validation result, not as a claim of
  universal SOTA on CICIoT2023.

## 10. Balanced Rare-Class Diagnostic

Balanced W&B was run to test whether BruteForce and Web could improve.

| Model | Accuracy | Macro-F1 | BruteForce F1 | Web F1 | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Final cap-50k proposed model | 76.58% | 71.13% | 42.56% | 46.29% | Best global proposed result |
| Best balanced W&B run `rgkdnwat` | 70.22% | 70.24% | 62.53% | 52.82% | Better rare classes, worse global result |

Interpretation:

> Balanced sampling improves BruteForce and Web, but the accuracy and global
> macro-F1 cost is too high for it to replace the final proposed model.

## 11. Second Architecture: BiGRU Variant

A lighter second architecture was implemented:

```text
Encoded IoT flow feature vector
  -> multi-scale Conv1D
  -> BiGRU encoder
  -> multi-head self-attention
  -> dense classifier
```

Seed-42 result:

| Model | Accuracy | Macro-F1 | Weighted-F1 | Parameters | Latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Final BiLSTM attention mean | 76.58% | 71.13% | 76.24% | 317,672 | 0.0129 ms/sample |
| BiGRU attention seed 42 | 76.57% | 70.96% | 76.08% | 280,424 | 0.0261 ms/sample |

Interpretation:

- BiGRU is very close to the BiLSTM version.
- It uses fewer parameters.
- The measured latency was worse in this run, so it is not automatically the
  better deployment model.
- It is useful as a second architecture / ablation.

## 12. FT-Transformer Baseline

The FT-Transformer baseline was trained on the same cap-50k + MI top-30
protocol used by the final proposed model.

| Model | Accuracy | Macro-F1 | Weighted-F1 | Parameters | Latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| FT-Transformer seed 42 | 74.29% | 68.47% | 73.82% | 247,208 | 0.0276 ms/sample |
| Final proposed mean | 76.58% | 71.13% | 76.24% | 317,672 | 0.0129 ms/sample |

Interpretation:

- the transformer baseline is valid, but weaker than the proposed model;
- it has fewer parameters, but higher measured latency;
- it is useful as a modern DL baseline rather than as the final selected
  CICIoT2023 model.

## 13. Comparison With Deep-Learning CICIoT2023 References

This table intentionally excludes classical ML methods and suspicious
near-perfect headline rows. The purpose is to compare against deep-learning or
transformer-style work with more interpretable multiclass reporting.

| Reference | Deep Architecture | Reported CICIoT2023 Result | Protocol Note | Relation To This Project |
| --- | --- | ---: | --- | --- |
| Official CICIoT2023 benchmark | ANN/DNN | 69.73% F1 reported for 8-class benchmark | Accuracy not used here because near-perfect accuracy is not informative for our macro-F1 focus | Our proposed model has higher macro-F1. |
| FT-Transformer 2026 | FT-Transformer | 80.96% accuracy; 72.40% macro-F1; 80.67% weighted-F1 | Multiclass tabular/IoT comparison | Closest DL target; our macro-F1 is about 1.27 points lower. |
| Local FT-Transformer | FT-Transformer-style tabular encoder | 74.29% accuracy; 68.47% macro-F1; 73.82% weighted-F1 | Same local protocol as proposed model | Useful negative baseline; below proposed model. |
| Sensors 2025 Transformer/DNN/CNN study | DNN/CNN/Transformer family | High accuracy reported | DDoS/multiclass setting; protocol differs and macro-F1 is not the same comparison basis | Useful transformer context, not direct ranking. |
| TransNeSt hybrid architecture | Split-attention + transformer | High F1 reported with augmentation | Uses WGAN/augmentation and multi-dataset setup | Useful architecture context; not direct strict comparison. |
| **This project** | **MS-CNN-BiLSTM-attention** | **76.58% +/- 0.32% accuracy; 71.13% +/- 0.29% macro-F1** | 8-class capped + MI protocol, three-seed verification | **Main proposed architecture transfer result.** |

## 14. Contribution Summary

The CICIoT2023 contribution is:

> A third-dataset external validation showing that the proposed feature-axis
> multi-scale CNN-BiLSTM-attention architecture transfers to modern IoT
> network-flow traffic, improves over internal deep-learning baselines, and
> reaches a stable 71.13% macro-F1 after W&B tuning.

This supports the thesis because it shows:

- the proposed architecture is not only tuned for UNSW-NB15;
- macro-F1 remains the right metric for imbalanced multiclass IDS;
- BruteForce and Web remain difficult categories;
- hyperparameter optimization improved both accuracy and macro-F1;
- a BiGRU variant gives a close lighter-parameter architecture;
- the FT-Transformer result confirms that a generic transformer is not
  automatically better than the proposed architecture under the local protocol.

## 15. Limitations And Next Work

Limitations:

- accuracy remains lower than several published CICIoT2023 reports;
- many published near-perfect results appear protocol-sensitive and are not
  used as direct comparisons here;
- BruteForce and Web remain weak in the final global model;
- current protocol uses capped class sampling for local feasibility;
- robustness/generalization tests are planned but not yet completed.

Next work:

1. Run the prepared DL-only W&B baselines: MLP, TabTransformer-style,
   TabNet-like, and optional FT-Transformer sweep.
2. Decide whether to run BiGRU seeds `7` and `123`.
3. Add unseen-family and shard-shift tests for Objective 3.

## References

- CICIoT2023 official paper:
  <https://www.mdpi.com/1424-8220/23/13/5941>
- CICIoT2023 dataset page:
  <https://www.unb.ca/cic/datasets/index.html>
- FT-Transformer CICIoT2023 paper:
  <https://www.mdpi.com/2079-9292/15/12/2516>
- Sensors 2025 Transformer/DNN/CNN study:
  <https://www.mdpi.com/1424-8220/25/15/4845>
- TransNeSt hybrid architecture:
  <https://www.techscience.com/CMES/v145n3/65008/html>
