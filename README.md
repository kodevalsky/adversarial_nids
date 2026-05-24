# Adversarial Robustness of Unsupervised NIDS

This repository contains the experimental framework accompanying the paper *"Adversarial Robustness and Performance of Anomaly Detection Models in Network Traffic Analysis using Masked FGSM and PGD Attacks"* (KES 2026).

The codebase evaluates the adversarial robustness and inference latency of three unsupervised anomaly detection architectures — **Deep Autoencoders**, **AnoGAN**, and **f-AnoGAN** — under domain-constrained **Masked FGSM** and **Masked PGD** attacks on the **UNSW-NB15** and **CIC-IDS2017** intrusion detection datasets.

## Overview

Signature-based intrusion detection systems are fundamentally blind to zero-day exploits, motivating the adoption of unsupervised anomaly detection that learns the manifold of benign network traffic. These models, however, are vulnerable to gradient-based evasion attacks. This project investigates the trade-off between adversarial resilience, computational latency, and initialization stability across three representative architectures, under a realistic threat model that enforces protocol-valid perturbations.

Key components implemented in this repository:

- **Unsupervised training** of all three models exclusively on benign traffic flows.
- **Masked FGSM** — a single-step gradient attack that locks categorical features via a binary mask and enforces integer rounding on discrete fields.
- **Masked PGD** — an iterative variant that projects perturbations onto an L∞ ε-ball while respecting the same domain constraints.
- **Multi-seed evaluation** across three independent random initializations to quantify stability.
- **Inference latency benchmarking** to assess viability for high-throughput deployment.

## Repository Structure

```
adversarial_nids/
├── src/
│   ├── autoencoder.py        # Autoencoder architecture and training
│   ├── gan.py                # AnoGAN Generator and Discriminator
│   ├── anogan.py             # Iterative latent optimization for AnoGAN inference
│   ├── f_anogan.py           # f-AnoGAN Encoder training (Generator frozen)
│   ├── preprocess_data.py    # Cleaning, one-hot encoding, Min-Max scaling
│   ├── evaluate.py           # Full attack and latency evaluation pipeline
│   ├── utils.py              # Deterministic seeding helpers
│   └── datasets/             # Place UNSW-NB15 and CIC-IDS2017 here
├── models/                   # Saved model checkpoints (per seed, per dataset)
├── plots/                    # Output figures (PDF)
├── test_gpu.py               # GPU availability check
├── requirements.txt
└── LICENSE
```

## Models and Hyperparameters

All models were trained from scratch using three independent random seeds (`42`, `123`, `2026`) on each dataset. Continuous features were Min-Max scaled to `[0, 1]`; categorical features were one-hot encoded.

### Autoencoder

Symmetric fully-connected architecture with the following layer widths:

| Stage | Layer dimensions |
|---|---|
| Encoder | input → 128 → 64 → 36 → 18 → **9** (latent) |
| Decoder | 9 → 18 → 36 → 64 → 128 → input |

- Activations: LeakyReLU (slope 0.01); Sigmoid output
- Optimizer: Adam, lr = 1e-3
- Batch size: 4096
- Up to 150 epochs with early stopping (patience 10) on validation loss
- Gradient clipping at max-norm 1.0
- Loss: MSE reconstruction

### AnoGAN

| Component | Layer dimensions |
|---|---|
| Generator | 64 (latent) → 64 → 128 → 256 → input |
| Discriminator | input → 256 → 128 → 64 → 1 |

- Activations: LeakyReLU (slope 0.2); Sigmoid output
- Optimizer: Adam, lr = 2e-4 (both networks)
- Batch size: 4096
- Epochs: 50
- Loss: BCE

**Inference (latent search):** Adam optimization over the latent vector z for 500 iterations (lr = 0.1), minimizing the weighted anomaly score `A(x) = (1 − λ)·R(x) + λ·D(x)` with `λ = 0.1`.

### f-AnoGAN

After AnoGAN training, the Generator and Discriminator weights are frozen and an Encoder is trained to invert the Generator.

| Component | Layer dimensions |
|---|---|
| Encoder | input → 128 → 64 → **64** (latent) |

- Activations: LeakyReLU (slope 0.2)
- Optimizer: Adam, lr = 1e-3
- Batch size: 2048
- Epochs: 15
- Loss: MSE on G(E(x)) vs x (izi mapping)

## Attack Parameters

| Parameter | Masked FGSM | Masked PGD |
|---|---|---|
| Perturbation budget ε | 0.1 | 0.05 |
| Step size α | — | 0.01 |
| Iterations | 1 | 10 |
| Norm | L∞ | L∞ |
| Categorical mask | yes | yes |
| Integer rounding | yes | yes |
| Clipping | `[0, 1]` | `[0, 1]` |

Time-based features are locked (mask value 0) since realistic temporal perturbations cannot be produced without invalidating network flow state. Masks lock 20 of 56 features for UNSW-NB15 and 16 of 78 features for CIC-IDS2017.

## Getting Started

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- [torch-directml](https://pypi.org/project/torch-directml/) for AMD GPU acceleration, or standard CUDA for NVIDIA hardware

### Installation

```bash
git clone https://github.com/kodevalsky/adversarial_nids.git
cd adversarial_nids
pip install -r requirements.txt
```

### Dataset Setup

Download the datasets from their official sources and place them as follows:

- UNSW-NB15 → `./src/datasets/unsw/`
- CIC-IDS2017 → `./src/datasets/cic/`

The `preprocess_data.py` module handles cleaning, one-hot encoding, and Min-Max scaling automatically when invoked by the training scripts.

### Training

Run each script from the `src/` directory. Each script trains the corresponding model across all three seeds and both datasets:

```bash
# 1. Train Autoencoders
python autoencoder.py

# 2. Train the GAN pairs (required before f-AnoGAN)
python gan.py

# 3. Train the f-AnoGAN Encoders (uses frozen Generators from step 2)
python f_anogan.py
```

Checkpoints are saved under `./models/` as `<model>_<dataset>_seed<N>.pth`.

### Evaluation

Run the full evaluation pipeline to execute both attacks, benchmark inference latency, and regenerate the figures:

```bash
python evaluate.py
```

This produces the following PDF outputs under `./plots/`:

- `feature_sensitivity_<dataset>.pdf` — per-feature gradient magnitudes under Masked FGSM
- `unified_evasion_comparison_<dataset>.pdf` — anomaly scores before vs. after attacks
- `inference_latency_<dataset>.pdf` — per-model forward-pass latency on a log scale

## Results Summary

Aggregated results over three seeds (full details in the paper):

| Dataset | Model | FGSM Evasion | PGD Evasion | FGSM ASR | PGD ASR | Inf. Time (ms) |
|---|---|---|---|---|---|---|
| UNSW-NB15 | Autoencoder | 25.9 ± 8.6% | 37.4 ± 12.1% | 0.0 ± 0.0% | 1.0 ± 1.8% | 0.0195 |
| UNSW-NB15 | AnoGAN | 9.9 ± 2.3% | 11.0 ± 1.2% | 0.0 ± 0.0% | 0.0 ± 0.0% | 9.7945 |
| UNSW-NB15 | f-AnoGAN | 35.4 ± 49.4% | 37.6 ± 50.8% | 34.1 ± 30.2% | 34.6 ± 30.7% | 0.0168 |
| CIC-IDS2017 | Autoencoder | −42.4 ± 23.4% | 38.2 ± 8.3% | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0055 |
| CIC-IDS2017 | AnoGAN | 6.4 ± 7.5% | 8.5 ± 9.2% | 0.0 ± 0.0% | 0.0 ± 0.0% | 10.1087 |
| CIC-IDS2017 | f-AnoGAN | 22.6 ± 86.5% | 21.7 ± 84.1% | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0081 |

**ASR** (Attack Success Rate) counts a sample as evaded if its anomaly score falls below the 95th-percentile threshold of benign samples. Negative evasion values indicate gradient overshoot — a phenomenon corrected by iterative PGD.

## Citation

If you use this code or build upon this work, please cite:

```bibtex
@inproceedings{kowalski2026adversarial,
  title     = {Adversarial Robustness and Performance of Anomaly Detection Models in Network Traffic Analysis using Masked FGSM and PGD Attacks},
  author    = {Kowalski, Jeremi and Nowak-Brzezi{\'n}ska, Agnieszka},
  booktitle = {Procedia Computer Science (KES 2026)},
  year      = {2026}
}
```

## License

This project is released under the MIT License — see [LICENSE](LICENSE) for details.

## Contributing

Issues and pull requests are welcome, particularly for:

- Additional unsupervised baselines (Isolation Forest, DeepSVDD, VAE-based detectors)
- Adversarial training defenses and regularization of the f-AnoGAN latent mapping
- Sparse L0 attacks (e.g., JSMA) for seed-agnostic robustness analysis
- Extensions to additional intrusion datasets

## Acknowledgements

The authors thank the UNSW Canberra Cyber research team and the Canadian Institute for Cybersecurity for releasing the UNSW-NB15 and CIC-IDS2017 datasets, respectively.
