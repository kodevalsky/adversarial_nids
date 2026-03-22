# Adversarial Resilience of Unsupervised IDS

This repository contains the experimental framework for evaluating the adversarial robustness of unsupervised anomaly detection models in Network Intrusion Detection Systems (IDS). 

The project compares **Deep Autoencoders (AE)**, **f-AnoGAN** and **AnoGAN** architectures against a domain-constrained **Masked Fast Gradient Sign Method (FGSM)** attack using the **UNSW-NB15** and **CIC-IDS2017** datasets and differences in inference time between aforementioned models.

## 📌 Project Overview

Traditional supervised IDS often fail against zero-day exploits. Unsupervised models learn the "normal" network manifold to detect deviations; however, these manifolds are vulnerable to adversarial evasion. This project implements:

* **Unsupervised Learning:** Models trained strictly on benign traffic.
* **f-AnoGAN:** Fast adversarial anomaly detection using a Generator-Encoder-Discriminator pipeline.
* **Masked FGSM:** A modified adversarial attack that preserves network protocol integrity by locking categorical features and rounding discrete integers.
* **AnoGAN** An industry-standard approach to anomaly detection based on the Generative Adversarial Network Architecture

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* PyTorch 2.0+
* [Torch-DirectML](https://pypi.org/project/torch-directml/) (for AMD GPU acceleration) or standard CUDA for NVIDIA.

### Installation
Clone the repository:

```bash
git clone [https://github.com/yourusername/ids-adversarial-resilience.git](https://github.com/yourusername/ids-adversarial-resilience.git)

cd ids-adversarial-resilience
```
Install dependencies:

```Bash
pip install -r requirements.txt
```

## 🛠️ Usage

1. **Preprocessing**

Ensure your datasets are placed in ./src/datasets/unsw and ./src/datasets/cic. The preprocess_data.py script handles cleaning, one-hot encoding, and scaling.

2. **Training**

Train the models for both datasets:
```Bash
# Train the Autoencoders
python autoencoder.py

# Train the GAN
python gan_train.py

# Train the Encoder for the f-AnoGAN
python f_anogan.py
```

3. **Evaluation & Attack**

Run the full evaluation suite to generate the adversarial results and figures:

```Bash
python evaluate.py
```

## 📊 Results & Visualizations

Running the evaluation script will generate several PDF reports used in the PP-RAI extended abstract:

- *feature_sensitivity_[dataset].pdf* - Visualizes which network features drive the anomaly score.

- *unified_evasion_comparison_[dataset].pdf* - Compares the detection loss before and after the evasion attack.

- *inference_latency_[dataset].pdf* - Shows the differences in inference time between the models

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🧠 Contributions

Feel free to propose PR's and issues to expand the research and project scope.