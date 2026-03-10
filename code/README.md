# NNTI Final Project – Language Identification from Audio

## Overview

Spoken language identification (SLID) system for **22 Indian languages** using
pretrained speech transformer models (≤ 600 M parameters).  The project
fine-tunes [facebook/w2v-bert-2.0](https://huggingface.co/facebook/w2v-bert-2.0)
on the `badrex/nnti-dataset-full` dataset and addresses speaker bias through
Domain Adversarial Neural Networks (DANN) and data augmentation.

### Tasks

| Task | Description |
|------|-------------|
| **Task 1** | Reproduce baseline & improve validation accuracy via hyperparameter tuning |
| **Task 2** | Mitigate speaker/dataset bias using DANN and speed-perturbation augmentation |
| **Task 3** | Analyse the model through confusion matrices, t-SNE, and speaker probes |

---

## Project Structure

```
code/
├── train_model.py              # Main training script (3 modes)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── src/
│   ├── __init__.py
│   ├── data.py                 # Dataset loading, preprocessing, collators
│   ├── model.py                # Model loading utilities
│   ├── dann.py                 # DANN: GRL, dual-head model, custom Trainer
│   ├── augmentation.py         # Speed perturbation augmentation
│   └── utils.py                # Metrics, embedding extraction, plotting
└── scripts/
    ├── make_plots.py           # Generate training curve plots from JSON logs
    ├── plot_model_comparison.py # 4-model accuracy comparison (run01)
    ├── make_tsne.py            # t-SNE from saved models
    └── make_confusion_matrix.py # Confusion matrix from saved models
```

---

## Requirements

- **Python** ≥ 3.9
- **GPU**: NVIDIA GPU with ≥ 16 GB VRAM recommended (e.g., T4, A100)
- **CUDA** ≥ 11.7

### Environment Setup

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Training

The unified `train_model.py` supports three modes:

### 1. Standard Fine-Tuning (Task 1)

```bash
python train_model.py \
    --mode standard \
    --model_id facebook/w2v-bert-2.0 \
    --lr 4e-5 \
    --epochs 6 \
    --warmup_ratio 0.15 \
    --batch_size 8 \
    --grad_accum 2 \
    --output_dir ./output_standard
```

### 2. DANN Training (Task 2 – Model-Based Bias Mitigation)

```bash
python train_model.py \
    --mode dann \
    --model_id facebook/w2v-bert-2.0 \
    --lr 4e-5 \
    --epochs 6 \
    --warmup_ratio 0.15 \
    --output_dir ./output_dann
```

### 3. DANN + Augmentation (Task 2 – Data-Centric Bias Mitigation)

```bash
python train_model.py \
    --mode dann_aug \
    --model_id facebook/w2v-bert-2.0 \
    --lr 4e-5 \
    --epochs 6 \
    --warmup_ratio 0.15 \
    --output_dir ./output_dann_aug
```

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | Training mode: `standard`, `dann`, `dann_aug` | `standard` |
| `--model_id` | HuggingFace model identifier | `facebook/w2v-bert-2.0` |
| `--lr` | Peak learning rate | `4e-5` |
| `--epochs` | Number of training epochs | `6` |
| `--warmup_ratio` | Fraction of steps for LR warmup | `0.15` |
| `--dropout` | Enable dropout (0.1) | `False` |
| `--batch_size` | Per-device batch size | `8` |
| `--grad_accum` | Gradient accumulation steps | `2` |
| `--hf_token` | HuggingFace API token | – |
| `--wandb_key` | Weights & Biases API key | – |

---

## Generating Plots

### All Training Curves

```bash
python scripts/make_plots.py \
    --results_dir /path/to/trainer_state_jsons \
    --output_dir ../report/figures
```

### 4-Model Comparison (Run01)

```bash
python scripts/plot_model_comparison.py \
    --results_dir /path/to/trainer_state_jsons \
    --output ../report/figures/model_comparison_accuracy.png
```

### t-SNE Visualisation

```bash
python scripts/make_tsne.py \
    --model_path ./output/final_model \
    --model_id facebook/w2v-bert-2.0 \
    --output ../report/figures/tsne.png
```

### Confusion Matrix

```bash
python scripts/make_confusion_matrix.py \
    --model_path ./output/final_model \
    --model_id facebook/w2v-bert-2.0 \
    --output ../report/figures/confusion_matrix.png
```

---

## Result Files

Trainer state JSON files follow the naming convention:

```
{model}_trainer_state_{checkpoint}_{run}.json
```

Each file contains:
- `log_history`: per-step training loss and periodic evaluation metrics
- `best_model_checkpoint`: path to the best checkpoint
- `best_metric`: best validation accuracy achieved

---

## Notes

- The dataset `badrex/nnti-dataset-full` is loaded automatically from HuggingFace.
- Audio is resampled to 16 kHz and truncated to 2 seconds.
- Effective batch size = `batch_size × grad_accum` = 16 by default.
- DANN uses the λ schedule from Ganin et al. (2016): λ(p) = 2/(1+exp(−10p)) − 1.
- Speed perturbation augmentation (0.9×/1.1× rate, 50% probability) uses librosa.
