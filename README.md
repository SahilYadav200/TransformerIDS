# TransformerIDS

A **Transformer-based Intrusion Detection System (IDS)** that leverages the self-attention mechanism of Transformer neural networks to detect network intrusions and cyberattacks with high accuracy.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Datasets](#datasets)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

TransformerIDS applies the Transformer architecture—originally designed for natural language processing—to the domain of network intrusion detection. By treating network traffic feature sequences as input tokens, the model uses multi-head self-attention to capture complex relationships between traffic features, enabling it to distinguish between benign traffic and a wide variety of attack types (e.g., DoS, DDoS, port scans, brute-force, and more).

---

## Architecture

```
Network Traffic Features
        │
        ▼
  Input Embedding
        │
        ▼
Positional Encoding
        │
        ▼
┌───────────────────┐
│  Transformer      │
│  Encoder Block    │ × N layers
│  ─────────────    │
│  Multi-Head       │
│  Self-Attention   │
│  ─────────────    │
│  Feed-Forward     │
│  Network          │
│  ─────────────    │
│  Layer Norm &     │
│  Residual Conn.   │
└───────────────────┘
        │
        ▼
  Global Average
    Pooling
        │
        ▼
Classification Head
(Benign / Attack Type)
```

### Key Components

| Component | Description |
|-----------|-------------|
| **Input Embedding** | Projects raw numerical traffic features into the model's hidden dimension |
| **Positional Encoding** | Injects sequential order information into the embeddings |
| **Multi-Head Self-Attention** | Captures dependencies between feature dimensions across the traffic sample |
| **Feed-Forward Network** | Two-layer MLP applied position-wise after attention |
| **Classification Head** | Fully-connected layer(s) producing class probabilities |

---

## Features

- ✅ Multi-class classification (benign + multiple attack types)
- ✅ Transformer encoder with configurable depth and width
- ✅ Support for popular benchmark IDS datasets (NSL-KDD, CICIDS2017/2018, UNSW-NB15)
- ✅ Data preprocessing pipeline (normalization, label encoding, train/val/test split)
- ✅ Training with early stopping and learning-rate scheduling
- ✅ Evaluation metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- ✅ Model checkpointing and loading

---

## Datasets

The following publicly available datasets are supported:

| Dataset | Classes | Description |
|---------|---------|-------------|
| [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html) | 5 | Improved version of the KDD Cup 1999 dataset |
| [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) | 15 | Realistic network traffic with modern attack types |
| [CICIDS2018](https://www.unb.ca/cic/datasets/ids-2018.html) | 7 | Large-scale dataset with diverse attack scenarios |
| [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) | 10 | Contemporary network traffic with nine attack families |

Download the desired dataset and place it in the `data/` directory before training.

---

## Requirements

- Python 3.8+
- PyTorch 1.12+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- tqdm

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/SahilYadav200/TransformerIDS.git
   cd TransformerIDS
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate       # Linux / macOS
   venv\Scripts\activate          # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Data Preprocessing

```bash
python preprocess.py --dataset cicids2017 --data_dir data/ --output_dir data/processed/
```

### Training

```bash
python train.py \
  --data_dir data/processed/ \
  --epochs 50 \
  --batch_size 256 \
  --lr 1e-4 \
  --d_model 128 \
  --n_heads 8 \
  --n_layers 4 \
  --dropout 0.1 \
  --save_dir checkpoints/
```

### Inference / Testing

```bash
python evaluate.py \
  --data_dir data/processed/ \
  --checkpoint checkpoints/best_model.pt
```

---

## Model Training

The training pipeline includes:

1. **Preprocessing**: Feature normalization (StandardScaler / MinMaxScaler) and categorical label encoding.
2. **Train/Validation/Test Split**: Stratified split to preserve class distribution.
3. **Optimizer**: AdamW with weight decay.
4. **Loss Function**: Cross-Entropy Loss.
5. **Scheduler**: Cosine annealing or ReduceLROnPlateau.
6. **Early Stopping**: Monitors validation loss to prevent overfitting.

---

## Evaluation

After training, the model is evaluated using:

- **Accuracy** – Overall percentage of correct predictions
- **Precision** – Ratio of true positives to all predicted positives (per class)
- **Recall** – Ratio of true positives to all actual positives (per class)
- **F1-Score** – Harmonic mean of Precision and Recall
- **Confusion Matrix** – Visual breakdown of predicted vs. actual classes

---

## Results

Example results on the CICIDS2017 dataset:

| Metric | Score |
|--------|-------|
| Accuracy | ~99% |
| Macro F1-Score | ~98% |
| False Positive Rate | < 1% |

> **Note:** Actual results may vary depending on hyperparameters, preprocessing choices, and dataset splits.

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

Please ensure your code follows the existing style and includes relevant tests or documentation updates.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

**Sahil Yadav** – [GitHub](https://github.com/SahilYadav200)

If you find this project useful, please consider giving it a ⭐!