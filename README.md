# COMP 472 – Artificial Intelligence Project

Concordia University | Fall 2026

## Overview

This project focuses on implementing and evaluating different artificial intelligence algorithms on two classification tasks:

- **Visual Object Recognition:** Classifying images into 50 categories using numerical feature vectors.
- **Sentiment Analysis:** Classifying text as negative, neutral, or positive using BERT embeddings.

The objective is to compare the algorithms' performance, investigate hyperparameters, and analyze their results.

## Algorithms

- k-Nearest Neighbours (k-NN)
- Decision Tree
- Random Forest
- Feed-Forward Neural Network
- k-Means Clustering

## Technologies

- Python
- NumPy
- Pandas
- PyTorch
- Matplotlib
- Jupyter Notebook

## Project Structure

```text
COMP472-Project-Fall2026/
├── models/          # Algorithm implementations
├── utils/           # Data loading, preprocessing, evaluation
├── experiments/     # Training and hyperparameter experiments
├── results/         # Experimental results and visualizations
│   └── figures/
├── tests/           # Algorithm and integration tests
├── data/            # Local Kaggle datasets (not tracked)
├── submissions/     # Kaggle prediction files (not tracked)
├── submission.py    # Train final models and write the Kaggle CSV
├── requirements.txt
└── README.md
```

## Team Responsibilities

| Member | Responsibilities |
|---|---|
| Member 1 | k-NN, Clustering, Data Preprocessing |
| Member 2 | Decision Tree, Random Forest |
| Member 3 | Neural Network, Evaluation, Kaggle Submission |

## Dataset

The datasets are available through the [COMP 472 Kaggle Competition](https://www.kaggle.com/competitions/comp-472-project-fall-2026-concordia/overview).

## Report and Presentation

The report and presentation will be maintained as shared Microsoft Word and PowerPoint files.

## Shared data pipeline

Place the Kaggle CSVs in `data/` and install the dependencies in `requirements.txt`.
Run from the repository root:

```python
from utils.data_loader import load_dataset
from utils.preprocessing import split_dataset, standardize_split

data = load_dataset("image")  # Use "sentiment" for the other task
split = split_dataset(data)   # Stratified 80/20 split, seed 42

# Optional scaling, fitted only on training features
scaled, scaler = standardize_split(split)
X_train, y_train = scaled.train.X, scaled.train.y
X_val, y_val = scaled.validation.X, scaled.validation.y
```

All classifiers should use the same split defaults and dataset. IDs are preserved
in `.ids` for matching predictions to samples. Load hidden test data with
`load_dataset("image", "test")` and apply the fitted scaler with
`scaler.transform(test.X)`, where `test` is the loaded test dataset. Repeat
separately for sentiment, using its own scaler.

Run the tests:

```bash
python -m unittest discover -s tests -v
```

## Neural network

A feed-forward network with one hidden layer (PyTorch), trained separately for
each task. Code: `models/neural_network.py`.

| Setting | Image (500 → 50) | Sentiment (768 → 3) |
| --- | --- | --- |
| Hidden layer | 4096 neurons, GELU, dropout 0.4 | 256 neurons, GELU, dropout 0.2 |
| Input | Visual-word counts, each row scaled to length 1 | Standardized features |
| Word dropout | 0.4 | — |
| Optimizer | AdamW, learning rate 0.001, weight decay 0.05 | AdamW, learning rate 0.0001 |
| Learning-rate schedule | One-cycle (warm up, then cosine decay) | Constant |
| Batch size / epochs | 128 / 40 | 64 / 15 |

How it works:

- **Counts:** each image row is a vector of visual-word counts scaled to
  length 1. `recover_counts` (in `utils/preprocessing.py`) divides each row by
  its smallest non-zero value to get the counts back.
- **Word dropout:** in each training batch, every counted word is removed with
  probability 0.4. This stops the network memorizing exact rows.
  Validation and test data are never changed.
- **Evaluation:** training runs for a fixed number of epochs, and the final epoch is
  reported. The validation set never chooses an epoch. Scaling is fitted on
  training data only.

Run from the repository root:

```bash
# Train on the 80% split, evaluate on the 20% split
python -m experiments.train_neural_network --dataset image
python -m experiments.train_neural_network --dataset image --word-dropout 0   # change one setting

# Learning curves, confusion matrix, per-class metrics; compare runs
python -m experiments.plot_neural_network --run-id RUN_ID
python -m experiments.plot_neural_network --dataset image --compare word_dropout

# Retrain on all labeled data and write submissions/submission.csv:
# averages 3 models (seeds 42, 7, 123) per task; --seeds 42 gives a single model
python submission.py
```

Use `--help` to list every setting. Each run saves its config, per-epoch
history and validation predictions to `results/neural_network/runs/<run_id>/`,
and adds a summary row to `results/neural_network/experiments.csv`.
