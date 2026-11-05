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
├── submission.py    # Final prediction generation
├── requirements.txt
└── README.m
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
