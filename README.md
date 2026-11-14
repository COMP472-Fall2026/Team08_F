# COMP 472 – Artificial Intelligence Project

Concordia University | Fall 2026

## Overview

This project implements and compares machine learning algorithms on two
independent classification tasks, using pre-extracted numerical feature vectors:

| Task | Features | Classes | Labeled examples | Test examples |
| --- | ---: | ---: | ---: | ---: |
| Visual object recognition | 500 | 50 (labels 0–49) | 20,000 | 5,000 |
| Sentiment analysis (frozen BERT embeddings) | 768 | 3 (0 = negative, 1 = neutral, 2 = positive) | 12,000 | 5,000 |

Algorithms: k-Nearest Neighbours, Decision Tree, Random Forest, a feed-forward
Neural Network, and k-Means clustering. A separate model is trained for each task.

## Team responsibilities

| Member | Responsibilities |
| --- | --- |
| Member 1 | k-NN, clustering, data preprocessing |
| Member 2 | Decision Tree, Random Forest |
| Member 3 | Neural Network, evaluation, Kaggle submission |

## Setup

1. **Python:** tested with Python 3.14.7. Exact library versions are in `requirements.txt`.
2. **Install the dependencies** from the repository root:
   ```bash
   pip install -r requirements.txt
   ```
3. **Download the data** from the
   [COMP 472 Kaggle competition](https://www.kaggle.com/competitions/comp-472-project-fall-2026-concordia/data)
   and place these five files in `data/`:
   ```text
   data/image_train.csv      data/image_test.csv
   data/sentiment_train.csv  data/sentiment_test.csv
   data/sample_submission.csv
   ```
4. **Check the installation** (no data needed, takes a few seconds):
   ```bash
   python -m unittest discover -s tests -v
   ```

Run every command from the repository root. A GPU is not required.

## Project structure

```text
COMP472-Project-Fall2026/
├── models/
│   └── neural_network.py         # Network, input preprocessing, training loop, prediction
├── utils/
│   ├── data_loader.py            # Load and validate the Kaggle CSVs
│   └── preprocessing.py          # Shared 80/20 split, standardization, word-count recovery
├── experiments/
│   └── train_neural_network.py   # Neural network experiments; results saved to CSV
├── tests/
│   ├── test_data_pipeline.py     # Data loading, split and preprocessing tests
│   └── test_neural_network.py    # Network, training, experiment and submission tests
├── submission.py                 # Train the final models and write the Kaggle submission
├── requirements.txt              # Python dependencies and tested versions
├── data/                         # Kaggle CSVs (downloaded, not tracked)
├── results/                      # Experiment results (generated, not tracked)
└── submissions/                  # Kaggle submission files (generated, not tracked)
```

## Data pipeline (shared by all classifiers)

- **Loading** (`utils/data_loader.py`): reads `{task}_{train|test}.csv`, checks the
  expected columns and row counts, and returns features `X` (float32), labels `y`
  and `ids`. The `id` column is never used as a feature, and `label` only as the target.
- **Validation split** (`utils/preprocessing.py`): a stratified 80% train / 20%
  validation split with seed 42, made separately for each task. Every classifier
  uses this same split, so their results are directly comparable.
- **No data leakage:** any learned preprocessing (such as standardization) is
  fitted on the training subset only, then applied unchanged to validation and test data.

```python
from utils.data_loader import load_dataset
from utils.preprocessing import split_dataset, standardize_split

data = load_dataset("image")               # or "sentiment"
split = split_dataset(data)                # stratified 80/20 split, seed 42
scaled, scaler = standardize_split(split)  # optional; fitted on training data only
X_train, y_train = scaled.train.X, scaled.train.y
X_val, y_val = scaled.validation.X, scaled.validation.y

test = load_dataset("image", "test")       # hidden test set: features and IDs, no labels
X_test = scaler.transform(test.X)
```

## Neural network

A feed-forward network written with PyTorch (`models/neural_network.py`):
input layer → one or more fully connected hidden layers (each followed by an
activation and dropout) → one output per class. It is trained with cross-entropy
loss and mini-batches. The number and size of hidden layers, activation (ReLU or
GELU), dropout, optimizer (Adam or AdamW), learning rate, learning-rate schedule,
weight decay, batch size and number of epochs are all configurable.

### Final configuration

| Setting | Image (500 → 50) | Sentiment (768 → 3) |
| --- | --- | --- |
| Hidden layers | 1 layer of 4096 neurons | 1 layer of 256 neurons |
| Activation / dropout | GELU / 0.4 | GELU / 0.2 |
| Input preprocessing | Visual-word counts, word dropout 0.4 | Standardized features |
| Optimizer | AdamW, learning rate 0.001, weight decay 0.05 | AdamW, learning rate 0.0001, no weight decay |
| Learning-rate schedule | One-cycle (warm-up, then cosine decay) | Constant |
| Batch size / epochs | 128 / 40 | 64 / 15 |
| Loss | Cross-entropy | Cross-entropy |

These defaults are defined in `DEFAULTS` in `experiments/train_neural_network.py`.

### Preprocessing

Preprocessing runs automatically inside training, so there is no separate step.
It is fitted on the training data only.

- **Image:** each feature row is a bag of visual-word counts scaled to length 1.
  `recover_counts` divides each row by its smallest non-zero value to get the
  integer counts back. During training, **word dropout** randomly removes each
  counted word with the given probability, giving the network a slightly different
  version of each image every epoch, which reduces overfitting. Each row is then
  rescaled to length 1. Validation and test data are never altered.
- **Sentiment:** each feature is standardized to mean 0 and standard deviation 1,
  using statistics from the training data.

### Train and evaluate

`experiments/train_neural_network.py` trains on the 80% split and evaluates on the
20% validation split. Training runs for a fixed number of epochs, and the model
after the final epoch is evaluated; validation data never influences training.

```bash
# All experiments used in the report, for one task
python -m experiments.train_neural_network --dataset image --all
python -m experiments.train_neural_network --dataset sentiment --all

# A single run with the final configuration
python -m experiments.train_neural_network --dataset image

# A single run with settings changed (see --help for every option)
python -m experiments.train_neural_network --dataset image --hidden-sizes 512 256 --learning-rate 0.01

# The final submitted model (average of 3 models with different seeds), on validation
python -m experiments.train_neural_network --dataset image --ensemble
```

`--all` runs every experiment in `EXPERIMENTS` (each changes one setting from the
final configuration), the final configuration again with two other seeds to measure
seed-to-seed variation, and the final ensemble. It takes about 16 minutes for image
and under a minute for sentiment on a laptop CPU.

### Results

Every run adds rows to three CSV files in `results/neural_network/`, linked by `run_id`:

| File | One row per | Contents |
| --- | --- | --- |
| `experiments.csv` | run | Experiment name, every setting, seed(s), final train and validation accuracy and loss, training time, parameter count |
| `history.csv` | run, seed and epoch | Train and validation accuracy and loss after each epoch (learning curves) |
| `validation_predictions.csv` | run and validation example | Example ID, true label, predicted label (confusion matrices, per-class metrics, error analysis) |

For an ensemble, accuracy, loss and predictions come from the averaged class
probabilities, and `history.csv` holds the curves of each model in it.

### Apply the model (Kaggle submission)

```bash
python submission.py              # 3-model ensemble per task (seeds 42, 7, 123)
python submission.py --seeds 42   # a single model per task
```

For each task, this retrains the final configuration on all labeled data (preprocessing
is refitted on that data), predicts the hidden test set, and averages the models' class
probabilities. Predictions are matched to `data/sample_submission.csv` by ID and checked
(every ID exactly once, labels in range), then written to `submissions/submission.csv`
(`id,label`). The settings used are saved to `submissions/submission_config.json`.

## Reproducibility

- **Seeds:** the validation split always uses seed 42. Each training run's seed is
  recorded in `experiments.csv`; the final ensemble uses seeds 42, 7 and 123.
- **Settings:** every setting of every run is recorded in `experiments.csv`.
- **Versions:** Python and library versions are listed in `requirements.txt`.

Results should match closely when rerun, though not necessarily bit for bit,
since PyTorch results can vary slightly between machines.
