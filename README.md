# COMP 472 – Artificial Intelligence Project

This project implements and compares machine learning algorithms on two independent
classification tasks, using pre-extracted numerical feature vectors:

| Task | Features | Classes | Labeled examples | Test examples |
| --- | ---: | ---: | ---: | ---: |
| Visual object recognition | 500 | 50 (labels 0–49) | 20,000 | 5,000 |
| Sentiment analysis (frozen BERT embeddings) | 768 | 3 (0 = negative, 1 = neutral, 2 = positive) | 12,000 | 5,000 |

Algorithms: k-nearest neighbours (k-NN), decision tree, random forest, a feed-forward
neural network, and k-means clustering. A separate model is trained for each task.

**Quick start** (from the repository root, after the setup below):

```bash
python experiments/knn_experiment.py              # train + evaluate k-NN
python experiments/tree_experiment.py --sweep final   # train + evaluate the final random forests (see note in section 4)
python experiments/neural_network_experiment.py   # train + evaluate the neural network
python experiments/kmeans_experiment.py           # k-means clustering
python experiments/analysis.py                    # figures and tables for all of the above
python submission.py                              # apply the final model: Kaggle submission
```

---

## 1. Repository contents

| File | Purpose |
| --- | --- |
| **Models** (`models/`) | Each algorithm as a class with `fit` and `predict` |
| `models/knn.py` | k-nearest neighbours, from scratch in NumPy: Euclidean or cosine distance, majority or distance-weighted vote |
| `models/decision_tree.py` | CART decision tree, from scratch in NumPy: Gini, entropy or misclassification criterion; histogram-based split search |
| `models/random_forest.py` | Random forest built on `models/decision_tree.py`: bootstrap samples, random features per split, vote over the trees, out-of-bag accuracy |
| `models/neural_network.py` | Feed-forward neural network in PyTorch, its input preprocessing (`InputPipeline`, word dropout), training loop and prediction |
| `models/kmeans.py` | k-means clustering, from scratch in NumPy |
| **Experiments** (`experiments/`) | Scripts that train and evaluate the models |
| `experiments/knn_experiment.py` | Trains and evaluates k-NN for every combination of k, distance, scaling and voting |
| `experiments/tree_experiment.py` | Trains and evaluates decision trees and random forests: depth, leaf size, criterion, bins, number of trees, features per split, tree vs forest, final forest |
| `experiments/neural_network_experiment.py` | Trains and evaluates the neural network: one setting changed at a time, seed repeats, final 3-network model |
| `experiments/kmeans_experiment.py` | Runs k-means on the sentiment data and compares the clusters with the true classes |
| `experiments/analysis.py` | Makes every figure and table from the saved results (no retraining), and compares the classifiers |
| **Shared code** (`utils/`) | Used by every algorithm |
| `utils/__init__.py` | Marks `utils` as a Python package |
| `utils/data_loader.py` | Loads a Kaggle CSV and checks it (columns, feature order, row count, IDs); returns features, labels and IDs |
| `utils/preprocessing.py` | The shared 80/20 train/validation split, standardization, and recovery of image word counts |
| `utils/metrics.py` | Accuracy, confusion matrix, per-class precision/recall/F1 |
| `utils/results.py` | Saves every run's settings, accuracy and validation predictions in the same CSV format |
| **Tests** (`tests/`) | Unit tests on small made-up data (no Kaggle files needed) |
| `tests/test_data_pipeline.py` | Data loading, the shared split, standardization, count recovery |
| `tests/test_neural_network.py` | Network, word dropout, training, experiment records, submission checks |
| `tests/test_tree_models.py` | Decision tree and random forest correctness and reproducibility, binning |
| `tests/test_results_and_analysis.py` | The shared results format and the analysis script |
| **Other** | |
| `submission.py` | Applies the final model (neural network) to the test data and writes the Kaggle submission |
| `requirements.txt` | Python packages, with the versions the project was tested with |
| `.gitignore` | Keeps data, results, submissions and caches out of Git |
| `data/` | The five Kaggle CSVs (downloaded; not in Git) |
| `results/` | Created by the experiment and analysis scripts (not in Git) |
| `submissions/` | Created by `submission.py` (not in Git) |

## 2. Setup

1. **Python 3.11 or newer** (tested on 3.11 and 3.14).
2. **Install the packages** from the repository root:
   ```bash
   pip install -r requirements.txt
   ```
3. **Download the data** from the
   [COMP 472 Kaggle competition](https://www.kaggle.com/competitions/comp-472-project-fall-2026-concordia/data)
   and put these five files in `data/`:
   ```text
   data/image_train.csv      data/image_test.csv
   data/sentiment_train.csv  data/sentiment_test.csv
   data/sample_submission.csv
   ```
4. **Check that everything works** (a few seconds; no data needed):
   ```bash
   python -m unittest discover -s tests -v
   ```

Run every command from the repository root. A GPU is not required.

## 3. Data preprocessing

**Preprocessing runs automatically as the first step of every experiment script and of
`submission.py`**, so there is no separate command to run before training. Each script:

1. **Loads and validates** each CSV with `load_dataset` (`utils/data_loader.py`): checks
   the expected columns and row counts, puts the features in numeric order
   (`feature_0`, `feature_1`, …), and separates them from the `id` and `label` columns,
   which are never used as features.
2. **Splits** the labeled data with `split_dataset` (`utils/preprocessing.py`): a
   stratified 80% training / 20% validation split with seed 42. Every algorithm uses this
   same split, so they are evaluated on the same examples.
3. **Applies the algorithm's own preprocessing**, learned from the training split only
   and then applied unchanged to the validation (and test) data, so no information leaks
   from the evaluation data:

| Algorithm | Preprocessing |
| --- | --- |
| k-NN | Both unscaled and standardized features are tested (each feature scaled to mean 0, standard deviation 1, using training statistics) |
| Decision tree, random forest | No scaling (a tree's splits do not change when a feature is rescaled). Features are binned onto a grid computed from the training rows, to speed up the split search |
| Neural network | Image: each row is turned back into integer visual-word counts (`recover_counts`), with word dropout during training. Sentiment: standardized |
| k-means | None: clusters the raw sentiment features |

To run the shared preprocessing steps on their own (for example, to check the data):

```bash
python -c "
from utils.data_loader import load_dataset
from utils.preprocessing import split_dataset, standardize_split
for task in ('image', 'sentiment'):
    split = split_dataset(load_dataset(task))      # steps 1 and 2
    scaled, scaler = standardize_split(split)      # step 3: standardization, fitted on training only
    print(task, 'train', split.train.X.shape, 'validation', split.validation.X.shape)
"
```

## 4. Train and evaluate

Every experiment script trains its models on the 80% training split and evaluates them
on the 20% validation split. It runs everything for both datasets with no arguments, and
saves the results to `results/<algorithm>/` (see [Results](#7-results-files)).

### k-nearest neighbours

```bash
python experiments/knn_experiment.py
```

Tries every combination of k (1, 3, 5, 10, 20), distance (Euclidean, cosine), scaling
(unscaled, standardized) and voting (majority, distance-weighted): 40 runs per dataset.

### Decision tree and random forest

**Recommended: run only the final forests** (a few minutes). This trains and evaluates the
final random forest for each task, the result used to compare the classifiers:

```bash
python experiments/tree_experiment.py --sweep final
```

**Full run: every tree experiment in the report.** This takes a few hours on a laptop CPU,
because it grows about 4,000 trees from scratch (mostly in the features-per-split sweep):

```bash
python experiments/tree_experiment.py
```

The full run contains single-tree sweeps of depth, minimum leaf size and criterion; forest sweeps of number of
bins, number of trees and features per split; tree vs forest at matched depths; and the
final forest for each task:

| Final forest | Image | Sentiment |
| --- | --- | --- |
| Trees / criterion | 300 / Gini | 300 / Gini |
| Maximum depth | unrestricted | 12 |
| Features per split | 0.05 (25 of 500) | sqrt (27 of 768) |

### Neural network

```bash
python experiments/neural_network_experiment.py                          # every experiment
python experiments/neural_network_experiment.py --dataset image --ensemble       # only the final model
python experiments/neural_network_experiment.py --dataset image --dropout 0.2    # one run, one setting changed
```

Changes one setting at a time from the final configuration (layer size, depth,
activation, dropout, word dropout, learning rate, epochs, preprocessing), repeats the
final configuration with two more seeds to measure seed-to-seed variation, and evaluates
the final model: the average of three networks (seeds 42, 7 and 123). Training always
runs for a fixed number of epochs. `--help` lists every option.

| Final configuration | Image (500 → 50) | Sentiment (768 → 3) |
| --- | --- | --- |
| Hidden layers | 1 layer of 4096 neurons | 1 layer of 256 neurons |
| Activation / dropout | GELU / 0.4 | GELU / 0.2 |
| Input preprocessing | visual-word counts, word dropout 0.4 | standardized |
| Optimizer | AdamW, learning rate 0.001, weight decay 0.05 | AdamW, learning rate 0.0001 |
| Learning-rate schedule | one-cycle (warm-up, then cosine decay) | constant |
| Batch size / epochs | 128 / 40 | 64 / 15 |
| Loss | cross-entropy | cross-entropy |

### k-means clustering

```bash
python experiments/kmeans_experiment.py
```

Clusters all sentiment training data (features only) with k = 1, 3, 5, ..., 25, then uses the
true labels to evaluate the clusters: purity (share of examples in their cluster's most
common class), WCSS (within-cluster sum of squared distances) and the class mix of each
cluster.

## 5. Analysis: figures and tables

```bash
python experiments/analysis.py                    # all algorithms
python experiments/analysis.py --algorithm knn    # one algorithm
```

Reads the saved results (no retraining) and writes to `results/<algorithm>/figures/`:
accuracy against each setting that was varied, the confusion matrix and per-class
precision/recall/F1 of the best run, learning curves (neural network) and cluster
composition charts (k-means). `results/comparison/` gets the validation accuracy of each
classifier's final model and how often their mistakes overlap.

## 6. Kaggle submission

The neural network is the final model, as it has the best validation accuracy on both
tasks.

```bash
python submission.py
```

For each task, this retrains the final configuration on all labeled data (preprocessing
is refitted on that data), trains three networks (seeds 42, 7 and 123), averages their
predicted class probabilities on the test data, matches the predictions to
`data/sample_submission.csv` by ID, checks them (every ID exactly once, labels in range)
and writes `submissions/submission.csv` (`id,label`).


## 7. Results files

Each experiment script saves to `results/<algorithm>/` (`knn`, `trees`, `neural_network`,
`kmeans`) through `utils/results.py`. A full run starts from empty files.

| File | One row per | Contents |
| --- | --- | --- |
| `experiments.csv` | run | dataset, sweep (which setting was varied), experiment name, every setting, validation (and train) accuracy, time |
| `validation_predictions.csv` | run and validation example | example ID, true label, predicted label |
| `history.csv` (neural network) | run, seed and epoch | train and validation accuracy and loss after each epoch |
| `cluster_composition.csv` (k-means) | k and cluster | number of examples of each class in each cluster |

