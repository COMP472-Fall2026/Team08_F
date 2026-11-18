
"""Generate a Kaggle submission using kNN."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.knn import KNNClassifier
from utils.data_loader import load_dataset


# Model settings.
K = 3
DISTANCE = "cosine"
WEIGHTED = True

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "submissions"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions = []

    for task in ["image", "sentiment"]:
        print(f"Processing {task} dataset...", flush=True)

        train = load_dataset(task, "train", data_dir=DATA_DIR)
        test = load_dataset(task, "test", data_dir=DATA_DIR)

        # Create the kNN model.
        model = KNNClassifier(
            k=K,
            distance=DISTANCE,
            weighted=WEIGHTED
        )

        model.fit(train.X, train.y)

        predicted = model.predict(test.X)

        results = pd.DataFrame({
            "id": test.ids,
            "label": predicted
        })

        predictions.append(results)

    combined = pd.concat(predictions, ignore_index=True)

    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

    submission = sample[["id"]].merge(
        combined,
        on="id",
        how="left",
        validate="one_to_one"
    )

    assert len(submission) == 10000
    assert submission["label"].notna().all()

    submission["label"] = submission["label"].astype(int)

    image_mask = submission["id"].str.startswith("img_")
    sentiment_mask = submission["id"].str.startswith("text_")

    assert image_mask.sum() == 5000
    assert sentiment_mask.sum() == 5000
    assert submission.loc[image_mask, "label"].between(0, 49).all()
    assert submission.loc[sentiment_mask, "label"].between(0, 2).all()

    output = OUTPUT_DIR / "knn_submission.csv"
    submission.to_csv(output, index=False)

    print(f"Submission saved to: {output}")


if __name__ == "__main__":
    main()
