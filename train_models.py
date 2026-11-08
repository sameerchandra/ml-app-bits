from pathlib import Path

import joblib
import pandas as pd

from data_prep import split_dataset
from modeling import (
    make_decision_tree_pipeline,
    make_knn_pipeline,
    make_logistic_regression_pipeline,
    make_naive_bayes_pipeline,
    make_svm_pipeline,
)


DATA_PATH = Path("online_shoppers_intention.csv")
MODEL_DIRECTORY = Path("model")
TARGET_COLUMN = "Revenue"
MODEL_BUILDERS = {
    "Logistic Regression": make_logistic_regression_pipeline,
    "Decision Tree": make_decision_tree_pipeline,
    "K-Nearest Neighbors": make_knn_pipeline,
    "Gaussian Naive Bayes": make_naive_bayes_pipeline,
    "Support Vector Machine": make_svm_pipeline,
}


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    train_features, test_features, train_labels, test_labels = split_dataset(
        data, TARGET_COLUMN
    )
    MODEL_DIRECTORY.mkdir(exist_ok=True)

    for model_name, build_pipeline in MODEL_BUILDERS.items():
        pipeline = build_pipeline(train_features)
        pipeline.fit(train_features, train_labels)
        file_stem = model_name.lower().replace(" ", "_").replace("-", "_")
        model_path = MODEL_DIRECTORY / f"{file_stem}.joblib"
        joblib.dump(pipeline, model_path)
        print(f"Saved {model_name}: {model_path}")

    test_data = test_features.copy()
    test_data[TARGET_COLUMN] = test_labels
    test_data.to_csv("test_data.csv", index=False)
    print(f"Saved labeled test rows: {len(test_data)}")


if __name__ == "__main__":
    main()
