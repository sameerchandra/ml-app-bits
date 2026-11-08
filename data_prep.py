import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(
    data: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if target_column not in data.columns:
        raise ValueError(f"Dataset must contain target column '{target_column}'")

    return train_test_split(
        data.drop(columns=[target_column]),
        data[target_column],
        test_size=0.2,
        random_state=2026,
        stratify=data[target_column],
    )
