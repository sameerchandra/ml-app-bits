import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(
    data: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if target_column not in data.columns:
        raise ValueError(f"Dataset must contain target column '{target_column}'")

    return train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=data[target_column],
    )
