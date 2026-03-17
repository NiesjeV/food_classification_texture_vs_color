import pandas as pd
from sklearn.model_selection import train_test_split


def make_train_val_split(labels_df, val_size=0.2, seed=42):
    if "label" not in labels_df.columns:
        raise ValueError("CSV must contain a 'label' column.")

    if len(labels_df) == 0:
        raise ValueError("labels_df is empty.")

    train_df, val_df = train_test_split(
        labels_df,
        test_size=val_size,
        random_state=seed,
        stratify=labels_df["label"]
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


if __name__ == "__main__":
    labels_path = "data/train_labels.csv"
    train_out = "data/train_split.csv"
    val_out = "data/val_split.csv"

    labels_df = pd.read_csv(labels_path)

    print("Loaded labels:", len(labels_df))
    print("Columns:", labels_df.columns.tolist())

    train_df, val_df = make_train_val_split(labels_df, val_size=0.2, seed=42)

    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)

    print(f"Saved train split to: {train_out} ({len(train_df)} rows)")
    print(f"Saved val split to:   {val_out} ({len(val_df)} rows)")