from sklearn.model_selection import train_test_split

def make_train_val_split(labels_df, val_size=0.2, seed=42):
    train_df, val_df = train_test_split(
        labels_df,
        test_size=val_size,
        random_state=seed,
        stratify=labels_df["label"]
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)