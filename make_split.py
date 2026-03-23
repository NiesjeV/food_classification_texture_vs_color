import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Laad de volledige labels van Kaggle
df = pd.read_csv('data/train_labels.csv')

# Controleer dat de kolommen kloppen
print("Kolommen:", df.columns)
# We gaan ervan uit dat ze heten: 'img_name' en 'label'

# 2. 80/20 stratified split op basis van het label
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['label']  # zorgt dat de label-verdeling ongeveer gelijk blijft
)

print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

# 3. Schrijf weg naar CSV in de data-map
train_df.to_csv('data/train_split.csv', index=False)
val_df.to_csv('data/val_split.csv', index=False)

print("Splits opgeslagen als data/train_split.csv en data/val_split.csv")
