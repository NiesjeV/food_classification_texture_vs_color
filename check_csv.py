import pandas as pd

# Laad de CSV
df = pd.read_csv('data/train_labels.csv')

print("Eerste 5 rijen van de CSV:")
print(df.head())

print("\nKolomnamen:")
print(df.columns.tolist())

print("\nEerste 5 bestandsnamen:")
print(df.iloc[:5, 0].tolist())  # Eerste kolom