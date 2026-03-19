import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from src.dataset import FoodDataset
from src.model import SimpleCNN
from sklearn.model_selection import train_test_split
from src.transforms import get_train_transforms, get_val_transforms

print("1. Data voorbereiden...")

# Labels laden
labels_df = pd.read_csv('data/train_labels.csv')

# Split data (80% train, 20% validation)
train_df, val_df = train_test_split(
    labels_df,
    test_size=0.2,
    stratify=labels_df['label'],
    random_state=42
)

# transforms
train_transform = get_train_transforms()
val_transform = get_val_transforms()

# Datasets
train_dataset = FoodDataset(
    img_dir='data/train_set',
    labels_df=train_df,
    transform=train_transform
)

val_dataset = FoodDataset(
    img_dir='data/train_set',
    labels_df=val_df,
    transform=val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

print(f"   Train: {len(train_dataset)} afbeeldingen")
print(f"   Val:   {len(val_dataset)} afbeeldingen")
print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches:   {len(val_loader)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("2. Model maken...")
model = SimpleCNN(num_classes=80)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
print(f"3. Training ({num_epochs} epochs)...")

best_val_acc = 0

for epoch in range(num_epochs):
    print(f"\n   Epoch {epoch+1}/{num_epochs}")
    print("-" * 40)
    
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        # Vooruit
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Accuracy berekenen
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Achteruit
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Elke 10 batches resultaten tonen
        if batch_idx % 10 == 0:
            current_acc = 100. * correct / total if total > 0 else 0
            print(f"      Batch {batch_idx:3d}: loss = {loss.item():.4f}, accuracy = {current_acc:.2f}%")
    
    avg_loss = total_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    print(f"   ▶ Epoch {epoch+1} afgerond: gem. loss = {avg_loss:.4f}, accuracy = {epoch_acc:.2f}%")

    # VALIDATION
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100. * val_correct / val_total
    print(f"   ▶ Validation accuracy: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'beste_model.pth')
        print("   ✓ Beste model opgeslagen")
    
print("="*50)
print(f"4. Klaar! Laatste accuracy: {epoch_acc:.2f}%")