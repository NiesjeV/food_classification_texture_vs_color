import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from src.dataset import FoodDataset
from src.model import SimpleCNN

print("1. Data voorbereiden...")

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Labels laden
labels_df = pd.read_csv('data/train_labels.csv')

# Gebruik ALLE data (niet alleen eerste 1000)
dataset = FoodDataset(
    img_dir='data/train_set',
    labels_df=labels_df,
    transform=transform
)

# Dataloader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

print(f"   {len(dataset)} afbeeldingen geladen")
print(f"   {len(dataloader)} batches van 32")

print("2. Model maken...")
model = SimpleCNN(num_classes=80)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
print(f"3. Training ({num_epochs} epochs)...")

for epoch in range(num_epochs):
    print(f"\n   Epoch {epoch+1}/{num_epochs}")
    print("-" * 40)
    
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
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
    
    avg_loss = total_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    print(f"   ▶ Epoch {epoch+1} afgerond: gem. loss = {avg_loss:.4f}, accuracy = {epoch_acc:.2f}%")

print("="*50)
print(f"4. Klaar! Laatste accuracy: {epoch_acc:.2f}%")

model_path = 'beste_model.pth'
torch.save(model.state_dict(), model_path)
print(f"Model opgeslagen als '{model_path}'")