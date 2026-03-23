import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import time
import os
from sklearn.model_selection import train_test_split
from src.dataset import FoodDataset
from src.model import FoodCNN

# ============================================
# CONFIGURATION - 256x256
# ============================================
config = {
    'img_size': 256,
    'batch_size': 32,
    'learning_rate': 0.01,   # keep SGD lr for now
    'epochs': 20,
    'num_classes': 80,
    'early_stop_patience': 5
}

print("="*60)
print(" food classification")
print("="*60)
print(f"Config: {config}")

# ============================================
# 1. DATA PREPARATION
# ============================================
print("\n1. Preparing data...")

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((config['img_size'], config['img_size'])),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

val_transform = transforms.Compose([
    transforms.Resize((config['img_size'], config['img_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

labels_df = pd.read_csv('data/train_labels.csv')

train_df, val_df = train_test_split(
    labels_df,
    test_size=0.2,
    random_state=42,
    stratify=labels_df['label']
)

print(f"   Train: {len(train_df)} images")
print(f"   Val:   {len(val_df)} images")

train_dataset = FoodDataset('data/train_set', train_df, train_transform)
val_dataset   = FoodDataset('data/train_set', val_df,   val_transform)

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=config['batch_size'], shuffle=False)

print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches:   {len(val_loader)}")

# ============================================
# 2. MODEL
# ============================================
print("\n2. Building model...")
model = FoodCNN(num_classes=config['num_classes'])
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3
)

device = torch.device("cpu")  # change to "cuda" if you have GPU
model.to(device)

# ============================================
# 3. TRAINING
# ============================================
print(f"\n3. Training for up to {config['epochs']} epochs...")

best_val_acc = 0.0
best_epoch = 0
patience_counter = 0
os.makedirs('checkpoints', exist_ok=True)

for epoch in range(config['epochs']):
    print(f"\n# Epoch {epoch+1}/{config['epochs']}")
    print("-" * 50)
    
    # --- TRAIN ---
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    start_time = time.time()
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        if batch_idx % 50 == 0:
            current_acc = 100.0 * train_correct / train_total
            print(f"      Batch {batch_idx:3d}/{len(train_loader)}: loss = {loss.item():.4f}, acc = {current_acc:.2f}%")
    
    train_acc = 100.0 * train_correct / train_total
    train_loss = train_loss / len(train_loader)
    
    # --- VALIDATION ---
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
    
    val_acc = 100.0 * val_correct / val_total
    epoch_time = time.time() - start_time
    
    scheduler.step(val_acc)
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"\n   > Train loss: {train_loss:.4f}, acc: {train_acc:.2f}%")
    print(f"   > Val acc: {val_acc:.2f}% ({epoch_time/60:.1f} min, lr={current_lr:.6f})")
    
    torch.save(model.state_dict(), f'checkpoints/epoch_{epoch+1}.pth')
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        torch.save(model.state_dict(), 'checkpoints/best_model.pth')
        print(f"   New best model ({val_acc:.2f}%)")
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= config['early_stop_patience']:
        print(f"\n~ Early stopping after {epoch+1} epochs (no improvement)")
        break

print("\n" + "="*60)
print("Training complete")
print("="*60)
print(f"Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
print("Best model stored in 'checkpoints/best_model.pth'")
