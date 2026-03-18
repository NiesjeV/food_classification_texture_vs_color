import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from src.dataset import FoodDataset
from src.model import SimpleCNN

#dit is voor inleveren kaggle

print("1. Model laden...")
model = SimpleCNN(num_classes=80)
model.load_state_dict(torch.load('beste_model.pth'))
model.eval()
print("Model geladen!")

print("2. Data voorbereiden...")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

test_dataset = FoodDataset(
    img_dir='data/test_set',
    labels_df=None,
    transform=transform,
    is_test=True
)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print(f"{len(test_dataset)} test afbeeldingen geladen")

print("3. Voorspellingen maken...")
predictions = []
image_names = []

with torch.no_grad():
    for batch_idx, (images, names) in enumerate(test_loader):
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.tolist())
        image_names.extend(names)
        
        if batch_idx % 10 == 0:
            print(f"   Batch {batch_idx}/{len(test_loader)}")

print("4. CSV maken...")
submission = pd.DataFrame({
    'img_name': image_names,
    'label': [p + 1 for p in predictions]
})

submission.to_csv('submission.csv', index=False)
print("submission.csv is gemaakt!")
print(submission.head())