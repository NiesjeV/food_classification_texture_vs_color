print("Testen of imports werken...")
try:
    from src.dataset import FoodDataset
    from src.model import SimpleCNN
    import torch
    import pandas as pd
    from torchvision import transforms
    print("✅ Alle imports werken!")
except Exception as e:
    print(f"❌ Fout bij imports: {e}")

print("\nTesten of data te laden is...")
try:
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    
    labels_df = pd.read_csv('data/train_labels.csv')
    dataset = FoodDataset('data/train_set', labels_df[:5], transform)
    
    for i in range(3):
        img, label = dataset[i]
        print(f"   Foto {i}: shape {img.shape}, label {label}")
    print("✅ Data laden werkt!")
except Exception as e:
    print(f"❌ Fout bij data laden: {e}")

print("\nTesten of model werkt...")
try:
    model = SimpleCNN(num_classes=80)
    test_input = torch.randn(2, 3, 224, 224)
    test_output = model(test_input)
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {test_output.shape}")
    print("✅ Model werkt!")
except Exception as e:
    print(f"❌ Fout bij model: {e}")