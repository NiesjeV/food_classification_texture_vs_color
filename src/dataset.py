import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class FoodDataset(Dataset):
    """Laadt food afbeeldingen en labels"""
    
    def __init__(self, img_dir, labels_df, transform=None):
        self.img_dir = img_dir
        self.labels_df = labels_df
        self.transform = transform
    
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        # Haal info op
        rij = self.labels_df.iloc[idx]
        img_name = rij['img_name']
        label = rij['label'] - 1  # 1-80 -> 0-79
        
        # Laad foto
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Transformeer
        if self.transform:
            image = self.transform(image)
        
        return image, label