import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

#deze dataclass vertelt PyTorch hoe het bij elk plaatje het juiste label kan vinden.

class FoodDataset(Dataset):
    """Laadt food afbeeldingen en labels"""
    
    def __init__(self, img_dir, labels_df, transform=None):
        self.img_dir = img_dir #hier zijn de fotos
        self.labels_df = labels_df #hier zijn de labels
        self.transform = transform #hier kan je bijvoorbeeld grijs of blur toevoegen
    
    def __len__(self):
        return len(self.labels_df) #hoeveel afbeeldingen we hebben
    
    def __getitem__(self, idx):
        # Haal info op
        rij = self.labels_df.iloc[idx] 
        img_name = rij['img_name']
        label = rij['label'] - 1
        
        # Laad foto, haalt path op
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        #past transformatie toe
        if self.transform:
            image = self.transform(image)
        
        return image, label