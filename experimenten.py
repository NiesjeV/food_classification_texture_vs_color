import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from src.dataset import FoodDataset
from src.model import SimpleCNN
import time
import os

# Het test 1 onze getrainde model op 4 verschillende manieren om te zien of kleur of textuur belangrijker is
# experiment weg halen van kleur -> dus hoe belangrijk is kleur?
# experiment toevoegen van blur -> hoe belangrijk is textuur?
# experiment met toevoegne blur en weghalen van keur
# en je ziet dus de accuracy over alle data (bij gerecht.py zie je hoe deze experimenten elk gerecht beinvloeden)

print("-"*60)
print("experimenten: is kleur of textuur belangrijk?")
print("-"*60)

# Laad het getrainde model
print("\nModel laden...")
model = SimpleCNN(num_classes=80)
model.load_state_dict(torch.load('beste_model.pth'))
model.eval()
print("Model geladen")

# 2. Data voorbereiden
print("\nData voorbereiden...")
labels_df = pd.read_csv('data/train_labels.csv')

# Kies hoeveel afbeeldingen je wilt testen
# bijvoorbeeld 2000 = snel 
# None = alle 30.612 duurde ongeveer 10 minuutjes

aantal_afbeeldingen = None  # <- Pas dit aan naar wens
if aantal_afbeeldingen:
    test_df = labels_df[:aantal_afbeeldingen]
    print(f"Testen op {len(test_df)} afbeeldingen")
else:
    test_df = labels_df
    print(f"Testen op alle {len(test_df)} afbeeldingen")

# de 4 experimenten
experimenten = [
    {
        'naam': 'Experiment 1: Normaal (baseline)',
        'transform': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    },
    {
        'naam': 'Experiment 2: Grijze plaatjes (zonder kleur)',
        'transform': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
    },
    {
        'naam': 'Experiment 3: Plaatjes met blur (Zonder textuur)',
        'transform': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.GaussianBlur(kernel_size=5),
            transforms.ToTensor(),
        ])
    },
    {
        'naam': 'Experiment 4: plaatjes die grijs zijn en blur hebben (zonder kleur en textuur)',
        'transform': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.GaussianBlur(kernel_size=5),
            transforms.ToTensor(),
        ])
    }
]

#Resultaten opslaan
results = []

#experimenten uitvoeren
print("\nExperimenten uitvoeren...")
print("^"*60)

for exp in experimenten:
    print(f"\n {exp['naam']}")
    print("-" * 50)
    
    start_time = time.time()
    
    # dataset maken met de transformatie
    dataset = FoodDataset(
        img_dir='data/train_set',
        labels_df=test_df,
        transform=exp['transform'])
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # accuracy testen
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Voortgang tonen
            if (batch_idx + 1) % 20 == 0:
                current_acc = 100. * correct / total
                print(f"   Batch {batch_idx+1}/{len(dataloader)}: huidige acc = {current_acc:.2f}%")
    
    accuracy = 100. * correct / total
    elapsed = time.time() - start_time
    
    results.append({
        'experiment': exp['naam'],
        'accuracy': accuracy,
        'tijd': elapsed
    })
    
    print(f"\n   {exp['naam']}: {accuracy:.2f}% ({elapsed/60:.1f} minuten)")

# 6. Resultaten vergelijken
print("\n" + "="*60)
print("resultaten")
print("="*60)

baseline = results[0]['accuracy']

for r in results:
    verschil = r['accuracy'] - baseline
    if verschil > 0:
        pijltje = "beter"
    elif verschil < 0:
        pijltje = "slechter"
    else:
        pijltje = "gelijk"
    
    print(f"{r['experiment']}: {r['accuracy']:.2f}% ({verschil:+.2f}%) {pijltje}")



# Kleur (vergelijk A met B)
kleur_verschil = baseline - results[1]['accuracy']

# Textuur (vergelijk A met C)
textuur_verschil = baseline - results[2]['accuracy']

# Beide
beide_verschil = baseline - results[3]['accuracy']


#Opslaan van resultaten
os.makedirs("Resultaten", exist_ok=True)
bestand_pad = os.path.join("Resultaten", "resultaten_experimenten.txt")

with open(bestand_pad, 'w') as f:
    f.write("EXPERIMENTEN: KLEUR vs TEXTUUR\n")
    f.write("="*50 + "\n")
    f.write(f"Datum: {time.strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"Getest op: {len(test_df)} afbeeldingen\n\n")
    
    for r in results:
        f.write(f"{r['experiment']}: {r['accuracy']:.2f}%\n")
    
    f.write("\nCONCLUSIE\n")
    f.write("="*50 + "\n")
    f.write(f"Kleur belangrijk: {kleur_verschil > 1}\n")
    f.write(f"Textuur belangrijk: {textuur_verschil > 1}\n")

print("\n Resultaten opgeslagen in 'resultaten_experimenten.txt'")