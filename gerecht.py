import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from src.dataset import FoodDataset
from src.model import SimpleCNN
import matplotlib.pyplot as plt
import os

# dit doet de ezperimenten uit experimenten.py maar dan kijken we naar elk gerecht
# dus bij welk gerecht heeft kleur de meeste invloed en welk gerecht heeft textuur de meeste invloed?

print("\n")
print("="*60)
print("Welke gerechten zijn afhankelijk van kleur en textuur?")
print("="*60)

# Laad class namen (pasta, pizza, etc)
with open('data/class_list_food.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]
print(f"\n {len(class_names)} gerechten geladen")

# Laad het model
model = SimpleCNN(num_classes=80)
model.load_state_dict(torch.load('beste_model.pth'))
model.eval()
print(" Model geladen")

# Data voorbereiden 
labels_df = pd.read_csv('data/train_labels.csv')

aantal_afbeeldingen = None  # None = alle, of getal zoals 5000, 10000

if aantal_afbeeldingen:
    test_df = labels_df[:aantal_afbeeldingen]
    print(f"Analyse op {len(test_df)} afbeeldingen")
else:
    test_df = labels_df
    print(f"Analyse op alle {len(test_df)} afbeeldingen")

# de experimenten
experimenten = [
    {
        'naam': 'RGB',
        'transform': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    },
    {
        'naam': 'GRIJS',
        'transform': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
    },
    {
        'naam': 'BLUR',
        'transform': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.GaussianBlur(kernel_size=5),
            transforms.ToTensor(),
        ])
    }
]

# Resultaten per gerecht opslaan
results_per_class = {class_name: {'RGB': 0, 'GRIJS': 0, 'BLUR': 0, 'totaal': 0} 
                     for class_name in class_names}

# Voer experimenten uit
for exp in experimenten:
    print(f"\nExperiment: {exp['naam']}")
    
    dataset = FoodDataset(
        img_dir='data/train_set',
        labels_df=test_df,
        transform=exp['transform']
    )
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            # Per afbeelding bijhouden
            for i in range(len(labels)):
                label = labels[i].item()
                class_name = class_names[label]
                correct = (predicted[i] == labels[i]).item()
                
                results_per_class[class_name][exp['naam']] += correct
                results_per_class[class_name]['totaal'] += 1

# Bereken accuracies per gerecht
print("\n" + "="*760)
print("resultaten per gerecht")
print("="*60)

class_results = []
for class_name, data in results_per_class.items():
    if data['totaal'] > 0:  # Alleen als er plaatjes van zijn
        rgb_acc = 100. * data['RGB'] / data['totaal']
        grijs_acc = 100. * data['GRIJS'] / data['totaal']
        blur_acc = 100. * data['BLUR'] / data['totaal']
        
        kleur_verlies = rgb_acc - grijs_acc
        textuur_verlies = rgb_acc - blur_acc
        
        class_results.append({
            'gerecht': class_name,
            'rgb': rgb_acc,
            'grijs': grijs_acc,
            'blur': blur_acc,
            'kleur_verlies': kleur_verlies,
            'textuur_verlies': textuur_verlies,
            'totaal': data['totaal']
        })
#map maken voor grafiek
os.makedirs('Resultaten', exist_ok=True)

#Sorteer op kleurverlies (meest afhankelijk van kleur)
class_results.sort(key=lambda x: x['kleur_verlies'], reverse=True)

print("\ntop 10 gerechten meest afhankelijk van kleur:")
print("-" * 70)
print(f"{'Gerecht':<25} {'RGB':>6} {'Grijs':>6} {'Verlies':>8} | {'Blur':>6}")
print("-" * 70)
for r in class_results[:10]:
    print(f"{r['gerecht']:<25} {r['rgb']:>5.1f}% {r['grijs']:>5.1f}% {r['kleur_verlies']:>7.1f}% | {r['blur']:>5.1f}%")

with open('Resultaten/top10_kleur.txt', 'w') as f:
    f.write("TOP 10 GERECHTEN MEEST AFHANKELIJK VAN KLEUR\n")
    f.write("-" * 70 + "\n")
    f.write(f"{'Gerecht':<25} {'RGB':>6} {'Grijs':>6} {'Verlies':>8} | {'Blur':>6}\n")
    f.write("-" * 70 + "\n")
    for r in class_results[:10]:
        f.write(f"{r['gerecht']:<25} {r['rgb']:>5.1f}% {r['grijs']:>5.1f}% {r['kleur_verlies']:>7.1f}% | {r['blur']:>5.1f}%\n")
print("Top 10 kleur tabel opgeslagen in 'Resultaten/top10_kleur.txt'")

#Sorteer op kleurverlies (meest afhankelijk van kleur)

print("\ntop 10 gerechten meest afhankelijk van textuur:")
print("-" * 70)
class_results.sort(key=lambda x: x['textuur_verlies'], reverse=True)
print(f"{'Gerecht':<25} {'RGB':>6} {'Blur':>6} {'Verlies':>8} | {'Grijs':>6}")
print("-" * 70)
for r in class_results[:10]:
    print(f"{r['gerecht']:<25} {r['rgb']:>5.1f}% {r['blur']:>5.1f}% {r['textuur_verlies']:>7.1f}% | {r['grijs']:>5.1f}%")

# Na het printen van de top 10 textuur tabel, voeg dit toe:
with open('Resultaten/top10_textuur.txt', 'w') as f:
    f.write("TOP 10 GERECHTEN MEEST AFHANKELIJK VAN TEXTUUR\n")
    f.write("-" * 70 + "\n")
    f.write(f"{'Gerecht':<25} {'RGB':>6} {'Blur':>6} {'Verlies':>8} | {'Grijs':>6}\n")
    f.write("-" * 70 + "\n")
    for r in class_results[:10]:
        f.write(f"{r['gerecht']:<25} {r['rgb']:>5.1f}% {r['blur']:>5.1f}% {r['textuur_verlies']:>7.1f}% | {r['grijs']:>5.1f}%\n")
print("Top 10 textuur tabel opgeslagen in 'Resultaten/top10_textuur.txt'")

# Grafiek maken
print("\nGrafiek maken...")



# Selecteer top 10 voor kleur
top_kleur = sorted(class_results, key=lambda x: x['kleur_verlies'], reverse=True)[:10]
top_textuur = sorted(class_results, key=lambda x: x['textuur_verlies'], reverse=True)[:10]

# Kleurgrafiek
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
gerechten = [r['gerecht'][:15] for r in top_kleur]  # Afkorten voor leesbaarheid
x = range(len(gerechten))
plt.bar(x, [r['rgb'] for r in top_kleur], width=0.4, label='RGB', alpha=0.8)
plt.bar([i+0.4 for i in x], [r['grijs'] for r in top_kleur], width=0.4, label='Grijs', alpha=0.8)
plt.xlabel('Gerecht')
plt.ylabel('Accuracy (%)')
plt.title('Gerechten meest afhankelijk van KLEUR')
plt.xticks([i+0.2 for i in x], gerechten, rotation=45, ha='right')
plt.legend()

# Textuurgrafiek
plt.subplot(1, 2, 2)
gerechten = [r['gerecht'][:15] for r in top_textuur]
x = range(len(gerechten))
plt.bar(x, [r['rgb'] for r in top_textuur], width=0.4, label='RGB', alpha=0.8)
plt.bar([i+0.4 for i in x], [r['blur'] for r in top_textuur], width=0.4, label='Blur', alpha=0.8)
plt.xlabel('Gerecht')
plt.ylabel('Accuracy (%)')
plt.title('Gerechten meest afhankelijk van TEXTUUR')
plt.xticks([i+0.2 for i in x], gerechten, rotation=45, ha='right')
plt.legend()

plt.tight_layout()
plt.savefig('Resultaten/analyse_per_gerecht.png', dpi=150, bbox_inches='tight')
print(" Grafiek opgeslagen als 'analyse_per_gerecht.png'")

# 10. Opslaan als CSV
df_results = pd.DataFrame(class_results)
df_results.to_csv('Resultaten/resultaten_per_gerecht.csv', index=False)
print(" Resultaten opgeslagen in 'resultaten_per_gerecht.csv'")

# 11. Conclusies
print("\n" + "="*70)
print("conclusie per gerecht")
print("="*70)

# Gerechten die het meest op kleur vertrouwen
meest_kleur = max(class_results, key=lambda x: x['kleur_verlies'])
print(f"\n Meest afhankelijk van kleur: {meest_kleur['gerecht']}")
print(f"   RGB: {meest_kleur['rgb']:.1f}% → Grijs: {meest_kleur['grijs']:.1f}% (verlies {meest_kleur['kleur_verlies']:.1f}%)")

# Gerechten die het meest op textuur vertrouwen
meest_textuur = max(class_results, key=lambda x: x['textuur_verlies'])
print(f"\n Meest afhankelijk van textuur: {meest_textuur['gerecht']}")
print(f"   RGB: {meest_textuur['rgb']:.1f}% → Blur: {meest_textuur['blur']:.1f}% (verlies {meest_textuur['textuur_verlies']:.1f}%)")

# Gerechten die het minst verliezen (robust)
minst_kleur = min(class_results, key=lambda x: x['kleur_verlies'])
print(f"\n Minst afhankelijk van kleur: {minst_kleur['gerecht']}")
print(f"   RGB: {minst_kleur['rgb']:.1f}% → Grijs: {minst_kleur['grijs']:.1f}% (verschil {minst_kleur['kleur_verlies']:.1f}%)")