"""Étape 4 — DataLoader + WeightedRandomSampler

Fichier extrait du notebook notebooks/AgroShield_VSCode_CPU.ipynb.
Le notebook original reste la référence; ce script rend l'étape visible dans VS Code.
"""

# ══════════════════════════════════════════════════════════
# ÉTAPE 4 — Dataset, transforms, DataLoader
# NOTE CPU : num_workers=0 sur Windows pour éviter les erreurs
# ══════════════════════════════════════════════════════════
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import numpy as np

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

class PlantDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples   = samples
        self.transform = transform
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except (IOError, UnidentifiedImageError):
            return self.__getitem__((idx + 1) % len(self.samples))
        if self.transform:
            img = self.transform(img)
        return img, label

train_dataset = PlantDataset(train_samples, transform=data_transforms['train'])
val_dataset   = PlantDataset(val_samples,   transform=data_transforms['val'])
test_dataset  = PlantDataset(test_samples,  transform=data_transforms['val'])

# WeightedRandomSampler
labels          = [lbl for _, lbl in train_samples]
class_counts    = Counter(labels)
weights_per_cls = {cls: 1.0 / count for cls, count in class_counts.items()}
sample_weights  = torch.tensor([weights_per_cls[lbl] for _, lbl in train_samples])
sampler         = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

# Class weights pour CrossEntropyLoss
cw_array  = np.array([1.0 / class_counts[i] for i in range(len(classes))])
cw_array  = cw_array / cw_array.sum() * len(classes)
cw_tensor = torch.tensor(cw_array, dtype=torch.float)

# num_workers=0 sur Windows pour éviter les erreurs de multiprocessing
train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler,    num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False,      num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=16, shuffle=False,      num_workers=0)

print(f'Train  : {len(train_loader)} batches  ({len(train_dataset)} images)')
print(f'Val    : {len(val_loader)} batches  ({len(val_dataset)} images)')
print(f'Test   : {len(test_loader)} batches  ({len(test_dataset)} images)')
print(f'Batch  : 16  (réduit pour CPU)')
print(f'✅ DataLoaders prêts')