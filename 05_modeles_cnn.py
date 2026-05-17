"""Étape 5 — Modèles CNN

Fichier extrait du notebook notebooks/AgroShield_VSCode_CPU.ipynb.
Le notebook original reste la référence; ce script rend l'étape visible dans VS Code.
"""

# ══════════════════════════════════════════════════════════
# ÉTAPE 5 — EfficientNet-B0 et ResNet-50
# ══════════════════════════════════════════════════════════
from torchvision import models

num_classes = len(classes)

def build_efficientnet(n):
    m = models.efficientnet_b0(weights='IMAGENET1K_V1')
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, n)
    return m  # CPU : pas de .to(device) nécessaire

def build_resnet50(n):
    m = models.resnet50(weights='IMAGENET1K_V1')
    m.fc = nn.Linear(m.fc.in_features, n)
    return m

eff_model = build_efficientnet(num_classes)
res_model = build_resnet50(num_classes)

eff_params = sum(p.numel() for p in eff_model.parameters()) / 1e6
res_params = sum(p.numel() for p in res_model.parameters()) / 1e6

print(f'✅ EfficientNet-B0 : {eff_params:.1f}M paramètres')
print(f'✅ ResNet-50       : {res_params:.1f}M paramètres')
print(f'   Classes         : {num_classes}')
print(f'   Device          : CPU')