"""Étape 0 — Installation et vérification

Fichier extrait du notebook notebooks/AgroShield_VSCode_CPU.ipynb.
Le notebook original reste la référence; ce script rend l'étape visible dans VS Code.
"""

# ══════════════════════════════════════════════════════════
# ÉTAPE 0A — Installer toutes les bibliothèques
# Lancez cette cellule UNE SEULE FOIS, puis redémarrez le kernel
# ══════════════════════════════════════════════════════════
import subprocess, sys

packages = [
    'torch', 'torchvision',
    'albumentations',
    'flask', 'flask-cors',
    'scikit-learn',
    'matplotlib', 'seaborn',
    'pillow', 'opencv-python',
    'tqdm',
]

for pkg in packages:
    print(f'Installation : {pkg}...')
    subprocess.run([sys.executable, '-m', 'pip', 'install', pkg, '-q'], check=True)

print('\n✅ Toutes les bibliothèques sont installées')
print('→ Redémarrez le kernel VS Code : Ctrl+Shift+P → Restart Kernel')

# ══════════════════════════════════════════════════════════
# ÉTAPE 0B — Vérification de l'environnement
# ══════════════════════════════════════════════════════════
import torch, torchvision, sklearn, cv2, albumentations
import os, sys
from pathlib import Path

# ── Device : CPU forcé ───────────────────────────────────
device = torch.device('cpu')
print(f'Device       : {device}')
print(f'PyTorch      : {torch.__version__}')
print(f'Torchvision  : {torchvision.__version__}')
print(f'Python       : {sys.version.split()[0]}')
print(f'OpenCV       : {cv2.__version__}')
print(f'Albumentations: {albumentations.__version__}')
print(f'sklearn      : {sklearn.__version__}')

# ── Dossiers du projet ───────────────────────────────────
# Adaptez ce chemin si votre projet est ailleurs
PROJECT_DIR = Path(r'.')
DATA_DIR_RAW      = PROJECT_DIR / 'data' / 'dataset_raw'
DATA_DIR_BALANCED = PROJECT_DIR / 'data' / 'dataset_balanced'
MODELS_DIR        = PROJECT_DIR / 'models'

for d in [DATA_DIR_RAW, DATA_DIR_BALANCED, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f'\nProjet       : {PROJECT_DIR}')
print(f'Data raw     : {DATA_DIR_RAW}')
print(f'Data balanced: {DATA_DIR_BALANCED}')
print(f'Models       : {MODELS_DIR}')
print('\n✅ Environnement prêt')