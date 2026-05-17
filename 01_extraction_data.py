"""Étape 1 — Extraction de DATA.zip

Fichier extrait du notebook notebooks/AgroShield_VSCode_CPU.ipynb.
Le notebook original reste la référence; ce script rend l'étape visible dans VS Code.
"""

from pathlib import Path

# =====================================================
# CHEMIN DU DATASET
# =====================================================

PLANT_DIR = Path(r'data/raw/PlantVillage')

# Vérification
if not PLANT_DIR.exists():
    raise FileNotFoundError(f'Dataset introuvable : {PLANT_DIR}')

print(f'Dataset trouvé : {PLANT_DIR}\n')

# =====================================================
# AFFICHER LES CLASSES
# =====================================================

print(f'{"Classe":<45} {"Images":>8}')
print('-' * 58)

total = 0

for class_dir in sorted(PLANT_DIR.iterdir()):

    # ignorer fichiers
    if not class_dir.is_dir():
        continue

    # compter rapidement les images
    n = sum(
        1 for f in class_dir.iterdir()
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
    )

    total += n

    print(f'{class_dir.name:<45} {n:>8}')

print('-' * 58)
print(f'{"TOTAL":<45} {total:>8}')