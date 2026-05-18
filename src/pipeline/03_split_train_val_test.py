"""Étape 3 — Split 70 / 15 / 15

Fichier extrait du notebook notebooks/AgroShield_VSCode_CPU.ipynb.
Le notebook original reste la référence; ce script rend l'étape visible dans VS Code.
"""

# ══════════════════════════════════════════════════════════
# ÉTAPE 3 — Split train / validation / test
# ══════════════════════════════════════════════════════════
import json
from collections import Counter

classes      = sorted([d.name for d in DATA_DIR_BALANCED.iterdir() if d.is_dir()])
class_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_class = {i: c for c, i in class_to_idx.items()}

print(f'Nombre de classes : {len(classes)}')

all_samples = []
for cls in classes:
    for p in (DATA_DIR_BALANCED / cls).glob('*.jpg'):
        all_samples.append((str(p), class_to_idx[cls]))

random.seed(42)
random.shuffle(all_samples)

n       = len(all_samples)
n_train = int(0.70 * n)
n_val   = int(0.15 * n)

train_samples = all_samples[:n_train]
val_samples   = all_samples[n_train : n_train + n_val]
test_samples  = all_samples[n_train + n_val:]

print(f'Train : {len(train_samples):>6} images  ({len(train_samples)/n*100:.0f}%)')
print(f'Val   : {len(val_samples):>6} images  ({len(val_samples)/n*100:.0f}%)')
print(f'Test  : {len(test_samples):>6} images  ({len(test_samples)/n*100:.0f}%)')
print(f'Total : {n:>6} images')