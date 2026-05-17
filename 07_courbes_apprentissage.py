"""Étape 7 — Courbes d'apprentissage

Fichier extrait du notebook notebooks/AgroShield_VSCode_CPU.ipynb.
Le notebook original reste la référence; ce script rend l'étape visible dans VS Code.
"""

# ══════════════════════════════════════════════════════════
# ÉTAPE 7 — Visualisation des courbes
# ══════════════════════════════════════════════════════════
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('AgroShield — Courbes d\'apprentissage (CPU)', fontsize=14, fontweight='bold')

configs = [
    (eff_history, 'EfficientNet-B0', axes[0]),
    (res_history, 'ResNet-50',       axes[1]),
]

for hist, name, axrow in configs:
    epochs = range(1, len(hist['train_acc']) + 1)

    axrow[0].plot(epochs, hist['train_loss'], 'b-o', markersize=4, label='Train')
    axrow[0].plot(epochs, hist['val_loss'],   'r-o', markersize=4, label='Validation')
    axrow[0].set_title(f'{name} — Loss')
    axrow[0].set_xlabel('Epoch')
    axrow[0].set_ylabel('Loss')
    axrow[0].legend()
    axrow[0].grid(True, alpha=0.3)

    axrow[1].plot(epochs, hist['train_acc'], 'b-o', markersize=4, label='Train')
    axrow[1].plot(epochs, hist['val_acc'],   'r-o', markersize=4, label='Validation')
    axrow[1].set_title(f'{name} — Accuracy')
    axrow[1].set_xlabel('Epoch')
    axrow[1].set_ylabel('Accuracy')
    axrow[1].legend()
    axrow[1].grid(True, alpha=0.3)

plt.tight_layout()
save_path = MODELS_DIR / 'courbes_apprentissage.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'✅ Courbes sauvegardées : {save_path}')