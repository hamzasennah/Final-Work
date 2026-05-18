"""Étape 8 — Évaluation complète sur le test set

Fichier extrait du notebook notebooks/AgroShield_VSCode_CPU.ipynb.
Le notebook original reste la référence; ce script rend l'étape visible dans VS Code.
"""

# ══════════════════════════════════════════════════════════
# ÉTAPE 8A — Rapport de classification
# ══════════════════════════════════════════════════════════
from sklearn.metrics import classification_report, accuracy_score

def evaluate_model(model, model_name):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(test_loader, desc=f'Test {model_name}', ncols=70):
            preds = model(imgs).argmax(1).tolist()
            all_preds.extend(preds)
            all_labels.extend(lbls.tolist())

    acc = accuracy_score(all_labels, all_preds)
    print(f'\n{"="*60}')
    print(f'{model_name} — Test Accuracy : {acc*100:.2f}%')
    print(f'{"="*60}')
    print(classification_report(all_labels, all_preds, target_names=classes, digits=3))
    return all_labels, all_preds, acc

eff_labels, eff_preds, eff_acc = evaluate_model(eff_model, 'EfficientNet-B0')
res_labels, res_preds, res_acc = evaluate_model(res_model, 'ResNet-50')

print(f'\n📊 Résumé :')
print(f'   EfficientNet-B0 : {eff_acc*100:.2f}%')
print(f'   ResNet-50       : {res_acc*100:.2f}%')
print(f'   Meilleur        : {"EfficientNet-B0" if eff_acc >= res_acc else "ResNet-50"}')

# ══════════════════════════════════════════════════════════
# ÉTAPE 8B — Matrice de confusion
# ══════════════════════════════════════════════════════════
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion(labels, preds, model_name):
    cm      = confusion_matrix(labels, preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    short   = [c.split('__')[-1].replace('_', ' ')[:18] for c in classes]

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens',
                xticklabels=short, yticklabels=short, ax=ax, linewidths=0.3)
    ax.set_title(f'{model_name} — Matrice de confusion (normalisée)', fontsize=13, pad=12)
    ax.set_xlabel('Prédit', fontsize=11)
    ax.set_ylabel('Réel',   fontsize=11)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0,  fontsize=8)
    plt.tight_layout()
    save_path = MODELS_DIR / f'confusion_{model_name.replace("-","_")}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'✅ Matrice sauvegardée : {save_path}')

plot_confusion(eff_labels, eff_preds, 'EfficientNet-B0')
plot_confusion(res_labels, res_preds, 'ResNet-50')