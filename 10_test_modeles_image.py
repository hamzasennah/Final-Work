"""Étape 10 — Test du modèle sur une image

Fichier extrait du notebook notebooks/AgroShield_VSCode_CPU.ipynb.
Le notebook original reste la référence; ce script rend l'étape visible dans VS Code.
"""

# ══════════════════════════════════════════════════════════
# ÉTAPE 10 — Tester le diagnostic sur une image
# Changez IMAGE_PATH pour tester vos propres photos
# ══════════════════════════════════════════════════════════
from torchvision import transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_image(pil_image):
    """Diagnostic complet avec double vérification."""
    tensor = preprocess(pil_image).unsqueeze(0)
    with torch.no_grad():
        eff_out = torch.softmax(eff_model(tensor), dim=1)[0]
        res_out = torch.softmax(res_model(tensor), dim=1)[0]

    eff_top  = eff_out.topk(3)
    res_top  = res_out.topk(3)
    eff_pred = classes[eff_top.indices[0].item()]
    res_pred = classes[res_top.indices[0].item()]
    eff_conf = float(eff_top.values[0])
    res_conf = float(res_top.values[0])

    agree      = (eff_pred == res_pred)
    final_pred = eff_pred if eff_conf >= res_conf else res_pred
    final_conf = max(eff_conf, res_conf)
    parts      = final_pred.split('__')
    plant      = parts[0].replace('_', ' ')
    disease    = parts[1].replace('_', ' ') if len(parts) > 1 else 'Healthy'

    return {
        'efficientnet'   : {'prediction': eff_pred, 'confidence': eff_conf},
        'resnet'         : {'prediction': res_pred, 'confidence': res_conf},
        'agreement'      : agree,
        'agreement_score': '2/2' if agree else '1/2',
        'final_prediction': final_pred,
        'plant'          : plant,
        'disease'        : disease,
        'confidence'     : final_conf,
        'is_healthy'     : 'healthy' in final_pred.lower(),
    }

# ── Tester sur une image du dataset ──────────────────────
# Prenez n'importe quelle image JPG pour tester
test_img_path, test_label = test_samples[0]
print(f'Image testée : {Path(test_img_path).name}')
print(f'Classe réelle : {classes[test_label]}')

img    = Image.open(test_img_path).convert('RGB')
result = predict_image(img)

# Affichage
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(img)
axes[0].axis('off')
axes[0].set_title('Image testée', fontsize=12)

color = '#22c55e' if result['is_healthy'] else '#ef4444'
info  = (
    f"Plante   : {result['plant']}\n"
    f"Maladie  : {result['disease']}\n"
    f"Confiance: {result['confidence']*100:.1f}%\n"
    f"Accord   : {result['agreement_score']}\n\n"
    f"EfficientNet : {result['efficientnet']['prediction'].split('__')[-1][:30]}\n"
    f"  conf {result['efficientnet']['confidence']*100:.1f}%\n\n"
    f"ResNet-50    : {result['resnet']['prediction'].split('__')[-1][:30]}\n"
    f"  conf {result['resnet']['confidence']*100:.1f}%"
)
axes[1].text(0.05, 0.95, info, transform=axes[1].transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor=color+'33', edgecolor=color, linewidth=2))
axes[1].axis('off')
axes[1].set_title('Diagnostic double vérification', fontsize=12)

plt.tight_layout()
plt.show()
print('\nPour tester une autre image :')
print('  img = Image.open(r"C:\\chemin\\vers\\votre\\image.jpg").convert("RGB")')
print('  result = predict_image(img)')
print('  print(result)')