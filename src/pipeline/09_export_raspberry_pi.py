"""Étape 9 — Export pour Raspberry Pi

Fichier extrait du notebook notebooks/AgroShield_VSCode_CPU.ipynb.
Le notebook original reste la référence; ce script rend l'étape visible dans VS Code.
"""

# ══════════════════════════════════════════════════════════
# ÉTAPE 9 — Export TorchScript + métadonnées JSON
# ══════════════════════════════════════════════════════════
import json

def export_model(model, model_name, classes):
    name_clean = model_name.lower().replace('-', '_')

    # 1) Poids bruts — pour reprendre l'entraînement
    pth_path = MODELS_DIR / f'{name_clean}.pth'
    torch.save(model.state_dict(), pth_path)

    # 2) TorchScript — pour la Raspberry Pi
    model.eval()
    dummy  = torch.randn(1, 3, 224, 224)
    traced = torch.jit.trace(model, dummy)
    pt_path = MODELS_DIR / f'{name_clean}_rpi.pt'
    traced.save(str(pt_path))

    # 3) Métadonnées JSON
    meta = {
        'classes'    : classes,
        'num_classes': len(classes),
        'input_size' : 224,
        'mean'       : [0.485, 0.456, 0.406],
        'std'        : [0.229, 0.224, 0.225],
        'model_name' : model_name,
    }
    meta_path = MODELS_DIR / f'{name_clean}_meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    size_pth = pth_path.stat().st_size / 1e6
    size_pt  = pt_path.stat().st_size  / 1e6
    print(f'✅ {model_name}')
    print(f'   .pth (reprise entraînement) : {size_pth:.1f} MB → {pth_path.name}')
    print(f'   .pt  (Raspberry Pi)         : {size_pt:.1f} MB  → {pt_path.name}')
    print(f'   .json (métadonnées)         : {meta_path.name}')

export_model(eff_model, 'EfficientNet-B0', classes)
print()
export_model(res_model, 'ResNet-50', classes)

print(f'\n📁 Contenu de {MODELS_DIR} :')
for f in sorted(MODELS_DIR.iterdir()):
    print(f'   {f.name:<45} {f.stat().st_size/1e6:>6.1f} MB')