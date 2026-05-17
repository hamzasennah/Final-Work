# AgroShield - Centrale Casablanca

Projet PLBD Groupe 3: protection climatique des cultures avec diagnostic foliaire IA et mécanisme de plaques piloté par capteurs.

## Structure VS Code

```text
final work/
├── server.py                  # API Flask + chargement EfficientNet-B0 / ResNet-50
├── index.html                 # Interface web professionnelle
├── assets/                    # Logo Centrale Casablanca et ressources visuelles
├── models/                    # Modèles entraînés locaux (.pth/.pt ignorés par Git)
├── notebooks/                 # Notebook complet d'import, équilibrage, entraînement, test
├── src/pipeline/              # Étapes extraites du notebook: import, équilibrage, split, entraînement...
├── src/                       # Scripts utilitaires VS Code
├── docs/                      # Documentation technique
├── data/raw/PlantVillage/     # Dataset brut local
├── data/balanced/             # Dataset équilibré local
├── requirements.txt
└── .vscode/tasks.json
```

## Lancer dans VS Code

1. Ouvrir ce dossier dans VS Code: `C:\Users\pc\Desktop\final work`
2. Installer les dépendances:

```bash
pip install -r requirements.txt
```

3. Lancer l'interface:

```bash
python server.py
```

4. Ouvrir:

```text
http://localhost:5000
```

## Travail ML visible

Le notebook original est conservé dans `notebooks/AgroShield_VSCode_CPU.ipynb`.
Pour que le travail soit lisible dans l'explorateur VS Code, ses étapes ont aussi été extraites dans `src/pipeline/`:

- installation et vérification
- extraction de la data
- équilibrage Albumentations
- split train / validation / test
- DataLoader + WeightedRandomSampler
- modèles EfficientNet-B0 et ResNet-50
- entraînement CPU
- courbes d'apprentissage
- évaluation test set
- export Raspberry Pi
- test modèle sur image

Accès interface:

- email: `prenom.nom@centrale-casablanca.ma`
- code: `10101010`

## Modèles IA

Les poids entraînés sont attendus dans `models/`:

- `efficientnet_b0.pth`
- `efficientnet_b0_meta.json`
- `resnet_50.pth`
- `resnet_50_meta.json`

Le serveur accepte aussi les variantes `*_best.pth` et les exports Raspberry `*_rpi.pt` restent disponibles pour déploiement embarqué.

## GitHub

Les datasets et poids `.pth/.pt` sont ignorés par `.gitignore` car ils sont trop lourds pour GitHub. Pour partager les modèles, utiliser Git LFS ou un lien Drive/OneDrive dans la documentation.
