# Carte complète du projet AgroShield

Ce fichier sert de point d'entrée dans VS Code. Il relie toutes les parties du travail sans supprimer le notebook original.

## 1. Notebook original

- `notebooks/AgroShield_VSCode_CPU.ipynb`

Il contient l'historique complet: installation, vérification, extraction, équilibrage, split, entraînement, évaluation, exports Raspberry Pi, génération de serveur et interface.

## 2. Données

- `data/raw/PlantVillage/` : dataset brut utilisé pour l'apprentissage.
- `data/balanced/` : dataset équilibré par augmentation.
- `models/` : poids EfficientNet-B0, ResNet-50, exports Raspberry Pi et métadonnées.

## 3. Pipeline extrait du notebook

Les scripts suivants rendent le travail visible fichier par fichier dans VS Code:

- `src/pipeline/00_installation_verification.py`
- `src/pipeline/01_extraction_data.py`
- `src/pipeline/02_equilibrage_albumentations.py`
- `src/pipeline/03_split_train_val_test.py`
- `src/pipeline/04_dataloaders_weighted_sampler.py`
- `src/pipeline/05_modeles_cnn.py`
- `src/pipeline/06_entrainement_cpu.py`
- `src/pipeline/07_courbes_apprentissage.py`
- `src/pipeline/08_evaluation_test_set.py`
- `src/pipeline/09_export_raspberry_pi.py`
- `src/pipeline/10_test_modeles_image.py`
- `src/pipeline/11_generation_server_archive.py`
- `src/pipeline/12_generation_index_archive.py`

## 4. Application web

- `server.py` : API Flask, capteurs simulés, hystérésis, chargement EfficientNet-B0 et ResNet-50, diagnostic, règles A/B/C.
- `index.html` : interface web avec dashboard, mécanisme animé, diagnostic et tableau des seuils par culture.
- `assets/centrale-casablanca-logo.svg` : logo utilisé dans l'interface.

## 5. Scripts de vérification

- `src/check_project.py` : vérifie datasets, modèles et métadonnées.
- `src/test_api.py` : teste l'API locale après lancement de `server.py`.
- `src/extract_notebook_pipeline.py` : régénère les fichiers `src/pipeline/` depuis le notebook.

## 6. Tâches VS Code

Dans VS Code: `Terminal > Run Task...`

- `AgroShield: lancer l'app web`
- `AgroShield: verifier projet`
- `AgroShield: tester API locale`
- `AgroShield: extraire pipeline notebook`

## 7. Décision A/B/C

La détection ne s'arrête pas à une probabilité. Elle renvoie:

- la classe prédite;
- la zone observée;
- les zones ou demi-zones à traiter;
- une recommandation agronomique.

Exemples:

- maladie locale/modérée en zone A: `Zone A complète`;
- maladie moyenne en zone A: `Zone A complète + moitié gauche de la zone B`;
- maladie très propagative: `Zone A complète + Zone B complète + Zone C complète`.
