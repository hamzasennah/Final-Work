# AgroShield - lire le projet complet

Ouvrez ce fichier en premier dans VS Code.

## Travail complet disponible

1. Données brutes: `data/raw/PlantVillage/`
2. Données équilibrées: `data/balanced/`
3. Notebook original complet: `notebooks/AgroShield_VSCode_CPU.ipynb`
4. Étapes extraites du notebook: `src/pipeline/`
5. Modèles entraînés: `models/`
6. Application web: `server.py` + `index.html`
7. Documentation technique: `docs/`

## Étapes visibles dans `src/pipeline`

- `00_installation_verification.py`
- `01_extraction_data.py`
- `02_equilibrage_albumentations.py`
- `03_split_train_val_test.py`
- `04_dataloaders_weighted_sampler.py`
- `05_modeles_cnn.py`
- `06_entrainement_cpu.py`
- `07_courbes_apprentissage.py`
- `08_evaluation_test_set.py`
- `09_export_raspberry_pi.py`
- `10_test_modeles_image.py`
- `11_generation_server_archive.py`
- `12_generation_index_archive.py`

## Commandes utiles

```bash
python src/check_project.py
python server.py
```

Puis ouvrir:

```text
http://localhost:5000
```
