# Protocole split strict et benchmark modèles

Ce document fixe la règle méthodologique à défendre pour AgroShield.

## 1. Pourquoi corriger le split

Le risque principal est la fuite de données. Si l'on augmente tout le dataset puis que l'on sépare ensuite train, validation et test, une image augmentée peut se retrouver dans le train pendant que l'image originale, ou une variante très proche, se retrouve dans le test. Le score de test devient alors trop optimiste.

La règle finale est donc :

1. séparer les images brutes par classe en `70 % train`, `15 % validation`, `15 % test` ;
2. figer les manifestes ;
3. appliquer l'augmentation hors ligne uniquement sur le train ;
4. garder validation et test indépendants ;
5. utiliser le test une seule fois pour l'évaluation finale.

## 2. Script de split strict

Commande :

```bash
python src/pipeline/03b_split_strict_no_leakage.py
```

Sorties :

```text
data/splits_strict/train.jsonl
data/splits_strict/val.jsonl
data/splits_strict/test.jsonl
data/splits_strict/split_manifest.json
```

Sur le dataset local actuel :

```text
Train : 14 444 images sources
Val   :  3 097 images sources
Test  :  3 097 images sources
Total : 20 638 images sources
```

Remarque importante : ces nombres viennent du dataset brut, pas du dossier `data/balanced`. Le test strict est donc plus fiable scientifiquement. Le dossier équilibré reste utile pour expliquer le rééquilibrage, mais le benchmark final doit être relancé avec les manifestes stricts.

## 3. Benchmark multicritère

Commande :

```bash
python src/pipeline/08b_benchmark_modeles.py
```

Le script compare `EfficientNet-B0` et `ResNet-50` avec :

- accuracy ;
- macro precision ;
- macro recall ;
- macro F1 ;
- weighted F1 ;
- taille du modèle ;
- nombre de paramètres ;
- latence CPU moyenne par image.

Sorties :

```text
outputs/benchmarks/model_benchmark.json
outputs/benchmarks/model_benchmark.csv
```

## 4. Interprétation attendue

EfficientNet-B0 est le candidat prioritaire pour Raspberry Pi grâce à son compromis performance, taille et coût d'inférence. ResNet-50 reste utile comme modèle de référence hors ligne, car il est plus lourd mais robuste pour comparaison.

Les scores semi-finaux restent utiles pour montrer le potentiel du prototype. En revanche, devant un jury, il faut préciser que la validation finale la plus rigoureuse est celle qui utilise le split strict avant augmentation.

## 5. Références méthodologiques

- Rohan Banerjee, *Hands-on TinyML*, 2023 : utile pour justifier l'inférence embarquée et l'optimisation edge.
- Jason Brownlee, *Imbalanced Classification with Python*, 2020 : utile pour justifier les métriques au-delà de l'accuracy, le traitement des classes déséquilibrées et l'importance du test indépendant.
