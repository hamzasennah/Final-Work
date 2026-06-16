# Benchmark des architectures de classification

Ce document justifie le choix d'`EfficientNet-B0` et `ResNet-50` pour AgroShield. Le benchmark est volontairement court : il compare uniquement des architectures reconnues et pertinentes pour une tâche de classification d'images foliaires.

## 1. Règle méthodologique

Deux types de métriques ne doivent pas être mélangés.

Le benchmark littérature compare les architectures avec des chiffres publics comparables : Top-1 ImageNet, nombre de paramètres et GFLOPS. Ces valeurs viennent de la documentation officielle TorchVision.

Le benchmark expérimental AgroShield compare uniquement les modèles entraînés sur le même dataset et le même split. C'est ici que l'on doit discuter accuracy, précision macro, rappel macro, F1 macro, F1 pondéré, matrice de confusion, rejet hors domaine, taille du modèle et latence CPU.

## 2. Benchmark littérature

| Architecture reconnue | Nature | Top-1 ImageNet | Paramètres | GFLOPS | Décision AgroShield |
|---|---|---:|---:|---:|---|
| VGG16 | CNN historique | 71,59 % | 138,4 M | 15,47 | Trop lourd et peu efficace pour l'embarqué. |
| Inception-v3 | CNN multi-échelle | 77,29 % | 27,2 M | 5,71 | Reconnu, mais moins simple et plus coûteux qu'un modèle léger. |
| DenseNet-121 | CNN à connexions denses | 74,43 % | 8,0 M | 2,83 | Crédible, mais moins avantageux qu'EfficientNet-B0 en rapport précision/coût. |
| MobileNetV2 | CNN mobile | 72,15 % | 3,5 M | 0,30 | Très bon repère mobile, mais précision publique inférieure à EfficientNet-B0. |
| EfficientNet-B0 | CNN efficient | 77,69 % | 5,3 M | 0,39 | Modèle principal : bon compromis précision, taille et coût de calcul. |
| ResNet-50 | CNN résiduel | 80,86 % | 25,6 M | 4,09 | Modèle de référence : robuste, reconnu, utile pour contrôle hors ligne. |

## 3. Résultats AgroShield disponibles

| Modèle entraîné | Accuracy semi-finale | Taille locale `.pth` | Rôle projet |
|---|---:|---:|---|
| EfficientNet-B0 | 99,42 % | 15,64 Mo | Modèle prioritaire pour Raspberry Pi. |
| ResNet-50 | 98,62 % | 90,09 Mo | Modèle de référence hors ligne et contrôle de cohérence. |

Ces valeurs sont les résultats semi-finaux disponibles. Les métriques F1, rappel et précision sont calculées uniquement sur le même test set strict afin de rester défendables. Le script prévu pour cette évaluation est :

```bash
python src/pipeline/08b_benchmark_modeles.py
```

Il génère `outputs/benchmarks/model_benchmark.json` et `outputs/benchmarks/model_benchmark.csv` avec accuracy, précision macro, rappel macro, F1 macro, F1 pondéré, paramètres, taille et latence CPU.

## 4. Pourquoi YOLO n'est pas comparé ici

YOLO est une famille de modèles de détection d'objets. Elle répond à la question : "où est l'objet ou la lésion dans l'image ?" et s'évalue avec des métriques comme mAP ou IoU. AgroShield traite ici une classification foliaire : "quelle classe de maladie décrit cette image ?".

YOLO serait pertinent dans une future version si le dataset contient des annotations de lésions par boîtes englobantes ou masques. Sans ces annotations, le comparer directement à EfficientNet-B0 ou ResNet-50 serait méthodologiquement incorrect.

## 5. Conclusion défendable

EfficientNet-B0 est retenu comme modèle de terrain parce qu'il garde un coût de calcul faible tout en restant performant. ResNet-50 est retenu comme modèle de référence parce qu'il est très reconnu et robuste, même s'il est plus lourd. Cette combinaison donne un choix équilibré : un modèle pour le déploiement Raspberry Pi et un modèle pour la validation technique.

Sources principales : documentation officielle TorchVision, papiers originaux EfficientNet, ResNet, VGG, Inception-v3, DenseNet, MobileNetV2 et YOLO.
