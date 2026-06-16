# Benchmark des architectures de classification

Ce benchmark sert à justifier le choix d'`EfficientNet-B0` et `ResNet-50` dans AgroShield. Les chiffres publics ci-dessous viennent des métriques TorchVision sur ImageNet. Ils ne remplacent pas les résultats AgroShield, mais ils permettent de comparer objectivement les architectures les plus reconnues avant l'entraînement spécifique sur les 15 classes foliaires.

| Architecture | Top-1 ImageNet | Paramètres | GFLOPS | Lecture pour AgroShield |
|---|---:|---:|---:|---|
| MobileNetV2 | 72,15 % | 3,5 M | 0,30 | Très léger, mais précision publique plus faible qu'EfficientNet-B0. |
| MobileNetV3-Large | 75,27 % | 5,5 M | 0,22 | Excellent candidat futur pour embarqué, mais moins performant publiquement qu'EfficientNet-B0 à complexité proche. |
| EfficientNet-B0 | 77,69 % | 5,3 M | 0,39 | Meilleur compromis précision / taille parmi les modèles légers retenus. |
| ResNet-18 | 69,76 % | 11,7 M | 1,81 | Simple et robuste, mais moins précis et plus coûteux qu'EfficientNet-B0. |
| DenseNet-121 | 74,43 % | 8,0 M | 2,83 | Intéressant mais plus coûteux et moins performant publiquement qu'EfficientNet-B0. |
| Inception-v3 | 77,29 % | 27,2 M | 5,71 | Précision proche d'EfficientNet-B0, mais trop lourd pour une boucle Raspberry simple. |
| ResNet-50 | 80,86 % | 25,6 M | 4,09 | Référence robuste et reconnue pour comparaison hors ligne. |
| VGG16 | 71,59 % | 138,4 M | 15,47 | Modèle historique, mais trop lourd pour un système embarqué moderne. |

## Résultats AgroShield

| Modèle entraîné | Précision semi-finale | Taille locale `.pth` | Rôle projet |
|---|---:|---:|---|
| EfficientNet-B0 | 99,42 % | 15,64 Mo | Modèle prioritaire pour Raspberry Pi. |
| ResNet-50 | 98,62 % | 90,09 Mo | Modèle de référence hors ligne et contrôle de cohérence. |

## Conclusion défendable

Le choix est volontairement complémentaire :

- `EfficientNet-B0` valorise le déploiement embarqué : il est léger, efficace et suffisamment performant pour le terrain.
- `ResNet-50` valorise la crédibilité comparative : c'est une architecture reconnue, plus lourde, utile pour vérifier que les prédictions ne dépendent pas d'un seul modèle.

Ainsi, AgroShield ne choisit pas simplement deux modèles populaires. Le projet combine un modèle de terrain et un modèle de référence, ce qui donne une justification plus solide en soutenance.

Source principale : PyTorch/TorchVision, documentation officielle des modèles de classification.
