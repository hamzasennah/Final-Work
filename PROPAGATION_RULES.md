# Règles A/B/C de propagation

La culture est divisée en trois zones: A, B et C. Après une détection foliaire, l'application estime le périmètre de traitement selon deux informations:

- le type de maladie prédit par le modèle;
- la confiance du modèle.

## Politique utilisée

| Maladie détectée | Risque | Décision |
|---|---:|---|
| Late blight / mildiou | Très élevé | A + B + C |
| Bacterial spot | Élevé | Zone détectée + zones voisines; A + B + C si confiance très forte |
| Yellow Leaf Curl Virus | Élevé | A + B + C, avec contrôle des aleurodes |
| Mosaic virus | Élevé | A + B + C si la confiance est suffisante |
| Septoria, leaf mold, target spot | Moyen à élevé | Zone détectée + zones voisines |
| Early blight | Moyen | Zone détectée; zones voisines si confiance forte |
| Spider mites | Moyen | Zone détectée; zones voisines si confiance forte |
| Healthy | Faible | Surveillance uniquement |

Ces règles sont volontairement prudentes pour une soutenance: elles évitent de sous-traiter une maladie à forte propagation, tout en gardant les traitements localisés lorsque le risque est modéré.

## Sources utilisées

- University of Minnesota Extension, late blight tomato/potato: https://extension.umn.edu/disease-management/late-blight
- Iowa State Extension, bacterial spot tomato/pepper: https://yardandgarden.extension.iastate.edu/encyclopedia/bacterial-spot-pepper-and-tomatoes
- University of Minnesota Extension, leaf mold: https://extension.umn.edu/diseases/leaf-mold-tomato
- NC State Extension, tomato yellow leaf curl virus: https://content.ces.ncsu.edu/tomato-yellow-leaf-curl-virus
