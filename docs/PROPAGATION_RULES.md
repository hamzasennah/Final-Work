# Règles A/B/C de propagation

La culture est divisée en trois zones: A, B et C. Après chaque détection foliaire, l'application ne donne pas seulement un pourcentage: elle retourne le périmètre précis d'intervention.

## Décision de traitement

| Maladie détectée | Conditions aggravantes | Décision |
|---|---|---|
| Late blight / mildiou | temps frais, humide, feuilles mouillées | traiter A, B et C |
| Bacterial spot | chaleur + pluie/eclaboussures | traiter zone détectée + voisines; A/B/C si confiance très forte |
| Septoria | pluie, éclaboussures, mouillure prolongée | zone détectée complète + demi-zone voisine |
| Leaf mold | humidité relative élevée | zone détectée complète + demi-zone voisine |
| Early blight | temps chaud et humide | zone détectée; demi-zones voisines si confiance ou météo forte |
| Yellow Leaf Curl Virus | pression aleurodes en conditions chaudes | surveillance A/B/C et contrôle vecteurs |
| Mosaic virus | contact/vecteurs | zone détectée; A/B/C si confiance forte |
| Spider mites | chaud et sec | zone détectée + inspection voisines |
| Healthy | aucun signal maladie | surveillance uniquement |

## Influence de la météo

Le serveur calcule un `weather_modifier` de 0 à 0.30 selon la maladie et les capteurs. Ce modificateur augmente la probabilité de propagation et peut orienter le mode mécanique automatique.

Exemple: une Septoria détectée en zone A avec forte pluie donne `Zone A complète + moitié gauche de la zone B`, puis augmente le risque parce que l'eau disperse les spores par éclaboussures.

## Sources

- University of Minnesota Extension, late blight: https://extension.umn.edu/disease-management/late-blight
- Penn State Extension, late blight of potato and tomato: https://extension.psu.edu/late-blight-of-potato-and-tomato
- Iowa State University Extension, bacterial spot of pepper and tomatoes: https://yardandgarden.extension.iastate.edu/encyclopedia/bacterial-spot-pepper-and-tomatoes
- University of Minnesota Extension, tomato leaf mold: https://extension.umn.edu/disease-management/tomato-leaf-mold
- NC State Extension, Tomato yellow leaf curl virus: https://content.ces.ncsu.edu/tomato-yellow-leaf-curl-virus
