# Seuils climatiques utilisés dans AgroShield

L'application choisit automatiquement la position des plaques selon la culture sélectionnée et les capteurs:

- `repos`: plaques abaissées à 0°, aucune couverture.
- `chaleur`: plaques horizontales à 90°, couverture de la culture.
- `pluie`: plaques inclinées vers le canal central et le réservoir.

## Seuils intégrés

| Culture | Zone idéale | Déclenchement chaleur | Déclenchement pluie/humidité |
|---|---|---:|---:|
| Tomate | 21-27 °C, 60-75 % RH | ≥ 29 °C ou luminosité ≥ 720 lx | pluie ≥ 18 mm/h ou humidité ≥ 82 % |
| Poivron | 21-28 °C, 60-75 % RH | ≥ 30 °C ou luminosité ≥ 700 lx | pluie ≥ 16 mm/h ou humidité ≥ 80 % |
| Pomme de terre | 16-22 °C, 70-85 % RH | ≥ 26 °C ou luminosité ≥ 600 lx | pluie ≥ 20 mm/h ou humidité ≥ 85 % |

Les seuils `off` sont plus bas que les seuils `on`: c'est l'hystérésis. Elle évite que le système oscille toutes les secondes autour d'une limite.

## Sources consultées

- University of Minnesota Extension, tomato growing conditions and disease management.
- Iowa State University Extension, bacterial spot of pepper and tomatoes.
- FAO crop water and climate guidance for tomato, pepper and potato.
- University of Minnesota Extension, potato late blight and potato crop sensitivity to warm, humid disease conditions.
