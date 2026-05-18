# Seuils climatiques et hystérésis

Le mode `Auto` applique des seuils par culture. Chaque déclenchement possède un seuil `on` et un seuil `off`: c'est l'hystérésis, indispensable pour éviter qu'un servo oscille autour d'une limite.

| Culture | Plage normale | Chaleur: déclenchement | Chaleur: retour | Pluie/humidité: déclenchement | Pluie/humidité: retour |
|---|---|---:|---:|---:|---:|
| Tomate | 21-27 °C, 60-75 % RH | ≥ 29 °C ou ≥ 720 lx | ≤ 26 °C | ≥ 18 mm/h ou ≥ 82 % RH | ≤ 8 mm/h et ≤ 72 % RH |
| Poivron | 21-28 °C, 60-75 % RH | ≥ 30 °C ou ≥ 700 lx | ≤ 27 °C | ≥ 16 mm/h ou ≥ 80 % RH | ≤ 7 mm/h et ≤ 70 % RH |
| Pomme de terre | 16-22 °C, 70-85 % RH | ≥ 26 °C ou ≥ 600 lx | ≤ 23 °C | ≥ 20 mm/h ou ≥ 85 % RH | ≤ 9 mm/h et ≤ 75 % RH |

## Modes mécaniques

- `repos`: plaques abaissées à 0°, culture ouverte.
- `chaleur`: plaques horizontales à 90°, couverture contre excès thermique et luminosité.
- `pluie`: plaques inclinées vers le canal central, eau dirigée vers le réservoir.

## Sources

- University of Maryland Extension, high temperature and heat injury in vegetables: https://extension.umd.edu/resource/high-temperature-and-heat-injury-vegetables/
- University of Minnesota Extension, vegetable production resources: https://extension.umn.edu/vegetables
- Penn State Extension, late blight of potato and tomato: https://extension.psu.edu/late-blight-of-potato-and-tomato
