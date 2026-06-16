# AgroShield - Centrale Casablanca

AgroShield est un centre de contrôle agricole intelligent réalisé par le Groupe PLBD 3. Le projet combine diagnostic foliaire par IA, capteurs climatiques reliés à une Raspberry Pi, servomoteurs et plaques mécaniques orientables pour protéger les cultures contre la forte chaleur, les pluies intenses et les conditions qui accélèrent la propagation des maladies.

## Ce que contient le dépôt

```text
final work/
├── server.py                         # API Flask, modèles IA, logique capteurs, Raspberry et décision mécanique
├── index.html                        # Application web professionnelle
├── assets/                           # Logos Centrale Casablanca, AgroShield et favicon
├── notebooks/                        # Notebook complet conservé et synchronisé
├── src/pipeline/                     # Étapes extraites du notebook pour lecture dans VS Code
├── src/prepare_raspberry_bundle.py   # Préparation EfficientNet-B0 TorchScript pour Raspberry Pi
├── raspberry/                        # Code embarqué capteurs, caméra, servos et client API
├── docs/                             # Documentation technique complémentaire
├── data/raw/PlantVillage/            # Dataset brut local, ignoré par Git
├── data/balanced/                    # Dataset équilibré local, ignoré par Git
└── models/                           # Métadonnées suivies, poids lourds gardés localement
```

Les scripts extraits du notebook couvrent l’installation, l’extraction des données, l’équilibrage Albumentations, le split train/validation/test, les DataLoaders avec `WeightedRandomSampler`, EfficientNet-B0, ResNet-50, l’entraînement CPU, les courbes, l’évaluation test, l’export Raspberry Pi et la génération des services web.

## Lancer l’application

```bash
pip install -r requirements.txt
python server.py
```

Ouvrir ensuite `http://localhost:5000`.

Accès interface:

- Email: `prenom.nom@centrale-casablanca.ma`
- Code projet: `10101010`

Pour un accès depuis un autre appareil du même réseau, utiliser `http://<adresse-ip-du-pc>:5000`.

## Fonctionnalités principales

- Dashboard capteurs avec température, humidité, précipitation, luminosité, état global et mini tendances.
- Tableau de décision agronomique: météo réelle Bouskoura, GDD, risque maladie à 7 jours, NDVI proxy RGB/GCC, centre d'alertes et rapport téléchargeable.
- Deux modes d'usage: `Fermier` pour les actions simples et immédiates, `Technicien` pour les seuils, historiques, indicateurs détaillés et diagnostic complet.
- Rapports agronomiques PDF réels et séparés: `/api/report.pdf?audience=farmer` pour le fermier, `/api/report.pdf?audience=technician` pour le technicien.
- Système d'alertes multi-niveaux visible dans l'application et dans les rapports: `Info` vert, `Attention` orange, `Critique` rouge, avec capteur déclencheur, zone et état recommandé.
- Animation mécanique fidèle au prototype: repos à 0°, plaques horizontales à 90° en forte chaleur, plaques inclinées en pluie pour guider l’eau vers le canal central puis le réservoir.
- Diagnostic IA par image avec aperçu, classe prédite, confiance, zones A/B/C à traiter et historique avec date + heure.
- Contrôle hors domaine: si la caméra capture un PC, un document, un objet ou une image qui ne ressemble pas à une feuille du dataset, l'application renvoie `Hors_dataset_non_vegetal` et ne déclenche aucune intervention maladie.
- Diagnostic instantané maladie-climat pour les 15 classes: effet chaleur, effet pluie/humidité, risque ajusté et action mécanique proposée.
- Seuils par culture: tomate, poivron et pomme de terre avec hystérésis pour éviter les oscillations des servomoteurs.
- Boucle intégrée maladie-météo: la pluie, l’humidité et la chaleur ne sont pas traitées séparément du diagnostic, elles modifient le risque de propagation et peuvent déclencher une action mécanique.
- Endpoints Raspberry Pi pour les capteurs et la caméra.

## Modules agronomiques avancés

- `Météo Bouskoura`: température, humidité, précipitation et rayonnement estimé via Open-Meteo lorsque la Raspberry Pi ne transmet pas encore de capteurs.
- `GDD`: accumulation de degrés-jours depuis une date de semis paramétrée par culture, avec stade phénologique estimé.
- `Risque maladie 7 jours`: projection locale à court terme combinant humidité, pluie, chaleur et dernière maladie active.
- `NDVI proxy caméra`: estimation RGB basée sur Green Chromatic Coordinate, utile comme indicateur de tendance quand la caméra n'est pas multispectrale.
- `Assistant agronomique`: réponses locales sur les plaques, la pluie, la chaleur, l'irrigation et le risque maladie.
- `Carte zonale SVG`: plan visuel de la serre/champ; les zones A/B/C changent de couleur selon le risque et ouvrent le détail capteurs, maladies et actions au clic.
- `Rapport fermier`: PDF simple avec logo Centrale Casablanca, action immédiate, état des plaques, maladie détectée, heure de l'image, zone touchée et consigne claire.
- `Rapport technicien`: PDF détaillé avec logo Centrale Casablanca, capteurs, météo réelle, seuils, mécanisme, GDD, NDVI proxy, risque 7 jours, diagnostic IA et historique utile au suivi technique.
- `Alertes`: trois fenêtres opérationnelles par niveau; chaque alerte porte le capteur déclencheur, la zone et l'état associé.

## Modèles IA

Le projet utilise deux modèles entraînés:

- `EfficientNet-B0`
- `ResNet-50`

Les poids `.pth` et `.pt` sont ignorés par Git pour éviter un dépôt trop lourd. Les métadonnées `.json` restent suivies. Pour préparer EfficientNet-B0 pour Raspberry Pi:

```bash
python src/prepare_raspberry_bundle.py
```

Le script prépare:

```text
raspberry/deploy/models/efficientnet_b0_rpi.pt
raspberry/deploy/models/efficientnet_b0_meta.json
raspberry/deploy/manifest.json
```

### Split strict sans fuite de données

Point méthodologique important: le test set doit être isolé avant toute augmentation hors ligne. Le dépôt contient donc un protocole corrigé:

```bash
python src/pipeline/03b_split_strict_no_leakage.py
```

Ce script crée des manifestes reproductibles à partir des images brutes:

```text
data/splits_strict/train.jsonl
data/splits_strict/val.jsonl
data/splits_strict/test.jsonl
data/splits_strict/split_manifest.json
```

Règle à défendre en soutenance: augmentation uniquement sur le train; validation et test restent indépendants et ne servent jamais au réglage manuel du modèle.

### Benchmark des architectures IA

Le choix des modèles est justifié par un benchmark multicritère disponible dans `docs/MODEL_BENCHMARK.md`. Il compare des architectures reconnues: MobileNetV2, MobileNetV3-Large, EfficientNet-B0, ResNet-18, DenseNet-121, Inception-v3, ResNet-50 et VGG16.

Conclusion projet:

- `EfficientNet-B0`: meilleur compromis précision publique / complexité / taille pour Raspberry Pi.
- `ResNet-50`: modèle de référence robuste pour comparaison hors ligne.
- Les modèles plus anciens ou plus lourds, comme VGG16 ou Inception-v3, sont moins adaptés à une boucle embarquée simple.

Le script suivant reste disponible pour produire un benchmark expérimental sur les poids entraînés localement:

```bash
python src/pipeline/08b_benchmark_modeles.py
```

Sorties attendues:

```text
outputs/benchmarks/model_benchmark.json
outputs/benchmarks/model_benchmark.csv
```

Interprétation projet: EfficientNet-B0 est prioritaire pour la Raspberry Pi grâce à son compromis performance/poids; ResNet-50 reste une référence comparative hors ligne.

## Déploiement Raspberry Pi

La Raspberry Pi peut envoyer les capteurs et photos directement à l’application:

- `POST /api/raspberry/sensors`
- `POST /api/raspberry/photo`
- `POST /upload` pour le test camera USB simple montre dans RealVNC/Thonny.

Code embarqué:

```bash
cd raspberry
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-rpi.txt
cp config.example.json config.json
python agroshield_rpi_client.py --config config.json
```

Le client lit les capteurs, applique une décision locale avec hystérésis si le serveur est indisponible, actionne les servos et envoie les photos caméra pour affichage immédiat dans l’interface.

### Flux camera USB sans upload manuel

Le serveur AgroShield analyse maintenant automatiquement les images recues par Raspberry:

1. Directement via `POST /upload`.
2. Ou par surveillance du dossier Windows `C:\Users\pc\Desktop\reception des images`.

Il faut lancer le serveur principal du projet:

```bash
python server.py
```

Puis, sur Raspberry, envoyer une capture USB:

```bash
cd raspberry
python3 usb_camera_upload.py --server http://ADRESSE_IP_DU_PC:5000 --zone A
```

L'image est sauvegardee dans `C:\Users\pc\Desktop\reception des images`, analysee par EfficientNet-B0/ResNet-50 si les modeles sont charges, puis affichee automatiquement dans l'onglet Diagnostic de l'application. Il n'est donc plus necessaire de choisir le fichier manuellement.

Quand une image détecte une maladie, le serveur renvoie aussi une commande exploitable par la Raspberry:

```json
{
  "actuator_command": {
    "mode": "pluie",
    "source": "disease_climate_loop",
    "reason": "Maladie detectee en zone A: risque maladie-climat 98%",
    "apply_on_raspberry": true
  }
}
```

Ainsi les plaques ne réagissent pas seulement à un seuil météo brut. Elles réagissent aussi quand la météo actuelle aggrave la maladie détectée: inclinaison en cas de pluie favorisant une maladie par éclaboussures, couverture en cas de chaleur favorisant stress ou vecteurs.

## Logique de décision maladie-climat

Le mode automatique combine:

1. Seuils climatiques de la culture.
2. Hystérésis haut/bas pour stabiliser les plaques.
3. Dernière maladie détectée.
4. Facteur météo de propagation de la maladie.

Exemples:

- Pluie intense ou humidité élevée + Septoria: plaques inclinées, canal actif, risque de propagation augmenté.
- Forte chaleur + pression de virus transmis par aleurodes: plaques horizontales si le seuil chaleur est dépassé.
- Late blight en climat humide: traitement A/B/C, car la propagation est très élevée.

Après chaque image, l'application affiche deux blocs séparés:

- `Influence chaleur`: active ou non, niveau d'aggravation, action adaptée.
- `Influence pluie / humidité`: active ou non, niveau d'aggravation, action adaptée.

La matrice couvre les 15 classes du dataset PlantVillage utilisées par EfficientNet-B0 et ResNet-50, y compris les classes `healthy` où le système conclut à une protection climatique sans propagation maladie.

Formule de confiance IA:

```text
P(y = k | z) = exp(z_k) / sum_j exp(z_j)
```

Indice de risque environnemental utilisé comme base conceptuelle:

```text
I_risk = alpha * H_r + beta * 1 / (|T - T_opt| + 1)
avec alpha + beta = 1
```

## Règles zones A/B/C

- Maladies très propagatives: A, B et C.
- Maladies à propagation par pluie ou éclaboussures: zone détectée + demi-zone voisine.
- Confiance IA très élevée ou météo favorable: extension de surveillance et parfois traitement global.
- Feuille saine: surveillance uniquement.

Le résultat ne s’arrête donc pas à un pourcentage: l’application donne les segments précis à traiter, par exemple `Zone A complète + moitié gauche de la zone B`.

## Sources et ressources utilisées

Documents projet fournis:

- `PLBD_Rapport_Groupe3_Janvier2026.pdf`
- `SOUTENANCE PRESENTATION.pdf`
- `Beige Minimalistic Pros and Cons List Decision Table A4 Document.pdf`
- `notebooks/AgroShield_VSCode_CPU.ipynb`
- `Rohan Banerjee - Hands-on TinyML`
- `Jason Brownlee - Imbalanced Classification with Python`

Références méthodologiques IA / embarqué:

- Rohan Banerjee, `Hands-on TinyML`: justification de l'inférence embarquée, de l'optimisation edge et du déploiement sur dispositifs limités.
- Jason Brownlee, `Imbalanced Classification with Python`: justification du traitement des classes déséquilibrées, des métriques macro/F1 et de la prudence face à l'accuracy seule.

Références agronomiques consultées pour relier maladies et météo:

- University of Minnesota Extension, late blight: https://extension.umn.edu/disease-management/late-blight
- Penn State Extension, potato/tomato late blight and cool wet conditions: https://extension.psu.edu/late-blight-of-potato-and-tomato
- Iowa State University Extension, bacterial spot of pepper and tomatoes: https://yardandgarden.extension.iastate.edu/encyclopedia/bacterial-spot-pepper-and-tomatoes
- University of Minnesota Extension, tomato leaf mold: https://extension.umn.edu/disease-management/tomato-leaf-mold
- NC State Extension, Tomato yellow leaf curl virus: https://content.ces.ncsu.edu/tomato-yellow-leaf-curl-virus
- University of Maryland Extension, tomato heat stress and high-temperature problems: https://extension.umd.edu/resource/high-temperature-and-heat-injury-vegetables/
- FAO Irrigation and Drainage Paper 56, Penman-Monteith / ET0 reference: https://www.fao.org/4/X0490E/x0490e08.htm
- Oregon State University Extension, vegetable degree-day models: https://extension.oregonstate.edu/catalog/em-9305-vegetable-degree-day-models-introduction-farmers-gardeners
- USDA Climate Hubs, AgroClimate Growing Degree Days Monitoring: https://www.climatehubs.usda.gov/hubs/southeast/tools/agroclimate-growing-degree-days-monitoring
- Camera vegetation indices / Green Chromatic Coordinate context: https://www.sciencedirect.com/science/article/pii/S0168192317303763
- University of Minnesota Extension, growing tomatoes, peppers and potatoes: https://extension.umn.edu/vegetables

## GitHub

Le dépôt garde le code, la documentation, les métadonnées et l’architecture VS Code. Les datasets et poids lourds sont conservés localement et ignorés par `.gitignore`.

Pour partager les modèles complets, utiliser Git LFS ou un lien Drive/OneDrive dans la documentation du projet.
