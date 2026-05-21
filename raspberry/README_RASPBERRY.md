# AgroShield sur Raspberry Pi

Ce dossier contient le code embarque pour fermer la boucle capteurs -> decision -> servomoteurs -> application web.

## Principe

1. La Raspberry Pi lit temperature, humidite, luminosite, pluie, humidite du sol et niveau reservoir.
2. Elle applique une decision locale avec hysteresis pour eviter les oscillations des plaques.
3. Elle envoie les mesures a l'application PC via `POST /api/raspberry/sensors`.
4. Elle capture une image avec la camera puis l'envoie a `POST /api/raspberry/photo`.
5. L'application classe l'image avec EfficientNet-B0/ResNet-50 et renvoie les zones a traiter.
6. Si la maladie detectee est aggravee par la pluie ou la chaleur instantanee, le serveur renvoie aussi `actuator_command`. La Raspberry applique alors directement `pluie` ou `chaleur`.

## Camera USB simple

Pour le test montre dans RealVNC/Thonny, lancez le serveur principal sur le PC:

```bash
python server.py
```

Puis executez sur Raspberry:

```bash
python3 usb_camera_upload.py --server http://ADRESSE_IP_DU_PC:5000 --zone A
```

La route `/upload` sauvegarde l'image dans `C:\Users\pc\Desktop\reception des images`,
lance l'analyse IA et met a jour l'onglet Diagnostic automatiquement.

Si la capture USB contient un PC, un document ou un objet hors culture, le serveur
renvoie `out_of_domain=true`. Dans ce cas la Raspberry ne doit appliquer aucune
commande maladie: elle continue seulement la logique capteurs/climat normale.

La commande des plaques vient donc de deux sources:

- seuil climatique simple: temperature, luminosite, pluie, humidite;
- boucle maladie-climat: maladie detectee + meteo qui augmente son risque.

Si le serveur devient temporairement indisponible, le client garde pendant `disease_command_ttl_seconds` la derniere commande maladie-climat issue de l'analyse image. Cela permet de continuer a proteger la culture localement.

## Installation rapide

```bash
cd raspberry
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-rpi.txt
cp config.example.json config.json
python agroshield_rpi_client.py --config config.json
```

Dans `config.json`, remplacer `server_url` par l'adresse IP du PC qui lance `server.py`.

## Mode service

```bash
sudo cp agroshield-rpi.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable agroshield-rpi
sudo systemctl start agroshield-rpi
```

## Modele IA

Le script `src/prepare_raspberry_bundle.py` prepare `raspberry/deploy/models/efficientnet_b0_rpi.pt`.
Ce fichier TorchScript peut etre charge localement par `model_inference.py` si on veut faire une inference directement sur la Raspberry. Par defaut, la Raspberry envoie l'image a l'application web pour que le test apparaisse immediatement dans l'interface.
