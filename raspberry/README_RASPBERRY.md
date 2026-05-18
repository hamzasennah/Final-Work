# AgroShield sur Raspberry Pi

Ce dossier contient le code embarque pour fermer la boucle capteurs -> decision -> servomoteurs -> application web.

## Principe

1. La Raspberry Pi lit temperature, humidite, luminosite, pluie, humidite du sol et niveau reservoir.
2. Elle applique une decision locale avec hysteresis pour eviter les oscillations des plaques.
3. Elle envoie les mesures a l'application PC via `POST /api/raspberry/sensors`.
4. Elle capture une image avec la camera puis l'envoie a `POST /api/raspberry/photo`.
5. L'application classe l'image avec EfficientNet-B0/ResNet-50 et renvoie les zones a traiter.

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
