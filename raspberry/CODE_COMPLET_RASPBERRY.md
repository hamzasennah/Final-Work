# Codes complets a implementer sur Raspberry Pi

Ce dossier `raspberry/` est le code a copier sur la Raspberry Pi pour faire fonctionner AgroShield:

- lecture capteurs;
- envoi des donnees vers l'application web;
- capture camera;
- envoi de l'image pour diagnostic IA;
- reception de la commande plaques;
- actionnement des servomoteurs;
- maintien local de la derniere commande maladie-climat si le serveur devient indisponible.

## 1. Structure a avoir sur la Raspberry

```text
raspberry/
├── agroshield_rpi_client.py
├── control_logic.py
├── sensors.py
├── actuators.py
├── camera_capture.py
├── model_inference.py
├── config.json
├── requirements-rpi.txt
├── install_service.sh
└── agroshield-rpi.service
```

Le plus simple: copier tout le dossier `raspberry/` depuis le projet PC vers:

```bash
/home/pi/agroshield/raspberry
```

## 2. Installation Raspberry

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip libgpiod2

cd /home/pi/agroshield/raspberry
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-rpi.txt
cp config.example.json config.json
```

Dans `config.json`, remplacer `server_url` par l'adresse IP du PC qui lance `server.py`.

Exemple:

```json
{
  "server_url": "http://10.100.12.8:5000",
  "pi_id": "agroshield-pi-01",
  "crop": "tomato",
  "zone_id": "A",
  "sensor_interval_seconds": 5,
  "photo_interval_seconds": 60,
  "disease_command_ttl_seconds": 1800,
  "pins": {
    "dht": 4,
    "rain_digital": 17,
    "servo_left": 12,
    "servo_right": 13,
    "servo_camera": 18
  },
  "servo_angles": {
    "repos": 0,
    "chaleur": 90,
    "pluie": 55
  }
}
```

## 3. Lancement manuel

```bash
cd /home/pi/agroshield/raspberry
source .venv/bin/activate
python agroshield_rpi_client.py --config config.json
```

## 4. Fichier `requirements-rpi.txt`

```text
requests
Pillow
gpiozero
lgpio
adafruit-circuitpython-dht
adafruit-circuitpython-bh1750
picamera2
torch
torchvision
```

## 5. Fichier principal `agroshield_rpi_client.py`

```python
import argparse
import json
import time
from pathlib import Path

import requests

from actuators import PlateActuators
from camera_capture import capture_image
from control_logic import HysteresisController
from sensors import SensorReader


def post_sensors(server_url, payload):
    response = requests.post(f"{server_url}/api/raspberry/sensors", json=payload, timeout=8)
    response.raise_for_status()
    return response.json()


def post_photo(server_url, image_path, zone_id):
    with Path(image_path).open("rb") as fh:
        response = requests.post(
            f"{server_url}/api/raspberry/photo",
            files={"image": (Path(image_path).name, fh, "image/jpeg")},
            data={"zone_id": zone_id},
            timeout=30,
        )
    response.raise_for_status()
    return response.json()


def main(config_path):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    server_url = config["server_url"].rstrip("/")
    reader = SensorReader(config["pins"])
    actuators = PlateActuators(config["pins"], config.get("servo_angles"))
    controller = HysteresisController(config.get("crop", "default"))
    next_photo = 0
    disease_command = None

    while True:
        sensors = reader.read()
        sensors.update({"pi_id": config.get("pi_id", "agroshield-pi"), "crop": config.get("crop", "default")})
        local_mode = controller.decide(sensors, disease_command)
        try:
            result = post_sensors(server_url, sensors)
            command = result.get("actuator_command", {})
            mode = command.get("mode") or result.get("mechanism", {}).get("mode", local_mode)
            print(f"[SERVER] commande plaques={mode} source={command.get('source', 'unknown')} raison={command.get('reason', '')}")
        except Exception as exc:
            print(f"[WARN] Serveur indisponible, decision locale appliquee: {exc}")
            mode = local_mode
        actuators.apply(mode)

        if time.time() >= next_photo:
            image_path = capture_image("latest_leaf.jpg")
            try:
                analysis = post_photo(server_url, image_path, config.get("zone_id", "A"))
                print(f"[IA] {analysis.get('predicted_class')} confiance={analysis.get('confidence')}")
                climate = analysis.get("climate_diagnosis") or {}
                disease_command = {
                    "timestamp": time.time(),
                    "ttl_seconds": int(config.get("disease_command_ttl_seconds", 1800)),
                    "recommended_mechanism": climate.get("recommended_mechanism", "repos"),
                    "instant_risk": climate.get("instant_risk", 0),
                    "heat_active": climate.get("heat", {}).get("active", False),
                    "rain_active": climate.get("rain", {}).get("active", False),
                }
                print(
                    "[IA->PLAQUES] "
                    f"mode={disease_command['recommended_mechanism']} "
                    f"risque={round(disease_command['instant_risk'] * 100)}% "
                    f"chaleur={disease_command['heat_active']} pluie={disease_command['rain_active']}"
                )
            except Exception as exc:
                print(f"[WARN] Envoi image impossible: {exc}")
            next_photo = time.time() + int(config.get("photo_interval_seconds", 60))

        time.sleep(int(config.get("sensor_interval_seconds", 5)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    main(args.config)
```

## 6. Logique locale `control_logic.py`

Ce fichier permet a la Raspberry de continuer localement si le serveur ne repond plus.

```python
import time


CROP_THRESHOLDS = {
    "tomato": {"temp_on": 29.0, "temp_off": 26.0, "humidity_on": 82.0, "humidity_off": 72.0, "rain_on": 18.0, "rain_off": 8.0, "luminosity_on": 720.0},
    "pepper": {"temp_on": 30.0, "temp_off": 27.0, "humidity_on": 80.0, "humidity_off": 70.0, "rain_on": 16.0, "rain_off": 7.0, "luminosity_on": 700.0},
    "potato": {"temp_on": 26.0, "temp_off": 23.0, "humidity_on": 85.0, "humidity_off": 75.0, "rain_on": 20.0, "rain_off": 9.0, "luminosity_on": 600.0},
    "default": {"temp_on": 29.0, "temp_off": 26.0, "humidity_on": 80.0, "humidity_off": 70.0, "rain_on": 18.0, "rain_off": 8.0, "luminosity_on": 650.0},
}


class HysteresisController:
    def __init__(self, crop="default", delay_seconds=8):
        self.crop = crop if crop in CROP_THRESHOLDS else "default"
        self.mode = "repos"
        self.last_change = 0
        self.delay_seconds = delay_seconds

    def decide(self, sensors, disease_command=None):
        th = CROP_THRESHOLDS[self.crop]
        rain_alert = sensors["precipitation"] >= th["rain_on"] or sensors["humidity"] >= th["humidity_on"]
        heat_alert = sensors["temperature"] >= th["temp_on"] or sensors["luminosity"] >= th["luminosity_on"]
        clear_rain = sensors["precipitation"] <= th["rain_off"] and sensors["humidity"] <= th["humidity_off"]
        clear_heat = sensors["temperature"] <= th["temp_off"] and sensors["luminosity"] < th["luminosity_on"] - 80

        disease_mode = self._disease_climate_mode(disease_command, heat_alert, rain_alert)
        if disease_mode:
            target = disease_mode
        elif self.mode == "pluie" and not clear_rain:
            target = "pluie"
        elif rain_alert:
            target = "pluie"
        elif self.mode == "chaleur" and not clear_heat:
            target = "chaleur"
        elif heat_alert:
            target = "chaleur"
        else:
            target = "repos"

        if target != self.mode and time.time() - self.last_change >= self.delay_seconds:
            self.mode = target
            self.last_change = time.time()
        return self.mode

    def _disease_climate_mode(self, disease_command, heat_alert, rain_alert):
        if not disease_command:
            return None
        if time.time() - disease_command.get("timestamp", 0) > disease_command.get("ttl_seconds", 1800):
            return None
        mode = disease_command.get("recommended_mechanism", "repos")
        heat_active = disease_command.get("heat_active", False) and heat_alert
        rain_active = disease_command.get("rain_active", False) and rain_alert
        risk = disease_command.get("instant_risk", 0)
        if risk < 0.70:
            return None
        if mode == "pluie" and rain_active:
            return "pluie"
        if mode == "chaleur" and heat_active:
            return "chaleur"
        return None
```

## 7. Lecture capteurs `sensors.py`

```python
import random


class SensorReader:
    def __init__(self, pins):
        self.pins = pins
        self.dht = None
        self.light = None
        self.rain_input = None
        try:
            import board
            import adafruit_dht

            self.dht = adafruit_dht.DHT22(getattr(board, f"D{pins['dht']}"))
        except Exception as exc:
            print(f"[WARN] DHT22 en mode simulation: {exc}")
        try:
            import board
            import adafruit_bh1750

            self.light = adafruit_bh1750.BH1750(board.I2C())
        except Exception as exc:
            print(f"[WARN] BH1750 en mode simulation: {exc}")
        try:
            from gpiozero import DigitalInputDevice

            self.rain_input = DigitalInputDevice(pins["rain_digital"])
        except Exception as exc:
            print(f"[WARN] Capteur pluie en mode simulation: {exc}")

    def read(self):
        if self.dht:
            temperature = float(self.dht.temperature)
            humidity = float(self.dht.humidity)
        else:
            temperature = random.uniform(22, 33)
            humidity = random.uniform(50, 88)

        luminosity = float(self.light.lux) if self.light else random.uniform(250, 850)
        if self.rain_input:
            precipitation = 22.0 if not self.rain_input.value else 0.0
        else:
            precipitation = random.choice([0.0, 2.0, 6.0, 20.0])

        return {
            "temperature": round(temperature, 1),
            "humidity": round(humidity, 1),
            "precipitation": round(precipitation, 1),
            "luminosity": round(luminosity, 1),
            "soil_moisture": 0.0,
            "reservoir_level": 0.0,
        }
```

## 8. Servomoteurs `actuators.py`

```python
class PlateActuators:
    def __init__(self, pins, servo_angles=None):
        self.pins = pins
        self.servo_angles = servo_angles or {"repos": 0, "chaleur": 90, "pluie": 55}
        self.enabled = False
        self.left = self.right = self.camera = None
        try:
            from gpiozero import AngularServo

            self.left = AngularServo(pins["servo_left"], min_angle=0, max_angle=90)
            self.right = AngularServo(pins["servo_right"], min_angle=0, max_angle=90)
            self.camera = AngularServo(pins.get("servo_camera", pins["servo_left"]), min_angle=0, max_angle=90)
            self.enabled = True
        except Exception as exc:
            print(f"[WARN] Servos en mode simulation: {exc}")

    def apply(self, mode):
        angle = self.servo_angles.get(mode, 0)
        if mode == "pluie":
            left_angle, right_angle, camera_angle = angle, angle, 35
        elif mode == "chaleur":
            left_angle, right_angle, camera_angle = 90, 90, 20
        else:
            left_angle, right_angle, camera_angle = 0, 0, 0

        if self.enabled:
            self.left.angle = left_angle
            self.right.angle = right_angle
            self.camera.angle = camera_angle
        print(f"[ACTUATOR] mode={mode} left={left_angle} right={right_angle} camera={camera_angle}")
```

## 9. Camera `camera_capture.py`

```python
from pathlib import Path
from PIL import Image


def capture_image(output_path="latest_leaf.jpg"):
    output = Path(output_path)
    try:
        from picamera2 import Picamera2

        camera = Picamera2()
        camera.configure(camera.create_still_configuration(main={"size": (1280, 720)}))
        camera.start()
        camera.capture_file(str(output))
        camera.stop()
    except Exception as exc:
        print(f"[WARN] Camera en mode image de test: {exc}")
        Image.new("RGB", (640, 480), (76, 130, 72)).save(output)
    return output
```

## 10. Inference locale optionnelle `model_inference.py`

Par defaut, la Raspberry envoie l'image au serveur. Ce fichier sert seulement si vous voulez tester EfficientNet-B0 localement sur Raspberry.

```python
import json
from pathlib import Path

from PIL import Image
import torch
from torchvision import transforms


def load_classifier(model_path="deploy/models/efficientnet_b0_rpi.pt", meta_path="deploy/models/efficientnet_b0_meta.json"):
    model = torch.jit.load(str(model_path), map_location="cpu")
    model.eval()
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    classes = meta["classes"]
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, classes, preprocess


def predict_image(image_path, model, classes, preprocess):
    img = Image.open(image_path).convert("RGB")
    with torch.no_grad():
        probs = torch.softmax(model(preprocess(img).unsqueeze(0)), dim=1)[0]
    confidence, index = torch.max(probs, dim=0)
    return {"class": classes[int(index)], "confidence": float(confidence)}
```

## 11. Service systemd

Fichier `agroshield-rpi.service`:

```ini
[Unit]
Description=AgroShield Raspberry Pi client
After=network-online.target
Wants=network-online.target

[Service]
WorkingDirectory=/home/pi/agroshield/raspberry
ExecStart=/home/pi/agroshield/raspberry/.venv/bin/python agroshield_rpi_client.py --config config.json
Restart=always
RestartSec=5
User=pi

[Install]
WantedBy=multi-user.target
```

Installation:

```bash
sudo cp agroshield-rpi.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable agroshield-rpi
sudo systemctl start agroshield-rpi
sudo systemctl status agroshield-rpi
```

Logs:

```bash
journalctl -u agroshield-rpi -f
```

## 12. Verification importante

Avant de lancer sur Raspberry, le PC doit lancer:

```bash
python server.py
```

Tester depuis la Raspberry:

```bash
curl http://ADRESSE_IP_DU_PC:5000/api/health
```

Si la reponse contient `"status": "ok"`, la connexion Raspberry -> application est bonne.

## 13. Test camera USB vers l'application sans upload manuel

Si vous utilisez une camera USB avec OpenCV, utilisez le fichier
`usb_camera_upload.py`. Il envoie l'image vers `POST /upload`, qui est
maintenant integre au serveur AgroShield principal.

```bash
cd raspberry
python3 usb_camera_upload.py --server http://ADRESSE_IP_DU_PC:5000 --zone A
```

Code minimal equivalent:

```python
import cv2
import requests

PC_IP = "10.116.177.98"
URL = f"http://{PC_IP}:5000/upload"
IMAGE_NAME = "capture.jpg"

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

if ret:
    cv2.imwrite(IMAGE_NAME, frame)
    with open(IMAGE_NAME, "rb") as img:
        response = requests.post(
            URL,
            files={"image": img},
            data={"zone_id": "A"},
            timeout=30,
        )
    print(response.text)
else:
    print("Erreur camera")

cap.release()
```

Important: lancer `C:\Users\pc\Desktop\final work\server.py`, pas le petit
serveur de test qui sauvegarde seulement les fichiers. AgroShield possede
maintenant les routes `/upload`, `/api/sensors`, `/api/health` et l'analyse IA.
Si une image est quand meme seulement deposee dans
`C:\Users\pc\Desktop\reception des images`, l'interface la detecte aussi par
scan automatique et affiche le test sans selection manuelle.
