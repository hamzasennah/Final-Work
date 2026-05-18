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

    while True:
        sensors = reader.read()
        sensors.update({"pi_id": config.get("pi_id", "agroshield-pi"), "crop": config.get("crop", "default")})
        local_mode = controller.decide(sensors)
        try:
            result = post_sensors(server_url, sensors)
            mode = result.get("mechanism", {}).get("mode", local_mode)
        except Exception as exc:
            print(f"[WARN] Serveur indisponible, decision locale appliquee: {exc}")
            mode = local_mode
        actuators.apply(mode)

        if time.time() >= next_photo:
            image_path = capture_image("latest_leaf.jpg")
            try:
                analysis = post_photo(server_url, image_path, config.get("zone_id", "A"))
                print(f"[IA] {analysis.get('predicted_class')} confiance={analysis.get('confidence')}")
            except Exception as exc:
                print(f"[WARN] Envoi image impossible: {exc}")
            next_photo = time.time() + int(config.get("photo_interval_seconds", 60))

        time.sleep(int(config.get("sensor_interval_seconds", 5)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    main(args.config)
