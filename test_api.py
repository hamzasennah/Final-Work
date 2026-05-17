from pathlib import Path
from urllib import request
import json

ROOT = Path(__file__).resolve().parents[1]


def get_json(url):
    with request.urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def main():
    base = "http://127.0.0.1:5000"
    print("Test API AgroShield")
    health = get_json(f"{base}/api/health")
    sensors = get_json(f"{base}/api/sensors")
    print(f"[OK] serveur: {health['status']}")
    print(f"[INFO] modeles: {health['model_status']['message']}")
    print(f"[OK] capteurs: {sensors['temperature']} C, {sensors['humidity']} %, mode {sensors['mechanism']['mode']}")


if __name__ == "__main__":
    main()
