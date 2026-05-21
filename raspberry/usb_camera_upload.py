"""
Capture USB simple pour Raspberry Pi.

Usage:
    python3 usb_camera_upload.py --server http://ADRESSE_IP_DU_PC:5000 --zone A

Le serveur AgroShield principal possede deja la route /upload. Il sauvegarde
l'image dans C:\\Users\\pc\\Desktop\\reception des images, lance le diagnostic IA
et l'affiche automatiquement dans l'interface web.
"""

import argparse
from pathlib import Path

import cv2
import requests


def capture_and_send(server_url, zone_id, camera_index, image_name):
    capture_path = Path(image_name)
    cap = cv2.VideoCapture(camera_index)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Erreur camera: aucune image capturee")

    cv2.imwrite(str(capture_path), frame)
    with capture_path.open("rb") as img:
        response = requests.post(
            f"{server_url.rstrip('/')}/upload",
            files={"image": (capture_path.name, img, "image/jpeg")},
            data={"zone_id": zone_id},
            timeout=30,
        )
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser(description="Envoi camera USB Raspberry vers AgroShield")
    parser.add_argument("--server", required=True, help="Exemple: http://10.116.177.98:5000")
    parser.add_argument("--zone", default="A", choices=["A", "B", "C"])
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--image", default="capture.jpg")
    args = parser.parse_args()

    result = capture_and_send(args.server, args.zone, args.camera, args.image)
    print(result)


if __name__ == "__main__":
    main()
