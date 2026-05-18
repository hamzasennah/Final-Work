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
