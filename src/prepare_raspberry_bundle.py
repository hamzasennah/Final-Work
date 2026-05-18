from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
DEPLOY = ROOT / "raspberry" / "deploy"


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main() -> None:
    model_dst = DEPLOY / "models" / "efficientnet_b0_rpi.pt"
    meta_dst = DEPLOY / "models" / "efficientnet_b0_meta.json"
    copied_model = copy_if_exists(MODELS / "efficientnet_b0_rpi.pt", model_dst)
    copied_meta = copy_if_exists(MODELS / "efficientnet_b0_meta.json", meta_dst)
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": str(model_dst.relative_to(ROOT)) if copied_model else None,
        "metadata": str(meta_dst.relative_to(ROOT)) if copied_meta else None,
        "target": "Raspberry Pi CPU TorchScript EfficientNet-B0",
        "server_endpoint_photo": "/api/raspberry/photo",
        "server_endpoint_sensors": "/api/raspberry/sensors",
    }
    DEPLOY.mkdir(parents=True, exist_ok=True)
    (DEPLOY / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
