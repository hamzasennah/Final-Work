from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_MODELS = [
    "models/efficientnet_b0.pth",
    "models/efficientnet_b0_meta.json",
    "models/resnet_50.pth",
    "models/resnet_50_meta.json",
]


def count_dataset_classes(path: Path):
    if not path.exists():
        return None
    classes = [p for p in path.iterdir() if p.is_dir()]
    images = {
        file.resolve()
        for ext in ("*.jpg", "*.jpeg", "*.png")
        for file in path.rglob(ext)
    }
    if not classes and not images:
        return None
    return len(classes), len(images)


def main():
    print("AgroShield - verification du projet")
    print(f"Racine: {ROOT}")

    raw = ROOT / "data" / "raw" / "PlantVillage"
    if not raw.exists():
        raw = ROOT / "PlantVillage"
    balanced = ROOT / "data" / "balanced"
    for label, path in [("dataset brut", raw), ("dataset equilibre", balanced)]:
        stats = count_dataset_classes(path)
        if stats:
            print(f"[OK] {label}: {stats[0]} classes, {stats[1]} images - {path}")
        else:
            print(f"[INFO] {label} non trouve: {path}")

    for rel in EXPECTED_MODELS:
        path = ROOT / rel
        print(f"[{'OK' if path.exists() else 'MANQUANT'}] {rel}")

    meta_path = ROOT / "models" / "efficientnet_b0_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print(f"[OK] metadata: {len(meta.get('classes', []))} classes")

    print("\nPour lancer l'interface: python server.py")


if __name__ == "__main__":
    main()
