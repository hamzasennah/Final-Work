"""Etape 3b - Split strict sans fuite entre train, validation et test.

Ce script corrige le point methodologique sensible signale en soutenance :
le sous-ensemble de test doit etre isole avant toute augmentation susceptible
de creer des variantes proches d'une image d'entrainement.

Principe retenu :
1. partir des images brutes, classe par classe ;
2. faire un split stratifie 70 % train, 15 % validation, 15 % test ;
3. ecrire des manifestes reproductibles ;
4. reserver l'augmentation hors ligne au train uniquement.

Validation et test restent donc des images sources independantes. Pendant
l'evaluation, elles ne doivent recevoir que les transformations deterministes :
resize, crop/center crop, normalisation.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path


RAW_DIR = Path("data/raw/PlantVillage")
OUT_DIR = Path("data/splits_strict")
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def list_images(class_dir: Path) -> list[Path]:
    return sorted(
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def split_one_class(images: list[Path], seed_key: str) -> tuple[list[Path], list[Path], list[Path]]:
    rng = random.Random(f"{SEED}:{seed_key}")
    shuffled = images[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_test = max(1, round(TEST_RATIO * n))
    n_val = max(1, round(VAL_RATIO * n))
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError(f"Classe trop petite pour un split 70/15/15 : {seed_key} ({n} images)")

    test = shuffled[:n_test]
    val = shuffled[n_test:n_test + n_val]
    train = shuffled[n_test + n_val:]
    return train, val, test


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Dataset brut introuvable : {RAW_DIR.resolve()}")

    classes = sorted(d.name for d in RAW_DIR.iterdir() if d.is_dir())
    class_to_idx = {name: idx for idx, name in enumerate(classes)}

    manifests = {"train": [], "val": [], "test": []}
    summary = []

    for class_name in classes:
        class_dir = RAW_DIR / class_name
        images = list_images(class_dir)
        if not images:
            continue

        train, val, test = split_one_class(images, class_name)
        parts = {"train": train, "val": val, "test": test}

        for split, paths in parts.items():
            for p in paths:
                manifests[split].append({
                    "path": str(p.as_posix()),
                    "class_name": class_name,
                    "label": class_to_idx[class_name],
                    "split": split,
                    "source": "raw_before_augmentation",
                })

        summary.append({
            "class_name": class_name,
            "raw": len(images),
            "train_originals": len(train),
            "val_originals": len(val),
            "test_originals": len(test),
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, rows in manifests.items():
        write_jsonl(OUT_DIR / f"{split}.jsonl", rows)

    metadata = {
        "seed": SEED,
        "ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        "raw_dir": str(RAW_DIR.as_posix()),
        "policy": (
            "Split source images first. Offline augmentation is allowed only for train. "
            "Validation and test are never used for augmentation, tuning by hand, or training."
        ),
        "class_to_idx": class_to_idx,
        "summary": summary,
        "totals": {
            split: len(rows) for split, rows in manifests.items()
        },
    }

    with (OUT_DIR / "split_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Split strict cree dans", OUT_DIR.resolve())
    print("Totaux :", metadata["totals"])
    print("\nControle par split :")
    for split, rows in manifests.items():
        counts = Counter(row["class_name"] for row in rows)
        min_count = min(counts.values()) if counts else 0
        max_count = max(counts.values()) if counts else 0
        print(f"- {split:<5}: {len(rows):>5} images | min/classe={min_count} max/classe={max_count}")


if __name__ == "__main__":
    main()
