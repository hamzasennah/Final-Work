"""Benchmark comparatif EfficientNet-B0 vs ResNet-50.

Le but n'est pas seulement d'afficher une accuracy. Pour choisir le modele
embarque, on compare aussi :
- precision globale ;
- precision / rappel / F1 macro ;
- F1 pondere ;
- nombre de parametres ;
- taille du fichier modele ;
- latence CPU moyenne.

Le script attend de preference le manifeste strict cree par
03b_split_strict_no_leakage.py. Ainsi, les images de test sont des images
sources isolees avant toute augmentation hors ligne.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "data/splits_strict/test.jsonl"
DEFAULT_OUT = ROOT / "outputs/benchmarks/model_benchmark.json"
DEVICE = torch.device("cpu")


class ManifestDataset(Dataset):
    def __init__(self, manifest_path: Path, transform):
        self.rows = []
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        path = ROOT / row["path"]
        try:
            img = Image.open(path).convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            raise RuntimeError(f"Image illisible dans le manifeste : {path}") from exc
        return self.transform(img), int(row["label"])


def read_classes(meta_path: Path) -> list[str]:
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)["classes"]


def build_model(model_name: str, num_classes: int):
    if model_name == "EfficientNet-B0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "ResNet-50":
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Modele inconnu : {model_name}")
    return model


def load_state(model, weights_path: Path):
    state = torch.load(weights_path, map_location=DEVICE)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    return model.to(DEVICE).eval()


def parameter_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def evaluate(model, loader):
    labels, preds = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().tolist()
            labels.extend(y.tolist())
            preds.extend(pred)

    acc = accuracy_score(labels, preds)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    return {
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "weighted_f1": weighted_f1,
    }


def measure_latency(model, image_size: int = 224, warmup: int = 5, runs: int = 30) -> float:
    x = torch.randn(1, 3, image_size, image_size, device=DEVICE)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        start = time.perf_counter()
        for _ in range(runs):
            model(x)
        end = time.perf_counter()
    return (end - start) * 1000 / runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if not args.manifest.exists():
        raise FileNotFoundError(
            f"Manifeste test introuvable : {args.manifest}. "
            "Lancer d'abord src/pipeline/03b_split_strict_no_leakage.py."
        )

    classes = read_classes(ROOT / "models/efficientnet_b0_meta.json")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = ManifestDataset(args.manifest, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    candidates = [
        {
            "name": "EfficientNet-B0",
            "weights": first_existing([
                ROOT / "models/efficientnet_b0.pth",
                ROOT / "efficientnet_b0.pth",
            ]),
            "role": "modele prioritaire pour Raspberry Pi",
        },
        {
            "name": "ResNet-50",
            "weights": first_existing([
                ROOT / "models/resnet_50.pth",
                ROOT / "resnet_50.pth",
            ]),
            "role": "modele de comparaison hors ligne",
        },
    ]

    results = []
    for candidate in candidates:
        if not candidate["weights"].exists():
            results.append({
                "model": candidate["name"],
                "status": "weights_missing",
                "weights": str(candidate["weights"]),
                "role": candidate["role"],
            })
            continue

        model = build_model(candidate["name"], len(classes))
        model = load_state(model, candidate["weights"])
        metrics = evaluate(model, loader)
        results.append({
            "model": candidate["name"],
            "status": "evaluated",
            "role": candidate["role"],
            "test_manifest": str(args.manifest),
            "test_images": len(dataset),
            "params_million": round(parameter_count(model) / 1_000_000, 2),
            "model_size_mb": round(file_size_mb(candidate["weights"]), 2),
            "cpu_latency_ms_image": round(measure_latency(model), 2),
            **{k: round(v * 100, 2) for k, v in metrics.items()},
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    csv_path = args.out.with_suffix(".csv")
    keys = sorted({k for row in results for k in row})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nBenchmark sauvegarde : {args.out}")
    print(f"CSV sauvegarde       : {csv_path}")


if __name__ == "__main__":
    main()
