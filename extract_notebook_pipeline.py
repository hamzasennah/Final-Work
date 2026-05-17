"""Extract the important notebook steps into visible VS Code pipeline files."""

from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "AgroShield_VSCode_CPU.ipynb"
OUT = ROOT / "src" / "pipeline"

NAME_MAP = {
    "Etape 0": "00_installation_verification.py",
    "Etape 1": "01_extraction_data.py",
    "Etape 2": "02_equilibrage_albumentations.py",
    "Etape 3": "03_split_train_val_test.py",
    "Etape 4": "04_dataloaders_weighted_sampler.py",
    "Etape 5": "05_modeles_cnn.py",
    "Etape 6": "06_entrainement_cpu.py",
    "Etape 7": "07_courbes_apprentissage.py",
    "Etape 8": "08_evaluation_test_set.py",
    "Etape 9": "09_export_raspberry_pi.py",
    "Etape 10": "10_test_modeles_image.py",
    "Etape 11": "11_generation_server_archive.py",
    "Etape 12": "12_generation_index_archive.py",
}


def ascii_key(title: str) -> str:
    return (
        title.replace("Étape", "Etape")
        .replace("—", "-")
        .replace("–", "-")
        .split("-")[0]
        .strip()
    )


def normalized_paths(source: str) -> str:
    return (
        source.replace(r"C:\Users\pc\Desktop\PlantVillage_balanced", "data/balanced")
        .replace(r"C:\Users\pc\Desktop\PlantVillage", "data/raw/PlantVillage")
        .replace(r"C:\Users\pc\Desktop\AgroShield", ".")
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    current = None
    steps = []
    code_by_step = {}

    for cell in nb["cells"]:
        source = "".join(cell.get("source", ""))
        if cell.get("cell_type") == "markdown" and source.strip().startswith("## Étape"):
            current = source.strip().splitlines()[0].replace("## ", "")
            steps.append(current)
            code_by_step[current] = []
        elif cell.get("cell_type") == "code" and current:
            code_by_step[current].append(source)

    for title in steps:
        key = ascii_key(title)
        filename = NAME_MAP.get(key)
        if filename is None:
            filename = re.sub(r"[^a-zA-Z0-9]+", "_", title).strip("_") + ".py"
        body = "\n\n".join(code_by_step[title])
        body = normalized_paths(body)
        header = (
            f'"""{title}\n\n'
            "Fichier extrait du notebook notebooks/AgroShield_VSCode_CPU.ipynb.\n"
            "Le notebook original reste la référence; ce script rend l'étape visible dans VS Code.\n"
            '"""\n\n'
        )
        (OUT / filename).write_text(header + body, encoding="utf-8")

    print(f"{len(steps)} fichiers créés dans {OUT}")


if __name__ == "__main__":
    main()
