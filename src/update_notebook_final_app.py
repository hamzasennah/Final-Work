"""Synchronize the notebook generator cells with the final app files."""

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "AgroShield_VSCode_CPU.ipynb"


def generator_source(target_file: str, content: str, title: str) -> str:
    return (
        "# ══════════════════════════════════════════════════════════\n"
        f"# {title}\n"
        "# ══════════════════════════════════════════════════════════\n\n"
        "from pathlib import Path\n\n"
        f"{Path(target_file).stem}_code = {content!r}\n\n"
        f"Path({target_file!r}).write_text({Path(target_file).stem}_code, encoding='utf-8')\n"
        f"print('✅ {target_file} synchronisé avec la version finale')\n"
    )


def main():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    server_code = (ROOT / "server.py").read_text(encoding="utf-8")
    index_code = (ROOT / "index.html").read_text(encoding="utf-8")

    nb["cells"][26]["source"] = generator_source(
        "server.py",
        server_code,
        "ÉTAPE 11 — Générer server.py final",
    ).splitlines(keepends=True)
    nb["cells"][28]["source"] = generator_source(
        "index.html",
        index_code,
        "ÉTAPE 12 — Générer index.html final",
    ).splitlines(keepends=True)

    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Notebook synchronisé avec server.py et index.html.")


if __name__ == "__main__":
    main()
