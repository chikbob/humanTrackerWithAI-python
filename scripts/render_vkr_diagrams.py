from __future__ import annotations

from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
DIAGRAMS_DIR = ROOT / "docs" / "vkr" / "diagrams"
OUT_DIR = DIAGRAMS_DIR / "rendered"
KROKI_URL = "https://kroki.io/plantuml/png"


def render_diagram(src_path: Path, out_path: Path) -> None:
    response = requests.post(KROKI_URL, data=src_path.read_text(encoding="utf-8").encode("utf-8"), timeout=60)
    response.raise_for_status()
    out_path.write_bytes(response.content)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for src_path in sorted(DIAGRAMS_DIR.glob("*.puml")):
        out_path = OUT_DIR / f"{src_path.stem}.png"
        render_diagram(src_path, out_path)
        print(f"rendered {src_path.name} -> {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
