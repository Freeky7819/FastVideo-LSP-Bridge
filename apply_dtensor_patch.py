# apply_dtensor_patch.py
from pathlib import Path
import re

ROOT = Path(__file__).parent
TARGET = ROOT / "fastvideo" / "pipelines" / "lora_pipeline.py"

TRYEXCEPT_BLOCK = (
    "try:\n"
    "    from torch.distributed.tensor import DTensor  # torch >= 2.5\n"
    "except Exception:  # torch <= 2.4\n"
    "    from torch.distributed._tensor.api import DTensor\n"
)

def main():
    if not TARGET.exists():
        raise SystemExit(f"[err] File not found: {TARGET}")

    src = TARGET.read_text(encoding="utf-8")

    # Če je naš try/except že prisoten, končamo.
    if "from torch.distributed._tensor.api import DTensor" in src or \
       "from torch.distributed.tensor import DTensor  # torch >= 2.5" in src:
        print("[ok] lora_pipeline.py already patched.")
        return

    # Zamenjaj enovrstični import z našim blokom.
    pattern = re.compile(r"from\s+torch\.distributed\.tensor\s+import\s+DTensor\s*\n")
    if pattern.search(src):
        new_src = pattern.sub(TRYEXCEPT_BLOCK, src, count=1)
        TARGET.write_text(new_src, encoding="utf-8")
        print(f"[done] Replaced DTensor import in: {TARGET}")
        return

    # Če enovrstični import ni najden, poskusi dodati naš blok med importe na vrhu
    lines = src.splitlines(keepends=True)
    insert_at = 0
    # preskoči shebang/komentarje/blank vrstice
    while insert_at < len(lines) and (lines[insert_at].lstrip().startswith("#") or lines[insert_at].strip() == ""):
        insert_at += 1
    # vtakni po prvih importih
    while insert_at < len(lines) and (lines[insert_at].startswith("import ") or lines[insert_at].startswith("from ")):
        insert_at += 1
    lines.insert(insert_at, TRYEXCEPT_BLOCK)
    TARGET.write_text("".join(lines), encoding="utf-8")
    print(f"[done] Injected DTensor compat block into: {TARGET}")

if __name__ == "__main__":
    main()
