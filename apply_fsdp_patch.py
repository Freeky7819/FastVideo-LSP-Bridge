# apply_fsdp_patch.py
import re
from pathlib import Path

ROOT = Path(__file__).parent
TARGET = ROOT / "fastvideo" / "models" / "loader" / "fsdp_load.py"

IMPORT_PATTERN = re.compile(
    r"from\s+torch\.distributed\.fsdp\s+import\s*\((?:.|\n)*?\)\s*",
    re.MULTILINE,
)

REPLACEMENT = (
    "from .fsdp_imports import (\n"
    "    FSDPModule,\n"
    "    CPUOffloadPolicy,\n"
    "    MixedPrecisionPolicy,\n"
    "    ShardingStrategy,\n"
    "    BackwardPrefetch,\n"
    "    StateDictType,\n"
    "    FullStateDictConfig,\n"
    "    LocalStateDictConfig,\n"
    ")\n"
)

def main():
    if not TARGET.exists():
        raise SystemExit(f"File not found: {TARGET}")

    src = TARGET.read_text(encoding="utf-8")

    if "from .fsdp_imports import (" in src:
        print("[ok] fsdp_load.py already patched.")
        return

    if IMPORT_PATTERN.search(src):
        new_src = IMPORT_PATTERN.sub(REPLACEMENT, src, count=1)
    else:
        # Fallback: vstavi za prvo 'import' sekcijo na vrhu datoteke.
        lines = src.splitlines(keepends=True)
        insert_at = 0
        # preskoči shebang/komentarje/blank
        while insert_at < len(lines) and (
            lines[insert_at].lstrip().startswith("#") or lines[insert_at].strip() == ""
        ):
            insert_at += 1
        # poišči konec uvodnih importov
        while insert_at < len(lines) and (
            lines[insert_at].startswith("import ") or lines[insert_at].startswith("from ")
        ):
            insert_at += 1
        lines.insert(insert_at, REPLACEMENT)
        new_src = "".join(lines)

    TARGET.write_text(new_src, encoding="utf-8")
    print(f"[done] Patched imports in: {TARGET}")

if __name__ == "__main__":
    main()
