#!/usr/bin/env python3
"""
Copy note images into ``hexo-site/source/imgs/``.

- Sources are root ``imgs/`` plus image directories owned by synced series.
- Only **adds/updates** files; does **not** delete files that exist only under
  Hexo (e.g. ``.gitkeep``).
- Skips hidden names (``.*``) and non-file entries.
- Copies recursively if a source contains subdirectories.

Run from anywhere; repo root is inferred from this script's location.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

IMAGE_SOURCE_DIRS = (
    Path("imgs"),
    Path("nccl_pcie_barex_learning") / "imgs",
    Path("cuda_cute_nvidia_learning") / "imgs",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def sync_imgs(*, dry_run: bool) -> int:
    root = repo_root()
    dest = root / "hexo-site" / "source" / "imgs"

    if not dest.parent.is_dir():
        print(f"Error: Hexo source directory not found: {dest.parent}", file=sys.stderr)
        return 1

    dest.mkdir(parents=True, exist_ok=True)

    copied = 0
    unchanged = 0

    def walk(directory: Path, rel: Path) -> None:
        nonlocal copied, unchanged
        for path in sorted(directory.iterdir()):
            if path.name.startswith("."):
                continue
            rel_child = rel / path.name
            if path.is_dir():
                walk(path, rel_child)
                continue
            if not path.is_file():
                continue
            out = dest / rel_child
            if out.exists():
                same_size = out.stat().st_size == path.stat().st_size
                same_mtime = int(out.stat().st_mtime) == int(path.stat().st_mtime)
                if same_size and same_mtime:
                    unchanged += 1
                    continue
            if dry_run:
                print(f"[dry-run] would copy {path.relative_to(root)} -> {out.relative_to(root)}")
                copied += 1
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, out)
            print(f"copy {path.relative_to(root)} -> {out.relative_to(root)}")
            copied += 1

    found_source = False
    for relative_source in IMAGE_SOURCE_DIRS:
        source = root / relative_source
        if not source.is_dir():
            print(f"Note: {relative_source} missing; skip image source.")
            continue
        found_source = True
        walk(source, Path("."))

    if not found_source:
        print("Note: no image source directories found; nothing to do.")
        return 0

    label = "[dry-run] " if dry_run else ""
    print(f"{label}Done: {copied} would copy/copied, {unchanged} unchanged.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without writing files",
    )
    args = parser.parse_args()
    return sync_imgs(dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
