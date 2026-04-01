#!/usr/bin/env python3
"""
Copy images from repository root ``imgs/`` into ``hexo-site/source/imgs/``.

- Only **adds/updates** files from root ``imgs/``; does **not** delete files
  that exist only under Hexo (e.g. ``.gitkeep``).
- Skips hidden names (``.*``) and non-file entries.
- Copies recursively if you add subdirectories under ``imgs/``.

Run from anywhere; repo root is inferred from this script's location.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def sync_imgs(*, dry_run: bool) -> int:
    root = repo_root()
    src = root / "imgs"
    dest = root / "hexo-site" / "source" / "imgs"

    if not dest.parent.is_dir():
        print(f"Error: Hexo source directory not found: {dest.parent}", file=sys.stderr)
        return 1

    if not src.is_dir():
        if dry_run:
            print(f"[dry-run] no source dir {src}; nothing to do")
        else:
            print(f"Note: {src} missing; skip image sync.")
        return 0

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

    walk(src, Path("."))

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
