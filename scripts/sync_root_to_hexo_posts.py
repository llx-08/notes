#!/usr/bin/env python3
"""
Sync *.md from repository root into hexo-site/source/_posts.

Convention:
  - Root `foo_bar.md` maps to slug `foo-bar` (underscores -> hyphens).
  - Existing post: `YYYY-MM-DD-<slug>.md` in _posts is updated; front matter is kept,
    body is replaced with the root file (minus optional front matter).
  - New post: creates `today-<slug>.md` with title from first `# heading` in root file.

Run from anywhere; repo root is inferred from this script's location.
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path

# Skip common non-article files at repo root
SKIP_STEMS = frozenset({"readme", "changelog", "license", "contributing"})


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def posts_dir(root: Path) -> Path:
    return root / "hexo-site" / "source" / "_posts"


def split_front_matter(text: str) -> tuple[str | None, str]:
    """Return (front_matter_inner_lines, body). front_matter_inner is without --- wrappers."""
    if not text.startswith("---"):
        return None, text
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None, text
    end = 1
    while end < len(lines):
        if lines[end].strip() == "---":
            break
        end += 1
    if end >= len(lines):
        return None, text
    inner = "\n".join(lines[1:end])
    body = "\n".join(lines[end + 1 :])
    return inner, body.lstrip("\n")


def slug_from_root_name(stem: str) -> str:
    return stem.replace("_", "-").lower()


def rewrite_hexo_img_paths(body: str) -> str:
    """Repo-root markdown uses ./imgs/ or imgs/; Hexo needs /imgs/ (see hexo-site README)."""
    body = body.replace("](./imgs/", "](/imgs/")
    body = body.replace("](imgs/", "](/imgs/")
    return body


def find_post_for_slug(d: Path, slug: str) -> Path | None:
    matches = sorted(d.glob(f"*-{slug}.md"))
    if not matches:
        return None
    if len(matches) > 1:
        print(
            f"Warning: multiple posts match slug {slug!r}: {[m.name for m in matches]}; using {matches[0].name}",
            file=sys.stderr,
        )
    return matches[0]


def title_from_body(body: str) -> str:
    for line in body.splitlines():
        line = line.strip()
        m = re.match(r"^#\s+(.+)$", line)
        if m:
            return m.group(1).strip()
    return ""


def default_front_matter(title: str, slug: str, today: date) -> str:
    safe_title = title or slug.replace("-", " ").title()
    return "\n".join(
        [
            "---",
            f"title: {safe_title}",
            f"date: {today.isoformat()}",
            "tags: []",
            "---",
            "",
        ]
    )


def sync_one(
    root_md: Path,
    posts: Path,
    *,
    dry_run: bool,
    today: date,
) -> str:
    stem = root_md.stem.lower()
    if stem in SKIP_STEMS:
        return "skip"

    slug = slug_from_root_name(root_md.stem)
    raw = root_md.read_text(encoding="utf-8")
    root_fm, root_body = split_front_matter(raw)
    if root_fm is not None:
        body = root_body
    else:
        body = raw
    body = rewrite_hexo_img_paths(body)

    existing = find_post_for_slug(posts, slug)
    if existing is not None:
        old = existing.read_text(encoding="utf-8")
        old_fm, _old_body = split_front_matter(old)
        if old_fm is None:
            print(f"Warning: {existing.name} has no front matter; rewriting with new file only", file=sys.stderr)
            merged = default_front_matter(title_from_body(body), slug, today) + body
        else:
            merged = "---\n" + old_fm + "\n---\n\n" + body.strip() + "\n"
        action = f"update {existing.name}"
    else:
        title = title_from_body(body)
        name = f"{today.isoformat()}-{slug}.md"
        merged = default_front_matter(title, slug, today) + body.strip() + "\n"
        existing = posts / name
        action = f"create {name}"

    if dry_run:
        print(f"[dry-run] would {action} <- {root_md.name}")
        return action.split()[0]

    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text(merged, encoding="utf-8", newline="\n")
    print(f"{action} <- {root_md.name}")
    return action.split()[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without writing files",
    )
    args = parser.parse_args()

    root = repo_root()
    posts = posts_dir(root)
    if not posts.is_dir():
        print(f"Error: Hexo _posts directory not found: {posts}", file=sys.stderr)
        return 1

    today = date.today()
    md_files = sorted(root.glob("*.md"))
    if not md_files:
        print("No *.md files in repository root.", file=sys.stderr)
        return 0

    updated = 0
    created = 0
    skipped = 0
    for p in md_files:
        r = sync_one(p, posts, dry_run=args.dry_run, today=today)
        if r == "update":
            updated += 1
        elif r == "create":
            created += 1
        elif r == "skip":
            skipped += 1

    if not args.dry_run:
        print(f"Done: {updated} updated, {created} created, {skipped} skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
