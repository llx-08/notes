#!/usr/bin/env python3
"""
Sync markdown notes into hexo-site/source/_posts.

Sources:
  1) Repository root ``*.md`` (unchanged behaviour).
  2) ``ep_learning/*.md`` series → posts with slug prefix ``ep-learning-``
     (``ep_learning/README.md`` → slug ``ep-learning``).

Convention:
  - Root ``foo_bar.md`` → slug ``foo-bar``.
  - ``ep_learning/01_ep_fundamentals.md`` → slug ``ep-learning-01-ep-fundamentals``.
  - Existing post ``YYYY-MM-DD-<slug>.md`` is updated (front matter kept).
  - New post: ``today-<slug>.md`` with title from first ``#`` heading.
  - Image links ``imgs/`` ``./imgs/`` ``../imgs/`` → ``/imgs/`` (Hexo root).
  - ``ep_learning`` posts get ``categories: [EP 学习笔记]`` and shared tags.
  - In-series ``*.md`` / ``README.md`` links → Hexo permalinks.
  - Root note links ``../foo_bar.md`` → matching _posts permalink when found.

Hexo permalink note (this site):
  ``permalink: :year/:month/:day/:title/`` uses the *full* post filename
  stem as ``:title`` (date is not stripped), e.g.
  ``2026-07-24-ep-learning.md`` → ``/notes/2026/07/24/2026-07-24-ep-learning/``.
  Markdown absolute links are not auto-prefixed with ``root``, so rewritten
  links include the configured ``root`` (default ``/notes/``).

Run from anywhere; repo root is inferred from this script's location.
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path

SKIP_STEMS = frozenset({"readme", "changelog", "license", "contributing"})

# Match markdown links whose URL ends with .md (optional #anchor)
_MD_LINK_RE = re.compile(
    r"\]\((?:\./|\.\./)?([^)\s#]+?\.md)(#[^)\s]*)?\)"
)


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def posts_dir(root: Path) -> Path:
    return root / "hexo-site" / "source" / "_posts"


def split_front_matter(text: str) -> tuple[str | None, str]:
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


def slug_from_ep_learning(stem: str) -> str:
    if stem.lower() == "readme":
        return "ep-learning"
    return "ep-learning-" + stem.replace("_", "-").lower()


def rewrite_hexo_img_paths(body: str) -> str:
    """Repo markdown uses imgs/ ./imgs/ ../imgs/; Hexo needs /imgs/."""
    body = body.replace("](./imgs/", "](/imgs/")
    body = body.replace("](imgs/", "](/imgs/")
    body = body.replace("](../imgs/", "](/imgs/")
    return body


def find_post_for_slug(d: Path, slug: str) -> Path | None:
    matches = sorted(d.glob(f"*-{slug}.md"))
    if not matches:
        return None
    if len(matches) > 1:
        print(
            f"Warning: multiple posts match slug {slug!r}: "
            f"{[m.name for m in matches]}; using {matches[0].name}",
            file=sys.stderr,
        )
    return matches[0]


def date_and_slug_from_post_name(name: str) -> tuple[str, str, str, str] | None:
    """Parse ``YYYY-MM-DD-slug.md`` → (year, month, day, slug)."""
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})-(.+)\.md$", name)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3), m.group(4)


def hexo_root_prefix(root: Path) -> str:
    """Read ``root:`` from hexo-site/_config.yml (e.g. ``/notes/``)."""
    cfg = root / "hexo-site" / "_config.yml"
    if not cfg.is_file():
        return "/notes/"
    for line in cfg.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^root:\s*['\"]?([^'\"#\s]+)", line)
        if m:
            val = m.group(1).strip()
            if not val.startswith("/"):
                val = "/" + val
            if not val.endswith("/"):
                val += "/"
            return val
    return "/notes/"


def permalink_for_post_file(post_path: Path, *, site_root: str) -> str | None:
    """Build site permalink matching this Hexo deploy's actual URLs."""
    parsed = date_and_slug_from_post_name(post_path.name)
    if not parsed:
        return None
    y, mo, d, _slug = parsed
    # :title is the full filename stem (includes YYYY-MM-DD-), not slug alone.
    stem = post_path.stem
    return f"{site_root}{y}/{mo}/{d}/{stem}/"


def build_slug_permalink_map(posts: Path, *, site_root: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for p in posts.glob("*.md"):
        parsed = date_and_slug_from_post_name(p.name)
        if not parsed:
            continue
        _y, _mo, _d, slug = parsed
        pl = permalink_for_post_file(p, site_root=site_root)
        if pl:
            mapping[slug] = pl
    return mapping


def title_from_body(body: str) -> str:
    for line in body.splitlines():
        line = line.strip()
        m = re.match(r"^#\s+(.+)$", line)
        if m:
            return m.group(1).strip()
    return ""


def yaml_inline_list(values: list[str]) -> str:
    return "[" + ", ".join(values) + "]"


def upsert_front_matter_list(fm: str, key: str, values: list[str] | None) -> str:
    """Set ``key: [a, b]`` in YAML front matter; insert if missing."""
    if values is None:
        return fm
    line = f"{key}: {yaml_inline_list(values)}"
    pattern = re.compile(rf"^{re.escape(key)}:\s*.*$", re.MULTILINE)
    if pattern.search(fm):
        return pattern.sub(line, fm)
    return fm.rstrip() + "\n" + line


def default_front_matter(
    title: str,
    slug: str,
    today: date,
    *,
    tags: list[str] | None = None,
    categories: list[str] | None = None,
) -> str:
    safe_title = title or slug.replace("-", " ").title()
    lines = [
        "---",
        f"title: {safe_title}",
        f"date: {today.isoformat()}",
    ]
    if categories is not None:
        lines.append(f"categories: {yaml_inline_list(categories)}")
    lines.append(f"tags: {yaml_inline_list(tags or [])}")
    lines.extend(["---", ""])
    return "\n".join(lines)


def rewrite_md_links(
    body: str,
    *,
    link_basename_to_slug: dict[str, str],
    slug_to_permalink: dict[str, str],
    today: date,
    site_root: str,
) -> str:
    """Rewrite ](foo.md) / ](../foo.md) / ](README.md) to Hexo permalinks."""

    def repl(m: re.Match[str]) -> str:
        target = m.group(1)
        anchor = m.group(2) or ""
        base = Path(target).name
        slug = link_basename_to_slug.get(base) or link_basename_to_slug.get(
            base.lower()
        )
        if slug is None:
            return m.group(0)
        permalink = slug_to_permalink.get(slug)
        if permalink is None:
            y, mo, d = today.isoformat().split("-")
            stem = f"{y}-{mo}-{d}-{slug}"
            permalink = f"{site_root}{y}/{mo}/{d}/{stem}/"
        return f"]({permalink}{anchor})"

    return _MD_LINK_RE.sub(repl, body)


def sync_one(
    root_md: Path,
    posts: Path,
    *,
    dry_run: bool,
    today: date,
    slug: str,
    tags: list[str] | None,
    categories: list[str] | None,
    link_basename_to_slug: dict[str, str],
    slug_to_permalink: dict[str, str],
    site_root: str,
) -> str:
    stem = root_md.stem.lower()
    # Skip root README etc.; allow ep_learning/README.md
    if stem in SKIP_STEMS and root_md.parent.name != "ep_learning":
        return "skip"

    raw = root_md.read_text(encoding="utf-8")
    root_fm, root_body = split_front_matter(raw)
    body = root_body if root_fm is not None else raw
    body = rewrite_hexo_img_paths(body)
    body = rewrite_md_links(
        body,
        link_basename_to_slug=link_basename_to_slug,
        slug_to_permalink=slug_to_permalink,
        today=today,
        site_root=site_root,
    )

    existing = find_post_for_slug(posts, slug)
    if existing is not None:
        old = existing.read_text(encoding="utf-8")
        old_fm, _old_body = split_front_matter(old)
        if old_fm is None:
            print(
                f"Warning: {existing.name} has no front matter; rewriting",
                file=sys.stderr,
            )
            merged = (
                default_front_matter(
                    title_from_body(body),
                    slug,
                    today,
                    tags=tags,
                    categories=categories,
                )
                + body
            )
        else:
            fm = upsert_front_matter_list(old_fm, "categories", categories)
            fm = upsert_front_matter_list(fm, "tags", tags)
            merged = "---\n" + fm + "\n---\n\n" + body.strip() + "\n"
        action = f"update {existing.name}"
        pl = permalink_for_post_file(existing, site_root=site_root)
        if pl:
            slug_to_permalink[slug] = pl
    else:
        title = title_from_body(body)
        name = f"{today.isoformat()}-{slug}.md"
        merged = (
            default_front_matter(
                title, slug, today, tags=tags, categories=categories
            )
            + body.strip()
            + "\n"
        )
        existing = posts / name
        action = f"create {name}"
        pl = permalink_for_post_file(existing, site_root=site_root)
        if pl:
            slug_to_permalink[slug] = pl

    if dry_run:
        print(f"[dry-run] would {action} <- {root_md.relative_to(repo_root())}")
        return action.split()[0]

    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text(merged, encoding="utf-8", newline="\n")
    print(f"{action} <- {root_md.relative_to(repo_root())}")
    return action.split()[0]


def collect_link_map(root: Path) -> dict[str, str]:
    """basename.md -> slug for root notes and ep_learning notes."""
    mapping: dict[str, str] = {}
    for p in root.glob("*.md"):
        mapping[p.name] = slug_from_root_name(p.stem)
        mapping[p.name.lower()] = slug_from_root_name(p.stem)
    ep_dir = root / "ep_learning"
    if ep_dir.is_dir():
        for p in ep_dir.glob("*.md"):
            slug = slug_from_ep_learning(p.stem)
            mapping[p.name] = slug
            mapping[p.name.lower()] = slug
            if p.stem.lower() == "readme":
                mapping["README.md"] = slug
                mapping["readme.md"] = slug
    return mapping


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
    site_root = hexo_root_prefix(root)
    link_map = collect_link_map(root)
    slug_to_permalink = build_slug_permalink_map(posts, site_root=site_root)

    updated = created = skipped = 0

    for p in sorted(root.glob("*.md")):
        r = sync_one(
            p,
            posts,
            dry_run=args.dry_run,
            today=today,
            slug=slug_from_root_name(p.stem),
            tags=None,
            categories=None,
            link_basename_to_slug=link_map,
            slug_to_permalink=slug_to_permalink,
            site_root=site_root,
        )
        if r == "update":
            updated += 1
        elif r == "create":
            created += 1
        elif r == "skip":
            skipped += 1

    ep_dir = root / "ep_learning"
    ep_tags = ["EP", "MoE", "学习笔记"]
    ep_categories = ["EP 学习笔记"]
    if ep_dir.is_dir():
        # First pass creates all posts / refreshes permalink map
        for p in sorted(ep_dir.glob("*.md")):
            r = sync_one(
                p,
                posts,
                dry_run=args.dry_run,
                today=today,
                slug=slug_from_ep_learning(p.stem),
                tags=ep_tags,
                categories=ep_categories,
                link_basename_to_slug=link_map,
                slug_to_permalink=slug_to_permalink,
                site_root=site_root,
            )
            if r == "update":
                updated += 1
            elif r == "create":
                created += 1
            elif r == "skip":
                skipped += 1

        # Second pass: rewrite in-series links with final permalink map
        if not args.dry_run:
            for p in sorted(ep_dir.glob("*.md")):
                sync_one(
                    p,
                    posts,
                    dry_run=False,
                    today=today,
                    slug=slug_from_ep_learning(p.stem),
                    tags=ep_tags,
                    categories=ep_categories,
                    link_basename_to_slug=link_map,
                    slug_to_permalink=slug_to_permalink,
                    site_root=site_root,
                )

    if not args.dry_run:
        print(f"Done: {updated} updated, {created} created, {skipped} skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
