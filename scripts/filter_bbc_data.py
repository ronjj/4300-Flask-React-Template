"""Filter BBC data files by size.

Walks ``data/bbc_data/`` and deletes any file whose size is at least
``MAX_SIZE_BYTES`` (default 50 MB). The relative paths of the deleted
files are appended to the project's ``.gitignore`` (inside a managed
block) so that if the files are ever regenerated they will not be
committed.

Usage:
    python scripts/filter_bbc_data.py            # delete & update .gitignore
    python scripts/filter_bbc_data.py --dry-run  # preview only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "bbc_data"
GITIGNORE = REPO_ROOT / ".gitignore"

MAX_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB

BLOCK_START = "# >>> bbc_data large files (auto-managed by scripts/filter_bbc_data.py)"
BLOCK_END = "# <<< bbc_data large files"


def human(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def find_large_files(root: Path, threshold: int) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.stat().st_size >= threshold)


def update_gitignore(gitignore_path: Path, ignored_paths: list[str]) -> None:
    """Write (or replace) the managed block with the given paths."""
    existing = gitignore_path.read_text() if gitignore_path.exists() else ""
    lines = existing.splitlines()

    new_lines: list[str] = []
    skip = False
    for line in lines:
        if line.strip() == BLOCK_START:
            skip = True
            continue
        if line.strip() == BLOCK_END:
            skip = False
            continue
        if not skip:
            new_lines.append(line)

    while new_lines and new_lines[-1] == "":
        new_lines.pop()

    if ignored_paths:
        new_lines.append("")
        new_lines.append(BLOCK_START)
        new_lines.extend(sorted(set(ignored_paths)))
        new_lines.append(BLOCK_END)

    gitignore_path.write_text("\n".join(new_lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be deleted without touching disk.",
    )
    parser.add_argument(
        "--threshold-mb",
        type=float,
        default=50.0,
        help="Size threshold in MB (files >= this are removed). Default: 50.",
    )
    args = parser.parse_args()

    threshold = int(args.threshold_mb * 1024 * 1024)

    if not DATA_DIR.exists():
        print(f"Data directory not found: {DATA_DIR}")
        return 0

    large_files = find_large_files(DATA_DIR, threshold)

    if not large_files:
        print(f"No files >= {args.threshold_mb} MB found under {DATA_DIR.relative_to(REPO_ROOT)}.")
        update_gitignore(GITIGNORE, [])
        return 0

    total_bytes = sum(p.stat().st_size for p in large_files)
    action = "Would delete" if args.dry_run else "Deleting"
    print(f"{action} {len(large_files)} file(s) totaling {human(total_bytes)} "
          f"(>= {args.threshold_mb} MB each):")

    ignored_rel_paths: list[str] = []
    for path in large_files:
        rel = path.relative_to(REPO_ROOT).as_posix()
        size_str = human(path.stat().st_size)
        print(f"  {rel}  ({size_str})")
        ignored_rel_paths.append(rel)
        if not args.dry_run:
            path.unlink()

    if args.dry_run:
        print("\nDry run: no files were deleted and .gitignore was not modified.")
        return 0

    update_gitignore(GITIGNORE, ignored_rel_paths)
    print(f"\nUpdated {GITIGNORE.relative_to(REPO_ROOT)} with {len(ignored_rel_paths)} entry(ies).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
