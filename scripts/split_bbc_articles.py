"""Split ``data/bbc_data/bbc_articles.csv`` into chunks below a size limit.

The source CSV has a column (``article_text``) whose values frequently contain
embedded newlines, so naive line-based splitting (``split -l``) would corrupt
records. This script streams the file through Python's ``csv`` module so parts
are always cut on record boundaries. Each output part gets a copy of the
header row.

Output files are written to ``data/bbc_data/bbc_articles_parts/`` and are named
``bbc_articles_part_XXX.csv`` (1-indexed, zero-padded to 3 digits).

Usage:
    python scripts/split_bbc_articles.py
    python scripts/split_bbc_articles.py --max-size-mb 45
    python scripts/split_bbc_articles.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "data" / "bbc_data" / "bbc_articles.csv"
OUTPUT_DIR = REPO_ROOT / "data" / "bbc_data" / "bbc_articles_parts"

# article_text can be huge; bump the CSV field size limit to the largest value
# the platform allows so csv.reader doesn't raise on long fields.
_MAX_INT = sys.maxsize
while True:
    try:
        csv.field_size_limit(_MAX_INT)
        break
    except OverflowError:
        _MAX_INT = int(_MAX_INT // 10)


def human(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def open_part(index: int, header: list[str]):
    """Create part file, write header, return (path, file_handle, writer)."""
    path = OUTPUT_DIR / f"bbc_articles_part_{index:03d}.csv"
    fh = path.open("w", encoding="utf-8", newline="")
    writer = csv.writer(fh)
    writer.writerow(header)
    fh.flush()
    return path, fh, writer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-size-mb",
        type=float,
        default=45.0,
        help=(
            "Target maximum size per part in MB. Defaults to 45 to leave "
            "headroom under a 50 MB ceiling."
        ),
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=SOURCE,
        help=f"Path to the source CSV (default: {SOURCE.relative_to(REPO_ROOT)}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=(
            "Directory to write part files into "
            f"(default: {OUTPUT_DIR.relative_to(REPO_ROOT)})."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stream through the source without writing parts (reports stats only).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing bbc_articles_part_*.csv files in the output dir first.",
    )
    args = parser.parse_args()

    source: Path = args.source
    out_dir: Path = args.output_dir
    max_bytes = int(args.max_size_mb * 1024 * 1024)

    if not source.exists():
        print(f"Source CSV not found: {source}", file=sys.stderr)
        return 1

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.overwrite:
            removed = 0
            for p in out_dir.glob("bbc_articles_part_*.csv"):
                p.unlink()
                removed += 1
            if removed:
                print(f"Removed {removed} existing part file(s) from {out_dir.relative_to(REPO_ROOT)}.")

    total_in_bytes = source.stat().st_size
    print(f"Source: {source.relative_to(REPO_ROOT)} ({human(total_in_bytes)})")
    print(f"Target max size per part: {args.max_size_mb} MB")
    if args.dry_run:
        print("Dry run: counting records without writing parts.\n")
    else:
        print(f"Writing parts to: {out_dir.relative_to(REPO_ROOT)}\n")

    part_index = 0
    part_path: Path | None = None
    part_fh = None
    part_writer = None

    total_records = 0
    current_part_records = 0
    records_per_part: list[int] = []

    with source.open("r", encoding="utf-8", newline="") as src:
        reader = csv.reader(src)
        try:
            header = next(reader)
        except StopIteration:
            print("Source CSV is empty.", file=sys.stderr)
            return 1

        if not args.dry_run:
            part_index = 1
            part_path, part_fh, part_writer = open_part(part_index, header)

        for row in reader:
            total_records += 1
            current_part_records += 1

            if args.dry_run:
                continue

            assert part_fh is not None and part_writer is not None and part_path is not None
            part_writer.writerow(row)

            # Roll over when the current part hits the size limit. Checked
            # after writing so every row lands somewhere; next record starts
            # a fresh part.
            if part_fh.tell() >= max_bytes:
                part_fh.close()
                size = part_path.stat().st_size
                records_per_part.append(current_part_records)
                print(
                    f"  {part_path.name}: {current_part_records:>8,} records, "
                    f"{human(size)}"
                )
                current_part_records = 0
                part_index += 1
                part_path, part_fh, part_writer = open_part(part_index, header)

    if not args.dry_run and part_fh is not None and part_path is not None:
        part_fh.close()
        if current_part_records == 0:
            # Nothing was written beyond the header; remove the empty trailing part.
            part_path.unlink(missing_ok=True)
            part_index -= 1
        else:
            size = part_path.stat().st_size
            records_per_part.append(current_part_records)
            print(
                f"  {part_path.name}: {current_part_records:>8,} records, "
                f"{human(size)}"
            )

    print()
    print(f"Total records (excluding header): {total_records:,}")
    if args.dry_run:
        return 0

    print(f"Wrote {part_index} part file(s).")

    # Sanity check: flag any parts that ended up >= 50 MB.
    ceiling = 50 * 1024 * 1024
    offenders = [
        p for p in sorted(out_dir.glob("bbc_articles_part_*.csv"))
        if p.stat().st_size >= ceiling
    ]
    if offenders:
        print(
            f"\nWarning: {len(offenders)} part file(s) are >= 50 MB. "
            "Consider lowering --max-size-mb:"
        )
        for p in offenders:
            print(f"  {p.name}: {human(p.stat().st_size)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
