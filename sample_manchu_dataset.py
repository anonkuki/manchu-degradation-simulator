"""Create a small example dataset by sampling IDs from an existing manchu_dataset.

Default behavior:
- Pick N IDs from `annotations/` (JSON files like 000001.json)
- Copy the corresponding files across sibling folders (clean_text, degraded_*, ground_truth, background_patch, annotations)
- Write a filtered `id_mapping.json` that only contains the sampled IDs

This is intended to create a structure-identical, smaller dataset folder next to the original.

Windows-friendly, pure-stdlib.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class DatasetPaths:
    src_root: Path
    dst_root: Path


def _iter_annotation_ids(annotations_dir: Path) -> list[str]:
    if not annotations_dir.exists() or not annotations_dir.is_dir():
        raise FileNotFoundError(f"annotations dir not found: {annotations_dir}")

    ids: list[str] = []
    for p in annotations_dir.iterdir():
        if p.is_file() and p.suffix.lower() == ".json":
            stem = p.stem
            if stem.isdigit():
                ids.append(stem)
    ids.sort()
    return ids


def _select_ids(all_ids: list[str], num: int, seed: int | None) -> list[str]:
    if num <= 0:
        raise ValueError("--num must be > 0")

    if len(all_ids) <= num:
        return all_ids

    rng = random.Random(seed)
    return sorted(rng.sample(all_ids, k=num))


def _expected_filename(folder_name: str, sample_id: str) -> str | None:
    # Known folder naming conventions in this repo's dataset_output/manchu_dataset
    if folder_name == "annotations":
        return f"{sample_id}.json"
    if folder_name == "clean_text":
        return f"clean_{sample_id}.png"
    if folder_name == "ground_truth":
        return f"gt_{sample_id}.png"
    if folder_name == "background_patch":
        return f"bg_{sample_id}.png"

    if folder_name.startswith("degraded_"):
        kind = folder_name.removeprefix("degraded_")
        # Common: degraded_ink -> ink_000001.png
        return f"{kind}_{sample_id}.png"

    return None


def _safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_sampled_files(
    paths: DatasetPaths,
    sampled_ids: Iterable[str],
    *,
    verbose: bool,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    src_root = paths.src_root
    dst_root = paths.dst_root

    for entry in src_root.iterdir():
        if entry.is_dir():
            folder_name = entry.name
            dst_dir = dst_root / folder_name
            dst_dir.mkdir(parents=True, exist_ok=True)

            copied = 0
            for sample_id in sampled_ids:
                expected = _expected_filename(folder_name, sample_id)
                if expected is None:
                    # Unknown folder: keep it empty (structure-only) to avoid accidental mismatch.
                    continue
                src_file = entry / expected
                if not src_file.exists():
                    # Some folders may be empty (e.g., degraded_damage) or missing files.
                    continue
                _safe_copy(src_file, dst_dir / expected)
                copied += 1

            counts[folder_name] = copied
            if verbose:
                print(f"[{folder_name}] copied {copied} files")

        elif entry.is_file():
            # Root-level files: we handle id_mapping.json specially; others are copied as-is.
            if entry.name == "id_mapping.json":
                continue
            _safe_copy(entry, dst_root / entry.name)
            counts.setdefault("__root_files__", 0)
            counts["__root_files__"] += 1

    return counts


def _filter_id_mapping(src_mapping_path: Path, sampled_ids: set[str]) -> dict:
    if not src_mapping_path.exists():
        return {}

    with src_mapping_path.open("r", encoding="utf-8") as f:
        mapping = json.load(f)

    if not isinstance(mapping, dict):
        raise ValueError("id_mapping.json is not a JSON object")

    return {k: v for k, v in mapping.items() if k in sampled_ids}


def _write_json(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _write_lines(lines: Iterable[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(f"{line}\n")


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_src = script_dir / "dataset_output" / "manchu_dataset"
    default_dst = script_dir / "dataset_output" / "manchu_dataset1"

    parser = argparse.ArgumentParser(
        description="Sample N IDs from a manchu_dataset and create a smaller, structure-identical dataset.",
    )
    parser.add_argument("--src", type=Path, default=default_src, help="source dataset root")
    parser.add_argument("--dst", type=Path, default=default_dst, help="destination dataset root")
    parser.add_argument("--num", type=int, default=20, help="number of samples (IDs) to take")
    parser.add_argument("--seed", type=int, default=42, help="random seed for sampling")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="delete destination folder if it already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only print what would be done, do not copy",
    )
    parser.add_argument("--verbose", action="store_true", help="print per-folder copy stats")

    args = parser.parse_args()

    src_root: Path = args.src
    dst_root: Path = args.dst

    if not src_root.exists() or not src_root.is_dir():
        raise FileNotFoundError(f"source dataset root not found: {src_root}")

    if dst_root.exists() and args.overwrite:
        shutil.rmtree(dst_root)

    annotations_dir = src_root / "annotations"
    all_ids = _iter_annotation_ids(annotations_dir)
    sampled_ids = _select_ids(all_ids, args.num, args.seed)

    if args.dry_run:
        print(f"SRC: {src_root}")
        print(f"DST: {dst_root}")
        print(f"Sampled {len(sampled_ids)} IDs (seed={args.seed}):")
        print(", ".join(sampled_ids))
        return 0

    dst_root.mkdir(parents=True, exist_ok=True)

    paths = DatasetPaths(src_root=src_root, dst_root=dst_root)
    counts = _copy_sampled_files(paths, sampled_ids, verbose=args.verbose)

    # Write sampled IDs list
    _write_lines(sampled_ids, dst_root / "sampled_ids.txt")

    # Filter and write id_mapping.json
    filtered_mapping = _filter_id_mapping(src_root / "id_mapping.json", set(sampled_ids))
    if filtered_mapping:
        _write_json(filtered_mapping, dst_root / "id_mapping.json")

    print("Done.")
    print(f"Sampled IDs: {len(sampled_ids)}")
    for k in sorted(counts.keys()):
        print(f"  {k}: {counts[k]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
