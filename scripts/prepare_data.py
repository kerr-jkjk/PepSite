#!/usr/bin/env python3
"""Validate and stage the CSV splits used by PepSite training.

The release intentionally does not ship the private/large training data.  Use
this command with externally obtained CSV files to produce a portable,
auditable staging directory.  Input rows are copied without reordering or
sequence mutation; only rows with ``binding_site_status != ok`` are filtered
by default, matching the historical training dataset classes.  Malformed
rows fail fast unless ``--drop-invalid`` is explicitly requested.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pick_column(fieldnames: list[str], *choices: str) -> str | None:
    for choice in choices:
        if choice in fieldnames:
            return choice
    return None


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        fieldnames = [str(item) for item in reader.fieldnames]
        rows = [{key: (value if value is not None else "") for key, value in row.items()} for row in reader]
    return fieldnames, rows


def validate_split(
    path: Path,
    *,
    allowed_alphabet: set[str],
    max_length: int,
    drop_non_ok: bool,
    drop_invalid: bool,
) -> tuple[list[str], list[dict[str, str]], dict[str, Any]]:
    fieldnames, rows = _read_rows(path)
    receptor_col = _pick_column(fieldnames, "protein_sequence", "Receptor Sequence")
    ligand_col = _pick_column(fieldnames, "ligand_sequence", "Binder")
    mask_col = _pick_column(fieldnames, "receptor_binding_site_mask")
    if receptor_col is None or ligand_col is None or mask_col is None:
        raise ValueError(
            f"{path} must contain receptor, ligand and receptor_binding_site_mask columns; "
            f"found {fieldnames}"
        )

    query_col = _pick_column(fieldnames, "query_id")
    kept: list[dict[str, str]] = []
    invalid: list[dict[str, Any]] = []
    filtered_non_ok = 0
    overlength = 0
    seen_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        if drop_non_ok and "binding_site_status" in row and row["binding_site_status"] != "ok":
            filtered_non_ok += 1
            continue

        receptor = row.get(receptor_col, "")
        ligand = row.get(ligand_col, "")
        mask = row.get(mask_col, "").strip()
        query_id = (row.get(query_col, "") if query_col else str(row_number - 2)).strip()
        problems: list[str] = []
        if not receptor:
            problems.append("empty receptor")
        if not ligand:
            problems.append("empty ligand")
        if not mask:
            problems.append("empty site mask")
        if len(mask) != len(receptor):
            problems.append("site mask length differs from receptor")
        if mask and any(char not in "01" for char in mask):
            problems.append("site mask is not binary")
        bad_receptor = sorted(set(receptor) - allowed_alphabet)
        bad_ligand = sorted(set(ligand) - allowed_alphabet)
        if bad_receptor:
            problems.append("receptor contains disallowed residues: " + "".join(bad_receptor))
        if bad_ligand:
            problems.append("ligand contains disallowed residues: " + "".join(bad_ligand))
        if query_id in seen_ids:
            problems.append("duplicate query_id")
        if len(receptor) + len(ligand) + 2 > int(max_length):
            overlength += 1
        if problems:
            invalid.append({"row": row_number, "query_id": query_id, "problems": problems})
            if not drop_invalid:
                continue
            continue
        seen_ids.add(query_id)
        kept.append(row)

    if invalid and not drop_invalid:
        details = "; ".join(f"row {item['row']}: {', '.join(item['problems'])}" for item in invalid[:5])
        suffix = "" if len(invalid) <= 5 else f"; ... and {len(invalid) - 5} more"
        raise ValueError(
            f"{path} has {len(invalid)} invalid row(s). {details}{suffix}. "
            "Fix the source or rerun with --drop-invalid (recorded in manifest)."
        )

    report = {
        "source": str(path),
        "source_sha256": sha256_file(path),
        "source_rows": len(rows),
        "kept_rows": len(kept),
        "filtered_non_ok_rows": filtered_non_ok,
        "invalid_rows": len(invalid),
        "overlength_rows_for_max_length": overlength,
        "receptor_column": receptor_col,
        "ligand_column": ligand_col,
        "site_mask_column": mask_col,
        "query_id_column": query_col,
        "invalid_examples": invalid[:20],
    }
    return fieldnames, kept, report


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", "--train_csv", required=True)
    parser.add_argument("--val-csv", "--val_csv", required=True)
    parser.add_argument("--test-csv", "--test_csv", default=None)
    parser.add_argument("--output-dir", "--output_dir", default="data/prepared")
    parser.add_argument("--max-length", "--max_length", type=int, default=552)
    parser.add_argument("--allowed-alphabet", default=DEFAULT_ALPHABET)
    parser.add_argument("--keep-non-ok", action="store_true", help="Do not apply historical binding_site_status == ok filter.")
    parser.add_argument("--drop-invalid", action="store_true", help="Drop malformed rows instead of failing fast.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_length <= 2:
        raise SystemExit("--max-length must be greater than 2")
    if not args.allowed_alphabet or len(set(args.allowed_alphabet)) != len(args.allowed_alphabet):
        raise SystemExit("--allowed-alphabet must be a non-empty string of unique residue symbols")
    allowed = set(args.allowed_alphabet)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_args = [("train", args.train_csv), ("val", args.val_csv)]
    if args.test_csv:
        split_args.append(("test", args.test_csv))
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "PepSite strict two-stage data preparation",
        "allowed_alphabet": args.allowed_alphabet,
        "max_length": args.max_length,
        "drop_non_ok": not args.keep_non_ok,
        "drop_invalid": bool(args.drop_invalid),
        "splits": {},
    }
    for name, raw_path in split_args:
        source = resolve_path(raw_path)
        if not source.is_file():
            raise SystemExit(f"Input CSV not found: {source}")
        fieldnames, rows, report = validate_split(
            source,
            allowed_alphabet=allowed,
            max_length=args.max_length,
            drop_non_ok=not args.keep_non_ok,
            drop_invalid=args.drop_invalid,
        )
        destination = output_dir / f"{name}.csv"
        write_csv(destination, fieldnames, rows)
        report.update({"output": str(destination), "output_sha256": sha256_file(destination)})
        manifest["splits"][name] = report

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Prepared data written to {output_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
