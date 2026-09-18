#!/usr/bin/env python3
"""Build the four PepSite sequence-route tables from score data.

This command deliberately does not load a checkpoint or invent model output.
The input CSV is the boundary between model inference and deterministic route
logic and must contain source mean log-probability, Site-chemistry, and source
masked-residue PPL for every candidate.  A real model backend can produce that
CSV using its own licensed checkpoint; this script records the supplied score
artifact and refuses rows that are missing required fields.

Usage::

    python scripts/generate_routes.py \
      --candidate-csv data/candidate_scores.csv --output-dir runs/pepsite/inference/routes \
      --xr-proposals data/xr_topk.csv \
      --xr-selected data/xr_selected.csv

The XR proposal table produced by ``scripts/infer_candidates.py`` has
``query_id``, ``route_id`` (the parent route), ``parent_sequence``,
``position`` (zero-based), and ``residue`` columns.  Rows may include
``rank``; lower ranks are tried first.  Parent-aware rows are required when
the baseline and two-score parents differ.  The old four-column schema
(``query_id,position,residue,rank``) is accepted only when the two parents are
identical; otherwise the command fails closed instead of crossing contexts.
For a parent containing X, omitting this table is an explicit error, because
silently copying an X sequence would not be an XR run.

When ``--xr-selected`` is provided, its gate-approved model-backed choices are
used in preference to rank-0 proposal assembly.  The selected artifact is
validated for exact parent linkage, canonical output, and X-only mutations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Mapping

# Keep the command runnable directly from a clean checkout without requiring a
# prior editable install.  No machine-local experiment path is imported.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pepsite.constants import CANONICAL_SET, GENERATION_ALPHABET_SET
from pepsite.routes import Candidate, Selection, select_two_score, xr_repair_once


ROUTE_FILES = {
    "baseline": "01_mlm_site.csv",
    "two_score": "02_two_score.csv",
    "xr_once": "03_mlm_site_xr1.csv",
    "two_score_xr_once": "04_two_score_xr1.csv",
}

ROUTE_IDS = {
    "baseline": "01_mlm_site",
    "two_score": "02_two_score",
    "xr_once": "03_mlm_site_xr1",
    "two_score_xr_once": "04_two_score_xr1",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _pick(row: Mapping[str, str], *names: str, default: str = "") -> str:
    for name in names:
        if name in row:
            return str(row[name])
    return default


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"input score table is empty: {path}")
    return rows


@dataclass(frozen=True)
class XRProposalTable:
    """Route-aware XR proposal lookup with a legacy compatibility view."""

    # The strict key is (query_id, parent_route_id, parent_sequence, position).
    # parent_route_id is 01_mlm_site or 02_two_score; the corresponding output
    # route is 03_mlm_site_xr1 or 04_two_score_xr1.
    strict: dict[tuple[str, str, str, int], list[str]]
    # Old files had only query_id, position, residue, rank.  They are usable
    # only when both XR parents are identical (or when no route-specific
    # distinction is needed).
    legacy: dict[tuple[str, int], list[str]]
    has_strict_rows: bool


@dataclass(frozen=True)
class XRSelectedTable:
    """Strict lookup of model-backed, gate-approved XR outputs.

    Keys include the exact parent sequence because one query has two XR arms
    and the two-score arm may have a different X context from the baseline.
    """

    strict: dict[tuple[str, str, str], dict[str, str]]


def _canonical_parent_route(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"01_mlm_site", "mlm_site", "baseline", "03_mlm_site_xr1"}:
        return "01_mlm_site"
    if normalized in {"02_two_score", "two_score", "04_two_score_xr1"}:
        return "02_two_score"
    raise ValueError(
        "XR route_id must identify the parent route (01_mlm_site/02_two_score) "
        f"or its XR alias, got {value!r}"
    )


def _read_xr_proposals(path: Path | None) -> XRProposalTable:
    if path is None:
        return XRProposalTable({}, {}, False)
    rows = _read_csv(path)
    required = {"query_id", "position"}
    if not required <= set(rows[0]):
        raise ValueError(f"XR proposal table requires {sorted(required)}")
    columns = set(rows[0])
    has_route_column = "route_id" in columns
    has_parent_column = "parent_sequence" in columns
    if has_route_column != has_parent_column:
        raise ValueError(
            "route-aware XR proposals must include both route_id and parent_sequence; "
            "provide neither for the legacy schema"
        )
    strict_result: dict[tuple[str, str, str, int], list[tuple[int, int, str]]] = {}
    legacy_result: dict[tuple[str, int], list[tuple[int, int, str]]] = {}
    for row_order, row in enumerate(rows):
        query_id = _pick(row, "query_id").strip()
        if not query_id:
            raise ValueError(f"XR query_id must not be empty: {row!r}")
        try:
            position = int(_pick(row, "position"))
        except ValueError as exc:
            raise ValueError(f"XR position must be an integer: {row!r}") from exc
        if position < 0:
            raise ValueError(f"XR position must be non-negative: {row!r}")
        residue = _pick(row, "residue", "aa", "amino_acid").strip().upper()
        if len(residue) != 1:
            raise ValueError(f"XR residue must be one character: {row!r}")
        try:
            rank = int(_pick(row, "rank", default="0"))
        except ValueError as exc:
            raise ValueError(f"XR rank must be an integer: {row!r}") from exc
        if has_route_column:
            route_id = _canonical_parent_route(_pick(row, "route_id"))
            parent = _pick(row, "parent_sequence").strip().upper()
            if not parent:
                raise ValueError(f"route-aware XR parent_sequence must not be empty: {row!r}")
            if set(parent) - GENERATION_ALPHABET_SET:
                raise ValueError(f"XR parent contains symbols outside the generation alphabet: {row!r}")
            if position >= len(parent) or parent[position] != "X":
                raise ValueError(
                    f"route-aware XR position must point to X in parent_sequence: {row!r}"
                )
            strict_result.setdefault((query_id, route_id, parent, position), []).append((rank, row_order, residue))
        else:
            legacy_result.setdefault((query_id, position), []).append((rank, row_order, residue))

    def finish(
        source: dict[tuple[Any, ...], list[tuple[int, int, str]]],
        label: str,
    ) -> dict[tuple[Any, ...], list[str]]:
        finished: dict[tuple[Any, ...], list[str]] = {}
        for key, values in source.items():
            # Equal ranks with different residues have no deterministic order
            # in a legacy CSV and therefore cannot be safely interpreted.
            by_rank: dict[int, set[str]] = {}
            for rank, _order, residue in values:
                by_rank.setdefault(int(rank), set()).add(residue)
            ambiguous = {rank: sorted(residues) for rank, residues in by_rank.items() if len(residues) > 1}
            if ambiguous:
                raise ValueError(f"ambiguous {label} XR ranks for {key}: {ambiguous}")
            ordered = sorted(values, key=lambda value: (int(value[0]), int(value[1]), str(value[2])))
            # Repeated identical residues are harmless but do not add signal to
            # top-k lookup; collapse them while preserving rank order.
            residues: list[str] = []
            for _rank, _order, residue in ordered:
                if residue not in residues:
                    residues.append(residue)
            finished[key] = residues
        return finished

    return XRProposalTable(
        strict=finish(strict_result, "route-aware"),
        legacy=finish(legacy_result, "legacy"),
        has_strict_rows=has_route_column,
    )


def _read_xr_selected(path: Path | None) -> XRSelectedTable:
    """Read final model-backed XR choices and validate the X-only contract."""

    if path is None:
        return XRSelectedTable({})
    rows = _read_csv(path)
    required = {"query_id", "route_id", "parent_sequence", "selected_sequence"}
    if not required <= set(rows[0]):
        raise ValueError(f"XR selected table requires {sorted(required)}")
    result: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        query_id = _pick(row, "query_id").strip()
        if not query_id:
            raise ValueError(f"XR selected query_id must not be empty: {row!r}")
        parent_route = _canonical_parent_route(_pick(row, "route_id"))
        parent = _pick(row, "parent_sequence").strip().upper()
        selected = _pick(row, "selected_sequence").strip().upper()
        if not parent:
            raise ValueError(f"XR selected parent_sequence must not be empty: {row!r}")
        if not selected:
            status = _pick(row, "xr_status", "status", default="failed").strip().lower()
            raise ValueError(f"XR selected row has no final sequence (status={status!r}): {row!r}")
        if set(parent) - GENERATION_ALPHABET_SET:
            raise ValueError(f"XR selected parent contains unsupported symbols: {row!r}")
        if set(selected) - CANONICAL_SET:
            raise ValueError(f"XR selected sequence must be canonical-20: {row!r}")
        if len(selected) != len(parent):
            raise ValueError(f"XR selected sequence length differs from parent: {row!r}")
        immutable = [index for index, residue in enumerate(parent) if residue != "X"]
        if any(selected[index] != parent[index] for index in immutable):
            raise ValueError(f"XR selected changed a non-X parent position: {row!r}")
        if "X" not in parent and selected != parent:
            raise ValueError(f"XR selected changed a parent with no X positions: {row!r}")
        status = _pick(row, "xr_status", "status", default="complete").strip().lower()
        if status in {"failed", "error", "incomplete", "pending"}:
            raise ValueError(f"XR selected row is not gate-approved (status={status!r}): {row!r}")
        if "xr_ppl_safe" in row and _pick(row, "xr_ppl_safe").strip().lower() in {"0", "false", "no", "n", "f"}:
            raise ValueError(f"XR selected row is marked PPL-unsafe: {row!r}")
        key = (query_id, parent_route, parent)
        if key in result:
            raise ValueError(f"duplicate XR selected key: {key}")
        parsed = dict(row)
        parsed["route_id"] = parent_route
        parsed["parent_sequence"] = parent
        parsed["selected_sequence"] = selected
        parsed["xr_status"] = status
        result[key] = parsed
    return XRSelectedTable(result)


def _selection_dict(selection: Selection) -> dict[str, Any]:
    return {
        "query_id": selection.query_id,
        "selected_sequence": selection.selected_sequence,
        "candidate_sequence": selection.candidate_sequence,
        "baseline_sequence": selection.baseline_sequence,
        "candidate_index": selection.candidate_index,
        "two_score_utility": selection.two_score_utility,
        "z_site_chemistry": selection.z_site_chemistry,
        "z_source_mean_logp": selection.z_source_mean_logp,
        "candidate_ppl": selection.candidate_ppl,
        "baseline_ppl": selection.baseline_ppl,
        "candidate_ppl_ratio": selection.candidate_ppl_ratio,
        "selected_ppl_ratio": selection.selected_ppl_ratio,
        "ppl_safe": selection.ppl_safe,
        "fallback": selection.fallback,
        "fallback_reason": selection.fallback_reason,
        "changed": selection.changed,
    }


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty route table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row | {field: "" for field in fields if field not in row} for row in rows)


def generate(
    candidate_csv: Path,
    output_dir: Path,
    *,
    xr_proposals: Path | None = None,
    xr_selected: Path | None = None,
    ppl_ratio_max: float = 1.05,
    composition_l1_max: float | None = 0.75,
) -> dict[str, Any]:
    rows = _read_csv(candidate_csv)
    required = {"query_id", "sequence", "source_mean_logp", "site_chemistry", "source_ppl"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"candidate score table missing columns: {sorted(missing)}")
    proposals = _read_xr_proposals(xr_proposals)
    selected_artifact = _read_xr_selected(xr_selected)
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        query_id = _pick(row, "query_id").strip()
        if not query_id:
            raise ValueError("query_id must not be empty")
        grouped.setdefault(query_id, []).append(row)
    if not grouped:
        raise ValueError("candidate score table has no queries")

    baseline_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    xr_baseline_rows: list[dict[str, Any]] = []
    xr_selected_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    used_selected_keys: set[tuple[str, str, str]] = set()
    for query_id, query_rows in grouped.items():
        baseline_values = {_pick(row, "baseline_sequence").strip().upper() for row in query_rows if _pick(row, "baseline_sequence").strip()}
        marked = [_pick(row, "sequence", "peptide", "generated_ligand_sequence").strip().upper() for row in query_rows if _pick(row, "is_baseline", "baseline").strip().lower() in {"1", "true", "yes", "y", "t"}]
        if len(baseline_values) > 1 or len(marked) > 1:
            raise ValueError(f"multiple baseline sequences for {query_id}")
        baseline = next(iter(baseline_values), marked[0] if marked else "")
        if not baseline:
            raise ValueError(f"baseline_sequence or is_baseline is required for {query_id}")
        candidates = [Candidate.from_mapping(row) for row in query_rows]
        selection = select_two_score(candidates, baseline, query_id=query_id, ppl_ratio_max=ppl_ratio_max, composition_l1_max=composition_l1_max)
        receptor = _pick(query_rows[0], "receptor_sequence", "Receptor Sequence", "receptor").strip().upper()
        if not receptor:
            raise ValueError(f"receptor_sequence is required for {query_id}")
        base_row = {"query_id": query_id, "receptor_sequence": receptor, "selected_sequence": baseline, "parent_sequence": baseline, "route": ROUTE_IDS["baseline"]}
        baseline_rows.append(base_row)
        selected_rows.append({"query_id": query_id, "receptor_sequence": receptor, "parent_sequence": selection.baseline_sequence, "route": ROUTE_IDS["two_score"], **_selection_dict(selection)})
        score_rows.extend(dict(row) | {"query_id": query_id} for row in selection.scored)

        # New model-backed inference writes parent-aware proposals.  The
        # parent route is mapped to the corresponding XR output route here;
        # never reuse a baseline proposal for a changed two-score parent.
        parent_route_for_output = {
            ROUTE_IDS["xr_once"]: ROUTE_IDS["baseline"],
            ROUTE_IDS["two_score_xr_once"]: ROUTE_IDS["two_score"],
        }

        if proposals.legacy and baseline != selection.selected_sequence and ("X" in baseline or "X" in selection.selected_sequence):
            raise ValueError(
                f"legacy XR proposals are ambiguous for {query_id}: baseline and "
                "two-score parents differ; provide route_id+parent_sequence"
            )

        def proposal_values(parent: str, output_route: str, position: int) -> list[str]:
            parent_route = parent_route_for_output[output_route]
            strict_key = (query_id, parent_route, parent, int(position))
            values = proposals.strict.get(strict_key)
            if values is not None:
                return values
            # Route aliases are normalized by _read_xr_proposals, so this
            # fallback is intentionally limited to the legacy schema.
            values = proposals.legacy.get((query_id, int(position)))
            if values is not None:
                return values
            return []

        def repair(parent: str, route: str) -> dict[str, Any]:
            parent_route = parent_route_for_output[route]
            selected_key = (query_id, parent_route, parent)
            selected_record = selected_artifact.strict.get(selected_key)
            if selected_record is not None:
                used_selected_keys.add(selected_key)
            # A model-backed final artifact takes precedence over proposal
            # assembly.  For an X-bearing parent, silently falling back to a
            # rank-0 proposal would discard its PPL/composition decision.
            if xr_selected is not None and "X" in parent:
                if selected_record is None:
                    raise ValueError(
                        f"XR selected artifact missing {query_id} route={parent_route} parent={parent!r}"
                    )
                repaired = selected_record["selected_sequence"]
                trace_value = _pick(selected_record, "xr_trace", "trace", default="[]")
                try:
                    trace = json.loads(trace_value) if trace_value else []
                except json.JSONDecodeError:
                    trace = trace_value
                result = {
                    "query_id": query_id,
                    "receptor_sequence": receptor,
                    "parent_sequence": parent,
                    "selected_sequence": repaired,
                    "route": route,
                    "xr_changed": repaired != parent,
                    "xr_mutable_positions": int(_pick(selected_record, "xr_mutable_positions", default=str(parent.count("X"))) or parent.count("X")),
                    "xr_repair_hamming": _pick(selected_record, "xr_repair_hamming", default=str(sum(a != b for a, b in zip(parent, repaired)))),
                    "xr_status": _pick(selected_record, "xr_status", "status", default="complete"),
                    "xr_selected_source": _pick(selected_record, "selected_source", default=""),
                    "xr_ppl_safe": _pick(selected_record, "xr_ppl_safe", default="true"),
                    "xr_ppl_ratio": _pick(selected_record, "xr_ppl_ratio", default=""),
                    "xr_composition_l1": _pick(selected_record, "xr_composition_l1", default=""),
                    "xr_trace": json.dumps(trace, ensure_ascii=False) if not isinstance(trace, str) else trace,
                }
                return result
            if "X" not in parent:
                repaired = parent
                trace: list[Mapping[str, Any]] = []
            else:
                missing_positions = [
                    index
                    for index, residue in enumerate(parent)
                    if residue == "X" and not proposal_values(parent, route, index)
                ]
                if missing_positions:
                    raise ValueError(f"XR proposals missing {query_id} positions {missing_positions}")
                def proposer(context: str, position: int, top_k: int):
                    del context
                    values = proposal_values(parent, route, position)
                    return values[:top_k]
                result = xr_repair_once(parent, proposer)
                repaired, trace = result.selected_sequence, result.trace
            return {
                "query_id": query_id,
                "receptor_sequence": receptor,
                "parent_sequence": parent,
                "selected_sequence": repaired,
                "route": route,
                "xr_changed": repaired != parent,
                "xr_mutable_positions": parent.count("X"),
                "xr_repair_hamming": sum(a != b for a, b in zip(parent, repaired)),
                "xr_status": "no_x" if "X" not in parent else "complete",
                "xr_selected_source": "parent_no_X" if "X" not in parent else "proposal_rank_0",
                "xr_ppl_safe": "true",
                "xr_ppl_ratio": "",
                "xr_composition_l1": "",
                "xr_trace": json.dumps(trace, ensure_ascii=False),
            }

        xr_baseline_rows.append(repair(baseline, ROUTE_IDS["xr_once"]))
        xr_selected_rows.append(repair(selection.selected_sequence, ROUTE_IDS["two_score_xr_once"]))

    if xr_selected is not None:
        extras = sorted(set(selected_artifact.strict) - used_selected_keys)
        if extras:
            raise ValueError(
                "XR selected artifact contains rows for unknown route parents: "
                f"{extras[:5]}" + (" ..." if len(extras) > 5 else "")
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / ROUTE_FILES["baseline"], baseline_rows)
    _write_csv(output_dir / ROUTE_FILES["two_score"], selected_rows)
    _write_csv(output_dir / ROUTE_FILES["xr_once"], xr_baseline_rows)
    _write_csv(output_dir / ROUTE_FILES["two_score_xr_once"], xr_selected_rows)
    _write_csv(output_dir / "candidate_scores_audited.csv", score_rows)
    manifest = {
        "protocol_version": "pepsite_routes_v1",
        "input_score_table": str(candidate_csv),
        "input_score_table_sha256": _sha256(candidate_csv),
        "query_count": len(grouped),
        "route_rows": {route: len(baseline_rows) for route in ROUTE_FILES},
        "two_score": {"terms": ["z_site_chemistry", "z_source_mean_logp"], "weights": [1.0, 1.0], "ppl_ratio_max": ppl_ratio_max, "composition_l1_max": composition_l1_max, "ppl_is_ranking_term": False, "fallback": "baseline"},
        "xr": {
            "passes": 1,
            "mutable_positions": "parent_X_only",
            "output_alphabet": "canonical_20",
            "non_X_invariant": True,
            "proposal_lookup": "(query_id,parent_route_id,parent_sequence,position)",
            "legacy_proposal_fallback": "only_when_baseline_equals_two_score_parent",
            "selected_artifact": str(xr_selected) if xr_selected is not None else None,
            "selected_artifact_preferred": xr_selected is not None,
        },
        "model_inference": {"performed_by_this_command": False, "required_external_artifacts": ["source_mean_logp", "site_chemistry", "source_ppl"], "checkpoint_loader": "caller-supplied; no checkpoint bundled"},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-csv", "--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--xr-proposals", type=Path, default=None)
    parser.add_argument(
        "--xr-selected",
        type=Path,
        default=None,
        help="model-backed gate-approved XR selections (preferred over proposal rank-0)",
    )
    parser.add_argument("--ppl-ratio-max", type=float, default=1.05)
    parser.add_argument("--composition-l1-max", type=float, default=0.75)
    args = parser.parse_args()
    if not 0 < args.ppl_ratio_max:
        parser.error("--ppl-ratio-max must be positive")
    result = generate(
        args.candidate_csv,
        args.output_dir,
        xr_proposals=args.xr_proposals,
        xr_selected=args.xr_selected,
        ppl_ratio_max=args.ppl_ratio_max,
        composition_l1_max=args.composition_l1_max,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
