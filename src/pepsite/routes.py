"""Pure sequence selection and proposal-only unknown-residue repair.

The route functions operate on score tables and a caller-supplied residue
proposer.  They do not import a checkpoint, read a machine-local path, or
silently fabricate model scores.  A production caller must provide scores
computed by the declared model backend (and can record the backend manifest
alongside the resulting CSV).  ``xr_repair_once`` is intentionally a small
proposal assembler for already ordered predictions; the strict model-backed
XR policy (all-mask reference, PPL/composition gates, and fallback) lives in
``pepsite.inference.xr_repair_one_round_model``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

from .constants import (
    CANONICAL_AMINO_ACIDS,
    CANONICAL_SET,
    COMPOSITION_L1_MAX,
    GENERATION_ALPHABET_SET,
    PPL_RATIO_MAX,
)


def _normalise(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"sequence must be a string, got {type(value).__name__}")
    result = "".join(value.split()).upper()
    if not result:
        raise ValueError("sequence must not be empty")
    return result


def zscore(values: Sequence[float]) -> list[float]:
    """Return population-standardized values, preserving input order.

    A constant vector maps to zeros.  This is the exact behavior needed for a
    per-query two-score ranking and avoids introducing a NaN tie-breaker.
    """

    numbers = [float(value) for value in values]
    if not numbers:
        return []
    mean = sum(numbers) / len(numbers)
    variance = sum((value - mean) ** 2 for value in numbers) / len(numbers)
    std = math.sqrt(variance)
    if not math.isfinite(std) or std < 1e-10:
        return [0.0] * len(numbers)
    return [(value - mean) / std for value in numbers]


def amino_acid_frequencies(sequence: str) -> dict[str, float]:
    """Return canonical residue frequencies (unknown ``X`` is ignored)."""

    sequence = _normalise(sequence)
    counts = {aa: 0 for aa in CANONICAL_AMINO_ACIDS}
    for residue in sequence:
        if residue in counts:
            counts[residue] += 1
    denominator = sum(counts.values())
    if denominator == 0:
        return counts
    return {aa: count / denominator for aa, count in counts.items()}


def composition_l1(reference: str, candidate: str) -> float:
    """Return L1 drift between canonical amino-acid compositions."""

    left = amino_acid_frequencies(reference)
    right = amino_acid_frequencies(candidate)
    return float(sum(abs(left[aa] - right[aa]) for aa in CANONICAL_AMINO_ACIDS))


@dataclass(frozen=True)
class Candidate:
    """One generated peptide and the scores required by two-score selection."""

    query_id: str
    sequence: str
    source_mean_logp: float
    site_chemistry: float
    source_ppl: float
    candidate_index: int = 0
    is_baseline: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "Candidate":
        """Construct from a CSV-like mapping with explicit field aliases."""

        def pick(*names: str, default: Any = None) -> Any:
            for name in names:
                if name in row:
                    return row[name]
            return default

        query_id = str(pick("query_id", "id", default="")).strip()
        sequence = _normalise(pick("sequence", "peptide", "generated_ligand_sequence"))
        source_logp = pick("source_mean_logp", "source_log_prob", "mlm_score")
        chemistry = pick("site_chemistry", "source_site_chemistry")
        source_ppl = pick("source_ppl", "ppl")
        if source_logp is None or chemistry is None or source_ppl is None:
            raise ValueError("candidate row requires source_mean_logp, site_chemistry and source_ppl")
        index = pick("candidate_index", "index", default=0)
        try:
            index_int = int(index)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"candidate_index must be an integer: {index!r}") from exc
        baseline = pick("is_baseline", "baseline", default=False)
        if isinstance(baseline, str):
            baseline = baseline.strip().lower() in {"1", "true", "yes", "y", "t"}
        known = {
            "query_id", "id", "sequence", "peptide", "generated_ligand_sequence",
            "source_mean_logp", "source_log_prob", "mlm_score", "site_chemistry",
            "source_site_chemistry", "source_ppl", "ppl", "candidate_index", "index",
            "is_baseline", "baseline",
        }
        return cls(
            query_id=query_id,
            sequence=sequence,
            source_mean_logp=float(source_logp),
            site_chemistry=float(chemistry),
            source_ppl=float(source_ppl),
            candidate_index=index_int,
            is_baseline=bool(baseline),
            metadata={str(k): value for k, value in row.items() if k not in known},
        )


@dataclass(frozen=True)
class Selection:
    """Deterministic two-score selection result plus audit rows."""

    query_id: str
    selected_sequence: str
    candidate_sequence: str
    baseline_sequence: str
    candidate_index: int
    two_score_utility: float
    z_site_chemistry: float
    z_source_mean_logp: float
    candidate_ppl: float
    baseline_ppl: float
    candidate_ppl_ratio: float
    selected_ppl_ratio: float
    ppl_safe: bool
    fallback: bool
    fallback_reason: str
    changed: bool
    scored: tuple[Mapping[str, Any], ...]


def _validate_candidate(candidate: Candidate, baseline: str, expected_length: int | None) -> tuple[bool, str]:
    if expected_length is not None and len(candidate.sequence) != expected_length:
        return False, "length_mismatch"
    if len(candidate.sequence) != len(baseline):
        return False, "length_mismatch"
    if set(candidate.sequence) - GENERATION_ALPHABET_SET:
        return False, "alphabet"
    if not math.isfinite(candidate.source_mean_logp) or not math.isfinite(candidate.site_chemistry):
        return False, "nonfinite_score"
    if not math.isfinite(candidate.source_ppl) or candidate.source_ppl <= 0:
        return False, "invalid_ppl"
    return True, ""


def select_two_score(
    candidates: Sequence[Candidate | Mapping[str, Any]],
    baseline_sequence: str | None = None,
    *,
    query_id: str | None = None,
    ppl_ratio_max: float = PPL_RATIO_MAX,
    composition_l1_max: float | None = COMPOSITION_L1_MAX,
) -> Selection:
    """Select one candidate using equal-weight z-scored chemistry and logP.

    The utility is strictly ``z(site_chemistry) + z(source_mean_logp)``.  PPL
    is evaluated only after ranking as an eligibility gate against the
    baseline PPL.  If the ranked candidate fails a gate, the baseline is
    returned; no PPL penalty or hidden third score is introduced.
    """

    parsed = [item if isinstance(item, Candidate) else Candidate.from_mapping(item) for item in candidates]
    if not parsed:
        raise ValueError("at least one candidate is required")
    qids = {item.query_id for item in parsed if item.query_id}
    if query_id is None:
        query_id = next(iter(qids), "")
    if qids and any(item.query_id and item.query_id != query_id for item in parsed):
        raise ValueError("all candidates must belong to one query")
    if baseline_sequence is None:
        marked = [item.sequence for item in parsed if item.is_baseline]
        if len(marked) != 1:
            raise ValueError("baseline_sequence is required unless exactly one candidate is_baseline")
        baseline_sequence = marked[0]
    baseline = _normalise(baseline_sequence)
    expected_length = len(baseline)
    if set(baseline) - GENERATION_ALPHABET_SET:
        raise ValueError("baseline contains symbols outside the generation alphabet")
    valid, reason = [], []
    for item in parsed:
        ok, why = _validate_candidate(item, baseline, expected_length)
        valid.append(ok)
        reason.append(why)
    if not any(valid):
        raise ValueError("candidate table contains no valid sequence")
    chemistry = zscore([item.site_chemistry for item in parsed])
    logp = zscore([item.source_mean_logp for item in parsed])
    utilities = [a + b for a, b in zip(chemistry, logp)]
    baseline_rows = [index for index, item in enumerate(parsed) if item.sequence == baseline]
    if len(baseline_rows) != 1:
        raise ValueError("baseline sequence must occur exactly once in candidate table")
    baseline_ppl = float(parsed[baseline_rows[0]].source_ppl)
    if not math.isfinite(baseline_ppl) or baseline_ppl <= 0:
        raise ValueError("baseline PPL must be finite and positive")
    scored: list[dict[str, Any]] = []
    for index, item in enumerate(parsed):
        ratio = float(item.source_ppl / baseline_ppl)
        ppl_safe = bool(math.isfinite(ratio) and ratio <= float(ppl_ratio_max))
        composition = composition_l1(baseline, item.sequence)
        gate_safe = bool(valid[index] and ppl_safe and (composition_l1_max is None or composition <= float(composition_l1_max)))
        scored.append({
            "query_id": item.query_id or query_id,
            "sequence": item.sequence,
            "candidate_index": int(item.candidate_index),
            "is_baseline": bool(item.is_baseline or item.sequence == baseline),
            "source_mean_logp": float(item.source_mean_logp),
            "site_chemistry": float(item.site_chemistry),
            "z_source_mean_logp": float(logp[index]),
            "z_site_chemistry": float(chemistry[index]),
            "two_score_utility": float(utilities[index]),
            "source_ppl": float(item.source_ppl),
            "baseline_source_ppl": baseline_ppl,
            "source_ppl_ratio": ratio,
            "ppl_safe": ppl_safe,
            "composition_l1": composition,
            "length_ok": bool(len(item.sequence) == expected_length),
            "alphabet_ok": bool(not (set(item.sequence) - GENERATION_ALPHABET_SET)),
            "safe": gate_safe,
            "invalid_reason": reason[index],
        })
    order = sorted(
        range(len(scored)),
        key=lambda index: (
            -float(scored[index]["two_score_utility"]),
            -float(scored[index]["source_mean_logp"]),
            int(scored[index]["candidate_index"]),
            str(scored[index]["sequence"]),
        ),
    )
    top_index = order[0]
    top = scored[top_index]
    candidate = str(top["sequence"])
    safe = bool(top["safe"])
    selected = candidate if safe else baseline
    selected_ppl = float(top["source_ppl"]) if safe else baseline_ppl
    if safe:
        fallback_reason = ""
    elif not bool(top["ppl_safe"]):
        fallback_reason = "ppl_ratio"
    elif reason[top_index]:
        fallback_reason = str(reason[top_index])
    elif composition_l1_max is not None and float(top["composition_l1"]) > float(composition_l1_max):
        fallback_reason = "composition_l1"
    else:
        fallback_reason = "eligibility_gate"
    scored_sorted = tuple(scored[index] | {"rank": rank + 1, "selected_top1": index == top_index} for rank, index in enumerate(order))
    return Selection(
        query_id=str(query_id or parsed[0].query_id),
        selected_sequence=selected,
        candidate_sequence=candidate,
        baseline_sequence=baseline,
        candidate_index=int(top["candidate_index"]),
        two_score_utility=float(top["two_score_utility"]),
        z_site_chemistry=float(top["z_site_chemistry"]),
        z_source_mean_logp=float(top["z_source_mean_logp"]),
        candidate_ppl=float(top["source_ppl"]),
        baseline_ppl=baseline_ppl,
        candidate_ppl_ratio=float(top["source_ppl_ratio"]),
        selected_ppl_ratio=float(selected_ppl / baseline_ppl),
        ppl_safe=safe,
        fallback=not safe,
        fallback_reason=fallback_reason,
        changed=selected != baseline,
        scored=scored_sorted,
    )


# A descriptive alias is useful for downstream scripts and keeps the public
# API independent of historical experiment names.
two_score_select = select_two_score


def _residue_from_prediction(value: Any) -> str:
    if isinstance(value, Mapping):
        for key in ("residue", "aa", "amino_acid", "token"):
            if key in value:
                return str(value[key]).strip().upper()
        raise ValueError("XR predictor mapping must contain residue/aa")
    if isinstance(value, (tuple, list)) and value and not isinstance(value, str):
        return str(value[0]).strip().upper()
    return str(value).strip().upper()


@dataclass(frozen=True)
class XRRepair:
    """Result and audit trace of one X-only coordinate sweep."""

    parent_sequence: str
    selected_sequence: str
    mutable_positions: tuple[int, ...]
    trace: tuple[Mapping[str, Any], ...]

    @property
    def changed(self) -> bool:
        return self.parent_sequence != self.selected_sequence


def xr_repair_once(
    parent_sequence: str,
    proposer: Callable[[str, int, int], Iterable[Any]],
    *,
    top_k: int = 3,
) -> XRRepair:
    """Assemble one ordered proposal sweep, preserving all non-X residues.

    ``proposer(context, position, top_k)`` must return candidates ordered from
    best to worst.  The first canonical residue is selected.  Unknown or
    special symbols are ignored; if no canonical proposal is available the
    function fails closed instead of changing a non-X position or emitting an
    invalid sequence.  This helper does not calculate PPL/composition gates;
    callers that have a checkpoint should use
    ``inference.xr_repair_one_round_model`` for the formal protocol.
    """

    parent = _normalise(parent_sequence)
    if set(parent) - GENERATION_ALPHABET_SET:
        raise ValueError("XR parent contains symbols outside the generation alphabet")
    mutable = tuple(index for index, residue in enumerate(parent) if residue == "X")
    current = list(parent)
    trace: list[Mapping[str, Any]] = []
    for position in mutable:
        context = "".join(current)
        predictions = list(proposer(context, position, int(top_k)))
        selected = None
        normalised_predictions: list[str] = []
        for value in predictions[: max(1, int(top_k))]:
            residue = _residue_from_prediction(value)
            normalised_predictions.append(residue)
            if len(residue) == 1 and residue in CANONICAL_SET:
                selected = residue
                break
        if selected is None:
            raise ValueError(f"XR proposer returned no canonical residue at position {position}")
        before = current[position]
        current[position] = selected
        trace.append({
            "position": int(position),
            "before": before,
            "chosen": selected,
            "context": context,
            "proposals": normalised_predictions,
        })
    result = "".join(current)
    if "X" in result or set(result) - CANONICAL_SET:
        raise RuntimeError("XR repair did not produce a canonical sequence")
    immutable = set(range(len(parent))) - set(mutable)
    if any(result[index] != parent[index] for index in immutable):
        raise RuntimeError("XR repair changed a non-X position")
    return XRRepair(parent, result, mutable, tuple(trace))


__all__ = [
    "Candidate",
    "Selection",
    "XRRepair",
    "amino_acid_frequencies",
    "composition_l1",
    "select_two_score",
    "two_score_select",
    "xr_repair_once",
    "zscore",
]
