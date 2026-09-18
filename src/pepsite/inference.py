"""Checkpoint-backed, label-free PepSite sequence inference.

The release repository deliberately keeps the model weights and benchmark
tables outside Git.  This module is the portable boundary between those
external artifacts and the deterministic route code: a caller supplies a
Phase-2 ``EsmForMaskedLMWithSiteHead`` checkpoint and a public context CSV,
and receives a candidate-score table plus one-round XR proposals.

No machine-local experiment module or workstation path is imported.  The only
model class used here is the self-contained class in
``training_harness.py`` and all pseudo-perplexities use the corrected FP32
implementation in ``ppl.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .constants import (
    CANONICAL_AMINO_ACIDS,
    CANONICAL_SET,
    GENERATION_ALPHABET,
    GENERATION_ALPHABET_SET,
    PPL_RATIO_MAX,
    COMPOSITION_L1_MAX,
)
from .ppl import PROTOCOL_VERSION as PPL_PROTOCOL_VERSION
from .ppl import masked_residue_ppl
from .training_harness import EsmForMaskedLMWithSiteHead


DEFAULT_PROPOSAL_COUNT = 64
DEFAULT_TOP_K = 3
DEFAULT_TEMPERATURE = 1.0
DEFAULT_SITE_THRESHOLD = 0.5


@dataclass(frozen=True)
class Backend:
    """Frozen model/tokenizer pair used by the inference functions."""

    model: Any
    tokenizer: Any
    device: torch.device


def _normalise(value: Any, *, label: str, alphabet: frozenset[str] = GENERATION_ALPHABET_SET) -> str:
    sequence = "".join(str(value).split()).upper()
    if not sequence:
        raise ValueError(f"{label} is empty")
    invalid = sorted(set(sequence) - alphabet)
    if invalid:
        raise ValueError(f"{label} contains unsupported symbols: {invalid}")
    return sequence


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"cannot encode {type(value).__name__}")


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_path(path: str | Path) -> str:
    """Hash a file or a sorted directory manifest.

    A directory digest includes relative names and each file digest, making it
    independent of filesystem traversal order.  Hashing is done only by the
    real CLI; unit tests can operate on an injected in-memory backend.
    """

    root = Path(path).expanduser().resolve()
    if root.is_file():
        return sha256_file(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    digest = hashlib.sha256()
    for child in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = child.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(sha256_file(child).encode("ascii"))
    return digest.hexdigest()


def stable_seed(seed: int, label: str) -> int:
    """Derive the protocol's reproducible 31-bit sub-seed.

    The modulo is part of the published sampling contract.  In particular,
    this is *not* a mask of the SHA-256 prefix: masking produces a different
    stream for many queries and would make an otherwise identical rerun use a
    different candidate bank.
    """

    digest = hashlib.sha256(f"{int(seed)}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def seed_everything(seed: int) -> None:
    """Seed CPU-side sampling without broadcasting state to other GPUs."""

    value = int(seed) & 0xFFFFFFFF
    os.environ["PYTHONHASHSEED"] = str(value)
    random.seed(value)
    np.random.seed(value)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(value)
    torch.set_rng_state(generator.get_state())
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device(spec: str) -> torch.device:
    requested = str(spec).strip().lower()
    if requested in {"", "auto"}:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is unavailable")
    return device


def load_backend(
    model_dir: str | Path,
    *,
    device: str = "auto",
    dtype: str = "float32",
) -> Backend:
    """Load one Phase-2 checkpoint with strict state-dict validation.

    ``local_files_only=True`` is intentional: downloading a checkpoint or
    tokenizer implicitly would make a public run non-reproducible.  The model
    is kept in FP32 by default; when CUDA FP16 is explicitly requested, the
    PPL function still promotes logits and accumulators to FP32/FP64.
    """

    path = Path(model_dir).expanduser().resolve()
    if not path.is_dir() or not (path / "config.json").is_file():
        raise FileNotFoundError(f"checkpoint directory/config.json is missing: {path}")
    requested = choose_device(device)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    loaded = EsmForMaskedLMWithSiteHead.from_pretrained(
        str(path), local_files_only=True, output_loading_info=True
    )
    if isinstance(loaded, tuple):
        model, loading_info = loaded
    else:  # defensive compatibility with older transformers releases
        model, loading_info = loaded, {}
    errors = {
        key: list(loading_info.get(key) or [])
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    }
    # The historical Phase-2 trainer registers the positive-class weight both
    # as a module buffer and inside ``BCEWithLogitsLoss``.  Transformers may
    # report those two non-parameter buffers as unexpected when the inference
    # model reconstructs the loss object from the checkpoint config.  They do
    # not affect inference (the site head logits are the only values consumed),
    # and the authoritative training audit explicitly allows these names.  All
    # other missing/unexpected/mismatched keys remain fatal so a wrong model
    # architecture or a partial checkpoint cannot pass silently.
    allowed_buffer_keys = {"site_pos_weight_tensor", "site_loss_fct.pos_weight"}
    unexpected_real = [key for key in errors["unexpected_keys"] if key not in allowed_buffer_keys]
    missing_real = [key for key in errors["missing_keys"] if key not in allowed_buffer_keys]
    errors["allowed_missing_keys"] = [key for key in errors["missing_keys"] if key in allowed_buffer_keys]
    errors["allowed_unexpected_keys"] = [key for key in errors["unexpected_keys"] if key in allowed_buffer_keys]
    if missing_real or unexpected_real or errors["mismatched_keys"] or errors["error_msgs"]:
        raise RuntimeError(f"checkpoint did not load strictly: {json.dumps(errors, default=str)}")
    if getattr(model.config, "model_type", None) != "esm":
        raise RuntimeError(f"expected ESM checkpoint, got model_type={getattr(model.config, 'model_type', None)!r}")
    if getattr(tokenizer, "mask_token_id", None) != getattr(model.config, "mask_token_id", None):
        raise RuntimeError("tokenizer/model mask_token_id mismatch")
    if int(getattr(tokenizer, "vocab_size", -1)) != int(getattr(model.config, "vocab_size", -2)):
        raise RuntimeError("tokenizer/model vocab_size mismatch")
    model.to(requested)
    mode = str(dtype).lower()
    if mode == "float16":
        if requested.type != "cuda":
            raise ValueError("float16 inference requires CUDA; use float32 on CPU")
        model.half()
    elif mode != "float32":
        raise ValueError(f"unsupported dtype={dtype!r}; use float32 or float16")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return Backend(model=model, tokenizer=tokenizer, device=requested)


def load_public_context(path: str | Path, *, expected_rows: int | None = None) -> pd.DataFrame:
    """Load only the label-free query context.

    The accepted schema is ``query_id,receptor_sequence,peptide_length``.
    Extra columns are rejected rather than merely ignored.  This makes the
    label-free boundary auditable: a public context exposes only the length
    needed to create masks and cannot accidentally carry a truth peptide or
    binding-site label into inference.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    frame = pd.read_csv(source, dtype=str, keep_default_na=False)
    required = {"query_id", "receptor_sequence", "peptide_length"}
    observed = set(frame.columns)
    if observed != required:
        missing = sorted(required - observed)
        unexpected = sorted(observed - required)
        raise ValueError(
            "context schema must be exactly "
            f"{sorted(required)}; missing={missing}, unexpected={unexpected}"
        )
    frame = frame.copy()
    frame["query_id"] = frame["query_id"].astype(str).str.strip()
    if frame["query_id"].eq("").any() or frame["query_id"].duplicated().any():
        raise ValueError("context query_id values must be non-empty and unique")
    frame["receptor_sequence"] = frame["receptor_sequence"].map(lambda value: _normalise(value, label="receptor_sequence"))
    frame["peptide_length"] = pd.to_numeric(frame["peptide_length"], errors="raise").astype(int)
    if (frame["peptide_length"] <= 0).any():
        raise ValueError("peptide_length must be positive")
    if expected_rows is not None and len(frame) != int(expected_rows):
        raise ValueError(f"context must contain {expected_rows} rows, got {len(frame)}")
    return frame


def _mask_positions(tokenizer: Any, encoded: Mapping[str, torch.Tensor], expected: int) -> torch.Tensor:
    positions = (encoded["input_ids"][0] == int(tokenizer.mask_token_id)).nonzero(as_tuple=False).flatten()
    if int(positions.numel()) != int(expected):
        raise RuntimeError(f"tokenizer mask layout mismatch: expected {expected}, got {int(positions.numel())}")
    return positions


@torch.no_grad()
def masked_logits(backend: Backend, receptor: str, peptide_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Return MLM logits at masked peptide positions and Site logits."""

    receptor = _normalise(receptor, label="receptor_sequence")
    length = int(peptide_length)
    if length <= 0:
        raise ValueError("peptide_length must be positive")
    masked = receptor + backend.tokenizer.mask_token * length
    encoded = backend.tokenizer(masked, return_tensors="pt")
    encoded = {key: value.to(backend.device) for key, value in encoded.items()}
    positions = _mask_positions(backend.tokenizer, encoded, length)
    outputs = backend.model.esm(**encoded, return_dict=True)
    hidden = outputs.last_hidden_state
    mlm_logits = backend.model.lm_head(hidden[0, positions]).float().cpu().numpy()
    site_head = backend.model.site_head(hidden[0, 1 : 1 + len(receptor)])
    site_logits = site_head.squeeze(-1).float().cpu().numpy()
    if len(site_logits) != len(receptor):
        raise RuntimeError("Site head/receptor length mismatch")
    return mlm_logits, site_logits


def canonical_token_ids(tokenizer: Any) -> list[int]:
    ids = [int(tokenizer.convert_tokens_to_ids(aa)) for aa in CANONICAL_AMINO_ACIDS]
    if len(set(ids)) != len(ids) or any(item < 0 for item in ids):
        raise RuntimeError("tokenizer does not expose 20 distinct canonical residue IDs")
    for aa, token_id in zip(CANONICAL_AMINO_ACIDS, ids):
        if str(tokenizer.convert_ids_to_tokens(token_id)).upper() != aa:
            raise RuntimeError(f"tokenizer residue round-trip failed for {aa!r}")
    return ids


def _decode_residue_ids(tokenizer: Any, token_ids: Sequence[int]) -> str:
    decoded = "".join(str(tokenizer.decode(list(map(int, token_ids)), skip_special_tokens=True)).split()).upper()
    return decoded


def sample_full_vocab_topk(
    logits: np.ndarray,
    tokenizer: Any,
    *,
    count: int,
    seed: int,
    top_k: int = DEFAULT_TOP_K,
    temperature: float = DEFAULT_TEMPERATURE,
    invalid_policy: str = "skip",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Draw independent per-position categorical samples from full-vocab top-3.

    Decoding is validated *after* sampling against canonical residues plus X;
    invalid proposal rows are rejected without resampling.  This preserves the
    full tokenizer vocabulary behavior while making the downstream alphabet
    explicit and auditable.
    """

    raw = torch.as_tensor(logits, dtype=torch.float32)
    if raw.ndim != 2 or int(raw.shape[0]) < 1 or int(raw.shape[1]) < 1:
        raise ValueError("logits must be a non-empty [length,vocabulary] matrix")
    policy = str(invalid_policy).lower()
    if policy not in {"skip", "raise"}:
        raise ValueError(f"invalid invalid_policy={invalid_policy!r}")
    k = min(max(1, int(top_k)), int(raw.shape[-1]))
    top_values, top_ids = torch.topk(raw, k=k, dim=-1)
    probabilities = torch.softmax(top_values / max(float(temperature), 1e-6), dim=-1)
    generator = torch.Generator(device="cpu").manual_seed(int(seed) & ((1 << 63) - 1))
    log_probs = torch.log_softmax(raw, dim=-1)
    positions = torch.arange(raw.shape[0], dtype=torch.long)
    accepted: dict[str, dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []
    for proposal_index in range(max(1, int(count))):
        choices = torch.multinomial(probabilities, 1, replacement=True, generator=generator).squeeze(-1)
        token_ids = top_ids[positions, choices]
        decoded = _decode_residue_ids(tokenizer, token_ids.tolist())
        invalid = sorted(set(decoded) - GENERATION_ALPHABET_SET)
        if len(decoded) != int(raw.shape[0]) or invalid:
            record = {
                "proposal_index": int(proposal_index),
                "sequence": decoded,
                "invalid_characters": invalid,
                "expected_length": int(raw.shape[0]),
                "reason": "non_residue_or_length",
                "sampling_seed": int(seed),
                "top_k": int(top_k),
                "temperature": float(temperature),
            }
            rejected.append(record)
            if policy == "raise":
                raise RuntimeError(f"invalid sampled sequence: {record}")
            continue
        score = float(log_probs[positions, token_ids].mean().item())
        candidate = {
            "sequence": decoded,
            "source_mean_logp": score,
            "proposal_index": int(proposal_index),
            "sampling_seed": int(seed),
            "top_k": int(top_k),
            "temperature": float(temperature),
            "sampling_vocab": "full_tokenizer_vocabulary",
        }
        old = accepted.get(decoded)
        if old is None or (score, -proposal_index) > (float(old["source_mean_logp"]), -int(old["proposal_index"])):
            accepted[decoded] = candidate
    values = sorted(
        accepted.values(),
        key=lambda row: (-float(row["source_mean_logp"]), int(row["proposal_index"]), str(row["sequence"])),
    )
    return values, rejected


def generate_candidate_bank(
    backend: Backend,
    context: pd.DataFrame,
    *,
    seed: int,
    proposal_count: int = DEFAULT_PROPOSAL_COUNT,
    top_k: int = DEFAULT_TOP_K,
    temperature: float = DEFAULT_TEMPERATURE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate one baseline-plus-proposal bank shared by every route arm."""

    if int(proposal_count) <= 0:
        raise ValueError("proposal_count must be positive")
    if int(top_k) != DEFAULT_TOP_K:
        raise ValueError("the release protocol fixes top_k=3")
    bank_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for item in context.itertuples(index=False):
        query_id = str(item.query_id)
        receptor = str(item.receptor_sequence)
        length = int(item.peptide_length)
        logits, _ = masked_logits(backend, receptor, length)
        baseline_values, _ = sample_full_vocab_topk(
            logits,
            backend.tokenizer,
            count=1,
            seed=stable_seed(seed, f"baseline:{query_id}"),
            top_k=top_k,
            temperature=temperature,
            invalid_policy="raise",
        )
        proposal_values, proposal_rejected = sample_full_vocab_topk(
            logits,
            backend.tokenizer,
            count=proposal_count,
            seed=stable_seed(seed, f"proposals:{query_id}"),
            top_k=top_k,
            temperature=temperature,
            invalid_policy="skip",
        )
        seen: set[str] = set()
        ordered: list[dict[str, Any]] = []
        for candidate in [baseline_values[0], *proposal_values]:
            sequence = _normalise(candidate["sequence"], label=f"candidate {query_id}")
            if len(sequence) != length or sequence in seen:
                continue
            seen.add(sequence)
            ordered.append(candidate)
            if len(ordered) >= int(proposal_count) + 1:
                break
        if not ordered:
            raise RuntimeError(f"no valid candidate for {query_id}")
        baseline = ordered[0]
        baseline_rows.append(
            {
                "query_id": query_id,
                "baseline_sequence": baseline["sequence"],
                "baseline_source_mean_logp": float(baseline["source_mean_logp"]),
                "baseline_sampling_seed": stable_seed(seed, f"baseline:{query_id}"),
            }
        )
        audit_rows.append(
            {
                "query_id": query_id,
                "baseline_sequence": baseline["sequence"],
                "candidate_count": len(ordered),
                "proposal_count": int(proposal_count),
                "invalid_decodes_rejected": len(proposal_rejected),
                "sampling_seed": stable_seed(seed, f"proposals:{query_id}"),
            }
        )
        for candidate_index, candidate in enumerate(ordered):
            sequence = str(candidate["sequence"])
            bank_rows.append(
                {
                    "query_id": query_id,
                    "receptor_sequence": receptor,
                    "peptide_length": length,
                    "sequence": sequence,
                    "source_mean_logp": float(candidate["source_mean_logp"]),
                    "candidate_index": int(candidate_index),
                    "proposal_index": int(candidate.get("proposal_index", -1)),
                    "is_baseline": bool(candidate_index == 0),
                    "contains_X": "X" in sequence,
                    "allowed_sequence_alphabet": GENERATION_ALPHABET,
                    "top_k": int(top_k),
                    "temperature": float(temperature),
                }
            )
    bank = pd.DataFrame(bank_rows)
    baseline = pd.DataFrame(baseline_rows)
    audit = pd.DataFrame(audit_rows)
    if len(baseline) != len(context) or bank.duplicated(["query_id", "sequence"]).any():
        raise RuntimeError("candidate bank coverage or uniqueness contract failed")
    return bank, baseline, audit


def _aa_frequency(sequence: str) -> np.ndarray:
    counts = np.zeros(len(CANONICAL_AMINO_ACIDS), dtype=np.float64)
    for residue in sequence:
        if residue in CANONICAL_SET:
            counts[CANONICAL_AMINO_ACIDS.index(residue)] += 1.0
    total = counts.sum()
    return counts / total if total else counts


def site_chemistry(receptor: str, peptide: str, site_mask: Sequence[float]) -> float:
    """Compute the historical source Site-chemistry score.

    Site positions are normalized weights.  The scalar is the canonical
    residue-frequency diagonal plus 0.25 times matching hydrophobic and
    charged fractions.  Unknown ``X`` residues are retained in the sequence
    but excluded from the canonical peptide denominator, exactly as in the
    audited implementation.
    """

    receptor = _normalise(receptor, label="receptor_sequence")
    peptide = _normalise(peptide, label="peptide")
    weights = np.clip(np.asarray(site_mask, dtype=np.float64), 0.0, None)
    if weights.size != len(receptor):
        raise ValueError("site mask/receptor length mismatch")
    if not float(weights.sum()):
        weights = np.ones(len(receptor), dtype=np.float64)
    weights /= float(weights.sum())
    receptor_frequency = np.zeros(len(CANONICAL_AMINO_ACIDS), dtype=np.float64)
    for residue, weight in zip(receptor, weights):
        if residue in CANONICAL_SET:
            receptor_frequency[CANONICAL_AMINO_ACIDS.index(residue)] += float(weight)
    peptide_frequency = _aa_frequency(peptide)
    diagonal = float(np.dot(receptor_frequency, peptide_frequency))
    hydrophobic = set("AILMFWVY")
    charged = set("DEKR")
    receptor_hydrophobic = sum(float(weight) for residue, weight in zip(receptor, weights) if residue in hydrophobic)
    receptor_charged = sum(float(weight) for residue, weight in zip(receptor, weights) if residue in charged)
    denominator = max(1, len(peptide))
    peptide_hydrophobic = sum(residue in hydrophobic for residue in peptide) / denominator
    peptide_charged = sum(residue in charged for residue in peptide) / denominator
    return diagonal + 0.25 * (receptor_hydrophobic * peptide_hydrophobic + receptor_charged * peptide_charged)


def _zscore(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return array
    scale = float(array.std(ddof=0))
    return np.zeros_like(array) if scale <= 1e-12 else (array - float(array.mean())) / scale


@torch.no_grad()
def source_site_prediction(
    backend: Backend,
    receptor: str,
    peptide_length: int,
    threshold: float = DEFAULT_SITE_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict source Site probabilities/mask from the all-mask context."""

    _, logits = masked_logits(backend, receptor, peptide_length)
    probabilities = (1.0 / (1.0 + np.exp(-np.clip(logits.astype(np.float64), -40.0, 40.0)))).astype(np.float32)
    return probabilities, (probabilities >= float(threshold)).astype(np.float32)


def score_candidate_bank(
    backend: Backend,
    bank: pd.DataFrame,
    baseline: pd.DataFrame,
    context: pd.DataFrame,
    *,
    ppl_ratio_max: float = PPL_RATIO_MAX,
    composition_l1_max: float | None = COMPOSITION_L1_MAX,
    site_threshold: float = DEFAULT_SITE_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute Site/logP/PPL scores and apply the equal-weight selector.

    PPL is the corrected masked-residue pseudo-PPL and is an eligibility gate,
    never a third ranking term.  Returned frames are the complete candidate
    score table, one selected row per query, and source Site diagnostics.
    """

    required_bank = {"query_id", "sequence", "source_mean_logp", "candidate_index"}
    if not required_bank <= set(bank.columns):
        raise ValueError(f"candidate bank missing columns {sorted(required_bank - set(bank.columns))}")
    baseline_index = baseline.set_index("query_id", drop=False)
    selected_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    site_rows: list[dict[str, Any]] = []
    for item in context.itertuples(index=False):
        query_id = str(item.query_id)
        receptor = str(item.receptor_sequence)
        length = int(item.peptide_length)
        group = bank[bank["query_id"].astype(str) == query_id].copy()
        if group.empty:
            raise ValueError(f"candidate bank has no rows for {query_id}")
        group = group.sort_values(["candidate_index", "sequence"], kind="mergesort").reset_index(drop=True)
        sequences = [_normalise(value, label=f"candidate {query_id}") for value in group["sequence"].tolist()]
        if any(len(sequence) != length for sequence in sequences):
            raise ValueError(f"candidate length mismatch for {query_id}")
        base_sequence = str(baseline_index.loc[query_id, "baseline_sequence"])
        if base_sequence not in sequences:
            raise ValueError(f"baseline sequence is absent from candidate bank for {query_id}")
        probabilities, site_mask = source_site_prediction(backend, receptor, length, site_threshold)
        source_ppl = masked_residue_ppl(
            backend,
            receptor,
            sequences,
            # Match the release reference's operational packing.  The
            # corrected implementation is chunk-invariant, but keeping this
            # value explicit makes an audit comparison straightforward.
            candidate_chunk_size=16,
            position_batch_size=16,
        )
        base_ppl = float(source_ppl[sequences.index(base_sequence)])
        chemistry = np.asarray([site_chemistry(receptor, sequence, site_mask) for sequence in sequences], dtype=np.float64)
        source_logp = pd.to_numeric(group["source_mean_logp"], errors="raise").to_numpy(float)
        z_chemistry = _zscore(chemistry)
        z_logp = _zscore(source_logp)
        utility = z_chemistry + z_logp
        drift = np.asarray([
            # Keep the same definition as routes.composition_l1 without
            # requiring a second parse of the full Candidate table.
            float(np.abs(_aa_frequency(base_sequence) - _aa_frequency(sequence)).sum())
            for sequence in sequences
        ], dtype=np.float64)
        ratios = np.asarray(source_ppl, dtype=np.float64) / max(base_ppl, 1e-12)
        safe = (ratios <= float(ppl_ratio_max)) & (
            np.ones_like(ratios, dtype=bool)
            if composition_l1_max is None
            else drift <= float(composition_l1_max)
        )
        enriched = group.assign(
            source_ppl=source_ppl,
            baseline_source_ppl=base_ppl,
            source_ppl_ratio=ratios,
            site_chemistry=chemistry,
            z_site_chemistry=z_chemistry,
            z_source_mean_logp=z_logp,
            two_score_utility=utility,
            composition_l1=drift,
            ppl_safe=ratios <= float(ppl_ratio_max),
            safe=safe,
        )
        ordered = enriched.sort_values(
            ["two_score_utility", "source_mean_logp", "candidate_index", "sequence"],
            ascending=[False, False, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        top = ordered.iloc[0]
        candidate = str(top["sequence"])
        is_safe = bool(top["safe"])
        chosen = candidate if is_safe else base_sequence
        if is_safe:
            fallback_reason = ""
        elif not bool(top["ppl_safe"]):
            fallback_reason = "ppl_ratio"
        else:
            fallback_reason = "composition_l1"
        selected_rows.append(
            {
                "query_id": query_id,
                "receptor_sequence": receptor,
                "selected_sequence": chosen,
                "candidate_sequence": candidate,
                "baseline_sequence": base_sequence,
                "candidate_index": int(top["candidate_index"]),
                "two_score_utility": float(top["two_score_utility"]),
                "z_site_chemistry": float(top["z_site_chemistry"]),
                "z_source_mean_logp": float(top["z_source_mean_logp"]),
                "candidate_ppl": float(top["source_ppl"]),
                "baseline_ppl": base_ppl,
                "candidate_ppl_ratio": float(top["source_ppl_ratio"]),
                "selected_ppl_ratio": float((top["source_ppl"] if is_safe else base_ppl) / max(base_ppl, 1e-12)),
                "ppl_safe": is_safe,
                "fallback": not is_safe,
                "fallback_reason": fallback_reason,
                "changed": chosen != base_sequence,
                "contains_X": "X" in chosen,
                "candidate_contains_X": "X" in candidate,
                "composition_l1": float(top["composition_l1"]),
            }
        )
        site_rows.append(
            {
                "query_id": query_id,
                "receptor_sequence": receptor,
                "receptor_length": len(receptor),
                "decoded_mask": "".join(str(int(value)) for value in site_mask),
                "decoded_site_count": int(site_mask.sum()),
                "site_probability_mean": float(probabilities.mean()),
                "site_threshold": float(site_threshold),
                "site_probabilities": json.dumps([float(value) for value in probabilities], separators=(",", ":")),
            }
        )
        for row in ordered.itertuples(index=False):
            row_dict = row._asdict()
            candidate_rows.append(
                {
                    "query_id": query_id,
                    "receptor_sequence": receptor,
                    "peptide_length": length,
                    "sequence": str(row_dict["sequence"]),
                    "baseline_sequence": base_sequence,
                    "candidate_index": int(row_dict["candidate_index"]),
                    "proposal_index": int(row_dict.get("proposal_index", -1)),
                    "is_baseline": bool(row_dict.get("is_baseline", False) or str(row_dict["sequence"]) == base_sequence),
                    "contains_X": "X" in str(row_dict["sequence"]),
                    "source_mean_logp": float(row_dict["source_mean_logp"]),
                    "source_ppl": float(row_dict["source_ppl"]),
                    "baseline_source_ppl": base_ppl,
                    "source_ppl_ratio": float(row_dict["source_ppl_ratio"]),
                    "site_chemistry": float(row_dict["site_chemistry"]),
                    "z_site_chemistry": float(row_dict["z_site_chemistry"]),
                    "z_source_mean_logp": float(row_dict["z_source_mean_logp"]),
                    "two_score_utility": float(row_dict["two_score_utility"]),
                    "composition_l1": float(row_dict["composition_l1"]),
                    "ppl_safe": bool(row_dict["ppl_safe"]),
                    "safe": bool(row_dict["safe"]),
                    "selected_top1": str(row_dict["sequence"]) == candidate,
                }
            )
    return pd.DataFrame(candidate_rows), pd.DataFrame(selected_rows), pd.DataFrame(site_rows)


@torch.no_grad()
def _topk_canonical(
    backend: Backend,
    receptor: str,
    parent: str,
    position: int,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """Return canonical top-k logits for one parent-X coordinate."""

    parent = _normalise(parent, label="XR parent")
    if not 0 <= int(position) < len(parent) or parent[int(position)] != "X":
        raise ValueError("XR position must point to an X in the parent")
    masked_parent = parent[: int(position)] + backend.tokenizer.mask_token + parent[int(position) + 1 :]
    encoded = backend.tokenizer(receptor + masked_parent, return_tensors="pt")
    encoded = {key: value.to(backend.device) for key, value in encoded.items()}
    positions = _mask_positions(backend.tokenizer, encoded, 1)
    hidden = backend.model.esm(**encoded, return_dict=True).last_hidden_state
    logits = backend.model.lm_head(hidden[0, positions[0]]).float().cpu()
    ids = canonical_token_ids(backend.tokenizer)
    order = sorted(range(len(ids)), key=lambda index: (-float(logits[ids[index]]), index))[: min(int(top_k), len(ids))]
    return [
        {"residue": CANONICAL_AMINO_ACIDS[index], "logit": float(logits[ids[index]]), "rank": rank}
        for rank, index in enumerate(order)
    ]


@torch.no_grad()
def _all_mask_canonical_topk(
    backend: Backend,
    receptor: str,
    length: int,
    top_k: int = DEFAULT_TOP_K,
) -> list[list[dict[str, Any]]]:
    """Return canonical top-k logits with every peptide position masked.

    XR uses this independent all-mask pass to construct its reference
    sequence.  It is deliberately separate from :func:`_topk_canonical`,
    whose one-coordinate context is used by the sequential conditional sweep.
    Keeping both contexts explicit prevents accidentally using a conditional
    rank-0 sequence as the PPL reference.
    """

    receptor = _normalise(receptor, label="receptor_sequence")
    length = int(length)
    if length <= 0:
        raise ValueError("XR peptide length must be positive")
    masked = receptor + backend.tokenizer.mask_token * length
    encoded = backend.tokenizer(masked, return_tensors="pt")
    encoded = {key: value.to(backend.device) for key, value in encoded.items()}
    positions = _mask_positions(backend.tokenizer, encoded, length)
    hidden = backend.model.esm(**encoded, return_dict=True).last_hidden_state
    logits = backend.model.lm_head(hidden[0, positions]).float().cpu()
    ids = canonical_token_ids(backend.tokenizer)
    limit = min(max(1, int(top_k)), len(ids))
    result: list[list[dict[str, Any]]] = []
    for row in logits:
        order = sorted(
            range(len(ids)),
            key=lambda index: (-float(row[ids[index]]), index),
        )[:limit]
        result.append(
            [
                {
                    "residue": CANONICAL_AMINO_ACIDS[index],
                    "logit": float(row[ids[index]]),
                    "rank": rank,
                }
                for rank, index in enumerate(order)
            ]
        )
    return result


def xr_repair_one_round_model(
    backend: Backend,
    receptor: str,
    parent: str,
    *,
    ppl_ratio_max: float = PPL_RATIO_MAX,
    composition_l1_max: float | None = COMPOSITION_L1_MAX,
    top_k: int = DEFAULT_TOP_K,
) -> tuple[str, dict[str, Any]]:
    """Run the strict one-round, X-only model-backed XR repair.

    The formal policy first builds an all-mask ``reference_sequence`` (top-1
    at every parent-X coordinate), then performs one sequential conditional
    sweep to obtain the primary candidate.  The primary is tried first,
    followed by the all-mask reference and each conditional rank-1/rank-2
    coordinate alternative.  Every candidate is accepted only when corrected
    FP32 masked-residue PPL and composition gates pass.  Thus an unsafe
    conditional top-1 cannot silently become the published sequence.

    Only positions that were ``X`` in ``parent`` are mutable.  A parent with no
    ``X`` is returned unchanged with a complete trace.  If no canonical
    candidate passes the gates, the function raises ``RuntimeError`` (fail
    closed) rather than returning an unsafe or partially repaired sequence.
    """

    receptor = _normalise(receptor, label="receptor_sequence")
    parent = _normalise(parent, label="XR parent")
    if set(parent) - GENERATION_ALPHABET_SET:
        raise ValueError("XR parent contains symbols outside the generation alphabet")
    if float(ppl_ratio_max) <= 0:
        raise ValueError("ppl_ratio_max must be positive")
    if composition_l1_max is not None and float(composition_l1_max) < 0:
        raise ValueError("composition_l1_max must be non-negative or None")
    mutable = [index for index, residue in enumerate(parent) if residue == "X"]
    if not mutable:
        return parent, {
            "parent_sequence": parent,
            "reference_sequence": parent,
            "mutable_positions": [],
            "reference_trace": [],
            "coordinate_trace": [],
            "attempts": [],
            "changed": False,
            "selected_source": "parent_no_X",
            "fallback": False,
            "xr_status": "no_x",
        }

    all_mask = _all_mask_canonical_topk(backend, receptor, len(parent), top_k=top_k)
    reference_chars = list(parent)
    reference_trace: list[dict[str, Any]] = []
    for position in mutable:
        proposals = all_mask[position]
        if not proposals:
            raise RuntimeError(f"XR all-mask predictor returned no canonical residue at position {position}")
        reference_chars[position] = str(proposals[0]["residue"])
        reference_trace.append(
            {
                "position": int(position),
                "before": "X",
                "chosen": reference_chars[position],
                "top_k": proposals,
            }
        )
    reference = "".join(reference_chars)

    # One sequential conditional sweep.  ``current`` retains X at unvisited
    # coordinates and incorporates each earlier rank-0 choice into the next
    # context, exactly as the formal protocol specifies.
    current = list(parent)
    coordinate_trace: list[dict[str, Any]] = []
    last_top: dict[int, list[dict[str, Any]]] = {}
    for position in mutable:
        before = current[position]
        context = "".join(current)
        proposals = _topk_canonical(backend, receptor, context, position, top_k=top_k)
        if not proposals:
            raise RuntimeError(f"XR conditional predictor returned no canonical residue at position {position}")
        current[position] = str(proposals[0]["residue"])
        last_top[position] = proposals
        coordinate_trace.append(
            {
                "sweep": 1,
                "position": int(position),
                "before": before,
                "chosen": current[position],
                "top_k": proposals,
                "masked_context": context[:position] + "<MASK>" + context[position + 1 :],
            }
        )
    primary = "".join(current)
    immutable = [index for index in range(len(parent)) if index not in mutable]
    for label, sequence in (("all-mask reference", reference), ("coordinate primary", primary)):
        if (
            len(sequence) != len(parent)
            or set(sequence) - CANONICAL_SET
            or any(sequence[index] != parent[index] for index in immutable)
        ):
            raise RuntimeError(f"XR {label} changed a non-X position or produced a non-canonical sequence")

    alternatives: list[tuple[str, str]] = [
        (primary, "coordinate_primary"),
        (reference, "all_mask_control"),
    ]
    for position in sorted(last_top):
        # rank 0 is already the primary coordinate choice; rank 1 and rank 2
        # are the formal alternatives.  Keep their rank labels for an audit
        # reader and preserve deterministic position ordering.
        for proposal in last_top[position][1:]:
            chars = list(primary)
            chars[position] = str(proposal["residue"])
            alternatives.append(
                ("".join(chars), f"coordinate_position_{position}_rank_{proposal['rank']}")
            )

    attempts: list[dict[str, Any]] = []
    for rank, (candidate, source) in enumerate(alternatives):
        valid = bool(candidate) and len(candidate) == len(parent) and not (set(candidate) - CANONICAL_SET)
        if not valid:
            attempts.append(
                {
                    "rank": int(rank),
                    "source": source,
                    "sequence": candidate,
                    "safe": False,
                    "reason": "alphabet_or_length",
                }
            )
            continue
        ppls = masked_residue_ppl(
            backend,
            receptor,
            [reference, candidate],
            candidate_chunk_size=2,
            position_batch_size=8,
        )
        if len(ppls) != 2 or not all(np.isfinite(value) and float(value) > 0 for value in ppls):
            attempts.append(
                {
                    "rank": int(rank),
                    "source": source,
                    "sequence": candidate,
                    "reference_ppl": float(ppls[0]) if ppls else None,
                    "candidate_ppl": float(ppls[1]) if len(ppls) > 1 else None,
                    "ppl_ratio": None,
                    "composition_l1": None,
                    "safe": False,
                    "reason": "invalid_ppl",
                }
            )
            continue
        ratio = float(float(ppls[1]) / float(ppls[0]))
        composition = float(np.abs(_aa_frequency(reference) - _aa_frequency(candidate)).sum())
        safe = bool(
            np.isfinite(ratio)
            and ratio <= float(ppl_ratio_max)
            and (composition_l1_max is None or composition <= float(composition_l1_max))
        )
        reason = ""
        if not safe:
            if not np.isfinite(ratio) or ratio > float(ppl_ratio_max):
                reason = "ppl_ratio"
            elif composition_l1_max is not None and composition > float(composition_l1_max):
                reason = "composition_l1"
            else:
                reason = "eligibility_gate"
        attempts.append(
            {
                "rank": int(rank),
                "source": source,
                "sequence": candidate,
                "reference_ppl": float(ppls[0]),
                "candidate_ppl": float(ppls[1]),
                "ppl_ratio": ratio,
                "composition_l1": composition,
                "safe": safe,
                "reason": reason,
            }
        )
        if safe:
            if "X" in candidate or any(candidate[index] != parent[index] for index in immutable):
                raise RuntimeError("XR selected candidate violated X-only contract")
            return candidate, {
                "parent_sequence": parent,
                "reference_sequence": reference,
                "mutable_positions": mutable,
                "reference_trace": reference_trace,
                "coordinate_trace": coordinate_trace,
                "attempts": attempts,
                "changed": candidate != parent,
                "selected_source": source,
                "fallback": rank > 0,
                "xr_status": "complete",
                "xr_ppl_safe": True,
                "xr_ppl_ratio": ratio,
                "xr_composition_l1": composition,
            }
    raise RuntimeError("XR failed closed: no canonical PPL-safe repair")


def generate_xr_proposals(
    backend: Backend,
    parents: pd.DataFrame,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> pd.DataFrame:
    """Generate proposals for every X in each route parent, one sweep only.

    ``parents`` must contain ``query_id``, ``route_id``, ``receptor_sequence``
    and ``parent_sequence``.  Route IDs are retained because the baseline and
    two-score parent can have different X contexts; sharing one proposal table
    between those contexts would be an invalid shortcut.
    """

    required = {"query_id", "route_id", "receptor_sequence", "parent_sequence"}
    missing = required - set(parents.columns)
    if missing:
        raise ValueError(f"XR parents missing columns {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for item in parents.itertuples(index=False):
        query_id = str(item.query_id)
        route_id = str(item.route_id)
        receptor = _normalise(item.receptor_sequence, label="receptor_sequence")
        parent = _normalise(item.parent_sequence, label="XR parent")
        if not any(residue == "X" for residue in parent):
            continue
        current = parent
        # The context is updated after each coordinate, documenting the
        # sequential one-round sweep used by inference-time XR.
        for position, residue in enumerate(parent):
            if residue != "X":
                continue
            proposals = _topk_canonical(backend, receptor, current, position, top_k)
            for proposal in proposals:
                rows.append(
                    {
                        "query_id": query_id,
                        "route_id": route_id,
                        "parent_sequence": parent,
                        "context_sequence": current,
                        "position": int(position),
                        "residue": str(proposal["residue"]),
                        "rank": int(proposal["rank"]),
                        "logit": float(proposal["logit"]),
                        "top_k": int(top_k),
                        "input_alphabet": GENERATION_ALPHABET,
                        "output_alphabet": CANONICAL_AMINO_ACIDS,
                        "round": 1,
                    }
                )
            current = current[:position] + proposals[0]["residue"] + current[position + 1 :]
    return pd.DataFrame(
        rows,
        columns=[
            "query_id", "route_id", "parent_sequence", "context_sequence", "position",
            "residue", "rank", "logit", "top_k", "input_alphabet", "output_alphabet", "round",
        ],
    )


def run_inference(
    *,
    model_dir: str | Path,
    context_path: str | Path,
    output_dir: str | Path,
    seed: int,
    device: str = "auto",
    dtype: str = "float32",
    proposal_count: int = DEFAULT_PROPOSAL_COUNT,
    top_k: int = DEFAULT_TOP_K,
    temperature: float = DEFAULT_TEMPERATURE,
    ppl_ratio_max: float = PPL_RATIO_MAX,
    composition_l1_max: float | None = COMPOSITION_L1_MAX,
    site_threshold: float = DEFAULT_SITE_THRESHOLD,
    expected_rows: int | None = None,
) -> dict[str, Any]:
    """Run model inference and write portable candidate/XR artifacts."""

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    context = load_public_context(context_path, expected_rows=expected_rows)
    backend = load_backend(model_dir, device=device, dtype=dtype)
    seed_everything(seed)
    bank, baseline, generation_audit = generate_candidate_bank(
        backend,
        context,
        seed=seed,
        proposal_count=proposal_count,
        top_k=top_k,
        temperature=temperature,
    )
    candidate_scores, selected, site_masks = score_candidate_bank(
        backend,
        bank,
        baseline,
        context,
        ppl_ratio_max=ppl_ratio_max,
        composition_l1_max=composition_l1_max,
        site_threshold=site_threshold,
    )
    parents = pd.concat(
        [
            baseline.assign(
                route_id="01_mlm_site",
                receptor_sequence=baseline["query_id"].map(context.set_index("query_id")["receptor_sequence"]),
            ).rename(columns={"baseline_sequence": "parent_sequence"})[
                ["query_id", "route_id", "receptor_sequence", "parent_sequence"]
            ],
            selected.assign(route_id="02_two_score").rename(columns={"selected_sequence": "parent_sequence"})[
                ["query_id", "route_id", "receptor_sequence", "parent_sequence"]
            ],
        ],
        ignore_index=True,
    )
    xr_proposals = generate_xr_proposals(backend, parents, top_k=top_k)

    # The proposal table is useful for inspection, but it is not the final XR
    # route: the formal policy compares the sequential primary against an
    # all-mask reference and gated alternatives.  Materialize that decision
    # here so downstream route assembly never has to guess which proposal was
    # accepted.
    xr_selected_rows: list[dict[str, Any]] = []
    xr_trace_rows: list[dict[str, Any]] = []
    output_route_for_parent = {
        "01_mlm_site": "03_mlm_site_xr1",
        "02_two_score": "04_two_score_xr1",
    }
    for item in parents.itertuples(index=False):
        query_id = str(item.query_id)
        parent_route = str(item.route_id)
        output_route = output_route_for_parent.get(parent_route)
        if output_route is None:
            raise RuntimeError(f"unsupported XR parent route: {parent_route!r}")
        receptor = str(item.receptor_sequence)
        parent = str(item.parent_sequence)
        try:
            chosen, trace = xr_repair_one_round_model(
                backend,
                receptor,
                parent,
                ppl_ratio_max=ppl_ratio_max,
                composition_l1_max=composition_l1_max,
                top_k=top_k,
            )
        except RuntimeError as exc:
            # A model-backed XR gate failure is recorded explicitly and leaves
            # no unsafe sequence for the assembler to consume.  Other runtime
            # errors (tokenizer/model contract failures) remain fatal.
            if not str(exc).startswith("XR failed closed:"):
                raise
            chosen = ""
            trace = {
                "parent_sequence": parent,
                "reference_sequence": "",
                "mutable_positions": [index for index, residue in enumerate(parent) if residue == "X"],
                "reference_trace": [],
                "coordinate_trace": [],
                "attempts": [],
                "changed": False,
                "selected_source": "",
                "fallback": False,
                "xr_status": "failed",
                "error": str(exc),
            }
        trace = dict(trace)
        trace.update({"query_id": query_id, "route_id": parent_route, "output_route_id": output_route})
        mutable_positions = list(trace.get("mutable_positions") or [])
        attempts = list(trace.get("attempts") or [])
        selected_attempt = next((attempt for attempt in attempts if attempt.get("safe")), None)
        xr_selected_rows.append(
            {
                "query_id": query_id,
                "route_id": parent_route,
                "output_route_id": output_route,
                "receptor_sequence": receptor,
                "parent_sequence": parent,
                "selected_sequence": chosen,
                "reference_sequence": str(trace.get("reference_sequence", "")),
                "selected_source": str(trace.get("selected_source", "")),
                "xr_status": str(trace.get("xr_status", "complete" if chosen else "failed")),
                "xr_changed": bool(chosen and chosen != parent),
                "xr_mutable_positions": len(mutable_positions),
                "xr_repair_hamming": sum(a != b for a, b in zip(parent, chosen)) if chosen else "",
                "xr_ppl_safe": trace.get("xr_ppl_safe", True if chosen else False),
                "xr_ppl_ratio": (selected_attempt or {}).get("ppl_ratio", ""),
                "xr_composition_l1": (selected_attempt or {}).get("composition_l1", ""),
            }
        )
        xr_trace_rows.append(trace)

    _atomic_csv(destination / "candidate_bank.csv", bank)
    _atomic_csv(destination / "baseline_sequences.csv", baseline)
    _atomic_csv(destination / "candidate_generation_audit.csv", generation_audit)
    _atomic_csv(destination / "candidate_scores.csv", candidate_scores)
    _atomic_csv(destination / "selected_two_score.csv", selected)
    _atomic_csv(destination / "source_site_masks.csv", site_masks)
    _atomic_csv(destination / "xr_parents.csv", parents)
    _atomic_csv(destination / "xr_proposals.csv", xr_proposals)
    _atomic_csv(destination / "xr_selected.csv", pd.DataFrame(xr_selected_rows))
    (destination / "xr_repair_trace.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n" for row in xr_trace_rows),
        encoding="utf-8",
    )
    _atomic_csv(destination / "test_context.csv", context)

    manifest = {
        "protocol_version": "pepsite_inference_v1",
        "status": (
            "complete"
            if not any(row["xr_status"] == "failed" for row in xr_selected_rows)
            else "complete_with_xr_failures"
        ),
        "seed": int(seed),
        "model": {"path": str(Path(model_dir).expanduser().resolve()), "sha256": sha256_path(model_dir)},
        "context": {
            "path": str(Path(context_path).expanduser().resolve()),
            "sha256": sha256_file(context_path),
            "rows": int(len(context)),
            "label_free_fields_used": ["query_id", "receptor_sequence", "peptide_length"],
        },
        "sampling": {
            "proposal_count": int(proposal_count),
            "top_k": int(top_k),
            "temperature": float(temperature),
            "sampling_vocab": "full_tokenizer_vocabulary",
            "decoded_alphabet": GENERATION_ALPHABET,
            "x_allowed_before_xr": True,
            "invalid_proposal_policy": "reject_without_resampling",
            "deduplicate_within_query": True,
        },
        "site_prediction": {"threshold": float(site_threshold), "context": "receptor + fully masked peptide"},
        "screening": {
            "components": ["source_site_chemistry", "source_mean_logp"],
            "weights": {"source_site_chemistry": 1.0, "source_mean_logp": 1.0},
            "normalization": "per_query_population_z_score",
            "ppl_protocol": PPL_PROTOCOL_VERSION,
            "ppl_ratio_max": float(ppl_ratio_max),
            "composition_l1_max": None if composition_l1_max is None else float(composition_l1_max),
            "ppl_is_ranking_term": False,
            "fallback": "baseline_sequence",
        },
        "xr": {
            "rounds": 1,
            "mutable_positions": "parent_X_only",
            "top_k": int(top_k),
            "proposal_alphabet": CANONICAL_AMINO_ACIDS,
            "non_X_positions_invariant": True,
            "route_specific_parents": True,
            "reference_context": "all_mask_top1",
            "primary_context": "sequential_conditional_top1",
            "candidate_order": [
                "coordinate_primary",
                "all_mask_control",
                "coordinate_position_<pos>_rank_1_or_2",
            ],
            "ppl_gate": float(ppl_ratio_max),
            "composition_gate": None if composition_l1_max is None else float(composition_l1_max),
            "failed_policy": "fail_closed",
        },
        "outputs": {
            "candidate_bank": "candidate_bank.csv",
            "candidate_scores": "candidate_scores.csv",
            "selected_two_score": "selected_two_score.csv",
            "xr_proposals": "xr_proposals.csv",
            "xr_selected": "xr_selected.csv",
            "xr_repair_trace": "xr_repair_trace.jsonl",
        },
        "rows": {
            "queries": int(len(context)),
            "candidate_rows": int(len(bank)),
            "score_rows": int(len(candidate_scores)),
            "xr_proposal_rows": int(len(xr_proposals)),
            "xr_selected_rows": int(len(xr_selected_rows)),
            "xr_failed_rows": int(sum(row["xr_status"] == "failed" for row in xr_selected_rows)),
        },
    }
    _atomic_json(destination / "inference_manifest.json", manifest)
    return manifest


__all__ = [
    "Backend",
    "DEFAULT_PROPOSAL_COUNT",
    "DEFAULT_TOP_K",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_SITE_THRESHOLD",
    "canonical_token_ids",
    "generate_candidate_bank",
    "xr_repair_one_round_model",
    "generate_xr_proposals",
    "load_backend",
    "load_public_context",
    "masked_logits",
    "run_inference",
    "sample_full_vocab_topk",
    "score_candidate_bank",
    "site_chemistry",
    "source_site_prediction",
    "stable_seed",
]
