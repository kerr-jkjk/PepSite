"""Deterministic masked-residue pseudo-perplexity.

The implementation evaluates one masked peptide residue at a time.  Candidate
and position chunk sizes only control packing of independent rows and therefore
cannot alter a score or its candidate association.  Logits are converted to
``float32`` before cross entropy and NLL accumulation uses ``float64`` to keep
the result stable across chunking choices.

No checkpoint is bundled with this repository.  For a real run, provide a
backend exposing ``tokenizer``, ``device`` and ``model.esm``/``model.lm_head``.
The model loader belongs to the caller because paths and licenses differ by
deployment.
"""

from __future__ import annotations

import math
import numbers
import re
from typing import Any, Sequence

import torch
import torch.nn.functional as F


PROTOCOL_VERSION = "masked_residue_ppl_v2_flat_row_mapping_float32"


def parse_candidate_index(value: Any, *, label: str = "candidate_index") -> int:
    """Parse a decimal index without lexicographic CSV surprises."""

    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer, not boolean")
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not re.fullmatch(r"[+-]?\d+", text):
            raise ValueError(f"{label} must be a decimal integer, got {value!r}")
        return int(text, 10)
    if isinstance(value, numbers.Real) and math.isfinite(float(value)) and float(value).is_integer():
        return int(value)
    raise ValueError(f"{label} must be a decimal integer, got {value!r}")


def parse_bool(value: Any, *, label: str = "boolean") -> bool:
    """Parse CSV/JSON booleans explicitly; never use ``bool(str)``."""

    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral) and int(value) in (0, 1):
        return bool(int(value))
    if isinstance(value, numbers.Real) and math.isfinite(float(value)) and float(value) in (0.0, 1.0):
        return bool(int(value))
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "t"}:
            return True
        if text in {"0", "false", "no", "n", "f"}:
            return False
    raise ValueError(f"{label} must be an explicit boolean, got {value!r}")


def _clean_sequence(value: str, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string, got {type(value).__name__}")
    sequence = "".join(value.split()).upper()
    if not sequence:
        raise ValueError(f"{label} must not be empty")
    return sequence


def _encode_pair(backend: Any, receptor: str, peptide: str) -> torch.Tensor:
    """Encode one residue-tokenized receptor/peptide pair.

    The reference ESM tokenizer adds BOS/EOS.  Rejecting a changed layout is
    safer than silently masking an incorrect position.  A custom backend may
    implement the same ``encode(..., return_tensors='pt')`` contract.
    """

    encoded = backend.tokenizer.encode(receptor + peptide, return_tensors="pt")
    if not isinstance(encoded, torch.Tensor) or encoded.ndim != 2 or int(encoded.shape[0]) != 1:
        raise ValueError("tokenizer.encode must return a [1, sequence_length] tensor")
    expected = len(receptor) + len(peptide) + 2
    if int(encoded.shape[1]) != expected:
        raise ValueError(
            "tokenizer changed residue layout: "
            f"expected {expected} tokens for {len(receptor)}+{len(peptide)} residues, "
            f"got {int(encoded.shape[1])}"
        )
    return encoded.squeeze(0).to(backend.device)


def _forward_logits(backend: Any, masked: torch.Tensor) -> torch.Tensor:
    model = backend.model
    encoder = getattr(model, "esm", None)
    if encoder is None:
        raise AttributeError("backend.model must expose an .esm encoder")
    outputs = encoder(
        input_ids=masked,
        attention_mask=torch.ones_like(masked),
        return_dict=True,
    )
    hidden = outputs.last_hidden_state
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        raise AttributeError("backend.model must expose an .lm_head")
    return lm_head(hidden).float()


def masked_residue_ppl(
    backend: Any,
    receptor: str,
    peptides: Sequence[str],
    *,
    candidate_chunk_size: int = 8,
    position_batch_size: int = 16,
) -> list[float]:
    """Return masked-residue pseudo-PPL in exactly the input peptide order.

    All peptides in a call must have equal length because each packed position
    batch shares target positions.  The flattened row contract is
    candidate-major then position-major, and targets use that same order.
    """

    if not peptides:
        return []
    receptor = _clean_sequence(receptor, label="receptor")
    peptides_clean = [_clean_sequence(item, label="peptide") for item in peptides]
    peptide_length = len(peptides_clean[0])
    if any(len(item) != peptide_length for item in peptides_clean):
        raise ValueError("all peptides must have equal length")
    candidate_chunk_size = max(1, int(candidate_chunk_size))
    position_batch_size = max(1, int(position_batch_size))
    mask_id = getattr(backend.tokenizer, "mask_token_id", None)
    if mask_id is None:
        raise AttributeError("tokenizer must expose mask_token_id")
    device = backend.device
    model = backend.model
    previous_training = bool(model.training)
    model.eval()
    output: list[float] = []
    try:
        with torch.no_grad():
            for candidate_start in range(0, len(peptides_clean), candidate_chunk_size):
                chunk = peptides_clean[candidate_start : candidate_start + candidate_chunk_size]
                encoded = torch.stack([_encode_pair(backend, receptor, item) for item in chunk], dim=0)
                sequence_length = int(encoded.shape[1])
                receptor_offset = 1 + len(receptor)
                nll_sum = torch.zeros(len(chunk), dtype=torch.float64, device=device)
                nll_count = 0
                for position_start in range(0, peptide_length, position_batch_size):
                    positions = list(range(position_start, min(peptide_length, position_start + position_batch_size)))
                    position_count = len(positions)
                    row_count = len(chunk) * position_count
                    masked = (
                        encoded[:, None, :]
                        .expand(len(chunk), position_count, sequence_length)
                        .reshape(row_count, sequence_length)
                        .clone()
                    )
                    rows = torch.arange(row_count, device=device, dtype=torch.long)
                    target_positions = torch.tensor(
                        [receptor_offset + pos for _candidate in chunk for pos in positions],
                        device=device,
                        dtype=torch.long,
                    )
                    masked[rows, target_positions] = int(mask_id)
                    logits = _forward_logits(backend, masked)
                    selected_logits = logits[rows, target_positions]
                    targets = encoded[
                        :,
                        receptor_offset + position_start : receptor_offset + position_start + position_count,
                    ].reshape(-1)
                    token_nll = F.cross_entropy(selected_logits, targets, reduction="none")
                    nll_sum += token_nll.reshape(len(chunk), position_count).double().sum(dim=1)
                    nll_count += position_count
                values = torch.exp(nll_sum / float(nll_count)).detach().cpu().tolist()
                output.extend(float(value) for value in values)
    finally:
        model.train(previous_training)
    if len(output) != len(peptides_clean):
        raise RuntimeError(f"PPL output length mismatch: {len(output)} != {len(peptides_clean)}")
    if any(not math.isfinite(value) or value <= 0.0 for value in output):
        raise ValueError("pseudo-PPL returned a non-finite or non-positive value")
    return output


def pseudo_ppl(backend: Any, receptor: str, peptides: Sequence[str], **kwargs: Any) -> list[float]:
    """Public alias for :func:`masked_residue_ppl`."""

    return masked_residue_ppl(backend, receptor, peptides, **kwargs)


def pseudo_ppl_pair(
    backend: Any,
    receptor: str,
    peptides: Sequence[str],
    *,
    position_batch_size: int = 16,
) -> list[float]:
    """Compatibility alias that uses the corrected implementation.

    This function intentionally has no special two-candidate indexing path.
    """

    return masked_residue_ppl(
        backend,
        receptor,
        peptides,
        candidate_chunk_size=max(1, len(peptides)),
        position_batch_size=position_batch_size,
    )


__all__ = [
    "PROTOCOL_VERSION",
    "parse_candidate_index",
    "parse_bool",
    "masked_residue_ppl",
    "pseudo_ppl",
    "pseudo_ppl_pair",
]
