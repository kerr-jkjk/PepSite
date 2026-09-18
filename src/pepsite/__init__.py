"""PepSite training, inference, and deterministic peptide routing utilities."""

from .constants import (
    CANONICAL_AMINO_ACIDS,
    COMPOSITION_L1_MAX,
    GENERATION_ALPHABET,
    PPL_RATIO_MAX,
)
from .routes import (
    Candidate,
    Selection,
    select_two_score,
    xr_repair_once,
    zscore,
)

__all__ = [
    "CANONICAL_AMINO_ACIDS",
    "COMPOSITION_L1_MAX",
    "GENERATION_ALPHABET",
    "PPL_RATIO_MAX",
    "Candidate",
    "Selection",
    "select_two_score",
    "xr_repair_once",
    "zscore",
    "masked_residue_ppl",
    "PPL_PROTOCOL_VERSION",
    "pseudo_ppl",
    "pseudo_ppl_pair",
]


def __getattr__(name: str):
    """Load the torch-backed PPL implementation only when it is requested."""

    if name in {"PPL_PROTOCOL_VERSION", "masked_residue_ppl", "pseudo_ppl", "pseudo_ppl_pair"}:
        from . import ppl as _ppl

        values = {
            "PPL_PROTOCOL_VERSION": _ppl.PROTOCOL_VERSION,
            "masked_residue_ppl": _ppl.masked_residue_ppl,
            "pseudo_ppl": _ppl.pseudo_ppl,
            "pseudo_ppl_pair": _ppl.pseudo_ppl_pair,
        }
        return values[name]
    raise AttributeError(name)
